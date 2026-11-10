"""Time distributions on ``y = log1p(gap seconds)`` and joint (time, mark) distributions.

Every density here is a density over ``y`` (nats per unit of ``log1p`` seconds),
the same variable and units as the Stage 1 head and the benchmark baselines, so
likelihoods are directly comparable. The density over seconds follows by the
change of variables ``log p(gap) = log p(y) - y``.

Time families
-------------
``LogNormalMixture``   Gaussian mixture on ``y`` (the Stage 1 / baseline head).
``SplineFlow``         conditional monotone rational-quadratic spline (Durkan et
                       al. 2019) from ``y`` to a standard normal.
``PiecewiseHazard``    piecewise-constant hazard on ``y >= 0`` (a survival
                       decoder: exact cumulative hazard, survival and sampling).

Joint families (``p(y, k | H)``)
-------------------------------
``IndependentJoint``   ``f(y|H) pi(k|H)``
``TimeFirstJoint``     ``f_A(y|H) pi_A(k|y,H)``  (when -> what)
``EventFirstJoint``    ``pi_B(k|H) f_B(y|k,H)``  (what -> when)
``MixtureJoint``       ``1/2 p_A + 1/2 p_B``     (the dual decoder at inference)

Marginals that need an integral over time (``p_A(k|H)``) use exact bin masses
from CDF differences on a fixed ``y`` grid with the tails folded into the end
bins, so the discretised distributions are proper probability tables.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

LOG_2 = math.log(2.0)
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
#: Bisection bracket for quantiles, in y units (log1p seconds). Gaps above
#: e^21 s (~40 years) or below 0 do not occur; the bracket only has to contain them.
Y_LOW, Y_HIGH = -4.0, 24.0


def default_grid(bins: int = 48, low: float = 0.0, high: float = 19.2, device=None) -> torch.Tensor:
    """Bin edges in y used for time marginals, consistency and mean integrals."""

    return torch.linspace(low, high, bins + 1, device=device)


def _log_ndtr(value: torch.Tensor) -> torch.Tensor:
    return torch.special.log_ndtr(value)


class TimeDist:
    """Batched distribution over y. ``y`` may carry one extra trailing (grid) dim."""

    batch_dims: int

    def _grid(self, y: torch.Tensor) -> bool:
        if y.dim() == self.batch_dims:
            return False
        if y.dim() == self.batch_dims + 1:
            return True
        raise ValueError("y has {} dims for a batch of {}".format(y.dim(), self.batch_dims))

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cdf(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_survival(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        raise NotImplementedError

    def quantile(self, p: torch.Tensor) -> torch.Tensor:
        return bisect_quantile(self.cdf, p, self.batch_shape, self.device)

    def bin_masses(self, edges: torch.Tensor) -> torch.Tensor:
        """Probability of each grid bin, tails folded into the end bins. (..., Q)."""

        values = edges.expand(*self.batch_shape, edges.shape[-1])
        cdf = self.cdf(values)
        masses = cdf[..., 1:] - cdf[..., :-1]
        masses = masses.clamp(min=0.0)
        masses[..., 0] = masses[..., 0] + cdf[..., 0]
        masses[..., -1] = masses[..., -1] + (1.0 - cdf[..., -1]).clamp(min=0.0)
        return masses


def bisect_quantile(cdf: Callable, p, batch_shape, device, iterations: int = 40) -> torch.Tensor:
    """Generic quantile by bisection on a monotone CDF over [Y_LOW, Y_HIGH]."""

    if not torch.is_tensor(p):
        p = torch.full(batch_shape, float(p), device=device)
    low = torch.full_like(p, Y_LOW)
    high = torch.full_like(p, Y_HIGH)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        below = cdf(middle) < p
        low = torch.where(below, middle, low)
        high = torch.where(below, high, middle)
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Time families
# ---------------------------------------------------------------------------


class LogNormalMixture(TimeDist):
    """Gaussian mixture on y = log1p(gap): a log-normal-type mixture on gaps."""

    def __init__(self, logits: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor):
        self.log_weights = torch.log_softmax(logits.float(), dim=-1)
        self.means = means.float()
        self.log_scales = log_scales.float().clamp(-6.0, 4.0)
        self.batch_shape = tuple(logits.shape[:-1])
        self.batch_dims = len(self.batch_shape)
        self.device = logits.device

    def _params(self, y):
        if self._grid(y):
            return (
                y.unsqueeze(-1),
                self.log_weights.unsqueeze(-2),
                self.means.unsqueeze(-2),
                self.log_scales.unsqueeze(-2),
            )
        return y.unsqueeze(-1), self.log_weights, self.means, self.log_scales

    def log_prob(self, y):
        y, lw, mu, ls = self._params(y.float())
        z = (y - mu) * torch.exp(-ls)
        return torch.logsumexp(lw - HALF_LOG_2PI - ls - 0.5 * z * z, dim=-1)

    def cdf(self, y):
        y, lw, mu, ls = self._params(y.float())
        z = (y - mu) * torch.exp(-ls)
        return (torch.exp(lw) * torch.special.ndtr(z)).sum(-1).clamp(0.0, 1.0)

    def log_survival(self, y):
        y, lw, mu, ls = self._params(y.float())
        z = (y - mu) * torch.exp(-ls)
        return torch.logsumexp(lw + _log_ndtr(-z), dim=-1)

    def mean(self):
        return (torch.exp(self.log_weights) * self.means).sum(-1)

    def sample(self, generator=None):
        component = torch.multinomial(
            torch.exp(self.log_weights).reshape(-1, self.log_weights.shape[-1]),
            1,
            generator=generator,
        ).reshape(self.batch_shape + (1,))
        mu = self.means.gather(-1, component).squeeze(-1)
        sigma = torch.exp(self.log_scales.gather(-1, component).squeeze(-1))
        noise = torch.randn(mu.shape, device=mu.device, generator=generator)
        return mu + sigma * noise


# -- rational-quadratic spline (Durkan et al. 2019), ported from nflows (MIT) --

_MIN_BIN = 1e-3
_MIN_DERIVATIVE = 1e-3


def _searchsorted(edges: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    edges = edges.clone()
    edges[..., -1] += 1e-6
    return torch.sum(inputs[..., None] >= edges, dim=-1) - 1


def rq_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    bound: float,
    inverse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Monotone RQ spline on [-bound, bound], identity outside. Returns (out, log|d out/d in|).

    ``widths``/``heights`` (..., K) and ``derivatives`` (..., K-1) are unnormalised.
    """

    inside = (inputs >= -bound) & (inputs <= bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    if not inside.any():
        return outputs, logabsdet
    x = inputs[inside]
    w, h, d = widths[inside], heights[inside], derivatives[inside]
    bins = w.shape[-1]
    constant = math.log(math.exp(1.0 - _MIN_DERIVATIVE) - 1.0)
    d = F.pad(d, (1, 1), value=constant)
    w = torch.softmax(w, dim=-1)
    w = _MIN_BIN + (1.0 - _MIN_BIN * bins) * w
    cumwidths = F.pad(torch.cumsum(w, dim=-1), (1, 0), value=0.0)
    cumwidths = 2.0 * bound * cumwidths - bound
    cumwidths[..., 0], cumwidths[..., -1] = -bound, bound
    w = cumwidths[..., 1:] - cumwidths[..., :-1]
    d = _MIN_DERIVATIVE + F.softplus(d)
    h = torch.softmax(h, dim=-1)
    h = _MIN_BIN + (1.0 - _MIN_BIN * bins) * h
    cumheights = F.pad(torch.cumsum(h, dim=-1), (1, 0), value=0.0)
    cumheights = 2.0 * bound * cumheights - bound
    cumheights[..., 0], cumheights[..., -1] = -bound, bound
    h = cumheights[..., 1:] - cumheights[..., :-1]

    index = (_searchsorted(cumheights if inverse else cumwidths, x)).clamp(0, bins - 1)[..., None]
    in_cumwidths = cumwidths.gather(-1, index)[..., 0]
    in_bin_widths = w.gather(-1, index)[..., 0]
    in_cumheights = cumheights.gather(-1, index)[..., 0]
    delta = h / w
    in_delta = delta.gather(-1, index)[..., 0]
    in_derivatives = d.gather(-1, index)[..., 0]
    in_derivatives_plus = d[..., 1:].gather(-1, index)[..., 0]
    in_heights = h.gather(-1, index)[..., 0]

    if inverse:
        a = (x - in_cumheights) * (in_derivatives + in_derivatives_plus - 2 * in_delta) + in_heights * (
            in_delta - in_derivatives
        )
        b = in_heights * in_derivatives - (x - in_cumheights) * (
            in_derivatives + in_derivatives_plus - 2 * in_delta
        )
        c = -in_delta * (x - in_cumheights)
        discriminant = (b.pow(2) - 4 * a * c).clamp(min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        out = root * in_bin_widths + in_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = in_delta + (
            (in_derivatives + in_derivatives_plus - 2 * in_delta) * theta_one_minus_theta
        )
        derivative_numerator = in_delta.pow(2) * (
            in_derivatives_plus * root.pow(2)
            + 2 * in_delta * theta_one_minus_theta
            + in_derivatives * (1 - root).pow(2)
        )
        lad = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        outputs[inside] = out
        logabsdet[inside] = -lad
        return outputs, logabsdet

    theta = (x - in_cumwidths) / in_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = in_heights * (in_delta * theta.pow(2) + in_derivatives * theta_one_minus_theta)
    denominator = in_delta + (
        (in_derivatives + in_derivatives_plus - 2 * in_delta) * theta_one_minus_theta
    )
    out = in_cumheights + numerator / denominator
    derivative_numerator = in_delta.pow(2) * (
        in_derivatives_plus * theta.pow(2)
        + 2 * in_delta * theta_one_minus_theta
        + in_derivatives * (1 - theta).pow(2)
    )
    lad = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    outputs[inside] = out
    logabsdet[inside] = lad
    return outputs, logabsdet


class SplineFlow(TimeDist):
    """y -> u = (y - center)/scale -> v = RQS_theta(u) ~ N(0, 1).

    The spline is the identity outside [-bound, bound], so the tails are
    Gaussian in u; at initialisation (uniform bins, unit derivatives) the
    density is N(center, scale^2) on y.
    """

    center, scale, bound = 11.0, 3.5, 5.0

    def __init__(self, widths: torch.Tensor, heights: torch.Tensor, derivatives: torch.Tensor):
        self.widths, self.heights, self.derivatives = widths.float(), heights.float(), derivatives.float()
        self.batch_shape = tuple(widths.shape[:-1])
        self.batch_dims = len(self.batch_shape)
        self.device = widths.device

    def _expand(self, y):
        if self._grid(y):
            q = y.shape[-1]
            expand = lambda t: t.unsqueeze(-2).expand(*t.shape[:-1], q, t.shape[-1])  # noqa: E731
            return y, expand(self.widths), expand(self.heights), expand(self.derivatives)
        return y, self.widths, self.heights, self.derivatives

    def _forward(self, y):
        y, w, h, d = self._expand(y.float())
        u = (y - self.center) / self.scale
        v, lad = rq_spline(u, w, h, d, self.bound)
        return v, lad

    def log_prob(self, y):
        v, lad = self._forward(y)
        return -HALF_LOG_2PI - 0.5 * v * v + lad - math.log(self.scale)

    def cdf(self, y):
        v, _ = self._forward(y)
        return torch.special.ndtr(v)

    def log_survival(self, y):
        v, _ = self._forward(y)
        return _log_ndtr(-v)

    def _inverse(self, v):
        y, w, h, d = self._expand(v)
        u, _ = rq_spline(y, w, h, d, self.bound, inverse=True)
        return self.center + self.scale * u

    def quantile(self, p):
        if not torch.is_tensor(p):
            p = torch.full(self.batch_shape, float(p), device=self.device)
        return self._inverse(torch.special.ndtri(p.clamp(1e-6, 1 - 1e-6)))

    def mean(self, points: int = 64):
        levels = (torch.arange(points, device=self.device, dtype=torch.float32) + 0.5) / points
        grid = levels.expand(*self.batch_shape, points)
        return self._inverse(torch.special.ndtri(grid)).mean(-1)

    def sample(self, generator=None):
        noise = torch.randn(self.batch_shape, device=self.device, generator=generator)
        return self._inverse(noise)


class PiecewiseHazard(TimeDist):
    """Piecewise-constant hazard on y >= 0 with fixed edges; the last bin is open.

    ``Lambda(y) = sum_{m<b} r_m w_m + r_b (y - e_b)``, ``log f = log r_b - Lambda``,
    ``S = exp(-Lambda)``. Negative y (never a valid gap) is clamped to 0.
    """

    def __init__(self, raw_rates: torch.Tensor, edges: torch.Tensor):
        self.rates = F.softplus(raw_rates.float()) + 1e-6  # (..., M+1)
        self.edges = edges.float()  # (M+1,) with edges[0] = 0
        widths = self.edges[1:] - self.edges[:-1]
        self.cum = F.pad(torch.cumsum(self.rates[..., :-1] * widths, dim=-1), (1, 0), value=0.0)
        self.batch_shape = tuple(raw_rates.shape[:-1])
        self.batch_dims = len(self.batch_shape)
        self.device = raw_rates.device

    def _locate(self, y):
        grid = self._grid(y)
        y = y.float().clamp(min=0.0)
        index = (torch.bucketize(y, self.edges, right=True) - 1).clamp(0, self.edges.shape[0] - 1)
        rates, cum = self.rates, self.cum
        if grid:
            rates, cum = rates.unsqueeze(-2).expand(*y.shape, -1), cum.unsqueeze(-2).expand(*y.shape, -1)
        rate = rates.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        base = cum.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        return y, rate, base + rate * (y - self.edges[index])

    def log_prob(self, y):
        _, rate, hazard = self._locate(y)
        return torch.where(y.float() < 0.0, torch.full_like(hazard, -1e4), torch.log(rate) - hazard)

    def cdf(self, y):
        _, _, hazard = self._locate(y)
        return -torch.expm1(-hazard)

    def log_survival(self, y):
        _, _, hazard = self._locate(y)
        return -hazard

    def mean(self):
        widths = self.edges[1:] - self.edges[:-1]
        inner = torch.exp(-self.cum[..., :-1]) * (-torch.expm1(-self.rates[..., :-1] * widths)) / self.rates[..., :-1]
        tail = torch.exp(-self.cum[..., -1]) / self.rates[..., -1]
        return self.edges[0] + inner.sum(-1) + tail

    def _invert(self, hazard):
        position = torch.searchsorted(
            self.cum.contiguous(), hazard.unsqueeze(-1).contiguous(), right=True
        )
        index = (position.squeeze(-1) - 1).clamp(0, self.edges.shape[0] - 1)
        rate = self.rates.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        base = self.cum.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        return self.edges[index] + (hazard - base) / rate

    def quantile(self, p):
        if not torch.is_tensor(p):
            p = torch.full(self.batch_shape, float(p), device=self.device)
        return self._invert(-torch.log1p(-p.clamp(0.0, 1 - 1e-7)))

    def sample(self, generator=None):
        exponential = -torch.log(torch.rand(self.batch_shape, device=self.device, generator=generator).clamp(min=1e-12))
        return self._invert(exponential)


class DiscreteBins(TimeDist):
    """Categorical over gap bins with uniform density inside each bin (Kairos-style).

    ``edges`` are bin edges in y (strictly increasing, first 0); the last bin is
    closed at edges[-1]. Density in y is mass / width.
    """

    def __init__(self, logits: torch.Tensor, edges: torch.Tensor):
        self.log_masses = torch.log_softmax(logits.float(), dim=-1)
        self.edges = edges.float()
        self.widths = (self.edges[1:] - self.edges[:-1]).clamp(min=1e-6)
        self.batch_shape = tuple(logits.shape[:-1])
        self.batch_dims = len(self.batch_shape)
        self.device = logits.device
        cdf = torch.cumsum(torch.exp(self.log_masses), dim=-1)
        self.cdf_edges = F.pad(cdf, (1, 0), value=0.0).clamp(max=1.0)

    def _locate(self, y):
        grid = self._grid(y)
        y = y.float().clamp(self.edges[0], self.edges[-1] - 1e-6)
        index = (torch.bucketize(y, self.edges, right=True) - 1).clamp(0, self.widths.shape[0] - 1)
        log_masses, cdf_edges = self.log_masses, self.cdf_edges
        if grid:
            log_masses = log_masses.unsqueeze(-2).expand(*y.shape, -1)
            cdf_edges = cdf_edges.unsqueeze(-2).expand(*y.shape, -1)
        return y, index, log_masses, cdf_edges

    def log_prob(self, y):
        _, index, log_masses, _ = self._locate(y)
        value = log_masses.gather(-1, index.unsqueeze(-1)).squeeze(-1) - torch.log(self.widths[index])
        outside = (y.float() < self.edges[0]) | (y.float() > self.edges[-1])
        return torch.where(outside, torch.full_like(value, -1e4), value)

    def cdf(self, y):
        y, index, log_masses, cdf_edges = self._locate(y)
        base = cdf_edges.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        mass = torch.exp(log_masses.gather(-1, index.unsqueeze(-1)).squeeze(-1))
        return (base + mass * (y - self.edges[index]) / self.widths[index]).clamp(0.0, 1.0)

    def log_survival(self, y):
        return torch.log1p(-self.cdf(y).clamp(max=1 - 1e-7))

    def mean(self):
        centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        return (torch.exp(self.log_masses) * centres).sum(-1)

    def sample(self, generator=None):
        flat = torch.exp(self.log_masses).reshape(-1, self.log_masses.shape[-1])
        index = torch.multinomial(flat, 1, generator=generator).reshape(self.batch_shape)
        offset = torch.rand(self.batch_shape, device=self.device, generator=generator)
        return self.edges[index] + offset * self.widths[index]


# ---------------------------------------------------------------------------
# Joint (time, mark) distributions
# ---------------------------------------------------------------------------


class JointDist:
    """p(y, k | H) for a flat batch of N histories."""

    num_marks: int

    def log_prob(self, y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mark_log_probs(self) -> torch.Tensor:
        raise NotImplementedError

    def time_log_prob(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def time_cdf(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def time_log_survival(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def time_mean(self) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def mark_log_probs_given_time(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def discretized(self, edges: torch.Tensor) -> torch.Tensor:
        """P(bin q, mark k): (N, Q, K), a proper probability table."""

        raise NotImplementedError

    def time_quantile(self, p) -> torch.Tensor:
        batch = (self.batch_size,)
        return bisect_quantile(self.time_cdf, p, batch, self.device)

    def residual_log_prob(self, y_total, y_elapsed, k, y_residual) -> torch.Tensor:
        """Log density (in log1p-residual units) of the next event given no event for ``s``.

        ``p(v, k | gap > s) = p(gap = s + v, k) / S(s)``; the y-unit density
        converts with the Jacobians ``-y_total`` and ``+y_residual``.
        """

        return (
            self.log_prob(y_total, k)
            - self.time_log_survival(y_elapsed)
            - y_total.float()
            + y_residual.float()
        )


def _gather(log_probs: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return log_probs.gather(-1, k.long().clamp(min=0).unsqueeze(-1)).squeeze(-1)


class IndependentJoint(JointDist):
    def __init__(self, time: TimeDist, mark_logits: torch.Tensor):
        self.time = time
        self.mark = torch.log_softmax(mark_logits.float(), dim=-1)
        self.num_marks = mark_logits.shape[-1]
        self.batch_size = mark_logits.shape[0]
        self.device = mark_logits.device

    def log_prob(self, y, k):
        return self.time.log_prob(y) + _gather(self.mark, k)

    def mark_log_probs(self):
        return self.mark

    def time_log_prob(self, y):
        return self.time.log_prob(y)

    def time_cdf(self, y):
        return self.time.cdf(y)

    def time_log_survival(self, y):
        return self.time.log_survival(y)

    def time_mean(self):
        return self.time.mean()

    def time_quantile(self, p):
        return self.time.quantile(p)

    def sample(self, generator=None):
        y = self.time.sample(generator)
        k = torch.multinomial(torch.exp(self.mark), 1, generator=generator).squeeze(-1)
        return y, k

    def mark_log_probs_given_time(self, y):
        return self.mark

    def discretized(self, edges):
        masses = self.time.bin_masses(edges)
        return masses.unsqueeze(-1) * torch.exp(self.mark).unsqueeze(-2)


class TimeFirstJoint(JointDist):
    """f_A(y|H) pi_A(k|y,H); ``mark_fn(y)`` maps (N,) or (N,Q) times to logits."""

    def __init__(self, time: TimeDist, mark_fn: Callable, num_marks: int, edges: torch.Tensor):
        self.time = time
        self.mark_fn = mark_fn
        self.num_marks = num_marks
        self.edges = edges
        self.batch_size = time.batch_shape[0]
        self.device = edges.device
        self._marginal = None

    def log_prob(self, y, k):
        return self.time.log_prob(y) + _gather(torch.log_softmax(self.mark_fn(y).float(), -1), k)

    def mark_log_probs_given_time(self, y):
        return torch.log_softmax(self.mark_fn(y).float(), dim=-1)

    def discretized(self, edges):
        masses = self.time.bin_masses(edges)  # (N, Q)
        mids = 0.5 * (edges[1:] + edges[:-1])
        logits = self.mark_fn(mids.expand(self.batch_size, mids.shape[0]))  # (N, Q, K)
        return masses.unsqueeze(-1) * torch.softmax(logits.float(), dim=-1)

    def mark_log_probs(self):
        if self._marginal is None:
            table = self.discretized(self.edges)
            self._marginal = torch.log(table.sum(-2).clamp(min=1e-30))
        return self._marginal

    def time_log_prob(self, y):
        return self.time.log_prob(y)

    def time_cdf(self, y):
        return self.time.cdf(y)

    def time_log_survival(self, y):
        return self.time.log_survival(y)

    def time_mean(self):
        return self.time.mean()

    def time_quantile(self, p):
        return self.time.quantile(p)

    def sample(self, generator=None):
        y = self.time.sample(generator)
        probs = torch.softmax(self.mark_fn(y).float(), dim=-1)
        return y, torch.multinomial(probs, 1, generator=generator).squeeze(-1)


class EventFirstJoint(JointDist):
    """pi_B(k|H) f_B(y|k,H); ``time_fn(k)`` returns a TimeDist with batch (N,) for
    k of shape (N,), or batch (N, K) when called with ``None`` (all marks)."""

    def __init__(self, mark_logits: torch.Tensor, time_fn: Callable):
        self.mark = torch.log_softmax(mark_logits.float(), dim=-1)
        self.time_fn = time_fn
        self.num_marks = mark_logits.shape[-1]
        self.batch_size = mark_logits.shape[0]
        self.device = mark_logits.device
        self._all = None

    def _all_marks(self) -> TimeDist:
        if self._all is None:
            self._all = self.time_fn(None)
        return self._all

    def log_prob(self, y, k):
        return _gather(self.mark, k) + self.time_fn(k).log_prob(y)

    def mark_log_probs(self):
        return self.mark

    def _per_mark(self, y):
        dist = self._all_marks()
        if y.dim() == 1:
            return y.unsqueeze(-1).expand(-1, self.num_marks)
        return y.unsqueeze(-2).expand(-1, self.num_marks, -1)

    def time_log_prob(self, y):
        values = self._all_marks().log_prob(self._per_mark(y))
        weights = self.mark if y.dim() == 1 else self.mark.unsqueeze(-1)
        return torch.logsumexp(weights + values, dim=-2 if y.dim() > 1 else -1)

    def time_cdf(self, y):
        values = self._all_marks().cdf(self._per_mark(y))
        weights = torch.exp(self.mark) if y.dim() == 1 else torch.exp(self.mark).unsqueeze(-1)
        return (weights * values).sum(dim=-2 if y.dim() > 1 else -1).clamp(0.0, 1.0)

    def time_log_survival(self, y):
        values = self._all_marks().log_survival(self._per_mark(y))
        weights = self.mark if y.dim() == 1 else self.mark.unsqueeze(-1)
        return torch.logsumexp(weights + values, dim=-2 if y.dim() > 1 else -1)

    def time_mean(self):
        return (torch.exp(self.mark) * self._all_marks().mean()).sum(-1)

    def sample(self, generator=None):
        k = torch.multinomial(torch.exp(self.mark), 1, generator=generator).squeeze(-1)
        return self.time_fn(k).sample(generator), k

    def mark_log_probs_given_time(self, y):
        values = self._all_marks().log_prob(self._per_mark(y))
        if y.dim() == 1:
            joint = self.mark + values
            return joint - torch.logsumexp(joint, dim=-1, keepdim=True)
        joint = self.mark.unsqueeze(-1) + values  # (N, K, Q)
        return (joint - torch.logsumexp(joint, dim=-2, keepdim=True)).transpose(-1, -2)

    def discretized(self, edges):
        masses = self._all_marks().bin_masses(edges)  # (N, K, Q)
        return (torch.exp(self.mark).unsqueeze(-1) * masses).transpose(-1, -2)


class MixtureJoint(JointDist):
    """1/2 p_A + 1/2 p_B: the dual decoder's inference distribution."""

    def __init__(self, first: JointDist, second: JointDist):
        self.first, self.second = first, second
        self.num_marks = first.num_marks
        self.batch_size = first.batch_size
        self.device = first.device

    def log_prob(self, y, k):
        return torch.logaddexp(self.first.log_prob(y, k), self.second.log_prob(y, k)) - LOG_2

    def mark_log_probs(self):
        return torch.logaddexp(self.first.mark_log_probs(), self.second.mark_log_probs()) - LOG_2

    def time_log_prob(self, y):
        return torch.logaddexp(self.first.time_log_prob(y), self.second.time_log_prob(y)) - LOG_2

    def time_cdf(self, y):
        return 0.5 * (self.first.time_cdf(y) + self.second.time_cdf(y))

    def time_log_survival(self, y):
        return (
            torch.logaddexp(self.first.time_log_survival(y), self.second.time_log_survival(y))
            - LOG_2
        )

    def time_mean(self):
        return 0.5 * (self.first.time_mean() + self.second.time_mean())

    def sample(self, generator=None):
        y1, k1 = self.first.sample(generator)
        y2, k2 = self.second.sample(generator)
        pick = torch.rand(y1.shape, device=y1.device, generator=generator) < 0.5
        return torch.where(pick, y1, y2), torch.where(pick, k1, k2)

    def mark_log_probs_given_time(self, y):
        first = self.first.time_log_prob(y).unsqueeze(-1) + self.first.mark_log_probs_given_time(y)
        second = self.second.time_log_prob(y).unsqueeze(-1) + self.second.mark_log_probs_given_time(y)
        joint = torch.logaddexp(first, second)
        return joint - torch.logsumexp(joint, dim=-1, keepdim=True)

    def discretized(self, edges):
        return 0.5 * (self.first.discretized(edges) + self.second.discretized(edges))


def js_divergence(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence (nats) between probability tables over the last two dims."""

    first = first.clamp(min=0.0)
    second = second.clamp(min=0.0)
    middle = 0.5 * (first + second)
    left = (first * (torch.log(first + eps) - torch.log(middle + eps))).sum(dim=(-1, -2))
    right = (second * (torch.log(second + eps) - torch.log(middle + eps))).sum(dim=(-1, -2))
    return 0.5 * (left + right)
