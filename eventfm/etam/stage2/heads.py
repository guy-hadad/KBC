"""ETAM-2 output heads: joint decoders (D), multi-event heads (E), counts (G), user embedding (H)."""

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from eventfm.etam.model import RMSNorm
from eventfm.etam.stage2.config import Etam2Config
from eventfm.etam.stage2.distributions import (
    EventFirstJoint,
    IndependentJoint,
    JointDist,
    LogNormalMixture,
    MixtureJoint,
    PiecewiseHazard,
    SplineFlow,
    TimeDist,
    TimeFirstJoint,
    default_grid,
)

PRIOR_LOG_DELTA = math.log1p(6.0 * 3600.0)


def time_features(y: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Fourier features of y / 20 plus the scaled value: (..., 2F + 1)."""

    scaled = (y.float() / 20.0).unsqueeze(-1)
    angle = scaled * frequencies
    return torch.cat([scaled, torch.sin(angle), torch.cos(angle)], dim=-1)


def _frequency_buffer() -> torch.Tensor:
    return math.pi * torch.exp(torch.linspace(0.0, math.log(64.0), 16))


def default_hazard_edges() -> List[float]:
    return [0.0] + torch.linspace(4.0, 17.0, 32).tolist()


class TimeHead(nn.Module):
    """Hidden state (+ optional conditioning already added) -> TimeDist."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.family = config.time_dist
        if self.family == "lognormal":
            self.components = config.num_mixture
            outputs = 3 * self.components
        elif self.family == "spline":
            self.bins = config.spline_bins
            outputs = 3 * self.bins - 1
        else:
            edges = config.hazard_edges or default_hazard_edges()
            self.register_buffer("edges", torch.tensor(edges, dtype=torch.float32), persistent=False)
            outputs = len(edges)
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, outputs))
        self.reset_prior()

    @torch.no_grad()
    def reset_prior(self) -> None:
        last = self.net[-1]
        if self.family == "lognormal":
            last.weight.normal_(0.0, 0.01)
            last.bias.zero_()
            count = self.components
            last.bias[count : 2 * count] = PRIOR_LOG_DELTA + torch.linspace(-2.0, 2.0, count)
        elif self.family == "spline":
            last.weight.zero_()
            last.bias.zero_()
        else:
            # Equal-mass prior over the bins: hazard m = -log(1 - 1/(M - m)) / width_m.
            edges = self.edges
            bins = edges.shape[0] - 1
            widths = (edges[1:] - edges[:-1]).clamp(min=1e-3)
            remaining = torch.arange(bins + 1, 1, -1, dtype=torch.float32)
            rates = -torch.log1p(-1.0 / remaining) / widths
            rates = torch.cat([rates, torch.tensor([1.0])])
            last.weight.normal_(0.0, 0.01)
            last.bias.copy_(torch.log(torch.expm1(rates.clamp(min=1e-4))))

    def forward(self, hidden: torch.Tensor) -> TimeDist:
        params = self.net(hidden).float()
        if self.family == "lognormal":
            logits, means, scales = params.chunk(3, dim=-1)
            return LogNormalMixture(logits, means, scales)
        if self.family == "spline":
            bins = self.bins
            return SplineFlow(params[..., :bins], params[..., bins : 2 * bins], params[..., 2 * bins :])
        return PiecewiseHazard(params, self.edges)


class TimeConditionedMark(nn.Module):
    """pi_A(k | y, H): logits from GELU(W h + W_y phi(y))."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.register_buffer("frequencies", _frequency_buffer(), persistent=False)
        self.state = nn.Linear(width, width)
        self.time = nn.Linear(33, width, bias=False)
        self.output = nn.Linear(width, config.num_event_types)

    def forward(self, hidden: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state = self.state(hidden.float())
        timed = self.time(time_features(y, self.frequencies))
        if y.dim() == hidden.dim():  # grid: (N, Q) against (N, d)
            state = state.unsqueeze(-2)
        return self.output(F.gelu(state + timed))


class JointDecoder(nn.Module):
    """Independent / time-first / event-first / dual next-event decoders.

    ``mark_proj`` is the shared categorical projection: pi(k|H) for the
    independent and event-first branches, the cheap marginal pi_aux(k|H) of a
    time-first decoder, and the output layer of the multi-event heads.
    """

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        self.kind = config.decoder
        self.num_marks = config.num_event_types
        self.mark_proj = nn.Linear(config.hidden_size, config.num_event_types)
        if self.kind in ("independent", "time_first", "dual"):
            self.time_a = TimeHead(config)
        if self.kind in ("time_first", "dual"):
            self.mark_a = TimeConditionedMark(config)
        if self.kind in ("event_first", "dual"):
            self.time_b = TimeHead(config)
            self.mark_embed = nn.Embedding(config.num_event_types, config.hidden_size)
        self.register_buffer("grid", default_grid(config.grid_bins), persistent=False)

    def branch_a(self, hidden: torch.Tensor) -> JointDist:
        hidden = hidden.float()
        if self.kind == "independent":
            return IndependentJoint(self.time_a(hidden), self.mark_proj(hidden))
        mark_fn: Callable = lambda y: self.mark_a(hidden, y)  # noqa: E731
        return TimeFirstJoint(self.time_a(hidden), mark_fn, self.num_marks, self.grid)

    def branch_b(self, hidden: torch.Tensor) -> JointDist:
        hidden = hidden.float()

        def time_fn(marks: Optional[torch.Tensor]) -> TimeDist:
            if marks is None:
                return self.time_b(hidden.unsqueeze(-2) + self.mark_embed.weight)
            return self.time_b(hidden + self.mark_embed(marks.long().clamp(min=0)))

        return EventFirstJoint(self.mark_proj(hidden), time_fn)

    def forward(self, hidden: torch.Tensor) -> JointDist:
        if self.kind in ("independent", "time_first"):
            return self.branch_a(hidden)
        if self.kind == "event_first":
            return self.branch_b(hidden)
        return MixtureJoint(self.branch_a(hidden), self.branch_b(hidden))

    def cheap_mark_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mark_proj(hidden.float())


class FeatureHead(nn.Module):
    """p(a_f | k, y, H) for each attribute field: needed to generate events."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.register_buffer("frequencies", _frequency_buffer(), persistent=False)
        self.state = nn.Linear(width, width)
        self.mark = nn.Embedding(config.num_event_types, width)
        self.time = nn.Linear(33, width, bias=False)
        self.outputs = nn.ModuleList([nn.Linear(width, size) for size in config.field_sizes])

    def forward(self, hidden, marks, y) -> List[torch.Tensor]:
        mixed = F.gelu(
            self.state(hidden.float())
            + self.mark(marks.long().clamp(min=0))
            + self.time(time_features(y, self.frequencies))
        )
        return [head(mixed) for head in self.outputs]


class MultiEventHeads(nn.Module):
    """Draft mark heads for events i+2..i+L sharing the decoder's projection, plus
    optional cumulative-time heads q_l(log1p(t_{i+l} - t_i) | H_i)."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.horizon = config.mtp_horizon
        steps = max(0, self.horizon - 1)
        self.transforms = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)) for _ in range(steps)]
        )
        self.norms = nn.ModuleList([RMSNorm(width) for _ in range(steps)])
        self.with_time = config.mtp_time
        if self.with_time:
            self.time_head = TimeHead(config)
            self.horizon_embed = nn.Embedding(max(1, steps), width)

    def states(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        hidden = hidden.float()
        return [norm(hidden + transform(hidden)) for transform, norm in zip(self.transforms, self.norms)]

    def time_dist(self, hidden: torch.Tensor, step: int) -> TimeDist:
        index = torch.full(hidden.shape[:-1], step, dtype=torch.long, device=hidden.device)
        return self.time_head(hidden.float() + self.horizon_embed(index))


class CountHead(nn.Module):
    """Event counts per mark (and in total) in windows of 1 / 7 / 30 days after "now"."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.kind = config.count_dist
        self.horizons = len(config.count_horizons_days)
        self.marks = config.num_event_types
        cells = self.horizons * (self.marks + 1)
        if self.kind == "stage1":
            self.multi_hot = nn.Linear(width, self.marks)
            self.log_count = nn.Linear(width, 1)
        else:
            self.rate = nn.Linear(width, cells)
            if self.kind == "negbin":
                self.dispersion = nn.Linear(width, cells)

    def forward(self, user: torch.Tensor) -> Dict[str, torch.Tensor]:
        user = user.float()
        if self.kind == "stage1":
            return {"multi_hot": self.multi_hot(user), "log_count": self.log_count(user).squeeze(-1)}
        shape = user.shape[:-1] + (self.horizons, self.marks + 1)
        output = {"mean": F.softplus(self.rate(user)).view(shape) + 1e-6}
        if self.kind == "negbin":
            output["dispersion"] = F.softplus(self.dispersion(user)).view(shape) + 1e-4
        return output


def count_log_prob(params: Dict[str, torch.Tensor], counts: torch.Tensor) -> torch.Tensor:
    """log P(counts) under Poisson(mean) or NB(mean, dispersion r); elementwise."""

    mean = params["mean"]
    counts = counts.float()
    if "dispersion" not in params:
        return counts * torch.log(mean) - mean - torch.lgamma(counts + 1.0)
    r = params["dispersion"]
    return (
        torch.lgamma(counts + r)
        - torch.lgamma(r)
        - torch.lgamma(counts + 1.0)
        + r * (torch.log(r) - torch.log(r + mean))
        + counts * (torch.log(mean) - torch.log(r + mean))
    )


class UserHead(nn.Module):
    """u = W_u [u_E; u_T; u_J] for private mixers, u_J otherwise; plus the label head."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.private = config.private
        if self.private:
            self.combine = nn.Linear(3 * width, width)
            self.norm = RMSNorm(width)
        self.cls_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(width, 2))

    def embed(self, joint, event=None, time=None) -> torch.Tensor:
        if not self.private or event is None:
            return joint
        return self.norm(self.combine(torch.cat([event, time, joint], dim=-1)))


class ReprHeads(nn.Module):
    """RePR-style projections: <R(H), R(y_<=k) - R(y_<k)> ~ log p(y_k|y_<k,H) - log p(y_k|y_<k)."""

    def __init__(self, config: Etam2Config, modalities=("joint",)) -> None:
        super().__init__()
        width, dim = config.hidden_size, config.repr_dim
        self.modalities = tuple(modalities)
        self.history = nn.ModuleDict({name: nn.Linear(width, dim) for name in self.modalities})
        self.future = nn.ModuleDict({name: nn.Linear(width, dim) for name in self.modalities})
