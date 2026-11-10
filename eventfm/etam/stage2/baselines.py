"""External TPP baselines inside the ETAM-2 framework.

Backbones (``mixer``: plugged into ``Etam2Model``; tokenizers, decoder and
losses are unchanged, so these arms test the middle architecture only)

``s2p2``  S2P2 (Chang et al., NeurIPS 2025): a stack of latent linear Hawkes
          (LLH) layers. Layer l keeps P complex states and H channels; between
          events a backward zero-order hold
          ``x(t') = e^{Lambda dt} x(t) + (e^{Lambda dt} - I) B u(t'-)``, at an event
          the right limit adds the impulse ``E alpha_i``; ``y = 2 Re(C x) + D * u``
          and ``u^{l+1} = LayerNorm(GELU(y) + u)``, ``u^1 = 0``. Input-dependent
          rates ``Lambda_i = softplus(W' u_i^+ + b') * Lambda`` with
          ``Lambda = -exp(a) + i b``.
``itpp``  ITPP (Zhou et al., AAAI 2026): one latent ``Z_k`` per event type.
          Between events a GRU-ODE drift shared over types
          (``r, z = sigmoid(W Z_k)``, ``g = tanh(W'(r * Z_k))``,
          ``dZ_k/ds = tanh((1 - z) * (g - Z_k))``); an event of type m updates
          only ``Z_m`` (GRU jump); readout = attention over the K type tokens
          (shared W, per-type bias), residual + LayerNorm.

Stand-alone interleaved-token models (``InterleavedTPP``: own tokens and heads,
the ``Etam2Model`` interface, so ``Trainer2`` and the evaluation suite run
unchanged)

``flextpp``  FlexTPP (Draxler et al., NeurIPS 2025): one causal transformer over
             the scalar tokens (time, type) of each event in order; the time
             token embeds the continuous value with Linear(1 -> D); continuous
             head = rational-quadratic spline flow, discrete head = MLP softmax;
             ``p(tau | H) p(k | tau, H)``.
``kairos``   Kairos (Gupta et al., ACML 2025): one shared vocabulary of K event
             tokens and B + 1 equal-width time-bin tokens, the same causal
             transformer, cross-entropy on both token kinds.

Both interleaved models use types and times only (no attributes): the window
``[tau_1, k_1, ..., tau_n, k_n]`` (``tau_i`` = log1p seconds since the previous
event, ``log_gap``), a pre-LayerNorm causal transformer with learned position
(relative to the window start) and modality embeddings. ``h[:, i]`` is the
output at event i's type token (it predicts the next gap); the query slot equals
the last event's state, since these models have no "now" token.

Deviations from the papers (by design of this framework)
--------------------------------------------------------
S2P2
  * impulses ``alpha_i = Linear(z^E_i + z^T_i)`` (event token with attributes
    plus its time token) instead of a mark embedding only;
  * no intensity head / Monte-Carlo compensator: the states feed ETAM's
    decoder; ``h`` = top-layer ``u`` at each event's right limit and, at the
    query slot, its left limit at "now";
  * dt in hours; decay timescales initialised log-spaced over [10 min, 180 d],
    ``Re Lambda_n = -1/tau_n``, ``Im Lambda_n = 2 pi (n mod 4) / tau_n``
    (S4D-Lin ``(-1/2 + i pi m) * dt_n`` with ``dt_n = 2/tau_n``: 0-3
    oscillations per e-folding time), rate scale initialised at 1
    (official: HiPPO-LegS eigenvalues and unit-time data);
  * post-norm GELU residual as in the paper's equation (the official code's
    defaults are pre-norm and a GLU activation); learned initial state; the
    rate of the first interval reads the input of the initial state, so left
    padding is an exact no-op.
ITPP
  * the drift is integrated with 2 explicit Euler steps per interval on the
    reparameterised clock ``s = log1p(dt seconds / 60)`` (heavy-tailed gaps
    become O(1) spans), not an adaptive ODE solver on real time; the update
    gate's bias starts at +2 (``1 - z ~ 0.12``) so the 2-step scheme is stable
    at initialisation for gaps of years;
  * the jump is a GRUCell whose input is the event token ``z^E_i`` projected to
    ``itpp_dim`` (the official jump is an input-free GRU update of ``Z_m``);
  * no per-type intensity MLP / compensator ODE and no energy regulariser: the
    readout tokens are flattened (K d) -> Linear -> hidden, RMSNorm, then
    ETAM's decoder.
FlexTPP
  * no attributes; standard layout without the shifted-input BOS token (tau_1
    is input only, never predicted); learned position + modality embeddings;
    GELU FFN of width 4D (the reference default is Softplus);
  * spline = this repo's ``SplineFlow`` on y = log1p(seconds)
    (``config.spline_bins`` bins on (y - 11)/3.5 in [-5, 5], identity tails to
    a standard normal) instead of FrEIA's binned spline; parameter nets
    Linear(D, 2 max(D, out)) -> ReLU -> Linear as in the reference, the spline
    net's last layer zero-initialised.
Kairos
  * bins are floor bins ``[j d, (j + 1) d)``, ``d = tau_max / B``, plus an
    overflow bin ``[tau_max, oo)`` (Kairos: ``round(tau B / tau_max)``, i.e. a
    half-width first bin); ``tau_max`` = 99.5 % quantile of training gaps
    (Kairos: the maximum gap);
  * the softmax is restricted to the token kind being predicted (time bins
    after a type token, types after a time token): the alternation is
    deterministic here, so the conditionals are proper (Kairos: full-vocabulary
    softmax, alternation learned); a modality embedding is added (redundant
    with the shared vocabulary, so the model class is unchanged); the same
    transformer size as FlexTPP;
  * densities: a bin's mass is spread uniformly in y = log1p(seconds) inside
    the bin (edges ``log1p`` of the equal-width second edges), so the time NLL is
    in the same units as every other model; this is an approximation of the
    (uniform-in-seconds) bin and is coarse at short gaps.
Both interleaved models
  * the auxiliary marginal mark head (``aux_mark``) and the query term
    (``query``) are framework additions; with ``p_between > 0`` the query target
    is the residual gap from "now", which these models cannot condition on.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eventfm.etam.batching import IGNORE_INDEX
from eventfm.etam.model import RMSNorm
from eventfm.etam.stage2.config import Etam2Config, Etam2Weights
from eventfm.etam.stage2.distributions import DiscreteBins, SplineFlow, TimeDist, TimeFirstJoint, default_grid
from eventfm.etam.stage2.heads import UserHead

SECONDS_PER_HOUR = 3600.0
#: S2P2 decay timescales at initialisation, in hours (10 minutes .. 180 days).
S2P2_TIMESCALES_HOURS = (10.0 / 60.0, 180.0 * 24.0)
#: S4D-Lin harmonics per timescale (oscillations per e-folding time: 0 .. 3).
S4D_HARMONICS = 4
#: Closing edge (y = log1p seconds, ~15 years) of Kairos' overflow bin.
KAIROS_TOP_Y = 20.0
TIME_TOKENS = {"flextpp": "continuous", "kairos": "discrete"}


def _fp32(reference: torch.Tensor):
    """Run recurrences in float32 even inside a bf16 autocast region."""

    return torch.autocast(device_type=reference.device.type, enabled=False)


# ---------------------------------------------------------------------------
# S2P2: latent linear Hawkes layers
# ---------------------------------------------------------------------------


class LatentLinearHawkes(nn.Module):
    """One LLH layer (backward zero-order hold). Complex matrices are stored as (re, im) pairs."""

    def __init__(self, width: int, states: int, first: bool, dropout: float) -> None:
        super().__init__()
        self.first = first
        low, high = S2P2_TIMESCALES_HOURS
        tau = torch.logspace(math.log10(low), math.log10(high), states)
        harmonic = (torch.arange(states) % S4D_HARMONICS).float()
        self.log_rate = nn.Parameter(-torch.log(tau))  # Re Lambda = -exp(log_rate)
        self.freq = nn.Parameter(2.0 * math.pi * harmonic / tau)  # Im Lambda
        std = math.sqrt(1.0 / (states + width))  # xavier-normal variance, split over re / im
        self.E = nn.Parameter(torch.randn(2, states, width) * std)
        self.C = nn.Parameter(torch.randn(2, width, states) * std)
        if not first:
            self.B = nn.Parameter(torch.randn(2, states, width) * std)
            self.D = nn.Parameter(torch.randn(width))
            self.rate = nn.Linear(width, states, bias=False)
        # Separate from any nn.Linear so ETAM's re-initialisation keeps softplus(b') = 1.
        self.rate_bias = nn.Parameter(torch.full((states,), math.log(math.e - 1.0)))
        self.x0 = nn.Parameter(1e-3 * torch.randn(2, states))
        self.activation = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def _project(inputs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.complex(inputs @ matrix[0].t(), inputs @ matrix[1].t())

    def _next_input(self, state: torch.Tensor, inputs: Optional[torch.Tensor]) -> torch.Tensor:
        """u^{l+1} = LayerNorm(GELU(2 Re(C x) + D * u) + u)."""

        y = 2.0 * (state.real @ self.C[0].t() - state.imag @ self.C[1].t())
        if inputs is None:
            return self.norm(self.activation(y))
        return self.norm(self.activation(y + self.D * inputs) + inputs)

    def forward(self, inputs: Optional[Tuple[torch.Tensor, torch.Tensor]], start, kicks, delta):
        """One layer over all slots.

        ``inputs``: layer inputs (B, S, H) at each slot's (left, right) limit, or
        None for the first layer (u = 0); ``start`` (H,): input of the initial
        state; ``kicks`` (B, S, H): alpha_i, zero off events; ``delta`` (B, S):
        hours since the previous slot. Returns the next layer's (inputs, start).
        """

        batch, slots, _ = kicks.shape
        if inputs is None:
            scale = F.softplus(self.rate_bias)
        else:
            left, right = inputs
            # Rate of the interval ending at slot i: input at the right limit of slot i - 1.
            previous = torch.cat([start.expand(batch, 1, -1), right[:, :-1]], dim=1)
            scale = F.softplus(self.rate(previous) + self.rate_bias)
        span = scale * delta.unsqueeze(-1)
        decay = torch.exp(torch.complex(-torch.exp(self.log_rate) * span, self.freq * span))
        impulse = self._project(kicks, self.E)
        increment = impulse if inputs is None else impulse + (decay - 1.0) * self._project(left, self.B)
        initial = torch.complex(self.x0[0], self.x0[1])
        state = initial.expand(batch, -1)
        rights = []
        for index in range(slots):
            state = decay[:, index] * state + increment[:, index]
            rights.append(state)
        state_right = torch.stack(rights, dim=1)
        state_left = state_right - impulse
        if inputs is None:
            outputs = (self._next_input(state_left, None), self._next_input(state_right, None))
            return outputs, self._next_input(initial, None)
        outputs = (self._next_input(state_left, left), self._next_input(state_right, right))
        return outputs, self._next_input(initial, start)


class S2P2Backbone(nn.Module):
    """``config.s2p2_layers`` LLH layers with ``config.s2p2_states`` complex states, H = hidden_size."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.impulse = nn.Linear(width, width)
        self.layers = nn.ModuleList(
            [LatentLinearHawkes(width, config.s2p2_states, index == 0, config.dropout) for index in range(config.s2p2_layers)]
        )
        self.norm = RMSNorm(width)

    def forward(self, time, event, batch, context):
        with _fp32(event):
            valid = batch["slot_valid"]
            events = valid & ~batch["is_query"]
            # Padding: no time passes and no impulse, so the state stays at x0.
            delta = torch.expm1(batch["log_gap"].float()).clamp(min=0.0) / SECONDS_PER_HOUR * valid.float()
            kicks = self.impulse((event + time).float()) * events.unsqueeze(-1).float()
            inputs, start = None, kicks.new_zeros(kicks.shape[-1])
            for layer in self.layers:
                inputs, start = layer(inputs, start, kicks, delta)
            # Right limits at events; at the query slot (no impulse) this is the left limit at "now".
            return {"h": self.norm(inputs[1])}


# ---------------------------------------------------------------------------
# ITPP: one latent per event type
# ---------------------------------------------------------------------------


class ITPPBackbone(nn.Module):
    """K = num_event_types latents of size ``itpp_dim``; GRU-ODE drift, per-type GRU jumps, type attention."""

    euler_steps = 2
    heads = 2

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        marks, dim, width = config.num_event_types, config.itpp_dim, config.hidden_size
        bound = 1.0 / math.sqrt(dim)
        self.marks, self.dim = marks, dim
        self.initial = nn.Parameter(torch.empty(marks, dim).uniform_(-bound, bound))
        self.drift_gates = nn.Linear(dim, 2 * dim)
        self.drift_candidate = nn.Linear(dim, dim)
        # (reset, update) gate offsets; update starts at sigmoid(2) = 0.88 (stable Euler steps).
        self.gate_offset = nn.Parameter(torch.cat([torch.zeros(dim), torch.full((dim,), 2.0)]))
        self.jump_input = nn.Linear(width, dim)
        self.jump = nn.GRUCell(dim, dim)
        self.qkv = nn.Linear(dim, self.heads * 3 * dim, bias=False)
        self.qkv_bias = nn.Parameter(torch.empty(marks, self.heads * 3 * dim).uniform_(-bound, bound))
        self.attn_out = nn.Linear(self.heads * dim, dim)
        self.attn_norm = nn.LayerNorm(marks * dim)  # LayerNorm over the whole (K, d) block
        self.project = nn.Linear(marks * dim, width)
        self.norm = RMSNorm(width)

    def drift(self, state: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(self.drift_gates(state) + self.gate_offset).chunk(2, dim=-1)
        candidate = torch.tanh(self.drift_candidate(reset * state))
        return torch.tanh((1.0 - update) * (candidate - state))

    def readout(self, states: torch.Tensor) -> torch.Tensor:
        """(B, S, K, d) latents -> (B, S, hidden): attention over the K type tokens, flatten, project."""

        batch, slots, marks, dim = states.shape
        tokens = states.reshape(batch * slots, marks, dim)
        qkv = (self.qkv(tokens) + self.qkv_bias).view(batch * slots, marks, self.heads, 3 * dim).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch * slots, marks, self.heads * dim)
        mixed = self.attn_norm((self.attn_out(attended) + tokens).reshape(batch * slots, marks * dim))
        return self.norm(self.project(mixed).view(batch, slots, -1))

    def forward(self, time, event, batch, context):
        with _fp32(event):
            valid = batch["slot_valid"]
            events = valid & ~batch["is_query"]
            seconds = torch.expm1(batch["log_gap"].float()).clamp(min=0.0)
            # Reparameterised clock s = log1p(dt / 1 min); padding slots do not move.
            step = torch.log1p(seconds / 60.0) * valid.float() / self.euler_steps
            marks = batch["type_ids"].clamp(0, self.marks - 1)
            inputs = self.jump_input(event.float())
            state = self.initial.unsqueeze(0).expand(valid.shape[0], -1, -1)
            moving = (step > 0).any(0).tolist()
            jumping = events.any(0).tolist()
            states = []
            for index in range(valid.shape[1]):
                if moving[index]:
                    size = step[:, index].view(-1, 1, 1)
                    for _ in range(self.euler_steps):
                        state = state + size * self.drift(state)
                if jumping[index]:
                    where = marks[:, index].view(-1, 1, 1).expand(-1, 1, self.dim)
                    current = state.gather(1, where).squeeze(1)
                    updated = self.jump(inputs[:, index], current)
                    updated = torch.where(events[:, index].unsqueeze(-1), updated, current)
                    state = state.scatter(1, where, updated.unsqueeze(1))
                # Right limit at events; left limit at the query slot (no jump there).
                states.append(state)
            return {"h": self.readout(torch.stack(states, dim=1))}


def build_baseline_backbone(config: Etam2Config) -> nn.Module:
    if config.mixer == "s2p2":
        return S2P2Backbone(config)
    if config.mixer == "itpp":
        return ITPPBackbone(config)
    if config.mixer in TIME_TOKENS:
        raise ValueError("{} is a stand-alone model: build it with InterleavedTPP.".format(config.mixer))
    raise ValueError("Unknown mixer {!r}".format(config.mixer))


# ---------------------------------------------------------------------------
# FlexTPP / Kairos: interleaved (time, type) tokens, one causal transformer
# ---------------------------------------------------------------------------


@dataclass
class InterleavedConfig(Etam2Config):
    """``Etam2Config`` (mixer ``flextpp`` or ``kairos``) plus the interleaved models' own fields."""

    #: Kairos: equal-width time bins below tau_max (plus one overflow bin).
    time_bins: int = 256
    #: Kairos: tau_max in seconds (99.5 % quantile of the training gaps).
    tau_max_seconds: float = 0.0
    #: Longest window; the learned positions cover 2 (max_events + 1) tokens.
    max_events: int = 128


def gap_quantile(store, level: float = 0.995, max_gaps: int = 3_000_000) -> float:
    """Quantile (seconds) of the within-sequence gaps of a store (strided sample of large stores)."""

    times, offsets = store.timestamps, store.offsets
    starts = np.zeros(times.shape[0], dtype=bool)
    starts[offsets[:-1][offsets[:-1] < times.shape[0]]] = True
    gaps = np.maximum(np.diff(times.astype(np.float64))[~starts[1:]], 0.0)
    if gaps.size == 0:
        return 1.0
    gaps = gaps[:: max(1, gaps.size // max_gaps)]
    return max(1.0, float(np.quantile(gaps, level)))


def kairos_edges(tau_max_seconds: float, bins: int) -> torch.Tensor:
    """y-space edges of B equal-width second bins on [0, tau_max] plus the overflow bin."""

    edges = np.log1p(np.linspace(0.0, float(tau_max_seconds), bins + 1))
    edges = np.append(edges, max(KAIROS_TOP_Y, edges[-1] + 1.0))
    return torch.tensor(edges, dtype=torch.float32)


def _param_net(inputs: int, outputs: int) -> nn.Sequential:
    """FlexTPP's parameter net with one hidden layer."""

    hidden = 2 * max(inputs, outputs)
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class _Block(nn.Module):
    """Pre-LayerNorm transformer block whose attention can also extend a cached prefix."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.norm_attn = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

    def _heads(self, hidden: torch.Tensor):
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)  # q, k, v: (B, heads, L, head_dim)

    def _merge(self, attended: torch.Tensor) -> torch.Tensor:
        batch, heads, length, dim = attended.shape
        return self.proj(attended.transpose(1, 2).reshape(batch, length, heads * dim))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        q, k, v = self._heads(self.norm_attn(hidden))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        hidden = hidden + self.drop(self._merge(attended))
        hidden = hidden + self.drop(self.mlp(self.norm_mlp(hidden)))
        return hidden, (k, v)

    def extend(self, hidden: torch.Tensor, cache, key_valid: torch.Tensor) -> torch.Tensor:
        """``hidden`` (N, Q, D): Q independent extra tokens, each attending to the prefix and itself."""

        q, k, v = self._heads(self.norm_attn(hidden))
        past_k, past_v = (tensor.to(q.dtype) for tensor in cache)
        scale = self.head_dim ** -0.5
        past = torch.einsum("nhqd,nhtd->nhqt", q, past_k) * scale
        past = past.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        own = (q * k).sum(-1, keepdim=True) * scale
        weights = torch.softmax(torch.cat([past, own], dim=-1), dim=-1)
        weights = F.dropout(weights, self.dropout, self.training)
        attended = torch.einsum("nhqt,nhtd->nhqd", weights[..., :-1], past_v) + weights[..., -1:] * v
        hidden = hidden + self.drop(self._merge(attended))
        return hidden + self.drop(self.mlp(self.norm_mlp(hidden)))


class InterleavedDecoder(nn.Module):
    """Time head on the type-token state; marks from ONE extra time token through the cached transformer."""

    kind = "time_first"

    def __init__(self, config: InterleavedConfig, continuous: bool) -> None:
        super().__init__()
        width, marks = config.hidden_size, config.num_event_types
        self.num_marks = marks
        self.continuous = continuous
        if continuous:
            self.spline_bins = config.spline_bins
            self.time_net = _param_net(width, 3 * config.spline_bins - 1)
            self.mark_net = _param_net(width, marks)
        else:
            edges = kairos_edges(config.tau_max_seconds, config.time_bins)
            self.register_buffer("time_edges", edges, persistent=False)
            # One affine output layer over the shared vocabulary: K marks, then B + 1 time bins.
            self.token_head = nn.Linear(width, marks + edges.shape[0] - 1)
        self.mark_proj = nn.Linear(width, marks)  # auxiliary marginal head pi(k | H)
        self.register_buffer("grid", default_grid(config.grid_bins), persistent=False)
        self.__dict__["owner"] = None  # the InterleavedTPP (not a submodule)

    def time_dist(self, states: torch.Tensor) -> TimeDist:
        states = states.float()
        if self.continuous:
            params = self.time_net(states)
            bins = self.spline_bins
            return SplineFlow(params[..., :bins], params[..., bins : 2 * bins], params[..., 2 * bins :])
        return DiscreteBins(self.token_head(states)[..., self.num_marks :], self.time_edges)

    def mark_logits(self, time_states: torch.Tensor) -> torch.Tensor:
        """Logits of the mark from the output at a time token (which carries the true gap)."""

        time_states = time_states.float()
        if self.continuous:
            return self.mark_net(time_states)
        return self.token_head(time_states)[..., : self.num_marks]

    def forward(self, states: torch.Tensor) -> TimeFirstJoint:
        states = states.float()
        owner = self.owner

        def mark_fn(y: torch.Tensor) -> torch.Tensor:
            owner.check_last_states(states)
            return self.mark_logits(owner.extend(y))

        return TimeFirstJoint(self.time_dist(states), mark_fn, self.num_marks, self.grid)

    def cheap_mark_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.mark_proj(states.float())


class InterleavedTPP(nn.Module):
    """FlexTPP (``time_tokens="continuous"``) / Kairos (``"discrete"``) with the ``Etam2Model`` interface.

    Width, depth and heads are ``config.hidden_size / num_layers / num_heads``
    (defaults 256 / 6 / 8). Kairos needs ``tau_max_seconds`` (see ``from_store``).
    """

    def __init__(
        self,
        config: Etam2Config,
        time_tokens: Optional[str] = None,
        tau_max_seconds: Optional[float] = None,
        max_events: Optional[int] = None,
        time_bins: Optional[int] = None,
    ) -> None:
        super().__init__()
        time_tokens = time_tokens or TIME_TOKENS.get(config.mixer)
        if time_tokens not in ("continuous", "discrete"):
            raise ValueError("time_tokens must be continuous or discrete, got {!r}".format(time_tokens))
        values = config.to_dict()
        values["mixer"] = "flextpp" if time_tokens == "continuous" else "kairos"
        for name, value in (("tau_max_seconds", tau_max_seconds), ("max_events", max_events), ("time_bins", time_bins)):
            if value is not None:
                values[name] = value
        config = InterleavedConfig(**values)
        self.continuous = time_tokens == "continuous"
        if not self.continuous and config.tau_max_seconds <= 0:
            raise ValueError("Kairos needs tau_max_seconds > 0 (InterleavedTPP.from_store computes it).")
        self.config = config
        width, marks = config.hidden_size, config.num_event_types
        if self.continuous:
            self.time_value = nn.Linear(1, width)
            self.tokens = nn.Embedding(marks, width)
        else:
            # Shared vocabulary: K event tokens, then B + 1 time-bin tokens.
            self.tokens = nn.Embedding(marks + config.time_bins + 1, width)
        self.modality = nn.Embedding(2, width)  # 0: time token, 1: type token
        self.position = nn.Embedding(2 * (config.max_events + 1), width)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([_Block(width, config.num_heads, config.dropout) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(width)
        self.decoder = InterleavedDecoder(config, self.continuous)
        self.decoder.__dict__["owner"] = self
        self.user_head = UserHead(config)
        self.has_unimodal = False
        self.mtp = None
        self.counts = None
        self.feature_head = None
        self.repr = None
        self._cache: Optional[Dict[str, object]] = None
        self._init_weights()

    @classmethod
    def from_store(cls, config: Etam2Config, store, max_events: int = 128) -> "InterleavedTPP":
        """tau_max = 99.5 % quantile of the store's (training) gaps."""

        return cls(config, tau_max_seconds=gap_quantile(store, 0.995), max_events=max_events)

    def _init_weights(self) -> None:
        depth_scale = 1.0 / math.sqrt(2.0 * max(1, len(self.layers)))
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                residual = name.endswith(".proj") or name.endswith(".mlp.2")
                nn.init.normal_(module.weight, 0.0, 0.02 * (depth_scale if residual else 1.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.02)
        if self.continuous:
            # FlexTPP: equally spaced spline at initialisation.
            nn.init.zeros_(self.decoder.time_net[-1].weight)
            nn.init.zeros_(self.decoder.time_net[-1].bias)

    # -- tokens -------------------------------------------------------------

    def _time_tokens(self, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        if self.continuous:
            content = self.time_value(y.unsqueeze(-1))
        else:
            edges = self.decoder.time_edges
            bins = torch.bucketize(y.contiguous(), edges[1:-1].contiguous(), right=True)
            content = self.tokens(self.config.num_event_types + bins)
        return content + self.modality.weight[0]

    def _type_tokens(self, marks: torch.Tensor) -> torch.Tensor:
        return self.tokens(marks.clamp(0, self.config.num_event_types - 1)) + self.modality.weight[1]

    def encode(self, batch: Dict[str, torch.Tensor], extra_y: Optional[torch.Tensor] = None):
        """Final states of ``[tau_1, k_1, ..., tau_n, k_n]`` (+ one time token of value ``extra_y`` (B,) at the
        end), the per-layer K/V and the key-validity mask. Positions count from the window's first event."""

        valid = batch["slot_valid"][:, :-1]
        count = batch["position"][:, -1]
        if int(count.max()) > self.config.max_events:
            raise ValueError("Window of {} events exceeds max_events={}.".format(int(count.max()), self.config.max_events))
        batch_size, events = valid.shape
        step = batch["position"][:, :-1]
        times = self._time_tokens(batch["log_gap"][:, :-1]) + self.position(2 * step)
        types = self._type_tokens(batch["type_ids"][:, :-1]) + self.position(2 * step + 1)
        hidden = torch.stack([times, types], dim=2).reshape(batch_size, 2 * events, -1)
        key_valid = valid.repeat_interleave(2, dim=1)
        if extra_y is not None:
            extra = self._time_tokens(extra_y) + self.position(2 * count)
            hidden = torch.cat([hidden, extra.unsqueeze(1)], dim=1)
            key_valid = torch.cat([key_valid, torch.ones_like(key_valid[:, :1])], dim=1)
        length = hidden.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        eye = torch.eye(length, dtype=torch.bool, device=hidden.device)
        # Padding tokens only see themselves (otherwise they would attend to nothing).
        mask = ((causal.unsqueeze(0) & key_valid.unsqueeze(1)) | eye).unsqueeze(1)
        hidden = self.dropout(hidden)
        cache = []
        for block in self.layers:
            hidden, kv = block(hidden, mask)
            cache.append(kv)
        return self.final_norm(hidden).float(), cache, key_valid

    def forward(self, batch: Dict[str, torch.Tensor], record_gates: bool = False) -> Dict[str, object]:
        self._cache = None
        states, cache, key_valid = self.encode(batch)
        time_states, type_states = states[:, 0::2], states[:, 1::2]
        # No "now" token: the query slot repeats the last event's state.
        hidden = torch.cat([type_states, type_states[:, -1:]], dim=1)
        user = hidden[:, -1]
        self._cache = {
            "kv": cache,
            "key_valid": key_valid,
            "position": 2 * batch["position"][:, -1],
            "last": user.detach(),
        }
        return {"h": hidden, "time_states": time_states, "user": user, "user_joint": user, "aux": {}, "gates": []}

    def check_last_states(self, states: torch.Tensor) -> None:
        cache = self._cache
        if cache is None:
            raise RuntimeError("Run forward before evaluating marks: the extra token extends its cache.")
        last = cache["last"]
        if states.shape != last.shape or not torch.equal(states.detach(), last.to(states.dtype)):
            raise ValueError("The decoder's marks are only defined at h[:, -1] of the most recent forward.")

    def extend(self, y: torch.Tensor) -> torch.Tensor:
        """Final state of ONE extra time token of value ``y`` ((N,) or (N, Q), one token per column)
        appended after the cached window of the most recent forward."""

        cache = self._cache
        if cache is None:
            raise RuntimeError("Run forward before extending its cache.")
        single = y.dim() == 1
        y = y.unsqueeze(-1) if single else y
        hidden = self.dropout(self._time_tokens(y) + self.position(cache["position"]).unsqueeze(1))
        for block, kv in zip(self.layers, cache["kv"]):
            hidden = block.extend(hidden, kv, cache["key_valid"])
        hidden = self.final_norm(hidden).float()
        return hidden[:, 0] if single else hidden

    def classify(self, user: torch.Tensor) -> torch.Tensor:
        return self.user_head.cls_head(user.float())

    def _joint_log_prob(self, states, time_states, y, k) -> torch.Tensor:
        """log f(y | type-token state) + log pi(k | output at the time token carrying y)."""

        marks = torch.log_softmax(self.decoder.mark_logits(time_states), dim=-1)
        return self.decoder.time_dist(states).log_prob(y) + marks.gather(-1, k.unsqueeze(-1)).squeeze(-1)

    def compute_losses(self, outputs, batch, weights: Etam2Weights) -> Dict[str, torch.Tensor]:
        """Teacher-forced joint NLL (+ query, auxiliary mark and label terms), ``compute_losses2``'s format."""

        terms: Dict[str, torch.Tensor] = {}
        hidden = outputs["h"].float()
        next_type, next_y = batch["next_type"], batch["next_log_dt"].float()
        events = (next_type[:, :-1] != IGNORE_INDEX) & ~batch["is_query"][:, :-1]
        has_query = next_type[:, -1] != IGNORE_INDEX
        extra = None
        if (weights.joint and events.any()) or (weights.query and has_query.any()):
            # The true gap after the window's last event (dense term) and the query target (query
            # term), each as one extra time token: inputs to the loss only, never to ``h``.
            extra = self.extend(torch.stack([next_y[:, -2], next_y[:, -1]], dim=-1))
        if events.any():
            states, k = hidden[:, :-1][events], next_type[:, :-1][events]
            if weights.joint:
                # Slot i's mark is read at the time token of event i + 1 (in the window, or the extra token).
                following = torch.cat([outputs["time_states"][:, 1:], extra[:, :1]], dim=1)
                terms["joint"] = -self._joint_log_prob(states, following[events], next_y[:, :-1][events], k).mean()
            if weights.aux_mark:
                terms["aux_mark"] = F.cross_entropy(self.decoder.cheap_mark_logits(states), k)
        if weights.query and has_query.any():
            terms["query"] = -self._joint_log_prob(
                hidden[:, -1][has_query], extra[:, 1][has_query], next_y[:, -1][has_query], next_type[:, -1][has_query]
            ).mean()
        if weights.cls:
            labels = batch["labels"]
            labelled = labels >= 0
            if labelled.any():
                terms["cls"] = F.cross_entropy(self.classify(outputs["user"][labelled]), labels[labelled])
        total = None
        for name, value in terms.items():
            weighted = getattr(weights, name) * value
            total = weighted if total is None else total + weighted
        terms["loss"] = total if total is not None else hidden.sum() * 0.0
        return terms
