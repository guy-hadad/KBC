"""ETAM-2 tokenizers: predictive temporal encoding (A) and structured event tokens (B).

Time token  ``z^T = RMSNorm(W_c c + W_r r + W_m m)``
  * ``c`` calendar: hour/weekday/day/month embeddings, cyclic coordinates and
    Fourier features of the absolute day count with periods initialised at 1 h,
    1 d, 7 d, 1 mo, 1 q, 1 y (``learned`` lets each period move by up to +-25 %).
  * ``r`` relative: adaptive quantile bins of log gap / log age, plus a
    continuous residual inside the bin (``bins_residual``); ``stage1`` is the
    Stage 1 Fourier + linear-bucket encoder.
  * ``m`` event-dependent temporal memory evaluated just before each slot's
    time, ``m_r(t_i^-) = sum_{j<i} exp(-(t_i - t_j)/tau_r) V_r e_j`` (exact,
    parallel), or its alternatives: learned timescales, decayed counts only
    (event-agnostic), a learned complex-diagonal continuous-time SSM, or an
    ODE-RNN (GRU jumps, neural-ODE flow between events).
Event token ``z^E = E_k + sum_f E_f(a_f) + MLP_attr([a_f]_f, amount residual)``.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from eventfm.etam.batching import NUM_CALENDAR_CONT
from eventfm.etam.model import RMSNorm, _fourier
from eventfm.etam.stage2.config import Etam2Config

BASE_PERIODS_DAYS = [1.0 / 24.0, 1.0, 7.0, 30.4375, 91.3125, 365.25]
Y_TOP = 20.0


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


def _bin_edges(edges, bins: int) -> torch.Tensor:
    """Interior edges (sorted, unique) plus the outer bounds [0, Y_TOP]."""

    if edges:
        inner = sorted(set(float(value) for value in edges if 0.0 < float(value) < Y_TOP))
    else:
        inner = torch.linspace(0.0, 18.0, bins + 1)[1:-1].tolist()
    return torch.tensor([0.0] + inner + [Y_TOP], dtype=torch.float32)


class Calendar(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.learned = config.calendar == "learned"
        self.hour = nn.Embedding(24, width)
        self.weekday = nn.Embedding(7, width)
        self.day_of_month = nn.Embedding(31, width)
        self.month = nn.Embedding(12, width)
        periods = torch.tensor(BASE_PERIODS_DAYS, dtype=torch.float64)
        self.register_buffer("log_periods", torch.log(periods), persistent=False)
        if self.learned:
            self.period_offset = nn.Parameter(torch.zeros(len(BASE_PERIODS_DAYS)))
        inputs = NUM_CALENDAR_CONT + 2 * len(BASE_PERIODS_DAYS) + 1
        self.projection = mlp(inputs, width, width)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        days = batch["abs_days"]
        log_periods = self.log_periods
        if self.learned:
            log_periods = log_periods + 0.25 * torch.tanh(self.period_offset).double()
        frequencies = 2.0 * math.pi * torch.exp(-log_periods)
        angle = torch.remainder(days.unsqueeze(-1) * frequencies, 2.0 * math.pi)
        continuous = torch.cat(
            [
                batch["calendar_cont"],
                torch.sin(angle).float(),
                torch.cos(angle).float(),
                (days.float() / 365.25).unsqueeze(-1),
            ],
            dim=-1,
        )
        ids = batch["calendar_ids"]
        return (
            self.projection(continuous)
            + self.hour(ids[..., 0])
            + self.weekday(ids[..., 1])
            + self.day_of_month(ids[..., 2])
            + self.month(ids[..., 3])
        )


class Relative(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.kind = config.relative
        if self.kind == "stage1":
            self.register_buffer(
                "log_frequencies",
                torch.exp(torch.linspace(math.log(0.25), math.log(16.0), 16)),
                persistent=False,
            )
            self.projection = mlp(2 * 33 + 1, width, width)
            self.gap_buckets = nn.Embedding(64, width)
            return
        gap_edges = _bin_edges(config.gap_bin_edges, 64)
        age_edges = _bin_edges(config.age_bin_edges, 64)
        self.register_buffer("gap_edges", gap_edges, persistent=False)
        self.register_buffer("age_edges", age_edges, persistent=False)
        self.gap_bins = nn.Embedding(gap_edges.shape[0] - 1, width)
        self.age_bins = nn.Embedding(age_edges.shape[0] - 1, width)
        self.has_prev = nn.Embedding(2, width)
        if self.kind == "bins_residual":
            self.gap_residual = mlp(1, 64, width)
            self.age_residual = mlp(1, 64, width)

    def _binned(self, values, edges, table, residual):
        index = (torch.bucketize(values, edges[1:-1].contiguous(), right=True)).clamp(
            0, edges.shape[0] - 2
        )
        hidden = table(index)
        if residual is not None:
            centre = 0.5 * (edges[index] + edges[index + 1])
            hidden = hidden + residual((values - centre).unsqueeze(-1))
        return hidden

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        gap, age = batch["log_gap"], batch["log_age"]
        if self.kind == "stage1":
            continuous = torch.cat(
                [
                    _fourier(gap, self.log_frequencies),
                    _fourier(age, self.log_frequencies),
                    batch["has_prev"].unsqueeze(-1),
                ],
                dim=-1,
            )
            bucket = (gap / 20.0 * 63).long().clamp(0, 63)
            return self.projection(continuous) + self.gap_buckets(bucket)
        residual = self.kind == "bins_residual"
        return (
            self._binned(gap, self.gap_edges, self.gap_bins, self.gap_residual if residual else None)
            + self._binned(age, self.age_edges, self.age_bins, self.age_residual if residual else None)
            + self.has_prev(batch["has_prev"].long())
        )


def strict_past_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, S, S) True where slot j is an observed event strictly before slot i."""

    source = batch["slot_valid"] & ~batch["is_query"]
    slots = source.shape[1]
    causal = torch.ones(slots, slots, dtype=torch.bool, device=source.device).tril(-1)
    return causal.unsqueeze(0) & source.unsqueeze(1)


def lag_seconds(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, S, S) float32 seconds t_i - t_j (computed in float64, clamped at 0)."""

    days = batch["abs_days"]
    days = days - days[:, :1]
    lag = (days.unsqueeze(-1) - days.unsqueeze(-2)) * 86_400.0
    return lag.clamp(min=0.0).float()


class TemporalMemory(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.kind = config.memory
        self.event_input = config.memory_event_input and config.inputs != "time_only"
        scales = torch.tensor(config.memory_timescales, dtype=torch.float32)
        rates = len(config.memory_timescales)
        if self.kind == "exp_learned":
            self.log_tau = nn.Parameter(torch.log(scales))
        else:
            self.register_buffer("log_tau", torch.log(scales), persistent=False)
        if self.kind in ("exp", "exp_learned"):
            inputs = rates
            if self.event_input:
                self.value = nn.Linear(width, rates * config.memory_dim, bias=False)
                inputs += rates * config.memory_dim
        elif self.kind == "count":
            inputs = rates
        elif self.kind == "ssm":
            modes, channels = config.ssm_modes, config.ssm_channels
            lows = torch.logspace(math.log10(60.0), math.log10(15_552_000.0), modes)
            self.ssm_log_rate = nn.Parameter(-torch.log(lows))
            generator = torch.Generator().manual_seed(0)
            self.ssm_freq = nn.Parameter(math.pi * torch.rand(modes, generator=generator))
            self.ssm_in = nn.Linear(width, modes * channels * 2, bias=False)
            self.modes, self.channels = modes, channels
            inputs = modes * channels * 2 + modes
        elif self.kind == "ode":
            hidden = config.ode_dim
            self.ode_f = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.gru = nn.GRUCell(width, hidden)
            self.steps = config.ode_steps
            inputs = hidden
        else:
            raise ValueError(self.kind)
        self.output = nn.Linear(inputs, width)

    def _decays(self, lag, mask, log_tau):
        tau = torch.exp(log_tau).view(1, -1, 1, 1)
        decay = torch.exp(-lag.unsqueeze(1) / tau) * mask.unsqueeze(1).float()
        return decay  # (B, R, S, S)

    def forward(self, batch, content, lag, mask) -> torch.Tensor:
        if self.kind == "ode":
            return self.output(self._ode(batch, content))
        if self.kind == "ssm":
            return self.output(self._ssm(content, lag, mask))
        decay = self._decays(lag, mask, self.log_tau)
        count = decay.sum(-1).transpose(1, 2)  # (B, S, R)
        features = [torch.log1p(count)]
        if self.kind != "count" and self.event_input:
            batch_size, slots, _ = content.shape
            values = self.value(content).view(batch_size, slots, count.shape[-1], -1)
            state = torch.einsum("brij,bjrc->birc", decay, values.float())
            state = state / (count.unsqueeze(-1) + 1.0)
            features.insert(0, state.reshape(batch_size, slots, -1))
        return self.output(torch.cat(features, dim=-1))

    def _ssm(self, content, lag, mask):
        rate = torch.exp(self.ssm_log_rate).view(1, -1, 1, 1)
        freq = (rate * self.ssm_freq.view(1, -1, 1, 1))
        lagged = lag.unsqueeze(1)
        envelope = torch.exp(-lagged * rate) * mask.unsqueeze(1).float()
        cos_kernel = envelope * torch.cos(freq * lagged)
        sin_kernel = envelope * torch.sin(freq * lagged)
        batch_size, slots, _ = content.shape
        inputs = self.ssm_in(content).float().view(batch_size, slots, self.modes, self.channels, 2)
        real, imag = inputs[..., 0], inputs[..., 1]
        state_real = torch.einsum("bnij,bjnc->binc", cos_kernel, real) - torch.einsum(
            "bnij,bjnc->binc", sin_kernel, imag
        )
        state_imag = torch.einsum("bnij,bjnc->binc", cos_kernel, imag) + torch.einsum(
            "bnij,bjnc->binc", sin_kernel, real
        )
        count = envelope.sum(-1).transpose(1, 2)  # (B, S, N)
        scale = (count + 1.0).unsqueeze(-1)
        features = torch.cat(
            [
                (state_real / scale).reshape(batch_size, slots, -1),
                (state_imag / scale).reshape(batch_size, slots, -1),
                torch.log1p(count),
            ],
            dim=-1,
        )
        return features

    def _ode(self, batch, content):
        valid = batch["slot_valid"]
        events = valid & ~batch["is_query"]
        gap_seconds = torch.expm1(batch["log_gap"].float()).clamp(min=0.0)
        # Time-reparameterised clock: heavy-tailed gaps become O(1) integration spans.
        span = torch.log1p(gap_seconds / 60.0) * valid.float()
        # No evolution into the first observed slot: its predecessor's state is unknown.
        first = valid & ~torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], dim=1)
        span = span * (~first).float()
        batch_size, slots, _ = content.shape
        state = content.new_zeros(batch_size, self.gru.hidden_size, dtype=torch.float32)
        outputs = []
        for index in range(slots):
            step = (span[:, index] / self.steps).unsqueeze(-1)
            for _ in range(self.steps):
                k1 = torch.tanh(self.ode_f(state))
                k2 = torch.tanh(self.ode_f(state + 0.5 * step * k1))
                k3 = torch.tanh(self.ode_f(state + 0.5 * step * k2))
                k4 = torch.tanh(self.ode_f(state + step * k3))
                state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            outputs.append(state)
            jumped = self.gru(content[:, index].float(), state)
            state = torch.where(events[:, index].unsqueeze(-1), jumped, state)
        return torch.stack(outputs, dim=1)


class VectorQuantizer(nn.Module):
    """Straight-through VQ of time tokens (VQ-VAE losses, beta = 0.25)."""

    def __init__(self, codes: int, width: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codes, width)
        nn.init.normal_(self.codebook.weight, std=1.0)
        self.beta = beta

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor):
        flat = hidden[valid].float()
        if flat.numel() == 0:
            return hidden, hidden.sum() * 0.0, 0.0
        codes = self.codebook.weight.float()
        distance = (
            flat.pow(2).sum(-1, keepdim=True) - 2 * flat @ codes.t() + codes.pow(2).sum(-1).unsqueeze(0)
        )
        index = distance.argmin(-1)
        quantised = codes[index]
        loss = F.mse_loss(quantised, flat.detach()) + self.beta * F.mse_loss(quantised.detach(), flat)
        output = hidden.clone()
        output[valid] = (flat + (quantised - flat).detach()).to(hidden.dtype)
        usage = torch.bincount(index, minlength=codes.shape[0]).float()
        usage = usage / usage.sum()
        perplexity = float(torch.exp(-(usage * torch.log(usage + 1e-10)).sum()))
        return output, loss, perplexity


class TimeTokenizer2(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.config = config
        timed = config.inputs != "event_only"
        self.calendar = Calendar(config) if timed and config.calendar != "none" else None
        self.relative = Relative(config) if timed else None
        self.memory = TemporalMemory(config) if timed and config.memory != "none" else None
        self.base = nn.Parameter(torch.zeros(width))
        self.query = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width)
        self.vq = VectorQuantizer(config.time_vq, width) if config.time_vq else None

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        content: torch.Tensor,
        lag: torch.Tensor,
        past: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        hidden = self.base.expand(*batch["log_gap"].shape, -1)
        if self.calendar is not None:
            hidden = hidden + self.calendar(batch)
        if self.relative is not None:
            hidden = hidden + self.relative(batch)
        if self.memory is not None:
            hidden = hidden + self.memory(batch, content, lag, past).to(hidden.dtype)
        hidden = hidden + batch["is_query"].unsqueeze(-1).to(hidden.dtype) * self.query
        hidden = self.norm(hidden)
        aux: Dict[str, object] = {}
        if self.vq is not None:
            hidden, aux["vq_loss"], aux["vq_perplexity"] = self.vq(hidden, batch["slot_valid"])
        return hidden, aux


class EventTokenizer2(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width = config.hidden_size
        self.config = config
        self.types = nn.Embedding(config.num_event_types, width)
        self.fields = nn.ModuleList(
            [nn.Embedding(size, width, padding_idx=0) for size in config.field_sizes]
        )
        self.use_amount = (
            config.attr == "mlp" and config.amount_scalar and 0 <= config.amount_field < len(config.field_sizes)
        )
        if config.attr == "stage1":
            self.mix = mlp(width, width, width)
        else:
            small = 32
            self.attr_fields = nn.ModuleList(
                [nn.Embedding(size, small, padding_idx=0) for size in config.field_sizes]
            )
            extra = 2 if self.use_amount else 0
            self.attr_mlp = mlp(small * len(config.field_sizes) + extra, width, width)
            if self.use_amount:
                size = config.field_sizes[config.amount_field]
                centres = list(config.amount_bucket_centers)[:size]
                centres += [0.0] * (size - len(centres))
                self.register_buffer("amount_centres", torch.tensor(centres), persistent=False)
        self.user = nn.Parameter(torch.zeros(width))
        self.blank = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        config = self.config
        if config.inputs == "time_only":
            hidden = self.blank.expand(*batch["type_ids"].shape, -1)
        else:
            hidden = self.types(batch["type_ids"])
            features = batch["feature_ids"]
            for column, table in enumerate(self.fields):
                hidden = hidden + table(features[..., column])
            if config.attr == "stage1":
                hidden = hidden + self.mix(hidden)
            else:
                parts = [table(features[..., column]) for column, table in enumerate(self.attr_fields)]
                if self.use_amount:
                    has_amount = batch.get("has_amount")
                    if has_amount is None:
                        has_amount = torch.zeros_like(batch["log_gap"])
                        residual = torch.zeros_like(has_amount)
                    else:
                        centre = self.amount_centres[features[..., config.amount_field]]
                        residual = (batch["log_amount"] - centre) * has_amount
                    parts += [residual.unsqueeze(-1), has_amount.unsqueeze(-1)]
                hidden = hidden + self.attr_mlp(torch.cat(parts, dim=-1))
        query = batch["is_query"].unsqueeze(-1).to(hidden.dtype)
        hidden = hidden * (1.0 - query) + query * self.user
        return self.norm(hidden)
