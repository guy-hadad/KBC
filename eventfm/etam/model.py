"""ETAM: one transformer, two modalities (time and event), shared attention.

Token layout (``layout="token"``), per history slot ``i`` and a final query slot::

    tau_1 eps_1  tau_2 eps_2  ...  tau_n eps_n  tau_q USR

``tau`` tokens come from the time tokenizer (absolute calendar + absolute day
count + relative gap/age), ``eps`` tokens from the event tokenizer (type +
feature values), ``USR`` is a learned query token whose output is the user
embedding. Every layer routes each token to its modality's parameters
(Mixture-of-Transformers / Mono-InternVL style); attention is one causal
softmax over both modalities, the only place where they meet.

The interleaving gives the marked-TPP likelihood its natural factorisation
``p(t_{i+1}, k_{i+1} | H_i) = p(t_{i+1} | H_i) * p(k_{i+1} | t_{i+1}, H_i)``:
``eps_i`` predicts *when* the next event happens, ``tau_{i+1}`` (which has
seen that time but not the event) predicts *what* it is.

``layout="additive"`` is the single-token control: the time embedding is added
onto the event embedding, as in every baseline in the benchmark.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from eventfm.etam.batching import NUM_CALENDAR_CONT
from eventfm.models.modeling_flat import LogNormalMixtureHead

TIME, EVENT, USER = 0, 1, 2
PLACEMENTS = ("shared", "ffn", "attn_ffn", "deep")
ROPES = ("none", "index", "time", "index+time")


@dataclass
class EtamConfig:
    num_event_types: int = 53
    field_sizes: List[int] = field(default_factory=list)
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_multiplier: float = 8.0 / 3.0
    dropout: float = 0.1
    layout: str = "token"
    placement: str = "deep"
    user_expert: bool = False
    rope: str = "index"
    time_rope_fraction: float = 0.5
    time_relative: bool = True
    time_absolute: bool = True
    num_frequencies: int = 16
    num_gap_buckets: int = 64
    num_time_mixture: int = 8
    max_positions: int = 4096
    #: ``scratch`` trains the routed transformer above; anything else names a
    #: frozen pretrained LM (e.g. ``Qwen/Qwen3-0.6B-Base``) used as the mixer,
    #: with one LoRA expert per modality on every projection (RAM-style).
    backbone: str = "scratch"
    lora_rank: int = 64
    lora_alpha: float = 128.0

    def __post_init__(self) -> None:
        if self.backbone != "scratch":
            from eventfm.etam.llm_backbone import llm_hidden_size

            self.hidden_size = llm_hidden_size(self.backbone)
            # The LM is the mixer; no routed scratch layers are built.
            self.num_heads, self.num_layers = 8, 0
        if self.layout not in ("token", "additive"):
            raise ValueError("layout must be `token` or `additive`.")
        if self.placement not in PLACEMENTS:
            raise ValueError("placement must be one of {}.".format(PLACEMENTS))
        if self.rope not in ROPES:
            raise ValueError("rope must be one of {}.".format(ROPES))
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return int(math.ceil(self.hidden_size * self.ffn_multiplier / 64.0) * 64)

    @property
    def route_map(self) -> List[int]:
        """Expert index for each modality (time, event, user)."""

        if self.layout == "token":
            return [0, 1, 2 if self.user_expert else 1]
        # Single-token control: no time tokens exist, so no time expert either.
        return [0, 0, 1 if self.user_expert else 0]

    @property
    def num_experts(self) -> int:
        return max(self.route_map) + 1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * hidden).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(hidden).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class QKVProjection(nn.Module):
    """Fused Q/K/V projection with per-head QK-norm (Qwen3-style)."""

    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = self.q_norm(q.reshape(shape)).reshape(v.shape)
        k = self.k_norm(k.reshape(shape)).reshape(v.shape)
        return torch.cat([q, k, v], dim=-1)


class Routed(nn.Module):
    """One copy of a module per expert; each token runs through its own copy.

    Gather/scatter routing, so a token pays for exactly one expert (the
    Mono-InternVL reference code runs every expert on every token and masks).
    """

    def __init__(self, factory: Callable[[], nn.Module], num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([factory() for _ in range(num_experts)])

    def forward(self, hidden: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 1:
            return self.experts[0](hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        flat_route = route.reshape(-1)
        output: Optional[torch.Tensor] = None
        for index, expert in enumerate(self.experts):
            rows = (flat_route == index).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            result = expert(flat.index_select(0, rows))
            if output is None:
                output = result.new_zeros(flat.shape[0], result.shape[-1])
            output.index_copy_(0, rows, result.to(output.dtype))
        if output is None:
            raise ValueError("Routing produced no tokens.")
        return output.reshape(*hidden.shape[:-1], output.shape[-1])


def _routed(factory: Callable[[], nn.Module], routed: bool, num_experts: int) -> Routed:
    return Routed(factory, num_experts if routed else 1)


def _rotate(hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class Rotary(nn.Module):
    """Index RoPE, continuous-time RoPE, or both on disjoint head-dim slices.

    Time RoPE rotates by ``2*pi*days/period`` over periods from one hour to two
    years, so the attention score between two tokens depends on their *time
    difference* (and its phase within a day/week/month), not on how many events
    separate them. That is the relative-time channel in attention; the time
    tokens carry absolute calendar time and gaps in the residual stream.
    """

    def __init__(self, config: EtamConfig) -> None:
        super().__init__()
        head_dim = config.head_dim
        if config.rope == "none":
            time_dims, index_dims = 0, 0
        elif config.rope == "index":
            time_dims, index_dims = 0, head_dim
        elif config.rope == "time":
            time_dims, index_dims = head_dim, 0
        else:
            time_dims = int(round(head_dim * config.time_rope_fraction / 2.0)) * 2
            index_dims = head_dim - time_dims
        self.time_dims, self.index_dims = time_dims, index_dims
        if index_dims:
            inverse = 1.0 / (10_000.0 ** (torch.arange(0, index_dims, 2).double() / index_dims))
            self.register_buffer("index_freq", inverse, persistent=False)
        if time_dims:
            periods_days = torch.logspace(
                math.log10(1.0 / 24.0), math.log10(730.0), time_dims // 2, dtype=torch.float64
            )
            self.register_buffer("time_freq", 2.0 * math.pi / periods_days, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position: torch.Tensor, days: torch.Tensor):
        # q, k: (B, H, T, D); position (B, T) int; days (B, T) float64.
        pieces_q, pieces_k = [], []
        cursor = 0
        for width, coordinate, frequency in (
            (self.time_dims, days, getattr(self, "time_freq", None)),
            (self.index_dims, position, getattr(self, "index_freq", None)),
        ):
            if not width:
                continue
            angle = torch.remainder(
                coordinate.double().unsqueeze(-1) * frequency, 2.0 * math.pi
            ).to(q.dtype)
            cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)
            pieces_q.append(_rotate(q[..., cursor : cursor + width], cos, sin))
            pieces_k.append(_rotate(k[..., cursor : cursor + width], cos, sin))
            cursor += width
        if cursor < q.shape[-1]:
            pieces_q.append(q[..., cursor:])
            pieces_k.append(k[..., cursor:])
        return torch.cat(pieces_q, dim=-1), torch.cat(pieces_k, dim=-1)


class EtamLayer(nn.Module):
    def __init__(self, config: EtamConfig) -> None:
        super().__init__()
        width, experts = config.hidden_size, config.num_experts
        deep = config.placement == "deep"
        attention = config.placement in ("attn_ffn", "deep")
        ffn = config.placement != "shared"
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.dropout = config.dropout
        self.attn_norm = _routed(lambda: RMSNorm(width), deep, experts)
        self.qkv = _routed(lambda: QKVProjection(width, config.num_heads), attention, experts)
        self.out = _routed(lambda: nn.Linear(width, width, bias=False), attention, experts)
        self.ffn_norm = _routed(lambda: RMSNorm(width), deep, experts)
        self.ffn = _routed(lambda: SwiGLU(width, config.intermediate_size), ffn, experts)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        route: torch.Tensor,
        mask: torch.Tensor,
        rotary: Rotary,
        position: torch.Tensor,
        days: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attn_norm(hidden, route), route)
        q, k, v = (
            part.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        )
        q, k = rotary(q, k, position, days)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        hidden = hidden + self.residual_dropout(self.out(attended, route))
        hidden = hidden + self.residual_dropout(self.ffn(self.ffn_norm(hidden, route), route))
        return hidden


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


def _fourier(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    scaled = value.unsqueeze(-1) * frequencies
    return torch.cat([value.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)], dim=-1)


class TimeTokenizer(nn.Module):
    """Soft time token from absolute and relative time.

    *Relative*: log gap to the previous event (Fourier features plus a learned
    log-scale bucket embedding, the GEM finding that log bins suit
    heavy-tailed gaps) and log age since the first observed event.
    *Absolute*: hour, weekday, day-of-month and month embeddings, their cyclic
    coordinates, and Fourier features of the absolute day count (weekly to
    yearly periods plus a linear trend).
    For the query slot the gap is the time elapsed since the last event, and a
    learned query embedding marks the token as "now".
    """

    def __init__(self, config: EtamConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.config = config
        self.register_buffer(
            "log_frequencies",
            torch.exp(torch.linspace(math.log(0.25), math.log(16.0), config.num_frequencies)),
            persistent=False,
        )
        periods = torch.tensor([1.0, 7.0, 30.4375, 91.3125, 365.25], dtype=torch.float64)
        self.register_buffer("day_frequencies", 2.0 * math.pi / periods, persistent=False)
        inputs = 0
        if config.time_relative:
            inputs += 2 * (2 * config.num_frequencies + 1) + 1
            self.gap_buckets = nn.Embedding(config.num_gap_buckets, width)
        if config.time_absolute:
            inputs += NUM_CALENDAR_CONT + 2 * self.day_frequencies.numel() + 1
            self.hour = nn.Embedding(24, width)
            self.weekday = nn.Embedding(7, width)
            self.day_of_month = nn.Embedding(31, width)
            self.month = nn.Embedding(12, width)
        self.projection = (
            nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width))
            if inputs
            else None
        )
        self.base = nn.Parameter(torch.zeros(width))
        self.query = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        config = self.config
        hidden = self.base.expand(*batch["log_gap"].shape, -1)
        continuous = []
        if config.time_relative:
            gap = batch["log_gap"]
            continuous += [
                _fourier(gap, self.log_frequencies),
                _fourier(batch["log_age"], self.log_frequencies),
                batch["has_prev"].unsqueeze(-1),
            ]
            bucket = (gap / 20.0 * (config.num_gap_buckets - 1)).long()
            hidden = hidden + self.gap_buckets(bucket.clamp(0, config.num_gap_buckets - 1))
        if config.time_absolute:
            days = batch["abs_days"]
            angle = torch.remainder(days.unsqueeze(-1) * self.day_frequencies, 2.0 * math.pi)
            continuous += [
                batch["calendar_cont"],
                torch.sin(angle).float(),
                torch.cos(angle).float(),
                (days.float() / 365.25).unsqueeze(-1),
            ]
            ids = batch["calendar_ids"]
            hidden = (
                hidden
                + self.hour(ids[..., 0])
                + self.weekday(ids[..., 1])
                + self.day_of_month(ids[..., 2])
                + self.month(ids[..., 3])
            )
        if self.projection is not None:
            hidden = hidden + self.projection(torch.cat(continuous, dim=-1))
        hidden = hidden + batch["is_query"].unsqueeze(-1).to(hidden.dtype) * self.query
        return self.norm(hidden)


class EventTokenizer(nn.Module):
    """Soft event token: mark embedding plus one embedding per feature value."""

    def __init__(self, config: EtamConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.types = nn.Embedding(config.num_event_types, width)
        self.fields = nn.ModuleList(
            [nn.Embedding(size, width, padding_idx=0) for size in config.field_sizes]
        )
        self.mix = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.user = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.types(batch["type_ids"])
        for column, table in enumerate(self.fields):
            hidden = hidden + table(batch["feature_ids"][..., column])
        hidden = hidden + self.mix(hidden)
        query = batch["is_query"].unsqueeze(-1).to(hidden.dtype)
        hidden = hidden * (1.0 - query) + query * self.user
        return self.norm(hidden)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class EtamModel(nn.Module):
    def __init__(self, config: EtamConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.time_tokenizer = TimeTokenizer(config)
        self.event_tokenizer = EventTokenizer(config)
        self.rotary = Rotary(config)
        self.layers = nn.ModuleList([EtamLayer(config) for _ in range(config.num_layers)])
        self.final_norm = _routed(
            lambda: RMSNorm(width), config.placement == "deep", config.num_experts
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "route_map", torch.tensor(config.route_map, dtype=torch.long), persistent=False
        )

        # Heads. `eps`/USR outputs predict the next event; `tau` outputs predict
        # the mark of the event whose time they carry.
        self.type_head = nn.Linear(width, config.num_event_types)
        self.time_head = LogNormalMixtureHead(width, config.num_time_mixture)
        self.cond_type_head = nn.Linear(width, config.num_event_types)
        self.cond_feature_heads = nn.ModuleList(
            [nn.Linear(width, size) for size in config.field_sizes]
        )
        self.future_head = nn.Linear(width, config.num_event_types)
        self.count_head = nn.Linear(width, 1)
        self.cls_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(width, 2))
        self._init_weights()
        # Attached after initialisation so the pretrained weights are untouched.
        self.llm = None
        if config.backbone != "scratch":
            from eventfm.etam.llm_backbone import LlmBackbone

            self.layers = nn.ModuleList()
            self.llm = LlmBackbone(
                config.backbone, config.num_experts, config.lora_rank, config.lora_alpha
            )

    def _init_weights(self) -> None:
        # GPT-2 style: projections that write into the residual stream start
        # smaller so the stream's variance does not grow with depth.
        depth_scale = 1.0 / math.sqrt(2.0 * max(1, self.config.num_layers))
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                writes_residual = ".out.experts." in name or name.endswith(".down")
                std = 0.02 * (depth_scale if writes_residual else 1.0)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        self.time_head.reset_time_prior()

    # -- assembly -----------------------------------------------------------

    def _tokens(self, batch: Dict[str, torch.Tensor]):
        time_tokens = self.time_tokenizer(batch)
        event_tokens = self.event_tokenizer(batch)
        slot_valid = batch["slot_valid"]
        is_query = batch["is_query"]
        batch_size, slots = slot_valid.shape
        event_modality = torch.where(is_query, USER, EVENT)
        if self.config.layout == "token":
            hidden = torch.stack([time_tokens, event_tokens], dim=2).reshape(
                batch_size, 2 * slots, -1
            )
            modality = torch.stack(
                [torch.full_like(event_modality, TIME), event_modality], dim=2
            ).reshape(batch_size, 2 * slots)
            valid = slot_valid.repeat_interleave(2, dim=1)
            position = batch["position"].repeat_interleave(2, dim=1)
            days = batch["abs_days"].repeat_interleave(2, dim=1)
        else:
            hidden = event_tokens + time_tokens
            modality = event_modality
            valid = slot_valid
            position = batch["position"]
            days = batch["abs_days"]
        return hidden, modality, valid, position, days

    @staticmethod
    def _mask(valid: torch.Tensor) -> torch.Tensor:
        length = valid.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
        mask = causal.unsqueeze(0) & valid.unsqueeze(1)
        # Padding rows would otherwise attend to nothing and produce NaNs.
        mask = mask | torch.eye(length, dtype=torch.bool, device=valid.device).unsqueeze(0)
        return mask.unsqueeze(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden, modality, valid, position, days = self._tokens(batch)
        route = self.route_map[modality]
        hidden = self.embedding_dropout(hidden)
        if self.llm is None:
            mask = self._mask(valid)
            for layer in self.layers:
                hidden = layer(hidden, route, mask, self.rotary, position, days)
        else:
            hidden = self.llm(hidden, route, valid, position).float()
        hidden = self.final_norm(hidden, route)
        if self.config.layout == "token":
            time_out = hidden[:, 0::2]
            event_out = hidden[:, 1::2]
        else:
            time_out = None
            event_out = hidden
        # The query slot is always the last one (left padding).
        user = event_out[:, -1]
        return {"event_out": event_out, "time_out": time_out, "user": user}

    # -- heads ---------------------------------------------------------------

    def next_event(self, states: torch.Tensor):
        """Marginal next-mark logits and point estimate of log1p(next gap)."""

        states = states.float()
        return self.type_head(states), self.time_head.mean(states)

    def classify(self, user: torch.Tensor) -> torch.Tensor:
        return self.cls_head(user.float())


@dataclass
class LossWeights:
    time: float = 1.0
    type: float = 1.0
    cond_type: float = 1.0
    cond_feature: float = 0.5
    user_next: float = 1.0
    future: float = 1.0
    count: float = 0.1
    cls: float = 1.0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_losses(
    model: EtamModel,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
    objectives: List[str],
) -> Dict[str, torch.Tensor]:
    """Weighted sum of the requested objectives plus each term for logging.

    ``tpp``     dense next-event NLL at every event slot (time mixture + mark CE)
    ``cond``    mark/feature of event i from its own time token tau_i
    ``user``    next event from the query slot (time measured from "now")
    ``future``  multi-hot marks and log-count in the window after "now"
    ``cls``     sequence label from the user embedding
    """

    terms: Dict[str, torch.Tensor] = {}
    event_out = outputs["event_out"].float()
    next_type = batch["next_type"]
    target_dt = batch["next_log_dt"]
    is_query = batch["is_query"]
    has_next = next_type != -100

    if "tpp" in objectives:
        dense = has_next & ~is_query
        if dense.any():
            states = event_out[dense]
            terms["time"] = model.time_head.nll(states, target_dt[dense]).mean()
            terms["type"] = F.cross_entropy(model.type_head(states), next_type[dense])
    if "user" in objectives:
        user_has_next = has_next[:, -1]
        if user_has_next.any():
            states = outputs["user"].float()[user_has_next]
            nll = model.time_head.nll(states, target_dt[:, -1][user_has_next]).mean()
            ce = F.cross_entropy(model.type_head(states), next_type[:, -1][user_has_next])
            terms["user_next"] = nll + ce
    if "cond" in objectives and outputs["time_out"] is not None:
        time_out = outputs["time_out"].float()
        current = batch["cur_type"] != -100
        if current.any():
            states = time_out[current]
            terms["cond_type"] = F.cross_entropy(
                model.cond_type_head(states), batch["cur_type"][current]
            )
            feature_terms = [
                F.cross_entropy(head(states), batch["cur_features"][..., column][current])
                for column, head in enumerate(model.cond_feature_heads)
            ]
            terms["cond_feature"] = torch.stack(feature_terms).mean()
    if "future" in objectives:
        mask = batch["future_mask"]
        if mask.sum() > 0:
            user = outputs["user"].float()
            bce = F.binary_cross_entropy_with_logits(
                model.future_head(user), batch["future_types"], reduction="none"
            ).mean(dim=-1)
            terms["future"] = _masked_mean(bce, mask)
            squared = (model.count_head(user).squeeze(-1) - batch["future_log_count"]) ** 2
            terms["count"] = _masked_mean(squared, mask)
    if "cls" in objectives:
        labels = batch["labels"]
        labelled = labels >= 0
        if labelled.any():
            logits = model.classify(outputs["user"][labelled])
            terms["cls"] = F.cross_entropy(logits, labels[labelled])

    total = None
    for name, value in terms.items():
        weighted = getattr(weights, name) * value
        total = weighted if total is None else total + weighted
    if total is None:
        total = outputs["user"].sum() * 0.0
    terms["loss"] = total
    return terms


def checkpoint_state(model: EtamModel) -> Dict[str, torch.Tensor]:
    """State dict without the frozen LM weights, which reload from the hub cache."""

    state = model.state_dict()
    if model.llm is None:
        return state
    return {
        key: value
        for key, value in state.items()
        if not key.startswith("llm.llm.") or ".lora_" in key
    }


def load_model_state(model: EtamModel, state: Dict[str, torch.Tensor]) -> None:
    if model.llm is None:
        model.load_state_dict(state)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    stray = [key for key in missing if not key.startswith("llm.llm.") or ".lora_" in key]
    if stray or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch: missing {} unexpected {}".format(stray, unexpected)
        )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad or not trainable_only
        )
    )
