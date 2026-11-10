"""ETAM-2 middle architectures (C): where the event and time processes interact.

``shared``          one fused token per event (z^E + z^T), one causal transformer.
``stage1``          the Stage 1 interleaved two-token layout with routed experts.
``private``         an event pathway and a time pathway (each a causal
                    transformer over n slots) that meet only in lag-aware
                    cross-attention *bridges* after ``bridge_layers`` (none =
                    "two private pathways + simple gating"); fused at the end.
``private_shared``  private pathways plus a shared stream that reads both at
                    every layer (private/shared decomposition).

Bridge (one direction, E<-T shown)::

    alpha_ij = softmax_{j<=i}(<q_i, k_j>/sqrt(d) + b(t_i - t_j, k_i))
    h^E_i += g_i * W_o sum_j alpha_ij v_j,   g_i = sigmoid(W_g [h^E_i; h^T_i; phi(gap_i)])

Every module is causal: slot i only sees slots j <= i, and slot i's inputs are
the history H_i = {(t_j, k_j, a_j): j <= i}.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from eventfm.etam.model import EVENT, TIME, USER, EtamConfig, EtamLayer, EtamModel, RMSNorm, Rotary
from eventfm.etam.stage2.config import Etam2Config


def stage1_layer_config(config: Etam2Config, placement: str = "shared", width: Optional[int] = None) -> EtamConfig:
    return EtamConfig(
        num_event_types=config.num_event_types,
        field_sizes=list(config.field_sizes),
        hidden_size=width or config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_multiplier=config.ffn_multiplier,
        dropout=config.dropout,
        placement=placement,
        rope=config.rope,
    )


class LagBias(nn.Module):
    """b(t_i - t_j, k_i): RBF basis over log1p(lag seconds), per-head weights (+ per query mark)."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        heads, basis = config.num_heads, config.num_lag_basis
        self.per_mark = config.bridge_bias == "lag_mark"
        self.shared = nn.Parameter(torch.zeros(heads, basis))
        if self.per_mark:
            # Marks plus one row for the query (USR) slot.
            self.marks = nn.Embedding(config.num_event_types + 1, heads * basis)
            nn.init.zeros_(self.marks.weight)
        self.heads, self.basis = heads, basis

    def forward(self, basis: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        # basis (B, S, S, R); marks (B, S) -> bias (B, H, S, S)
        if not self.per_mark:
            return torch.einsum("bijr,hr->bhij", basis, self.shared.to(basis.dtype))
        weights = self.marks(marks).view(*marks.shape, self.heads, self.basis) + self.shared
        return torch.einsum("bijr,bihr->bhij", basis, weights.to(basis.dtype))


class CrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key_value = nn.Linear(width, 2 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, queries, keys, attn_mask) -> torch.Tensor:
        batch, slots, width = queries.shape
        q = self.q_norm(self.query(queries).view(batch, slots, self.heads, self.head_dim)).transpose(1, 2)
        k, v = self.key_value(keys).chunk(2, dim=-1)
        k = self.k_norm(k.view(batch, slots, self.heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, slots, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v.to(q.dtype), attn_mask=attn_mask.to(q.dtype), dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(attended.transpose(1, 2).reshape(batch, slots, width))


class Bridge(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        width, heads = config.hidden_size, config.num_heads
        self.norm_event = RMSNorm(width)
        self.norm_time = RMSNorm(width)
        self.event_from_time = CrossAttention(width, heads, config.dropout)
        self.time_from_event = CrossAttention(width, heads, config.dropout)
        self.with_bias = config.bridge_bias != "none"
        if self.with_bias:
            self.bias_event = LagBias(config)
            self.bias_time = LagBias(config)
        self.with_gate = config.bridge_gate
        if self.with_gate:
            self.gate_event = nn.Linear(2 * width + 33, width)
            self.gate_time = nn.Linear(2 * width + 33, width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, event, time, context) -> tuple:
        normed_event, normed_time = self.norm_event(event), self.norm_time(time)
        mask = context["attn_mask"]  # (B, 1, S, S) float: 0 or -inf
        mask_event, mask_time = mask, mask
        if self.with_bias and "lag_basis" in context:
            mask_event = mask + self.bias_event(context["lag_basis"], context["bias_marks"])
            mask_time = mask + self.bias_time(context["lag_basis"], context["bias_marks"])
        update_event = self.event_from_time(normed_event, normed_time, mask_event)
        update_time = self.time_from_event(normed_time, normed_event, mask_time)
        if self.with_gate:
            phi = context["gap_features"].to(normed_event.dtype)
            gate_event = torch.sigmoid(self.gate_event(torch.cat([normed_event, normed_time, phi], -1)))
            gate_time = torch.sigmoid(self.gate_time(torch.cat([normed_time, normed_event, phi], -1)))
            update_event = gate_event * update_event
            update_time = gate_time * update_time
            if context.get("record_gates"):
                context.setdefault("gate_means", []).append(
                    (gate_event.detach().float().mean(-1), gate_time.detach().float().mean(-1))
                )
        return event + self.dropout(update_event), time + self.dropout(update_time)


class Fusion(nn.Module):
    """h_J = RMSNorm(g * h^E + (1 - g) * h^T), g = sigmoid(W [h^E; h^T])."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * width, width)
        self.norm = RMSNorm(width)

    def forward(self, event, time):
        gate = torch.sigmoid(self.gate(torch.cat([event, time], dim=-1)))
        return self.norm(gate * event + (1.0 - gate) * time)


class SharedBackbone(nn.Module):
    def __init__(self, config: Etam2Config, width: Optional[int] = None) -> None:
        super().__init__()
        layer_config = stage1_layer_config(config, "shared", width)
        self.rotary = Rotary(layer_config)
        self.layers = nn.ModuleList([EtamLayer(layer_config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(layer_config.hidden_size)

    def forward(self, time, event, batch, context):
        hidden = event + time
        route = torch.zeros(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        mask = context["bool_mask"]
        for layer in self.layers:
            hidden = layer(hidden, route, mask, self.rotary, batch["position"], batch["abs_days"])
        return {"h": self.norm(hidden)}


class Stage1Backbone(nn.Module):
    """Stage 1's interleaved tau/eps layout with routed (deep) experts, as a mixer."""

    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        layer_config = stage1_layer_config(config, config.placement)
        self.layer_config = layer_config
        self.rotary = Rotary(layer_config)
        self.layers = nn.ModuleList([EtamLayer(layer_config) for _ in range(config.num_layers)])
        from eventfm.etam.model import _routed

        self.final_norm = _routed(
            lambda: RMSNorm(config.hidden_size), config.placement == "deep", layer_config.num_experts
        )
        self.register_buffer(
            "route_map", torch.tensor(layer_config.route_map, dtype=torch.long), persistent=False
        )

    def forward(self, time, event, batch, context):
        valid, is_query = batch["slot_valid"], batch["is_query"]
        batch_size, slots = valid.shape
        hidden = torch.stack([time, event], dim=2).reshape(batch_size, 2 * slots, -1)
        event_modality = torch.where(is_query, USER, EVENT)
        modality = torch.stack([torch.full_like(event_modality, TIME), event_modality], dim=2).reshape(
            batch_size, 2 * slots
        )
        route = self.route_map[modality]
        mask = EtamModel._mask(valid.repeat_interleave(2, dim=1))
        position = batch["position"].repeat_interleave(2, dim=1)
        days = batch["abs_days"].repeat_interleave(2, dim=1)
        for layer in self.layers:
            hidden = layer(hidden, route, mask, self.rotary, position, days)
        hidden = self.final_norm(hidden, route)
        return {"h": hidden[:, 1::2], "h_time_tokens": hidden[:, 0::2]}


class PrivateBackbone(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        layer_config = stage1_layer_config(config, "shared")
        width = config.hidden_size
        self.rotary = Rotary(layer_config)
        self.event_layers = nn.ModuleList([EtamLayer(layer_config) for _ in range(config.num_layers)])
        self.time_layers = nn.ModuleList([EtamLayer(layer_config) for _ in range(config.num_layers)])
        self.bridge_layers = sorted(set(config.bridge_layers)) if config.mixer == "private" else []
        self.bridges = nn.ModuleDict({str(layer): Bridge(config) for layer in self.bridge_layers})
        self.unimodal_at = self.bridge_layers[0] if self.bridge_layers else config.num_layers
        self.with_shared = config.mixer == "private_shared"
        if self.with_shared:
            self.shared_layers = nn.ModuleList([EtamLayer(layer_config) for _ in range(config.num_layers)])
            self.read_event = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(config.num_layers)])
            self.read_time = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(config.num_layers)])
            self.shared_norm = RMSNorm(width)
        self.norm_event = RMSNorm(width)
        self.norm_time = RMSNorm(width)
        self.norm_pre_event = RMSNorm(width)
        self.norm_pre_time = RMSNorm(width)
        self.fusion = Fusion(width)

    def forward(self, time, event, batch, context):
        route = torch.zeros(event.shape[:2], dtype=torch.long, device=event.device)
        mask, position, days = context["bool_mask"], batch["position"], batch["abs_days"]
        shared = event + time if self.with_shared else None
        pre_event = pre_time = None
        for index in range(len(self.event_layers)):
            layer = index + 1
            event = self.event_layers[index](event, route, mask, self.rotary, position, days)
            time = self.time_layers[index](time, route, mask, self.rotary, position, days)
            if layer == self.unimodal_at:
                pre_event, pre_time = event, time
            if str(layer) in self.bridges:
                event, time = self.bridges[str(layer)](event, time, context)
            if self.with_shared:
                shared = shared + self.read_event[index](event) + self.read_time[index](time)
                shared = self.shared_layers[index](shared, route, mask, self.rotary, position, days)
        event, time = self.norm_event(event), self.norm_time(time)
        fused = self.fusion(event, time)
        if self.with_shared:
            fused = fused + self.shared_norm(shared)
        return {
            "h": fused,
            "h_event": event,
            "h_time": time,
            "pre_event": self.norm_pre_event(pre_event),
            "pre_time": self.norm_pre_time(pre_time),
        }


def build_backbone(config: Etam2Config) -> nn.Module:
    if config.mixer == "shared":
        return SharedBackbone(config)
    if config.mixer == "stage1":
        return Stage1Backbone(config)
    if config.mixer in ("private", "private_shared"):
        return PrivateBackbone(config)
    from eventfm.etam.stage2.baselines import build_baseline_backbone

    return build_baseline_backbone(config)


def attention_context(batch: Dict[str, torch.Tensor], num_basis: int, need_basis: bool, num_marks: int,
                      inputs: str = "both"):
    """Masks and lag features shared by every bridge (computed once per forward).

    Restricted-input arms must not see the excluded modality anywhere: with
    ``event_only`` the gates get no gap features (and the model builds no lag bias),
    with ``time_only`` the lag bias gets no marks.
    """

    from eventfm.etam.stage2.tokenizers import lag_seconds

    valid = batch["slot_valid"]
    bool_mask = EtamModel._mask(valid)  # (B, 1, S, S): j <= i, valid j, plus the diagonal
    context = {"bool_mask": bool_mask}
    float_mask = torch.zeros(bool_mask.shape, device=valid.device, dtype=torch.float32)
    context["attn_mask"] = float_mask.masked_fill(~bool_mask, float("-inf"))
    gap = batch["log_gap"].float()
    frequencies = torch.exp(torch.linspace(0.0, 3.0, 16, device=gap.device))
    angle = (gap / 20.0).unsqueeze(-1) * frequencies * 3.14159
    context["gap_features"] = torch.cat([(gap / 20.0).unsqueeze(-1), torch.sin(angle), torch.cos(angle)], -1)
    if inputs == "event_only":
        context["gap_features"] = torch.zeros_like(context["gap_features"])
    if need_basis:
        lag = torch.log1p(lag_seconds(batch))
        centres = torch.linspace(0.0, 18.0, num_basis, device=lag.device)
        width = 18.0 / max(1, num_basis - 1)
        context["lag_basis"] = torch.exp(-0.5 * ((lag.unsqueeze(-1) - centres) / width) ** 2).to(torch.bfloat16 if lag.is_cuda else torch.float32)
        marks = batch["type_ids"].clone()
        if inputs == "time_only":
            marks.fill_(0)
        marks[batch["is_query"]] = num_marks
        context["bias_marks"] = marks
    return context
