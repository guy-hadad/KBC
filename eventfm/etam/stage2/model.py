"""ETAM-2: tokenizers -> (private mixers + bridges | alternatives) -> dual decoders."""

import math
from typing import Dict

import torch
from torch import nn

from eventfm.etam.stage2.backbone import LagBias, attention_context, build_backbone
from eventfm.etam.stage2.config import Etam2Config
from eventfm.etam.stage2.heads import (
    CountHead,
    FeatureHead,
    JointDecoder,
    MultiEventHeads,
    ReprHeads,
    TimeHead,
    UserHead,
)
from eventfm.etam.stage2.tokenizers import (
    EventTokenizer2,
    TimeTokenizer2,
    VectorQuantizer,
    lag_seconds,
    strict_past_mask,
)


class Etam2Model(nn.Module):
    def __init__(self, config: Etam2Config) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.event_tokenizer = EventTokenizer2(config)
        self.time_tokenizer = TimeTokenizer2(config)
        self.backbone = build_backbone(config)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.decoder = JointDecoder(config)
        self.feature_head = FeatureHead(config) if config.feature_head else None
        self.has_unimodal = config.unimodal_heads and config.private
        if self.has_unimodal:
            self.unimodal_event = nn.Linear(width, config.num_event_types)
            self.unimodal_time = TimeHead(config)
        self.mtp = MultiEventHeads(config) if config.mtp_horizon > 1 else None
        self.counts = CountHead(config) if config.count_dist != "none" else None
        self.user_head = UserHead(config)
        modalities = tuple(m for m in config.repr_modalities if m == "joint" or config.private)
        self.repr = ReprHeads(config, modalities) if config.repr_dim else None
        self._init_weights()

    def _init_weights(self) -> None:
        depth_scale = 1.0 / math.sqrt(2.0 * max(1, self.config.num_layers))
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                writes_residual = (
                    ".out.experts." in name or name.endswith(".down") or name.endswith(".out")
                )
                nn.init.normal_(module.weight, 0.0, 0.02 * (depth_scale if writes_residual else 1.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        for module in self.modules():
            if isinstance(module, TimeHead):
                module.reset_prior()
            elif isinstance(module, LagBias):
                nn.init.zeros_(module.shared)
                if module.per_mark:
                    nn.init.zeros_(module.marks.weight)
            elif isinstance(module, VectorQuantizer):
                nn.init.normal_(module.codebook.weight, std=1.0)

    @property
    def needs_lag(self) -> bool:
        return self.config.memory in ("exp", "exp_learned", "count", "ssm")

    @property
    def needs_basis(self) -> bool:
        config = self.config
        return (
            config.mixer == "private"
            and bool(config.bridge_layers)
            and config.bridge_bias != "none"
            and config.inputs != "event_only"
        )

    def forward(self, batch: Dict[str, torch.Tensor], record_gates: bool = False) -> Dict[str, object]:
        event = self.event_tokenizer(batch)
        lag = past = None
        if self.needs_lag and self.config.inputs != "event_only":
            lag, past = lag_seconds(batch), strict_past_mask(batch)
        time, aux = self.time_tokenizer(batch, event, lag, past)
        context = attention_context(
            batch, self.config.num_lag_basis, self.needs_basis, self.config.num_event_types, self.config.inputs
        )
        context["record_gates"] = record_gates
        outputs = self.backbone(self.embedding_dropout(time), self.embedding_dropout(event), batch, context)
        hidden = outputs["h"]
        user_joint = hidden[:, -1]
        if "h_event" in outputs:
            user = self.user_head.embed(user_joint, outputs["h_event"][:, -1], outputs["h_time"][:, -1])
        else:
            user = self.user_head.embed(user_joint)
        outputs.update(user=user, user_joint=user_joint, aux=aux, gates=context.get("gate_means", []))
        return outputs

    def classify(self, user: torch.Tensor) -> torch.Tensor:
        return self.user_head.cls_head(user.float())


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only))
