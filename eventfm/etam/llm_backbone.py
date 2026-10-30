"""Frozen pretrained LM as the ETAM mixer, with token-conditional LoRA experts.

This is RAM's design (`~/ST/docs_RAM/methodology.md` §4.5) with time and event
as the two non-text modalities: every Q/K/V/O and FFN projection of the frozen
LM gets one LoRA delta per modality, ``W^m = W + B^m A^m`` with ``B^m`` zero
initialised, so step 0 is the pretrained LM reading time and event soft
tokens. Norms stay the LM's own (the LoRA form of deep separation). Soft tokens
enter at the RMS of the LM's real input embeddings (RAM's calibration).
"""

from typing import Optional

import torch
from torch import nn

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class RouteContext:
    """Holds the current batch's per-token expert index for every LoRA layer."""

    def __init__(self) -> None:
        self.route: Optional[torch.Tensor] = None


class RoutedLoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, num_experts: int, rank: int, alpha: float, context: RouteContext
    ) -> None:
        super().__init__()
        self.base = base
        self.context = context
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(num_experts, rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(num_experts, base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a.view(-1, base.in_features), a=5**0.5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden)
        route = self.context.route
        if route is None:
            raise RuntimeError("RoutedLoRALinear called without a route.")
        delta = None
        for index in range(self.lora_a.shape[0]):
            mask = (route == index).unsqueeze(-1).to(hidden.dtype)
            low = hidden @ self.lora_a[index].to(hidden.dtype).t()
            update = (low @ self.lora_b[index].to(hidden.dtype).t()) * mask
            delta = update if delta is None else delta + update
        return output + self.scale * delta


class LlmBackbone(nn.Module):
    def __init__(self, name: str, num_experts: int, rank: int, alpha: float) -> None:
        super().__init__()
        from transformers import AutoModel

        self.llm = AutoModel.from_pretrained(name, dtype=torch.bfloat16)
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.context = RouteContext()
        targets = [
            (parent, child_name, child)
            for parent in self.llm.modules()
            for child_name, child in parent.named_children()
            if child_name in LORA_TARGETS and isinstance(child, nn.Linear)
        ]
        for parent, child_name, child in targets:
            setattr(
                parent, child_name, RoutedLoRALinear(child, num_experts, rank, alpha, self.context)
            )
        with torch.no_grad():
            table = self.llm.get_input_embeddings().weight.float()
            rho = table.pow(2).mean(dim=-1).sqrt().median()
        self.register_buffer("input_scale", rho.reshape(()), persistent=False)

    @property
    def hidden_size(self) -> int:
        return int(self.llm.config.hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,
        route: torch.Tensor,
        valid: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        self.context.route = route
        try:
            outputs = self.llm(
                inputs_embeds=(hidden * self.input_scale).to(torch.bfloat16),
                attention_mask=valid.long(),
                position_ids=position,
                use_cache=False,
                return_dict=True,
            )
        finally:
            self.context.route = None
        return outputs.last_hidden_state


def llm_hidden_size(name: str) -> int:
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(name).hidden_size)
