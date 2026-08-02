"""Language-model backbones for the three LM-based TPP entries.

A pretrained causal LM encodes the serialised history; a pooled representation
of the final position drives the same two heads the rest of the zoo uses, so
the tables compare like with like. What differs between the entries is the
*input*: TPP-LLM adds a continuous temporal embedding to a text-only prompt,
while Language-TPP and MM-TPP carry the timing as byte tokens inside the text.

Parameter-efficient adaptation follows TPP-LLM: the base weights stay frozen and
LoRA adapters plus the heads are trained.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput

from eventfm.models.modeling_flat import FourierTimeFeatures, LogNormalMixtureHead


@dataclass
class LlmNextEventOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    type_logits: torch.FloatTensor = None
    delta_log: torch.FloatTensor = None


class LlmTppConfig(PretrainedConfig):
    model_type = "llm_tpp"

    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        num_event_types: int = 16,
        num_labels: int = 2,
        num_added_tokens: int = 0,
        use_temporal_embedding: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        classifier_dropout: float = 0.1,
        num_time_mixture_components: int = 8,
        tpp_loss_weight: float = 1.0,
        freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(num_labels=num_labels, **kwargs)
        self.base_model_name = str(base_model_name)
        self.num_event_types = int(num_event_types)
        self.num_added_tokens = int(num_added_tokens)
        self.use_temporal_embedding = bool(use_temporal_embedding)
        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.classifier_dropout = float(classifier_dropout)
        self.num_time_mixture_components = int(num_time_mixture_components)
        self.tpp_loss_weight = float(tpp_loss_weight)
        self.freeze_base = bool(freeze_base)


def _apply_lora(model: nn.Module, config: LlmTppConfig) -> nn.Module:
    if not config.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:  # pragma: no cover - peft is a declared dependency
        return model

    target_modules = _lora_targets(model)
    if not target_modules:
        return model
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, peft_config)


def _lora_targets(model: nn.Module):
    """Pick attention/MLP projection names that exist in this architecture."""

    candidates = [
        {"q_proj", "k_proj", "v_proj", "o_proj"},
        {"c_attn", "c_proj"},
        {"query", "key", "value"},
    ]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    for group in candidates:
        matched = sorted(group & present)
        if matched:
            return matched
    return []


class LlmTppBackbone(nn.Module):
    """Frozen causal LM + LoRA, with an optional continuous temporal embedding."""

    def __init__(self, config: LlmTppConfig) -> None:
        super().__init__()
        self.config = config
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        # Load in float32 even when the checkpoint ships bfloat16 weights: the
        # task heads are float32, and mixed precision is the Trainer's job via
        # `bf16=True` autocast, not the checkpoint's.
        base = AutoModel.from_pretrained(
            config.base_model_name, config=base_config, dtype=torch.float32
        )
        if config.num_added_tokens:
            base.resize_token_embeddings(
                base.get_input_embeddings().weight.shape[0] + config.num_added_tokens
            )
        if config.freeze_base:
            for parameter in base.parameters():
                parameter.requires_grad = False
        self.hidden_size = int(
            getattr(base_config, "hidden_size", getattr(base_config, "n_embd", 768))
        )
        self.base = _apply_lora(base, config)

        if config.use_temporal_embedding:
            # TPP-LLM's temporal component: continuous time enters through the
            # embedding space rather than through the token stream.
            self.temporal_encoder = FourierTimeFeatures(self.hidden_size, num_frequencies=16)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recent_log_delta: Optional[torch.Tensor] = None,
        history_log_span: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state

        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        index = lengths.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        pooled = hidden.gather(dim=1, index=index).squeeze(1)

        if self.config.use_temporal_embedding and recent_log_delta is not None:
            pooled = pooled + self.temporal_encoder(recent_log_delta.to(dtype=pooled.dtype))
            if history_log_span is not None:
                pooled = pooled + self.temporal_encoder(history_log_span.to(dtype=pooled.dtype))
        # Under autocast the backbone may still hand back a reduced dtype; the
        # heads are float32, so normalise here rather than in three call sites.
        return pooled.float()


class LlmTppPreTrainedModel(PreTrainedModel):
    config_class = LlmTppConfig
    base_model_prefix = "llm_tpp"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class LlmTppForSequenceClassification(LlmTppPreTrainedModel):
    def __init__(self, config: LlmTppConfig) -> None:
        super().__init__(config)
        self.backbone = LlmTppBackbone(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(self.backbone.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recent_log_delta: Optional[torch.Tensor] = None,
        history_log_span: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        del kwargs
        pooled = self.backbone(input_ids, attention_mask, recent_log_delta, history_log_span)
        logits = self.classifier(self.dropout(pooled))
        loss = None if labels is None else F.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits)


class LlmTppForNextEventPrediction(LlmTppPreTrainedModel):
    def __init__(self, config: LlmTppConfig) -> None:
        super().__init__(config)
        self.backbone = LlmTppBackbone(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.type_head = nn.Linear(self.backbone.hidden_size, config.num_event_types)
        self.time_head = LogNormalMixtureHead(
            self.backbone.hidden_size, config.num_time_mixture_components
        )
        self.time_head.reset_time_prior()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recent_log_delta: Optional[torch.Tensor] = None,
        history_log_span: Optional[torch.Tensor] = None,
        next_event_type_labels: Optional[torch.Tensor] = None,
        next_delta_log: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LlmNextEventOutput:
        del kwargs
        pooled = self.dropout(
            self.backbone(input_ids, attention_mask, recent_log_delta, history_log_span)
        )
        type_logits = self.type_head(pooled)
        delta_log = self.time_head.mean(pooled)

        loss = None
        if next_event_type_labels is not None and next_delta_log is not None:
            type_loss = F.cross_entropy(type_logits, next_event_type_labels.long())
            time_loss = self.time_head.nll(pooled, next_delta_log.to(dtype=pooled.dtype)).mean()
            loss = type_loss + self.config.tpp_loss_weight * time_loss
        return LlmNextEventOutput(loss=loss, type_logits=type_logits, delta_log=delta_log)
