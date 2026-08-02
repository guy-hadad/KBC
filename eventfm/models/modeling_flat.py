"""Flat event-sequence backbones and task heads.

The architectures surveyed in the project note differ along two axes only:

* how the event history is mixed (recurrent, bidirectional attention, causal
  attention, or a state-space scan), and
* what objective sits on top (classification, next-event prediction with a
  proper inter-arrival density, contrastive, or masked reconstruction).

This module implements the axes separately so a fair comparison is possible:
the embedding of an event is identical everywhere and only the mixer and the
head change.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, GPT2Config, PreTrainedModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from eventfm.models.configuration_flat import FlatEventConfig


@dataclass
class FlatBackboneOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    pooled: torch.FloatTensor = None
    last_hidden: torch.FloatTensor = None


@dataclass
class NextEventOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    type_logits: torch.FloatTensor = None
    delta_log: torch.FloatTensor = None
    time_nll: Optional[torch.FloatTensor] = None


class FourierTimeFeatures(nn.Module):
    """Smooth encoding of a log-seconds scalar into ``hidden_size`` dimensions."""

    def __init__(self, hidden_size: int, num_frequencies: int = 16) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(512.0), num_frequencies))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(2 * num_frequencies + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, log_seconds: torch.Tensor) -> torch.Tensor:
        scaled = log_seconds.unsqueeze(-1) / self.frequencies.to(
            device=log_seconds.device, dtype=log_seconds.dtype
        )
        features = torch.cat(
            [log_seconds.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)], dim=-1
        )
        return self.projection(features)


class FlatEventEmbedding(nn.Module):
    """Mark + feature-value + time embedding for one event."""

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__()
        self.config = config
        self.mark_embeddings = nn.Embedding(config.num_event_types, config.hidden_size)
        self.value_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.field_embeddings = nn.Embedding(config.num_feature_fields, config.hidden_size)
        if config.use_time_features:
            self.delta_encoder = FourierTimeFeatures(
                config.hidden_size, config.time_encoding_frequencies
            )
            self.age_encoder = FourierTimeFeatures(
                config.hidden_size, config.time_encoding_frequencies
            )
        if config.use_calendar_features:
            self.calendar_encoder = nn.Sequential(
                nn.Linear(config.calendar_feature_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        event_type_ids: torch.Tensor,
        feature_value_ids: torch.Tensor,
        delta_log: torch.Tensor,
        time_since_start: torch.Tensor,
        calendar_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.mark_embeddings(event_type_ids)

        width = feature_value_ids.shape[-1]
        field_ids = torch.arange(width, device=feature_value_ids.device).clamp(
            max=self.config.num_feature_fields - 1
        )
        values = self.value_embeddings(feature_value_ids) + self.field_embeddings(field_ids)
        hidden = hidden + values.sum(dim=2)

        if self.config.use_time_features:
            hidden = hidden + self.delta_encoder(delta_log) + self.age_encoder(time_since_start)
        if self.config.use_calendar_features:
            hidden = hidden + self.calendar_encoder(calendar_features)
        return self.dropout(self.layer_norm(hidden))


def _extended_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class _GruMixer(nn.Module):
    """Recurrent history encoder, the default in the neural-TPP review."""

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.dropout if config.num_hidden_layers > 1 else 0.0,
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(hidden)
        return output * attention_mask.unsqueeze(-1).to(output.dtype)


class _BertMixer(nn.Module):
    """Bidirectional attention over events (TabBERT, Transformer-Hawkes-style)."""

    def __init__(self, config: FlatEventConfig, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.encoder = BertEncoder(
            BertConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
                max_position_embeddings=config.max_position_embeddings,
                type_vocab_size=1,
            )
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device).clamp(
            max=self.position_embeddings.num_embeddings - 1
        )
        hidden = hidden + self.position_embeddings(positions)
        extended = _extended_mask(attention_mask, hidden.dtype)
        if self.causal:
            causal = torch.tril(torch.ones(length, length, device=hidden.device, dtype=torch.bool))
            causal_bias = torch.where(
                causal,
                torch.zeros((), device=hidden.device, dtype=hidden.dtype),
                torch.full(
                    (), torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype
                ),
            )
            extended = extended + causal_bias[None, None, :, :]
        return self.encoder(hidden, attention_mask=extended, return_dict=True).last_hidden_state


class _Gpt2Mixer(nn.Module):
    """Decoder-only mixer, matching the NVIDIA blueprint and TabGPT."""

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__()
        gpt_config = GPT2Config(
            vocab_size=max(2, config.vocab_size),
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
        )
        self.transformer = GPT2Model(gpt_config)
        # Event embeddings are supplied directly, so the token table is unused.
        self.transformer.wte = nn.Identity()

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer(
            inputs_embeds=hidden, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state


class _MambaMixer(nn.Module):
    """Selective state-space mixer (Mambular / DeepTab).

    Uses the ``transformers`` Mamba implementation, which falls back to a pure
    PyTorch scan when the fused CUDA kernels are unavailable, so the same code
    runs on CPU smoke tests and on GPU jobs.
    """

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__()
        from transformers import MambaConfig
        from transformers.models.mamba.modeling_mamba import MambaModel

        mamba_config = MambaConfig(
            vocab_size=max(2, config.vocab_size),
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            state_size=16,
            expand=2,
            conv_kernel=4,
            use_cache=False,
        )
        self.mamba = MambaModel(mamba_config)
        self.mamba.embeddings = nn.Identity()

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.mamba(
            inputs_embeds=hidden, use_cache=False, return_dict=True
        ).last_hidden_state
        return output * attention_mask.unsqueeze(-1).to(output.dtype)


MIXERS = {
    "gru": lambda config: _GruMixer(config),
    "bert": lambda config: _BertMixer(config, causal=False),
    "bert-causal": lambda config: _BertMixer(config, causal=True),
    "gpt2": lambda config: _Gpt2Mixer(config),
    "mamba": lambda config: _MambaMixer(config),
}


class LogNormalMixtureHead(nn.Module):
    """Intensity-free inter-arrival density, as recommended by Shchur et al.

    A mixture over ``log1p(delta_seconds)`` gives a proper likelihood for the
    *when* half of the TPP task and a closed-form mean for the point estimate
    that the shared RMSE metric consumes.
    """

    # Inter-arrival times in these benchmarks sit around a few hours, i.e.
    # log1p(seconds) near 10. Starting the mixture means there instead of at
    # zero is the difference between a usable RMSE and one dominated by the
    # initial offset.
    prior_log_delta = math.log1p(6.0 * 3600.0)

    def __init__(self, hidden_size: int, num_components: int = 8) -> None:
        super().__init__()
        self.num_components = int(num_components)
        self.projection = nn.Linear(hidden_size, 3 * self.num_components)
        self.reset_time_prior()

    @torch.no_grad()
    def reset_time_prior(self) -> None:
        """Re-seat the mixture prior.

        Must be called *after* ``post_init``: the generic weight initialiser of
        the parent model zeroes every bias, which would otherwise place the
        whole mixture at ``log1p(delta) = 0``.
        """

        self.projection.weight.normal_(mean=0.0, std=0.01)
        self.projection.bias[: self.num_components].zero_()
        spread = torch.linspace(-2.0, 2.0, self.num_components)
        self.projection.bias[self.num_components : 2 * self.num_components] = (
            self.prior_log_delta + spread
        )
        self.projection.bias[2 * self.num_components :].fill_(0.0)

    def forward(self, hidden: torch.Tensor):
        weights_logits, means, log_scales = self.projection(hidden).chunk(3, dim=-1)
        log_scales = log_scales.clamp(-6.0, 4.0)
        return weights_logits, means, log_scales

    def mean(self, hidden: torch.Tensor) -> torch.Tensor:
        weights_logits, means, _ = self(hidden)
        return (torch.softmax(weights_logits, dim=-1) * means).sum(dim=-1)

    def nll(self, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights_logits, means, log_scales = self(hidden)
        log_weights = torch.log_softmax(weights_logits, dim=-1)
        target = target.unsqueeze(-1)
        scales = torch.exp(log_scales)
        log_prob = (
            -0.5 * math.log(2.0 * math.pi)
            - log_scales
            - 0.5 * ((target - means) / scales) ** 2
        )
        return -torch.logsumexp(log_weights + log_prob, dim=-1)


class FlatPreTrainedModel(PreTrainedModel):
    config_class = FlatEventConfig
    base_model_prefix = "flat"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FlatEventBackbone(nn.Module):
    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = FlatEventEmbedding(config)
        if config.backbone not in MIXERS:
            raise ValueError(
                "Unknown backbone `{}`. Available: {}".format(config.backbone, sorted(MIXERS))
            )
        self.mixer = MIXERS[config.backbone](config)

    def forward(
        self,
        event_type_ids: torch.Tensor,
        feature_value_ids: torch.Tensor,
        delta_log: torch.Tensor,
        time_since_start: torch.Tensor,
        calendar_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> FlatBackboneOutput:
        hidden = self.embeddings(
            event_type_ids=event_type_ids,
            feature_value_ids=feature_value_ids,
            delta_log=delta_log,
            time_since_start=time_since_start,
            calendar_features=calendar_features,
        )
        hidden = self.mixer(hidden, attention_mask)

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        last_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        last_hidden = hidden.gather(dim=1, index=last_index).squeeze(1)

        if self.config.pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / lengths.unsqueeze(-1).to(hidden.dtype)
        elif self.config.pooling == "max":
            pooled = hidden.masked_fill(mask == 0, torch.finfo(hidden.dtype).min).max(dim=1).values
        else:
            pooled = last_hidden
        return FlatBackboneOutput(hidden_states=hidden, pooled=pooled, last_hidden=last_hidden)


class FlatForSequenceClassification(FlatPreTrainedModel):
    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__(config)
        self.backbone = FlatEventBackbone(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        event_type_ids: torch.Tensor,
        feature_value_ids: torch.Tensor,
        delta_log: torch.Tensor,
        time_since_start: torch.Tensor,
        calendar_features: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        del kwargs
        outputs = self.backbone(
            event_type_ids=event_type_ids,
            feature_value_ids=feature_value_ids,
            delta_log=delta_log,
            time_since_start=time_since_start,
            calendar_features=calendar_features,
            attention_mask=attention_mask,
        )
        logits = self.classifier(self.dropout(outputs.pooled))
        loss = None if labels is None else F.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits)


class FlatForNextEventPrediction(FlatPreTrainedModel):
    """Marked next-event head: mark classifier plus inter-arrival density."""

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__(config)
        self.backbone = FlatEventBackbone(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.type_head = nn.Linear(config.hidden_size, config.num_event_types)
        if config.time_loss == "lognormal_mixture":
            self.time_head = LogNormalMixtureHead(
                config.hidden_size, config.num_time_mixture_components
            )
        else:
            self.time_head = nn.Linear(config.hidden_size, 1)
        self.post_init()
        # `post_init` runs the generic initialiser over every submodule, so the
        # time prior has to be re-applied afterwards.
        if isinstance(self.time_head, LogNormalMixtureHead):
            self.time_head.reset_time_prior()
        else:
            self.time_head.bias.data.fill_(math.log1p(6.0 * 3600.0))

    def forward(
        self,
        event_type_ids: torch.Tensor,
        feature_value_ids: torch.Tensor,
        delta_log: torch.Tensor,
        time_since_start: torch.Tensor,
        calendar_features: torch.Tensor,
        attention_mask: torch.Tensor,
        next_event_type_labels: Optional[torch.Tensor] = None,
        next_delta_log: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NextEventOutput:
        del kwargs
        outputs = self.backbone(
            event_type_ids=event_type_ids,
            feature_value_ids=feature_value_ids,
            delta_log=delta_log,
            time_since_start=time_since_start,
            calendar_features=calendar_features,
            attention_mask=attention_mask,
        )
        hidden = self.dropout(outputs.last_hidden)
        type_logits = self.type_head(hidden)

        if isinstance(self.time_head, LogNormalMixtureHead):
            predicted_delta = self.time_head.mean(hidden)
        else:
            predicted_delta = self.time_head(hidden).squeeze(-1)

        loss = None
        # Reported per example so the Trainer can concatenate it across batches.
        time_nll = None
        if next_event_type_labels is not None and next_delta_log is not None:
            type_loss = F.cross_entropy(type_logits, next_event_type_labels.long())
            target = next_delta_log.to(dtype=predicted_delta.dtype)
            if isinstance(self.time_head, LogNormalMixtureHead):
                time_nll = self.time_head.nll(hidden, target)
                time_loss = time_nll.mean()
            else:
                time_nll = 0.5 * (predicted_delta - target) ** 2
                time_loss = time_nll.mean()
            loss = type_loss + self.config.tpp_loss_weight * time_loss
        return NextEventOutput(
            loss=loss,
            type_logits=type_logits,
            delta_log=predicted_delta,
            time_nll=time_nll,
        )


class FlatForMaskedEventModeling(FlatPreTrainedModel):
    """Masked mark reconstruction, the TabBERT / PRAGMA pretraining objective."""

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__(config)
        self.backbone = FlatEventBackbone(config)
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.decoder = nn.Linear(config.hidden_size, config.num_event_types)
        self.post_init()

    def forward(
        self,
        event_type_ids: torch.Tensor,
        feature_value_ids: torch.Tensor,
        delta_log: torch.Tensor,
        time_since_start: torch.Tensor,
        calendar_features: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        outputs = self.backbone(
            event_type_ids=event_type_ids,
            feature_value_ids=feature_value_ids,
            delta_log=delta_log,
            time_since_start=time_since_start,
            calendar_features=calendar_features,
            attention_mask=attention_mask,
        )
        logits = self.decoder(self.transform(outputs.hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
        return ModelOutput(loss=loss, logits=logits)
