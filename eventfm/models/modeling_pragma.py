"""PRAGMA-style hierarchical transformer models for event sequences."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEncoder

from eventfm.models.configuration_pragma import PragmaConfig


@dataclass
class PragmaBackboneOutput(ModelOutput):
    user_embedding: torch.FloatTensor = None
    event_embeddings: torch.FloatTensor = None
    local_feature_embeddings: torch.FloatTensor = None
    last_event_embedding: torch.FloatTensor = None


@dataclass
class NextEventPredictionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    type_logits: torch.FloatTensor = None
    delta_log: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None


def _encoder_config(config: PragmaConfig, num_hidden_layers: int) -> BertConfig:
    return BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        pad_token_id=config.pad_token_id,
        max_position_embeddings=2,
        type_vocab_size=1,
        is_decoder=False,
    )


def _extended_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class FourierTimeEncoding(nn.Module):
    """Continuous timestamp encoding over log-seconds.

    This is intentionally simple and robust. It gives the history encoder a
    smooth representation of event age, while the calendar MLP captures periodic
    hour/day/month effects.
    """

    def __init__(self, hidden_size: int, num_frequencies: int = 16) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(365.0 * 24.0), num_frequencies))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(2 * num_frequencies + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        hours = torch.log1p(torch.clamp(seconds, min=0.0) / 3600.0)
        scaled = hours.unsqueeze(-1) / self.frequencies.to(device=seconds.device, dtype=seconds.dtype)
        features = torch.cat([hours.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.projection(features)


class PragmaPreTrainedModel(PreTrainedModel):
    config_class = PragmaConfig
    base_model_prefix = "pragma"
    supports_gradient_checkpointing = True

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


class PragmaBackbone(nn.Module):
    """Hierarchical encoder: feature tokens -> event embeddings -> user history."""

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.feature_position_embeddings = nn.Embedding(
            config.feature_position_vocab_size,
            config.hidden_size,
        )
        self.event_encoder = BertEncoder(_encoder_config(config, config.event_num_hidden_layers))
        self.profile_encoder = BertEncoder(_encoder_config(config, config.profile_num_hidden_layers))
        self.history_encoder = BertEncoder(_encoder_config(config, config.history_num_hidden_layers))
        self.calendar_encoder = nn.Sequential(
            nn.Linear(config.calendar_feature_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.time_encoder = FourierTimeEncoding(
            hidden_size=config.hidden_size,
            num_frequencies=config.time_encoding_frequencies,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def _special_token(self, token_id: int, shape: tuple, device: torch.device) -> torch.Tensor:
        ids = torch.full(shape, int(token_id), dtype=torch.long, device=device)
        return self.token_embeddings(ids)

    def _encode_profile(
        self,
        profile_key_ids: Optional[torch.Tensor],
        profile_value_ids: Optional[torch.Tensor],
        profile_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if profile_key_ids is None or profile_value_ids is None or profile_attention_mask is None:
            return self._special_token(self.config.usr_token_id, (batch_size, 1), device)

        token_embeddings = self.token_embeddings(profile_key_ids) + self.token_embeddings(profile_value_ids)
        usr = self._special_token(self.config.usr_token_id, (batch_size, 1), device)
        hidden_states = torch.cat([usr, token_embeddings], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=profile_attention_mask.dtype, device=device),
                profile_attention_mask,
            ],
            dim=1,
        )
        encoded = self.profile_encoder(
            hidden_states,
            attention_mask=_extended_attention_mask(attention_mask, hidden_states.dtype),
            return_dict=True,
        ).last_hidden_state
        return encoded[:, :1, :]

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        feature_position_ids: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_age_seconds: torch.Tensor,
        calendar_features: torch.Tensor,
        profile_key_ids: Optional[torch.Tensor] = None,
        profile_value_ids: Optional[torch.Tensor] = None,
        profile_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PragmaBackboneOutput:
        del kwargs
        batch_size, num_events, width = value_ids.shape
        device = value_ids.device
        flat_size = batch_size * num_events

        feature_position_ids = torch.clamp(
            feature_position_ids,
            min=0,
            max=self.config.feature_position_vocab_size - 1,
        )
        feature_hidden = (
            self.token_embeddings(key_ids)
            + self.token_embeddings(value_ids)
            + self.feature_position_embeddings(feature_position_ids)
        )
        feature_hidden = self.layer_norm(self.dropout(feature_hidden))
        flat_hidden = feature_hidden.reshape(flat_size, width, self.config.hidden_size)
        evt = self._special_token(self.config.evt_token_id, (flat_size, 1), device)
        event_hidden = torch.cat([evt, flat_hidden], dim=1)
        flat_feature_mask = feature_attention_mask.reshape(flat_size, width)
        event_token_mask = torch.cat(
            [
                torch.ones(flat_size, 1, dtype=flat_feature_mask.dtype, device=device),
                flat_feature_mask,
            ],
            dim=1,
        )
        encoded_event = self.event_encoder(
            event_hidden,
            attention_mask=_extended_attention_mask(event_token_mask, event_hidden.dtype),
            return_dict=True,
        ).last_hidden_state
        event_cls = encoded_event[:, 0, :].reshape(batch_size, num_events, self.config.hidden_size)
        local_features = encoded_event[:, 1:, :].reshape(
            batch_size, num_events, width, self.config.hidden_size
        )

        event_cls = event_cls + self.calendar_encoder(calendar_features)
        user_embedding = self._encode_profile(
            profile_key_ids=profile_key_ids,
            profile_value_ids=profile_value_ids,
            profile_attention_mask=profile_attention_mask,
            batch_size=batch_size,
            device=device,
        )

        history_hidden = torch.cat([user_embedding, event_cls], dim=1)
        history_seconds = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=event_age_seconds.dtype, device=device),
                event_age_seconds,
            ],
            dim=1,
        )
        history_hidden = history_hidden + self.time_encoder(history_seconds)
        history_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=event_attention_mask.dtype, device=device),
                event_attention_mask,
            ],
            dim=1,
        )
        encoded_history = self.history_encoder(
            history_hidden,
            attention_mask=_extended_attention_mask(history_mask, history_hidden.dtype),
            return_dict=True,
        ).last_hidden_state

        history_events = encoded_history[:, 1:, :]
        lengths = torch.clamp(event_attention_mask.sum(dim=1) - 1, min=0).long()
        gather_idx = lengths.view(batch_size, 1, 1).expand(batch_size, 1, self.config.hidden_size)
        last_event = history_events.gather(dim=1, index=gather_idx).squeeze(1)
        return PragmaBackboneOutput(
            user_embedding=encoded_history[:, 0, :],
            event_embeddings=history_events,
            local_feature_embeddings=local_features,
            last_event_embedding=last_event,
        )


class PragmaForMaskedEventModeling(PragmaPreTrainedModel):
    """Pretraining model that reconstructs masked event feature values."""

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__(config)
        self.backbone = PragmaBackbone(config)
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.lm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.post_init()
        self.lm_decoder.weight = self.backbone.token_embeddings.weight

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        feature_position_ids: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_age_seconds: torch.Tensor,
        calendar_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        backbone_outputs = self.backbone(
            key_ids=key_ids,
            value_ids=value_ids,
            feature_position_ids=feature_position_ids,
            feature_attention_mask=feature_attention_mask,
            event_attention_mask=event_attention_mask,
            event_age_seconds=event_age_seconds,
            calendar_features=calendar_features,
            **kwargs,
        )
        user = backbone_outputs.user_embedding[:, None, None, :].expand_as(
            backbone_outputs.local_feature_embeddings
        )
        history_events = backbone_outputs.event_embeddings[:, :, None, :].expand_as(
            backbone_outputs.local_feature_embeddings
        )
        features = torch.cat(
            [backbone_outputs.local_feature_embeddings, history_events, user],
            dim=-1,
        )
        hidden = self.mlm_transform(features)
        logits = self.lm_decoder(hidden) + self.lm_bias
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
        return MaskedLMOutput(loss=loss, logits=logits)


class PragmaForSequenceClassification(PragmaPreTrainedModel):
    """Downstream classifier over the user-history embedding."""

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__(config)
        self.backbone = PragmaBackbone(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        feature_position_ids: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_age_seconds: torch.Tensor,
        calendar_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(
            key_ids=key_ids,
            value_ids=value_ids,
            feature_position_ids=feature_position_ids,
            feature_attention_mask=feature_attention_mask,
            event_attention_mask=event_attention_mask,
            event_age_seconds=event_age_seconds,
            calendar_features=calendar_features,
            **kwargs,
        )
        logits = self.classifier(self.dropout(outputs.user_embedding))
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.to(dtype=logits.dtype))
            else:
                loss = F.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits)


class PragmaForNextEventPrediction(PragmaPreTrainedModel):
    """Prefix model for next event type and log inter-arrival time."""

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__(config)
        self.backbone = PragmaBackbone(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.type_head = nn.Linear(config.hidden_size, config.num_event_types)
        self.delta_head = nn.Linear(config.hidden_size, 1)
        self.post_init()
        self.delta_head.bias.data.fill_(math.log1p(6.0 * 3600.0))

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        feature_position_ids: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_age_seconds: torch.Tensor,
        calendar_features: torch.Tensor,
        next_event_type_labels: Optional[torch.Tensor] = None,
        next_delta_log: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NextEventPredictionOutput:
        outputs = self.backbone(
            key_ids=key_ids,
            value_ids=value_ids,
            feature_position_ids=feature_position_ids,
            feature_attention_mask=feature_attention_mask,
            event_attention_mask=event_attention_mask,
            event_age_seconds=event_age_seconds,
            calendar_features=calendar_features,
            **kwargs,
        )
        embedding = self.dropout(outputs.last_event_embedding)
        type_logits = self.type_head(embedding)
        delta_log = self.delta_head(embedding).squeeze(-1)

        loss = None
        if next_event_type_labels is not None and next_delta_log is not None:
            type_loss = F.cross_entropy(type_logits, next_event_type_labels.long())
            delta_loss = F.mse_loss(delta_log, next_delta_log.to(dtype=delta_log.dtype))
            loss = type_loss + self.config.tpp_loss_weight * delta_loss
        return NextEventPredictionOutput(
            loss=loss,
            type_logits=type_logits,
            delta_log=delta_log,
            embeddings=outputs.last_event_embedding,
        )
