"""Shared heads for reconstruction, autoregression, MOTOR, and marked TTE."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput

from eventfm.models.configuration_flat import FlatEventConfig
from eventfm.models.modeling_flat import FlatEventBackbone, FlatPreTrainedModel


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not bool((labels != -100).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


class FlatForEventPretraining(FlatPreTrainedModel):
    """One encoder with composable local/global temporal objectives."""

    def __init__(self, config: FlatEventConfig) -> None:
        super().__init__(config)
        self.backbone = FlatEventBackbone(config)
        self.mark_decoder = nn.Linear(config.hidden_size, config.num_input_event_types)
        self.feature_decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.next_mark_decoder = nn.Linear(config.hidden_size, config.num_input_event_types)
        self.next_feature_decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.next_time_decoder = nn.Linear(config.hidden_size, 1)
        self.marked_time_decoder = nn.Linear(2 * config.hidden_size, 1)
        self.code_rate_decoder = nn.Linear(config.hidden_size, config.num_input_event_types)
        self.post_init()

    def _feature_logits(self, hidden: torch.Tensor, decoder: nn.Module) -> torch.Tensor:
        width = self.config.num_feature_fields
        field_ids = torch.arange(width, device=hidden.device)
        field_context = self.backbone.embeddings.field_embeddings(field_ids)
        return decoder(hidden.unsqueeze(2) + field_context[None, None, :, :])

    def _weight(self, name: str) -> float:
        return float(self.config.pretraining_loss_weights.get(name, 1.0))

    def forward(
        self,
        event_type_ids: torch.Tensor,
        feature_value_ids: torch.Tensor,
        delta_log: torch.Tensor,
        time_since_start: torch.Tensor,
        calendar_features: torch.Tensor,
        attention_mask: torch.Tensor,
        mark_labels: Optional[torch.Tensor] = None,
        feature_labels: Optional[torch.Tensor] = None,
        next_mark_labels: Optional[torch.Tensor] = None,
        next_feature_labels: Optional[torch.Tensor] = None,
        next_delta_labels: Optional[torch.Tensor] = None,
        next_delta_mask: Optional[torch.Tensor] = None,
        code_time_labels: Optional[torch.Tensor] = None,
        code_time_observed: Optional[torch.Tensor] = None,
        code_time_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        del kwargs
        hidden = self.backbone(
            event_type_ids=event_type_ids,
            feature_value_ids=feature_value_ids,
            delta_log=delta_log,
            time_since_start=time_since_start,
            calendar_features=calendar_features,
            attention_mask=attention_mask,
        ).hidden_states
        objectives = set(self.config.pretraining_objectives)
        losses = []

        if {"autoencode", "masked"} & objectives and mark_labels is not None:
            losses.append(self._weight("reconstruct-mark") * _cross_entropy(
                self.mark_decoder(hidden), mark_labels
            ))
        if {"autoencode", "masked"} & objectives and feature_labels is not None:
            losses.append(self._weight("reconstruct-feature") * _cross_entropy(
                self._feature_logits(hidden, self.feature_decoder), feature_labels
            ))
        if "next" in objectives and next_mark_labels is not None:
            losses.append(self._weight("next-mark") * _cross_entropy(
                self.next_mark_decoder(hidden), next_mark_labels
            ))
        if "next-feature" in objectives and next_feature_labels is not None:
            losses.append(self._weight("next-feature") * _cross_entropy(
                self._feature_logits(hidden, self.next_feature_decoder), next_feature_labels
            ))
        if "next-time" in objectives and next_delta_labels is not None:
            prediction = self.next_time_decoder(hidden).squeeze(-1)
            valid = next_delta_mask.bool()
            if bool(valid.any()):
                losses.append(
                    self._weight("next-time")
                    * F.smooth_l1_loss(prediction[valid], next_delta_labels[valid])
                )
        if "marked-tte" in objectives and next_feature_labels is not None:
            valid_features = next_feature_labels != -100
            embedded = self.backbone.embeddings.value_embeddings(
                next_feature_labels.clamp(min=0)
            )
            denominator = valid_features.sum(dim=-1, keepdim=True).clamp(min=1)
            value_context = (embedded * valid_features.unsqueeze(-1)).sum(dim=2) / denominator
            prediction = self.marked_time_decoder(
                torch.cat([hidden, value_context], dim=-1)
            ).squeeze(-1)
            valid = next_delta_mask.bool()
            if bool(valid.any()):
                losses.append(
                    self._weight("marked-tte")
                    * F.smooth_l1_loss(prediction[valid], next_delta_labels[valid])
                )
        if "motor" in objectives and code_time_labels is not None:
            rates = F.softplus(self.code_rate_decoder(hidden) - 10.0).clamp(min=1e-8)
            seconds = torch.expm1(code_time_labels).clamp(min=0.0, max=1e12)
            observed = code_time_observed.to(rates.dtype)
            nll = rates * seconds - observed * torch.log(rates)
            valid = code_time_mask.bool()
            if bool(valid.any()):
                losses.append(self._weight("motor") * nll[valid].mean())

        loss = sum(losses) if losses else hidden.sum() * 0.0
        return ModelOutput(loss=loss)
