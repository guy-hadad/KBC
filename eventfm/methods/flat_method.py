"""Shared wrapper for every method that consumes the flat tensor contract."""

from typing import Any, Callable, Dict, Sequence

import torch
from torch.utils.data import Dataset

from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset
from eventfm.data.flat import FlatEventCollator
from eventfm.methods.base import TorchMethod
from eventfm.models.configuration_flat import FlatEventConfig
from eventfm.models.modeling_flat import (
    FlatForNextEventPrediction,
    FlatForSequenceClassification,
)


class FlatMethod(TorchMethod):
    """One backbone + one task head, sharing the whole training loop.

    Subclasses set ``backbone`` and may override :meth:`configure` to express
    what makes their paper different (pooling, causality, time features,
    pretraining objective).
    """

    backbone = "gru"
    pooling = "last"
    use_time_features = True
    use_calendar_features = True
    time_loss = "lognormal_mixture"

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        return config

    def build_config(self) -> FlatEventConfig:
        context = self.context
        config = FlatEventConfig(
            vocab_size=len(context.vocab),
            num_event_types=max(2, context.num_event_types),
            num_feature_fields=max(1, len(context.dataset.feature_fields)),
            hidden_size=context.hidden_size,
            num_hidden_layers=context.num_hidden_layers,
            num_attention_heads=context.num_attention_heads,
            intermediate_size=context.intermediate_size,
            dropout=context.dropout,
            backbone=self.backbone,
            pooling=self.pooling,
            use_time_features=self.use_time_features,
            use_calendar_features=self.use_calendar_features,
            time_loss=self.time_loss,
            max_position_embeddings=max(64, context.training.max_events + 8),
            num_labels=context.num_labels,
            pad_token_id=context.vocab.pad_token_id,
        )
        return self.configure(config)

    def build_model(self) -> torch.nn.Module:
        config = self.build_config()
        if self.context.task == "classification":
            return FlatForSequenceClassification(config)
        return FlatForNextEventPrediction(config)

    def build_collator(self) -> Callable[[Sequence[Any]], Dict[str, torch.Tensor]]:
        return FlatEventCollator(
            tokenizer=self.context.tokenizer,
            task=self.context.task,
            max_events=self.context.training.max_events,
            feature_fields=self.context.dataset.feature_fields,
        )

    def build_dataset(self, path: str, split: str) -> Dataset:
        if self.context.task == "classification":
            return JsonlEventDataset(path)
        return NextEventJsonlDataset(
            path,
            min_prefix_events=int(self.context.extra.get("tpp_min_prefix_events", 4)),
            max_prefixes_per_user=int(self.context.extra.get("tpp_max_prefixes_per_user", 8)),
            seed=self.context.training.seed + (0 if split == "train" else 1),
        )
