"""PRAGMA (Ostroukhov et al., arXiv:2604.08649), the in-repo baseline.

PRAGMA is the only entry with a genuinely hierarchical encoder: key/value/time
tokenisation per event, an event encoder that summarises the field tokens into
an ``[EVT]`` vector, and a history encoder over event summaries plus a ``[USR]``
profile token and continuous time-age encodings.

Two variants are benchmarked. ``pragma`` trains the downstream head directly;
``pragma-mlm`` first runs the paper's masked-value pretraining stage on the same
training split, which is the part of the recipe that makes it a foundation
model rather than a supervised encoder.
"""

from typing import Any, Callable, Dict, Sequence

import torch
from torch.utils.data import Dataset
from transformers import Trainer

from eventfm.data.collator import PragmaDataCollator
from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset
from eventfm.methods.base import (
    STATUS_APPROXIMATION,
    MethodSpec,
    TorchMethod,
    _training_arguments,
    register_method,
)
from eventfm.models.configuration_pragma import PragmaConfig
from eventfm.models.modeling_pragma import (
    PragmaForMaskedEventModeling,
    PragmaForNextEventPrediction,
    PragmaForSequenceClassification,
)


class Pragma(TorchMethod):
    name = "pragma"
    pretrain_epochs = 0.0

    def build_config(self) -> PragmaConfig:
        context = self.context
        vocab = context.vocab
        return PragmaConfig(
            vocab_size=len(vocab),
            hidden_size=context.hidden_size,
            intermediate_size=context.intermediate_size,
            num_attention_heads=context.num_attention_heads,
            profile_num_hidden_layers=1,
            event_num_hidden_layers=max(1, context.num_hidden_layers // 2 + 1),
            history_num_hidden_layers=context.num_hidden_layers,
            max_features_per_event=context.max_features_per_event,
            feature_position_vocab_size=max(8, context.max_features_per_event + 2),
            dropout=context.dropout,
            classifier_dropout=context.dropout,
            pad_token_id=vocab.pad_token_id,
            mask_token_id=vocab.mask_token_id,
            unk_token_id=vocab.unk_token_id,
            usr_token_id=vocab.usr_token_id,
            evt_token_id=vocab.evt_token_id,
            num_event_types=max(2, context.num_event_types),
            num_labels=context.num_labels,
        )

    def build_model(self) -> torch.nn.Module:
        config = self.build_config()
        if self.context.task == "classification":
            return PragmaForSequenceClassification(config)
        return PragmaForNextEventPrediction(config)

    def build_collator(self) -> Callable[[Sequence[Any]], Dict[str, torch.Tensor]]:
        return PragmaDataCollator(
            tokenizer=self.context.tokenizer,
            task=self.context.task,
            max_events=self.context.training.max_events,
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

    def pretrain(self, model: torch.nn.Module) -> None:
        context = self.context
        if self.pretrain_epochs <= 0.0:
            return
        pretrain_model = PragmaForMaskedEventModeling(self.build_config())
        arguments = _training_arguments(context, label_names=None)
        arguments.num_train_epochs = float(self.pretrain_epochs)
        arguments.max_steps = -1
        Trainer(
            model=pretrain_model,
            args=arguments,
            train_dataset=JsonlEventDataset(context.train_path),
            data_collator=PragmaDataCollator(
                tokenizer=context.tokenizer,
                task="pretrain",
                max_events=context.training.max_events,
            ),
        ).train()
        model.backbone.load_state_dict(pretrain_model.backbone.state_dict())


class PragmaMlm(Pragma):
    name = "pragma-mlm"
    pretrain_epochs = 3.0


register_method(
    MethodSpec(
        name="pragma",
        status=STATUS_APPROXIMATION,
        divergence="supervised PRAGMA-style hierarchy without foundation-scale pretraining",
        display_name="PRAGMA",
        reference="Ostroukhov et al., PRAGMA: Revolut foundation model (2026)",
        family="hierarchical",
        factory=Pragma,
        notes="Hierarchical key/value/time encoder, supervised only.",
    )
)

register_method(
    MethodSpec(
        name="pragma-mlm",
        status=STATUS_APPROXIMATION,
        divergence=(
            "masked-value pretraining on the downstream training split, not a separate large "
            "unlabelled corpus"
        ),
        display_name="PRAGMA + masked pretraining",
        reference="Ostroukhov et al. (2026), full recipe",
        family="hierarchical",
        factory=PragmaMlm,
        notes="Masked event-value pretraining on the training split, then the task head.",
    )
)
