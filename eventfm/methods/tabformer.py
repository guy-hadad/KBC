"""IBM TabFormer (Padhi et al., ICASSP 2021).

The paper offers two paradigms over tabular time series, and both are
benchmarked here:

``tabbert``
    The BERT-analogue. Fields of an event are pooled into one event embedding,
    a bidirectional transformer runs over the event sequence, and the model is
    pretrained with masked-field reconstruction before the downstream head is
    attached.

``tabgpt``
    The GPT-analogue: the same field encoding with a causal decoder, pretrained
    autoregressively on the next event of the sequence.

Unlike PRAGMA, TabFormer has no explicit continuous-time branch — position in
the sequence carries the temporal signal. Keeping that faithful is the point of
the comparison, so the time encoders are disabled for both entries.
"""

from typing import Any, Dict, Sequence

import torch
from transformers import Trainer

from eventfm.data.flat import FlatEventCollator
from eventfm.methods.base import (
    STATUS_APPROXIMATION,
    MethodSpec,
    _training_arguments,
    register_method,
)
from eventfm.methods.flat_method import FlatMethod
from eventfm.models.configuration_flat import FlatEventConfig
from eventfm.models.modeling_flat import FlatForMaskedEventModeling


class _MaskedMarkCollator(FlatEventCollator):
    """Mask a fraction of marks and ask the model to reconstruct them."""

    def __init__(self, *args, mask_probability: float = 0.15, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask_probability = float(mask_probability)

    def __call__(self, examples: Sequence[Any]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        marks = batch["event_type_ids"]
        valid = batch["attention_mask"].bool()
        selected = valid & (torch.rand(marks.shape) < self.mask_probability)
        if not bool(selected.any()) and bool(valid.any()):
            first = valid.nonzero(as_tuple=False)[0]
            selected[first[0], first[1]] = True
        labels = torch.full_like(marks, -100)
        labels[selected] = marks[selected]
        corrupted = marks.clone()
        corrupted[selected] = 0
        batch["event_type_ids"] = corrupted
        batch["labels"] = labels
        return batch


class _TabFormerBase(FlatMethod):
    use_time_features = False
    use_calendar_features = False
    pretrain_epochs = 3.0

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        # TabFormer relies on sequence position, not on wall-clock time.
        config.use_time_features = False
        config.use_calendar_features = False
        return config

    def pretrain(self, model: torch.nn.Module) -> None:
        """Masked / autoregressive field pretraining before the task head."""

        context = self.context
        if self.pretrain_epochs <= 0.0:
            return

        pretrain_model = FlatForMaskedEventModeling(self.build_config())
        collator = _MaskedMarkCollator(
            tokenizer=context.tokenizer,
            task="pretrain",
            max_events=context.training.max_events,
            feature_fields=context.dataset.feature_fields,
        )
        from eventfm.data.dataset import JsonlEventDataset

        arguments = _training_arguments(context, label_names=None)
        arguments.num_train_epochs = float(self.pretrain_epochs)
        arguments.max_steps = -1
        Trainer(
            model=pretrain_model,
            args=arguments,
            train_dataset=JsonlEventDataset(context.train_path),
            data_collator=collator,
        ).train()

        # Transfer the pretrained event encoder into the downstream model.
        model.backbone.load_state_dict(pretrain_model.backbone.state_dict())


class TabBert(_TabFormerBase):
    name = "tabbert"
    backbone = "bert"
    pooling = "mean"


class TabGpt(_TabFormerBase):
    name = "tabgpt"
    backbone = "gpt2"
    pooling = "last"


register_method(
    MethodSpec(
        name="tabbert",
        status=STATUS_APPROXIMATION,
        divergence="field-pooled events with masked-field pretraining at benchmark scale",
        display_name="TabFormer (TabBERT)",
        reference="Padhi et al., Tabular transformers for modeling multivariate time series (2021)",
        family="tabular-transformer",
        factory=TabBert,
        notes="Field-pooled events, bidirectional encoder, masked-field pretraining.",
    )
)

register_method(
    MethodSpec(
        name="tabgpt",
        status=STATUS_APPROXIMATION,
        divergence="field-pooled events with a causal decoder at benchmark scale",
        display_name="TabFormer (TabGPT)",
        reference="Padhi et al. (2021), generative variant",
        family="tabular-transformer",
        factory=TabGpt,
        notes="Field-pooled events with a causal GPT-2 decoder.",
    )
)
