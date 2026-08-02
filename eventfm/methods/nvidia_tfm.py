"""NVIDIA Transaction Foundation Model blueprint.

The blueprint is a decoder-only foundation model over a domain-specific tabular
tokenizer: merchant categories, amount buckets and time deltas each become
tokens, and the model is pretrained with causal language modelling before being
adapted to downstream tasks.

Reproduced here as a GPT-2 decoder over the same bucketed event tokens the rest
of the zoo uses, with a causal-LM pretraining stage over the mark stream so the
"pretrained checkpoint, then adapt" part of the blueprint is represented rather
than skipped.
"""

from typing import Any, Dict, Sequence

import torch
from transformers import Trainer

from eventfm.data.dataset import JsonlEventDataset
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


class _CausalMarkCollator(FlatEventCollator):
    """Shift the mark stream by one position for next-token pretraining."""

    def __call__(self, examples: Sequence[Any]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        marks = batch["event_type_ids"]
        mask = batch["attention_mask"].bool()
        labels = torch.full_like(marks, -100)
        labels[:, :-1] = torch.where(mask[:, 1:], marks[:, 1:], torch.full_like(marks[:, 1:], -100))
        batch["labels"] = labels
        return batch


class NvidiaTfm(FlatMethod):
    name = "nvidia-tfm"
    backbone = "gpt2"
    pooling = "last"
    pretrain_epochs = 3.0

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        # The blueprint feeds time deltas through the tokenizer as tokens, so
        # the continuous-time branch stays on alongside the causal decoder.
        config.use_time_features = True
        return config

    def pretrain(self, model: torch.nn.Module) -> None:
        context = self.context
        if self.pretrain_epochs <= 0.0:
            return

        pretrain_model = FlatForMaskedEventModeling(self.build_config())
        arguments = _training_arguments(context, label_names=None)
        arguments.num_train_epochs = float(self.pretrain_epochs)
        arguments.max_steps = -1
        Trainer(
            model=pretrain_model,
            args=arguments,
            train_dataset=JsonlEventDataset(context.train_path),
            data_collator=_CausalMarkCollator(
                tokenizer=context.tokenizer,
                task="pretrain",
                max_events=context.training.max_events,
                feature_fields=context.dataset.feature_fields,
            ),
        ).train()
        model.backbone.load_state_dict(pretrain_model.backbone.state_dict())


register_method(
    MethodSpec(
        name="nvidia-tfm",
        status=STATUS_APPROXIMATION,
        divergence=(
            "controlled implementation of the public blueprint, not the industrial pretrained "
            "checkpoint"
        ),
        display_name="NVIDIA TFM blueprint",
        reference="NVIDIA Transaction Foundation Model blueprint (decoder-only, causal LM)",
        family="tabular-transformer",
        factory=NvidiaTfm,
        notes="GPT-2 decoder over bucketed tabular tokens, causal-LM pretrained then adapted.",
    )
)
