"""Generative, reconstructive, contrastive, and time-to-event baselines."""

from typing import Sequence

import torch
from transformers import Trainer

from eventfm.data.dataset import JsonlEventDataset
from eventfm.data.pretraining import EventPretrainingCollator
from eventfm.methods.base import (
    STATUS_APPROXIMATION,
    STATUS_IMPLEMENTED,
    MethodSpec,
    _training_arguments,
    register_method,
)
from eventfm.methods.coles import Coles
from eventfm.methods.flat_method import FlatMethod
from eventfm.models.configuration_flat import FlatEventConfig
from eventfm.models.modeling_pretraining import FlatForEventPretraining


class SupervisedGru(FlatMethod):
    name = "supervised-gru"
    backbone = "gru"
    pooling = "last"


class _ObjectivePretrainedMethod(FlatMethod):
    objectives: Sequence[str] = ()
    pretrain_epochs = 3.0
    mask_probability = 0.15

    def resolved_objectives(self) -> Sequence[str]:
        return tuple(self.context.extra.get("pretraining_objectives", self.objectives))

    def pretraining_config(self) -> FlatEventConfig:
        config = self.build_config()
        config.pretraining_objectives = list(self.resolved_objectives())
        config.pretraining_loss_weights = dict(
            self.context.extra.get("pretraining_loss_weights", {})
        )
        return config

    def pretrain(self, model: torch.nn.Module) -> None:
        objectives = self.resolved_objectives()
        if self.pretrain_epochs <= 0 or not objectives:
            return
        context = self.context
        pretrain_model = FlatForEventPretraining(self.pretraining_config())
        arguments = _training_arguments(context, label_names=None)
        arguments.num_train_epochs = float(
            context.extra.get("pretrain_epochs", self.pretrain_epochs)
        )
        arguments.max_steps = int(context.extra.get("pretrain_max_steps", -1))
        Trainer(
            model=pretrain_model,
            args=arguments,
            train_dataset=JsonlEventDataset(
                str(context.extra.get("pretrain_path", context.train_path))
            ),
            data_collator=EventPretrainingCollator(
                tokenizer=context.tokenizer,
                task="pretrain",
                max_events=context.training.max_events,
                feature_fields=context.model_feature_fields,
                objectives=objectives,
                num_event_types=context.num_input_event_types,
                mask_probability=self.mask_probability,
            ),
        ).train()
        model.backbone.load_state_dict(pretrain_model.backbone.state_dict())


class EventAutoencoder(_ObjectivePretrainedMethod):
    name = "autoencoder"
    backbone = "gru"
    pooling = "mean"
    objectives = ("autoencode",)
    mask_probability = 0.30


class TransactionMlm(_ObjectivePretrainedMethod):
    name = "transaction-mlm"
    backbone = "bert"
    pooling = "mean"
    objectives = ("masked",)


class AutoregressiveTransformer(_ObjectivePretrainedMethod):
    name = "autoregressive-transformer"
    backbone = "gpt2"
    pooling = "last"
    objectives = ("next", "next-feature", "next-time")


class Nppr(_ObjectivePretrainedMethod):
    name = "nppr"
    backbone = "gpt2"
    pooling = "last"
    objectives = ("masked", "next", "next-feature", "next-time")


class _LocalGlobalHybrid(_ObjectivePretrainedMethod):
    def pretrain(self, model: torch.nn.Module) -> None:
        super().pretrain(model)
        if not bool(self.context.extra.get("use_contrastive", True)):
            return
        contrastive = Coles(self.context)
        contrastive.contrastive_epochs = int(self.context.extra.get("contrastive_epochs", 3))
        contrastive.pretrain(model)


class Mlem(_LocalGlobalHybrid):
    name = "mlem"
    backbone = "gru"
    pooling = "mean"
    objectives = ("next", "next-feature", "next-time")


class CmlmColes(_LocalGlobalHybrid):
    name = "cmlm-coles"
    backbone = "gru"
    pooling = "mean"
    objectives = ("masked",)


class Sohet(_ObjectivePretrainedMethod):
    name = "sohet"
    backbone = "bert-causal"
    pooling = "last"
    type_conditioned_features = True
    objectives = ("next", "next-feature", "next-time")


class Motor(_ObjectivePretrainedMethod):
    name = "motor"
    backbone = "bert-causal"
    pooling = "last"
    objectives = ("motor",)


class Ora(_ObjectivePretrainedMethod):
    name = "ora"
    backbone = "bert-causal"
    pooling = "last"
    type_conditioned_features = True
    objectives = ("next", "next-feature", "marked-tte")


class EventFmJoint(_LocalGlobalHybrid):
    name = "eventfm-joint"
    backbone = "bert-causal"
    pooling = "mean"
    type_conditioned_features = True
    objectives = ("masked", "next", "next-feature", "marked-tte")


register_method(
    MethodSpec(
        name="supervised-gru",
        status=STATUS_IMPLEMENTED,
        divergence="controlled supervised baseline with no paper-specific pretraining",
        display_name="Supervised GRU",
        reference="standard supervised recurrent baseline",
        family="supervised",
        factory=SupervisedGru,
        notes="Categorical events, continuous time, and calendar features; no self-supervision.",
    )
)

register_method(
    MethodSpec(
        name="autoencoder",
        status=STATUS_IMPLEMENTED,
        divergence="controlled denoising autoencoder rather than a named paper reproduction",
        display_name="Transaction autoencoder",
        reference="standard denoising sequence autoencoder baseline",
        family="reconstruction",
        factory=EventAutoencoder,
        notes="Denoising reconstruction of event marks and field values before adaptation.",
    )
)

register_method(
    MethodSpec(
        name="transaction-mlm",
        status=STATUS_IMPLEMENTED,
        divergence="generic controlled masked-transaction baseline",
        display_name="Transaction MLM",
        reference="generic masked event/field modeling baseline",
        family="reconstruction",
        factory=TransactionMlm,
        notes="Bidirectional masked mark and field reconstruction.",
    )
)

register_method(
    MethodSpec(
        name="autoregressive-transformer",
        status=STATUS_IMPLEMENTED,
        divergence="generic controlled next-event baseline",
        display_name="Autoregressive transaction Transformer",
        reference="generic causal next-event modeling baseline",
        family="generative",
        factory=AutoregressiveTransformer,
        notes="Causal next mark, field, and interval pretraining.",
    )
)

register_method(
    MethodSpec(
        name="nppr",
        status=STATUS_APPROXIMATION,
        divergence=(
            "combines past masked reconstruction and future event prediction under the shared "
            "EventFM decoder, without the original purchasing corpus or scale"
        ),
        display_name="NPPR",
        reference="Zuo et al., Towards a Foundation Purchasing Model (2024)",
        family="generative",
        factory=Nppr,
        notes="Past reconstruction plus next mark, field, and interval prediction.",
    )
)

register_method(
    MethodSpec(
        name="mlem",
        status=STATUS_APPROXIMATION,
        divergence=(
            "sequential local generative and global contrastive stages under one encoder rather "
            "than the reference dual-modality architecture"
        ),
        display_name="MLEM",
        reference="Rubachev et al., MLEM (2024)",
        family="hybrid",
        factory=Mlem,
        notes="Local next-event pretraining followed by same-customer contrastive alignment.",
    )
)

register_method(
    MethodSpec(
        name="cmlm-coles",
        status=STATUS_APPROXIMATION,
        divergence="sequential masked-event and CoLES stages under a shared benchmark encoder",
        display_name="CMLM + CoLES",
        reference="Babaev et al., Uniting contrastive and generative learning (2024)",
        family="hybrid",
        factory=CmlmColes,
        notes="Masked-event learning followed by subsequence contrastive learning.",
    )
)

register_method(
    MethodSpec(
        name="sohet",
        status=STATUS_APPROXIMATION,
        divergence=(
            "type-conditioned feature gating approximates event-specific FT-Transformer encoders "
            "within the shared parameter budget"
        ),
        display_name="SOHET",
        reference="SOHET: Sequence of Heterogeneous Events Transformer (2026)",
        family="heterogeneous-event",
        factory=Sohet,
        notes="Type-conditioned event encoding with next type, time, and feature objectives.",
    )
)

register_method(
    MethodSpec(
        name="motor",
        status=STATUS_APPROXIMATION,
        divergence=(
            "code-specific exponential time-to-event pretraining on banking events; the original "
            "medical ontology, censoring horizon, and scale are unavailable"
        ),
        display_name="MOTOR-style TTE",
        reference="Steinberg et al., MOTOR (2023)",
        family="time-to-event",
        factory=Motor,
        notes="Observed and censored code-specific time-to-event likelihoods.",
    )
)

register_method(
    MethodSpec(
        name="ora",
        status=STATUS_APPROXIMATION,
        divergence=(
            "joint next-time and bucketed feature-value factorization under the EventFM schema, "
            "rather than the reference EHR value distributions"
        ),
        display_name="ORA-style marked TTE",
        reference="One Loss to Rule Them All: Marked Time-to-Event (2026)",
        family="time-to-event",
        factory=Ora,
        notes="Next type/value prediction plus value-conditioned next-time regression.",
    )
)

register_method(
    MethodSpec(
        name="eventfm-joint",
        status=STATUS_APPROXIMATION,
        divergence="candidate EventFM contribution; not a published baseline",
        display_name="EventFM joint objective",
        reference="candidate method specified in docs/research/method_catalog.md",
        family="candidate",
        factory=EventFmJoint,
        notes="Masked, next-event, marked-TTE, and subsequence-contrastive pretraining.",
    )
)
