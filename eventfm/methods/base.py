"""Registry and shared plumbing for the benchmarked architectures.

A *method* is anything that can be fitted on a training split and scored on a
test split for one of the two tasks. Non-neural controls implement
:class:`Method` directly; everything else subclasses :class:`TorchMethod`,
supplies a ``transformers`` model plus a collator, and inherits the shared
``Trainer`` loop so training hyper-parameters stay identical across the zoo.
"""

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from eventfm.data.tokenizer import EventTokenizer, EventVocabulary
from eventfm.datasets.registry import DatasetMeta
from eventfm.training.metrics import classification_metrics, tpp_metrics

TASKS = ("classification", "tpp")


@dataclass
class TrainingSpec:
    """Optimisation settings shared by every neural method."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_train_epochs: float = 8.0
    max_steps: int = -1
    batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_events: int = 64
    seed: int = 13
    fp16: bool = False
    bf16: bool = False
    use_cpu: bool = False
    dataloader_num_workers: int = 2
    output_dir: Optional[str] = None
    # LLM methods need a smaller learning rate and a shorter schedule.
    learning_rate_override: Optional[float] = None


@dataclass
class MethodContext:
    """Everything a method needs to instantiate itself for one benchmark cell."""

    dataset: DatasetMeta
    task: str
    train_path: str
    validation_path: str
    test_path: str
    vocab: EventVocabulary
    tokenizer: EventTokenizer
    training: TrainingSpec = field(default_factory=TrainingSpec)
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 256
    dropout: float = 0.1
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_event_types(self) -> int:
        return self.dataset.num_event_types

    @property
    def num_labels(self) -> int:
        return self.dataset.num_labels

    @property
    def max_features_per_event(self) -> int:
        return self.tokenizer.max_features_per_event


# Fidelity vocabulary from docs/research/README.md. A registry entry named after
# a paper is not a reproduction of it, and the generated tables must say so.
STATUS_IMPLEMENTED = "Implemented"
STATUS_APPROXIMATION = "Approximation"
VALID_STATUSES = (STATUS_IMPLEMENTED, STATUS_APPROXIMATION)


@dataclass
class MethodSpec:
    name: str
    display_name: str
    reference: str
    family: str
    factory: Callable[[MethodContext], "Method"]
    supports: Tuple[str, ...] = TASKS
    requires_gpu: bool = False
    notes: str = ""
    #: `Implemented` only for the controls, which have no paper to be faithful
    #: to. Every architecture entry is an `Approximation`; `divergence` records
    #: how it differs from the cited recipe.
    status: str = STATUS_APPROXIMATION
    divergence: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                "Method `{}` has status `{}`; expected one of {}.".format(
                    self.name, self.status, VALID_STATUSES
                )
            )
        if self.status == STATUS_APPROXIMATION and not self.divergence:
            raise ValueError(
                "Method `{}` is an approximation and must disclose how it "
                "differs from the cited recipe via `divergence`.".format(self.name)
            )


METHOD_REGISTRY: Dict[str, MethodSpec] = {}


def register_method(spec: MethodSpec) -> MethodSpec:
    if spec.name in METHOD_REGISTRY:
        raise KeyError("Method `{}` is already registered.".format(spec.name))
    METHOD_REGISTRY[spec.name] = spec
    return spec


def method_names(task: Optional[str] = None, include_gpu_only: bool = True) -> List[str]:
    names = []
    for name, spec in sorted(METHOD_REGISTRY.items()):
        if task is not None and task not in spec.supports:
            continue
        if not include_gpu_only and spec.requires_gpu:
            continue
        names.append(name)
    return names


def build_method(name: str, context: MethodContext) -> "Method":
    if name not in METHOD_REGISTRY:
        raise KeyError("Unknown method `{}`. Available: {}".format(name, method_names()))
    spec = METHOD_REGISTRY[name]
    if context.task not in spec.supports:
        raise ValueError("Method `{}` does not support task `{}`.".format(name, context.task))
    return spec.factory(context)


class Method:
    """Minimal fit/evaluate contract shared by neural and non-neural methods."""

    name = "method"

    def __init__(self, context: MethodContext) -> None:
        self.context = context

    def run(self) -> Dict[str, float]:
        raise NotImplementedError

    def num_parameters(self) -> int:
        return 0


def compute_metrics_for(task: str):
    return classification_metrics if task == "classification" else tpp_metrics


def _training_arguments(context: MethodContext, label_names: Optional[Sequence[str]]):
    """Build ``TrainingArguments`` while tolerating differing transformers versions."""

    from transformers import TrainingArguments

    spec = context.training
    output_dir = spec.output_dir or str(
        Path("/tmp") / "eventfm_runs" / "{}_{}".format(context.dataset.name, context.task)
    )
    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": float(spec.learning_rate_override or spec.learning_rate),
        "weight_decay": float(spec.weight_decay),
        "warmup_ratio": float(spec.warmup_ratio),
        "num_train_epochs": float(spec.num_train_epochs),
        "max_steps": int(spec.max_steps),
        "per_device_train_batch_size": int(spec.batch_size),
        "per_device_eval_batch_size": int(spec.eval_batch_size),
        "gradient_accumulation_steps": int(spec.gradient_accumulation_steps),
        "logging_strategy": "no",
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": bool(spec.fp16),
        "bf16": bool(spec.bf16),
        "use_cpu": bool(spec.use_cpu),
        "disable_tqdm": True,
        "dataloader_num_workers": int(spec.dataloader_num_workers),
        "seed": int(spec.seed),
        "label_names": list(label_names) if label_names else None,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    if (
        "eval_strategy" not in signature.parameters
        and "evaluation_strategy" in signature.parameters
    ):
        supported["evaluation_strategy"] = "no"
    return TrainingArguments(**supported)


class TorchMethod(Method):
    """Shared HF ``Trainer`` loop for every neural architecture in the zoo."""

    label_names: Dict[str, Optional[Sequence[str]]] = {
        "classification": None,
        "tpp": ["next_event_type_labels", "next_delta_log"],
    }

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_collator(self) -> Callable[[Sequence[Any]], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def build_dataset(self, path: str, split: str) -> Dataset:
        raise NotImplementedError

    def pretrain(self, model: torch.nn.Module) -> None:
        """Optional self-supervised stage run before the supervised head."""

    def run(self) -> Dict[str, float]:
        from transformers import Trainer

        model = self.build_model()
        self._model = model
        self.pretrain(model)

        collator = self.build_collator()
        trainer = Trainer(
            model=model,
            args=_training_arguments(self.context, self.label_names.get(self.context.task)),
            train_dataset=self.build_dataset(self.context.train_path, "train"),
            eval_dataset=self.build_dataset(self.context.test_path, "test"),
            data_collator=collator,
            compute_metrics=compute_metrics_for(self.context.task),
        )
        trainer.train()
        metrics = trainer.evaluate()
        metrics["num_parameters"] = float(self.num_parameters())
        return {key.replace("eval_", ""): value for key, value in metrics.items()}

    def num_parameters(self) -> int:
        model = getattr(self, "_model", None)
        if model is None:
            return 0
        return int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
