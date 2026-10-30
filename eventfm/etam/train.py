"""Pretraining, fine-tuning, probing and benchmark-parity evaluation for ETAM.

Stages
------
``pretrain``  self-supervised on a store (the MBD train pool, or the labeled
              training subset for the matched-data arm): dense marked-TPP
              likelihood, time-conditioned marks, next event and future window
              from the query token. No labels.
``finetune``  ``task=cls`` (label from the user embedding), ``task=tpp``
              (next event), or ``task=multi`` (both at once: one model, one user
              embedding, both benchmark tasks). From a checkpoint or scratch.
``probe``     frozen user embeddings -> logistic regression for the label; the
              pretrained heads, unchanged, for TPP (zero-shot).

Evaluation always rebuilds the benchmark's own examples (all test histories
for classification; `NextEventJsonlDataset`'s prefixes with seed ``seed + 1``
for TPP) and scores them with `eventfm.training.metrics`, so the numbers sit in
the same table as every baseline.
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eventfm.datasets.paths import output_root
from eventfm.etam.batching import (
    ListDataset,
    RandomCutViews,
    View,
    ViewCollator,
    benchmark_tpp_views,
    full_views,
    worker_init,
)
from eventfm.etam.model import (
    EtamConfig,
    EtamModel,
    LossWeights,
    checkpoint_state,
    compute_losses,
    count_parameters,
    load_model_state,
)
from eventfm.etam.store import EventStore
from eventfm.training.metrics import classification_metrics, tpp_metrics

TASK_OBJECTIVES = {
    "pretrain": ["tpp", "cond", "user", "future"],
    "tpp": ["tpp", "cond", "user"],
    "cls": ["cls"],
    "multi": ["cls", "tpp", "cond", "user"],
}


@dataclass
class RunConfig:
    stage: str = "pretrain"
    task: str = "multi"
    run_name: str = "etam"
    method_name: str = "etam"
    variant: str = "default"
    dataset: str = "mbd"
    store_root: str = ""
    pretrain_store: str = "pool"
    init_checkpoint: str = ""
    objectives: List[str] = field(default_factory=list)
    model: Dict[str, object] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    seed: int = 13
    max_events: int = 128
    future_window_days: float = 30.0
    cls_query: str = "last"
    p_full: float = 0.5
    p_between: float = 0.5
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 20_000
    epochs: float = 0.0
    grad_clip: float = 1.0
    #: Steps between validations; 0 means once per epoch.
    eval_every: int = 1_000
    patience: int = 0
    num_workers: int = 6
    bf16: bool = True
    val_users: int = 10_000
    freeze_backbone: bool = False
    head_learning_rate: float = 0.0
    max_train_sequences: int = 0
    #: Random fraction of the pretraining store (nested across fractions: one
    #: fixed permutation, then a prefix), for the data-scaling ladder.
    pretrain_fraction: float = 1.0
    #: HPO trials set this false so no test number exists until a config is chosen.
    test_eval: bool = True
    #: Also score the final-step model on test (not written to the benchmark
    #: table): the baselines train a fixed schedule with no selection.
    eval_final_step: bool = True
    #: Age and first gap measured inside the event window (baseline parity for
    #: the short-context control).
    window_time: bool = False
    write_results: bool = True
    output_dir: str = ""
    #: Flat JSON of the stage's return value, for the campaign harvester.
    result_path: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RunConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise KeyError("Unknown run-config keys: {}".format(unknown))
        return cls(**payload)

    def resolved_objectives(self) -> List[str]:
        if self.objectives:
            return list(self.objectives)
        return list(TASK_OBJECTIVES["pretrain" if self.stage == "pretrain" else self.task])

    def run_dir(self) -> Path:
        root = Path(self.output_dir) if self.output_dir else output_root() / "etam" / "runs"
        return root / self.run_name

    def store_dir(self) -> Path:
        root = Path(self.store_root) if self.store_root else output_root() / "etam" / "stores"
        return root / self.dataset


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _autocast(config: RunConfig, device: torch.device):
    enabled = bool(config.bf16) and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def load_store(config: RunConfig, split: str) -> EventStore:
    path = config.store_dir() / split
    if not (path / "store.json").exists():
        raise FileNotFoundError(
            "ETAM store `{}` is missing; run `python scripts/etam.py build-store`.".format(path)
        )
    return EventStore.load(str(path))


def build_model(config: RunConfig, store: EventStore) -> EtamModel:
    model_config = EtamConfig(
        num_event_types=store.vocab.num_event_types,
        field_sizes=store.vocab.field_sizes,
        **config.model,
    )
    return EtamModel(model_config)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, object]:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(path: str) -> EtamModel:
    payload = load_checkpoint(path)
    model = EtamModel(EtamConfig(**payload["model_config"]))
    load_model_state(model, payload["model"])
    return model


def _parameter_groups(model: EtamModel, config: RunConfig):
    decay, no_decay, heads = [], [], []
    head_prefixes = ("cls_head",)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if config.head_learning_rate and name.startswith(head_prefixes):
            heads.append(parameter)
        elif parameter.ndim < 2:
            # Norm gains, biases and the learned base/query/user vectors.
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if heads:
        groups.append({"params": heads, "weight_decay": 0.0, "lr": config.head_learning_rate})
    return [group for group in groups if group["params"]]


def _schedule(optimizer: torch.optim.Optimizer, config: RunConfig, total_steps: int):
    warmup = max(1, min(config.warmup_steps, total_steps // 10 or 1))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _loader(dataset, collator, batch_size, shuffle, workers, seed=0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collator,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=shuffle,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(
    model: EtamModel,
    store: EventStore,
    views: Sequence[View],
    config: RunConfig,
    want: Sequence[str] = ("cls", "tpp"),
) -> Dict[str, np.ndarray]:
    """Run the model over fixed views; returns numpy arrays keyed by output."""

    device = _device()
    model.eval()
    collator = _collator(store, config)
    loader = _loader(
        ListDataset(views), collator, config.eval_batch_size, False, min(4, config.num_workers)
    )
    collected: Dict[str, List[np.ndarray]] = {}

    def push(name: str, value: torch.Tensor) -> None:
        collected.setdefault(name, []).append(value.detach().float().cpu().numpy())

    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast(config, device):
            outputs = model(batch)
        user = outputs["user"].float()
        if "embedding" in want:
            push("embedding", user)
        if "cls" in want:
            push("cls_logits", model.classify(user))
            push("labels", batch["labels"])
        if "tpp" in want:
            type_logits, delta = model.next_event(user)
            push("type_logits", type_logits)
            push("delta_log", delta)
            target_dt = batch["next_log_dt"][:, -1]
            target_type = batch["next_type"][:, -1]
            push("time_nll", model.time_head.nll(user, target_dt))
            push(
                "type_nll", F.cross_entropy(type_logits, target_type.clamp(min=0), reduction="none")
            )
            push("next_type", target_type)
            push("next_log_dt", target_dt)
            # The same prediction read off the last *event* token instead of the
            # query token: a diagnostic for what the user token adds.
            last_event = outputs["event_out"][:, -2].float()
            event_logits, event_delta = model.next_event(last_event)
            push("event_type_logits", event_logits)
            push("event_delta_log", event_delta)
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def cls_metrics_from(predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    return classification_metrics((predictions["cls_logits"], predictions["labels"].astype(int)))


def tpp_metrics_from(predictions: Dict[str, np.ndarray], source: str = "user") -> Dict[str, float]:
    prefix = "" if source == "user" else "event_"
    labels = (predictions["next_type"].astype(int), predictions["next_log_dt"])
    metrics = tpp_metrics(
        ((predictions[prefix + "type_logits"], predictions[prefix + "delta_log"]), labels)
    )
    if source == "user":
        metrics["time_nll"] = float(np.mean(predictions["time_nll"]))
        metrics["type_nll"] = float(np.mean(predictions["type_nll"]))
        metrics["joint_nll"] = metrics["time_nll"] + metrics["type_nll"]
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.device = _device()
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.jsonl"
        self.weights = LossWeights(**config.weights)
        self.objectives = config.resolved_objectives()

    def log(self, payload: Dict[str, object]) -> None:
        payload = dict(payload, wall=time.time())
        print(json.dumps(payload, default=float), flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=float) + "\n")

    def fit(
        self,
        model: EtamModel,
        train_dataset,
        collator: ViewCollator,
        evaluate,
        higher_is_better: bool,
    ) -> Dict[str, object]:
        """Generic loop: train, evaluate every ``eval_every`` steps, keep the best."""

        config = self.config
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            _parameter_groups(model, config), lr=config.learning_rate, betas=(0.9, 0.98)
        )
        steps_per_epoch = max(1, len(train_dataset) // config.batch_size)
        total_steps = (
            int(round(config.epochs * steps_per_epoch)) if config.epochs > 0 else config.max_steps
        )
        scheduler = _schedule(optimizer, config, total_steps)
        eval_every = config.eval_every or steps_per_epoch
        state = {
            "step": 0,
            "best": None,
            "best_step": -1,
            "best_metrics": {},
            "stale": 0,
            "epoch": 0,
        }

        last_path = self.run_dir / "last.pt"
        if last_path.exists():
            resumed = load_checkpoint(str(last_path), map_location=self.device)
            saved = dict(resumed.get("run_config", {}))
            current = asdict(config)
            for key in ("result_path",):
                saved.pop(key, None)
                current.pop(key, None)
            if saved != current:
                changed = sorted(
                    k for k in set(saved) | set(current) if saved.get(k) != current.get(k)
                )
                raise RuntimeError(
                    "{} belongs to a different run config ({}); use a new run_name.".format(
                        last_path, changed
                    )
                )
            load_model_state(model, resumed["model"])
            optimizer.load_state_dict(resumed["optimizer"])
            scheduler.load_state_dict(resumed["scheduler"])
            state.update(resumed["state"])
            self.log({"event": "resumed", "step": state["step"]})

        self.log(
            {
                "event": "start",
                "objectives": self.objectives,
                "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "trainable_parameters": count_parameters(model, trainable_only=True),
                "total_parameters": count_parameters(model),
            }
        )
        started = time.time()
        running: Dict[str, float] = {}
        done = state["step"] >= total_steps
        while not done:
            loader = _loader(
                train_dataset,
                collator,
                config.batch_size,
                True,
                config.num_workers,
                seed=config.seed * 7919 + state["epoch"],
            )
            for batch in loader:
                model.train()
                batch = _to_device(batch, self.device)
                with _autocast(config, self.device):
                    outputs = model(batch)
                terms = compute_losses(model, outputs, batch, self.weights, self.objectives)
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        "Non-finite gradient norm at step {}.".format(state["step"])
                    )
                optimizer.step()
                scheduler.step()
                state["step"] += 1
                for name, value in terms.items():
                    running[name] = running.get(name, 0.0) + float(value.detach())
                running["_count"] = running.get("_count", 0.0) + 1.0

                if state["step"] % eval_every == 0 or state["step"] >= total_steps:
                    count = running.pop("_count")
                    train_terms = {key: value / count for key, value in running.items()}
                    running = {}
                    score, metrics = evaluate(model)
                    improved = state["best"] is None or (
                        score > state["best"] if higher_is_better else score < state["best"]
                    )
                    if improved:
                        state.update(
                            best=score, best_step=state["step"], best_metrics=metrics, stale=0
                        )
                        self._save(model, "best.pt", state)
                    else:
                        state["stale"] += 1
                    self.log(
                        {
                            "event": "eval",
                            "step": state["step"],
                            "epoch": state["epoch"],
                            "lr": scheduler.get_last_lr()[0],
                            "train": train_terms,
                            "valid": metrics,
                            "score": score,
                            "best": state["best"],
                            "best_step": state["best_step"],
                            "elapsed_min": (time.time() - started) / 60.0,
                        }
                    )
                    self._save(model, "last.pt", state, optimizer, scheduler)
                    if config.patience and state["stale"] >= config.patience:
                        self.log({"event": "early_stop", "step": state["step"]})
                        done = True
                if state["step"] >= total_steps:
                    done = True
                if done:
                    break
            state["epoch"] += 1

        final_state = {
            key: value.detach().cpu().clone() for key, value in checkpoint_state(model).items()
        }
        best = load_checkpoint(str(self.run_dir / "best.pt"), map_location=self.device)
        load_model_state(model, best["model"])
        return {
            "final_state": final_state,
            "final_step": state["step"],
            "best_step": state["best_step"],
            "best_score": state["best"],
            "valid": state["best_metrics"],
            "total_steps": total_steps,
            "train_minutes": (time.time() - started) / 60.0,
        }

    def _save(self, model, name, state, optimizer=None, scheduler=None) -> None:
        payload = {
            "model": checkpoint_state(model),
            "model_config": model.config.to_dict(),
            "run_config": asdict(self.config),
            "state": dict(state),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
            payload["scheduler"] = scheduler.state_dict()
        temporary = self.run_dir / "{}.tmp".format(name)
        torch.save(payload, temporary)
        os.replace(temporary, self.run_dir / name)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _collator(store: EventStore, config: RunConfig) -> ViewCollator:
    return ViewCollator(
        store, config.max_events, config.future_window_days, window_time=config.window_time
    )


def _limit(store: EventStore, config: RunConfig, fraction: float = 1.0) -> EventStore:
    if 0.0 < fraction < 1.0:
        order = np.random.default_rng(2026).permutation(len(store))
        keep = max(1, int(round(fraction * len(store))))
        store = store.subset(np.sort(order[:keep]))
    if config.max_train_sequences and len(store) > config.max_train_sequences:
        return store.subset(np.arange(config.max_train_sequences))
    return store


def _validation_views(config: RunConfig, store: EventStore):
    users = min(len(store), config.val_users) if config.val_users else len(store)
    tpp_views = benchmark_tpp_views(store, seed=config.seed + 1, max_users=users)
    cls_views = full_views(store, query=config.cls_query, indices=range(users))
    return cls_views, tpp_views


def run_pretrain(config: RunConfig) -> Dict[str, object]:
    _seed(config.seed)
    train_store = _limit(
        load_store(config, config.pretrain_store), config, fraction=config.pretrain_fraction
    )
    valid_store = load_store(config, "validation")
    model = build_model(config, train_store)
    collator = _collator(train_store, config)
    dataset = RandomCutViews(
        train_store, p_full=config.p_full, p_between=config.p_between, seed=config.seed
    )
    _, tpp_views = _validation_views(config, valid_store)
    trainer = Trainer(config)

    def evaluate(current: EtamModel):
        predictions = predict(current, valid_store, tpp_views, config, want=("tpp",))
        metrics = tpp_metrics_from(predictions)
        return metrics["joint_nll"], metrics

    summary = trainer.fit(model, dataset, collator, evaluate, higher_is_better=False)
    summary.pop("final_state")
    summary["pretrain_sequences"] = len(train_store)
    trainer.log(dict(summary, event="done"))
    return summary


def run_finetune(config: RunConfig) -> Dict[str, object]:
    _seed(config.seed)
    train_store = _limit(load_store(config, "train"), config)
    valid_store = load_store(config, "validation")
    if config.init_checkpoint:
        model = model_from_checkpoint(config.init_checkpoint)
        # Regularisation may be re-tuned at fine-tuning time; nothing that
        # changes the parameter shapes can be.
        unsupported = sorted(set(config.model) - {"dropout"})
        if unsupported:
            raise ValueError("Cannot override {} on a checkpoint.".format(unsupported))
        if "dropout" in config.model:
            _set_dropout(model, float(config.model["dropout"]))
        if model.config.num_event_types != train_store.vocab.num_event_types:
            raise ValueError("Checkpoint and store disagree on the event vocabulary.")
    else:
        model = build_model(config, train_store)
    if config.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("cls_head", "type_head", "time_head"))

    collator = _collator(train_store, config)
    task = config.task
    if task == "cls":
        dataset = ListDataset(full_views(train_store, query=config.cls_query))
    else:
        dataset = RandomCutViews(
            train_store, p_full=config.p_full, p_between=config.p_between, seed=config.seed
        )
    cls_views, tpp_views = _validation_views(config, valid_store)
    # Classification is selected on every validation history: the positive
    # rate is ~1.2 %, so a 10k subset holds only ~120 positives.
    cls_views = full_views(valid_store, query=config.cls_query)
    trainer = Trainer(config)

    def evaluate(current: EtamModel):
        metrics: Dict[str, float] = {}
        if task in ("cls", "multi"):
            predictions = predict(current, valid_store, cls_views, config, want=("cls",))
            metrics.update(
                {"cls_" + key: value for key, value in cls_metrics_from(predictions).items()}
            )
        if task in ("tpp", "multi"):
            predictions = predict(current, valid_store, tpp_views, config, want=("tpp",))
            metrics.update(
                {"tpp_" + key: value for key, value in tpp_metrics_from(predictions).items()}
            )
        if task == "cls":
            return metrics["cls_auc"], metrics
        if task == "tpp":
            return -metrics["tpp_joint_nll"], metrics
        # One model for both tasks: select on the two headline metrics together.
        return metrics["cls_auc"] + metrics["tpp_next_type_accuracy"], metrics

    summary = trainer.fit(model, dataset, collator, evaluate, higher_is_better=True)
    final_state = summary.pop("final_state")
    results: Dict[str, object] = {}
    if config.test_eval:
        tasks = _tasks_for(task)
        results.update(evaluate_test(model, config, tasks=tasks, summary=summary))
        if config.eval_final_step:
            # The model as the baselines would report it: last step, no selection.
            if summary["final_step"] == summary["best_step"]:
                results["final_step"] = {key: results[key] for key in tasks}
            else:
                best_state = {
                    k: v.detach().cpu().clone() for k, v in checkpoint_state(model).items()
                }
                load_model_state(model, final_state)
                results["final_step"] = evaluate_test(model, config, tasks=tasks, write=False)
                load_model_state(model, best_state)
    results["valid"] = dict(summary["valid"], score=summary["best_score"])
    results["best_step"] = summary["best_step"]
    results["train_minutes"] = summary["train_minutes"]
    trainer.log({"event": "done", "summary": summary, "test": results})
    return results


def _set_dropout(model: EtamModel, rate: float) -> None:
    model.config.dropout = rate
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = rate
    for layer in model.layers:
        layer.dropout = rate


def _tasks_for(task: str) -> List[str]:
    return {"cls": ["classification"], "tpp": ["tpp"], "multi": ["classification", "tpp"]}[task]


def run_probe(config: RunConfig) -> Dict[str, object]:
    """Frozen user embeddings: linear probe for the label, pretrained heads for TPP."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from eventfm.training.metrics import binary_auc_score

    _seed(config.seed)
    model = model_from_checkpoint(config.init_checkpoint).to(_device())
    stores = {split: load_store(config, split) for split in ("train", "validation", "test")}
    embeddings = {}
    labels = {}
    for split, store in stores.items():
        predictions = predict(
            model,
            store,
            full_views(store, query=config.cls_query),
            config,
            want=("embedding", "cls"),
        )
        embeddings[split] = predictions["embedding"]
        labels[split] = predictions["labels"].astype(int)
    scaler = StandardScaler().fit(embeddings["train"])
    scaled = {split: scaler.transform(value) for split, value in embeddings.items()}
    best = None
    search = {}
    for strength in (0.001, 0.01, 0.1, 1.0, 10.0):
        probe = LogisticRegression(C=strength, max_iter=2000)
        probe.fit(scaled["train"], labels["train"])
        auc = binary_auc_score(labels["validation"], probe.decision_function(scaled["validation"]))
        search[str(strength)] = auc
        if best is None or auc > best[0]:
            best = (auc, strength, probe)
    _, strength, probe = best
    logits = probe.decision_function(scaled["test"])
    cls_logits = np.stack([np.zeros_like(logits), logits], axis=1)
    cls = classification_metrics((cls_logits, labels["test"]))
    cls.update({"probe_C": strength, "probe_valid_auc": best[0]})

    _, valid_tpp_views = _validation_views(config, stores["validation"])
    valid_tpp = tpp_metrics_from(
        predict(model, stores["validation"], valid_tpp_views, config, want=("tpp",))
    )
    valid = {
        "probe_auc": best[0],
        "tpp_next_type_accuracy": valid_tpp["next_type_accuracy"],
        "tpp_joint_nll": valid_tpp["joint_nll"],
        # Same two-headline criterion as multitask fine-tuning.
        "score": best[0] + valid_tpp["next_type_accuracy"],
    }
    summary = {"probe_search": search, "embedding_dim": int(embeddings["test"].shape[1])}
    results: Dict[str, object] = {"valid": valid}
    if config.test_eval:
        tpp_views = benchmark_tpp_views(stores["test"], seed=config.seed + 1)
        tpp = tpp_metrics_from(predict(model, stores["test"], tpp_views, config, want=("tpp",)))
        tests = {"classification": cls, "tpp": tpp}
        _write_results(config, tests, summary, model, regime="frozen")
        results.update(tests)
    return results


def evaluate_test(
    model: EtamModel,
    config: RunConfig,
    tasks: Sequence[str],
    summary: Optional[Dict[str, object]] = None,
    write: bool = True,
) -> Dict[str, Dict[str, float]]:
    test_store = load_store(config, "test")
    results: Dict[str, Dict[str, float]] = {}
    if "classification" in tasks:
        predictions = predict(
            model, test_store, full_views(test_store, query=config.cls_query), config, want=("cls",)
        )
        results["classification"] = cls_metrics_from(predictions)
    if "tpp" in tasks:
        views = benchmark_tpp_views(test_store, seed=config.seed + 1)
        predictions = predict(model, test_store, views, config, want=("tpp",))
        metrics = tpp_metrics_from(predictions)
        event_metrics = tpp_metrics_from(predictions, source="event")
        metrics.update({"last_event_" + key: value for key, value in event_metrics.items()})
        results["tpp"] = metrics
    if write:
        regime = "frozen" if config.freeze_backbone else "full"
        _write_results(config, results, summary or {}, model, regime=regime)
    return results


def _write_results(
    config: RunConfig,
    results: Dict[str, Dict[str, float]],
    summary: Dict[str, object],
    model: EtamModel,
    regime: str,
) -> None:
    """One benchmark-schema result file per task, next to the run directory."""

    from eventfm.benchmark.runner import BenchmarkCell, RunResult, _git_provenance

    with (config.run_dir() / "test_results.json").open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "summary": summary}, handle, indent=2, default=float)
    if not config.write_results:
        return
    train_store = load_store(config, "train")
    directory = output_root() / "paper_runs" / "etam" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    for task, metrics in results.items():
        cell = BenchmarkCell(
            dataset=config.dataset,
            task=task,
            method=config.method_name,
            sample_size=len(train_store),
            seed=config.seed,
            regime=regime,
            variant=config.variant,
        )
        metrics = dict(metrics)
        metrics["num_trainable_parameters"] = float(count_parameters(model, trainable_only=True))
        metrics["num_total_parameters"] = float(count_parameters(model))
        result = RunResult(
            cell=asdict(cell),
            metrics={key: float(value) for key, value in metrics.items()},
            status="ok",
            num_train_sequences=len(train_store),
            num_test_sequences=len(load_store(config, "test")),
            metadata={
                "method": {
                    "display_name": "ETAM ({})".format(config.variant),
                    "family": "candidate",
                    "reference": "docs/etam/methodology.md",
                    "status": "Implemented",
                    "divergence": "",
                },
                "run_config": asdict(config),
                "model_config": model.config.to_dict(),
                "summary": summary,
                "git": _git_provenance(),
            },
        )
        path = directory / "{}.json".format(cell.key)
        if path.exists():
            # Two arms sharing a cell key would silently overwrite each other;
            # only the same run (e.g. a requeue) may rewrite its own file.
            previous = json.loads(path.read_text()).get("metadata", {}).get("run_config", {})
            if (previous.get("output_dir"), previous.get("run_name")) != (
                config.output_dir,
                config.run_name,
            ):
                raise FileExistsError("{} was written by another run.".format(path))
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_json(), handle, indent=2, sort_keys=True, default=float)


STAGES = {"pretrain": run_pretrain, "finetune": run_finetune, "probe": run_probe}


def run(config: RunConfig) -> Dict[str, object]:
    if config.stage not in STAGES:
        raise ValueError("Unknown stage `{}`; expected {}.".format(config.stage, sorted(STAGES)))
    config.run_dir().mkdir(parents=True, exist_ok=True)
    with (config.run_dir() / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    results = STAGES[config.stage](config)
    if config.result_path:
        Path(config.result_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(config.result_path).open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=float)
    return results
