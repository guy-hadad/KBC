"""ETAM-2 stages: pretrain, finetune (full / LoRA / head-only, label fractions), evaluate, simulate.

Protocol is Stage 1's (benchmark-replay TPP prefixes with seed + 1, all 40,000
test histories for classification, validation-only selection). Classification
can run on the strict-prospective stores (``cls_suffix=_strict``: histories end
before 2022-09-01, the label window starts after them), TPP always on the
standard stores.
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
from torch import nn

from eventfm.datasets.paths import output_root
from eventfm.etam.batching import (
    ListDataset,
    benchmark_tpp_views,
    full_views,
    worker_init,
)
from eventfm.etam.store import EventStore
from eventfm.etam.stage2 import evaluate as E
from eventfm.etam.stage2 import metrics as M
from eventfm.etam.stage2.batching import Collator2, Views2, load_extras
from eventfm.etam.stage2.config import Etam2Config, Etam2Weights
from eventfm.etam.stage2.losses import compute_losses2
from eventfm.etam.stage2.model import Etam2Model, count_parameters
from eventfm.training.metrics import classification_metrics

SECONDS_PER_DAY = 86_400.0
HEAD_PREFIXES = (
    "decoder", "user_head", "feature_head", "unimodal", "mtp", "counts", "repr",
    # Stage 1 checkpoints wrapped in Stage1Adapter (``inner.*``): their output heads.
    "inner.cls_head", "inner.type_head", "inner.time_head", "inner.cond_type_head",
    "inner.cond_feature_heads", "inner.future_head", "inner.count_head",
)


@dataclass
class RunConfig2:
    stage: str = "pretrain"
    task: str = "multi"
    run_name: str = "etam2"
    method_name: str = "etam2"
    variant: str = "default"
    dataset: str = "mbd"
    store_root: str = ""
    pretrain_store: str = "pool"
    cls_suffix: str = ""
    init_checkpoint: str = ""
    model: Dict[str, object] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    seed: int = 13
    max_events: int = 128
    window_time: bool = False
    p_full: float = 0.5
    p_between: float = 0.5
    batch_size: int = 256
    #: Micro-batches per optimiser step (effective batch = batch_size * grad_accum).
    grad_accum: int = 1
    eval_batch_size: int = 512
    learning_rate: float = 6e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    max_steps: int = 20_670
    epochs: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 2000
    patience: int = 0
    num_workers: int = 5
    bf16: bool = True
    val_users: int = 10_000
    pretrain_fraction: float = 1.0
    max_train_sequences: int = 0
    # fine-tuning
    adapt: str = "full"
    lora_rank: int = 8
    label_fraction: float = 1.0
    head_lr_mult: float = 1.0
    bridge_lr_mult: float = 1.0
    encoder_lr_mult: float = 1.0
    log_grad_conflict: bool = False
    #: ``best`` keeps the validation-selected checkpoint; ``final`` reports the
    #: last step (the baselines' fixed-schedule protocol).
    select: str = "best"
    # evaluation
    test_eval: bool = True
    eval_suite: bool = True
    eval_test_users: int = 0
    dump_embeddings: bool = False
    simulate_users: int = 2000
    simulate_samples: int = 16
    simulate_days: float = 30.0
    output_dir: str = ""
    result_path: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RunConfig2":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise KeyError("Unknown run-config keys: {}".format(unknown))
        return cls(**payload)

    def run_dir(self) -> Path:
        root = Path(self.output_dir) if self.output_dir else output_root() / "etam2" / "runs"
        return root / self.run_name

    def store_dir(self) -> Path:
        root = Path(self.store_root) if self.store_root else output_root() / "etam" / "stores"
        return root / self.dataset


# ---------------------------------------------------------------------------
# Data and model construction
# ---------------------------------------------------------------------------


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_store2(config: RunConfig2, split: str) -> EventStore:
    path = config.store_dir() / split
    if not (path / "store.json").exists():
        raise FileNotFoundError("Store `{}` is missing.".format(path))
    return load_extras(EventStore.load(str(path)), str(path))


def has_store(config: RunConfig2, split: str) -> bool:
    return (config.store_dir() / split / "store.json").exists()


def subset_store(store: EventStore, indices) -> EventStore:
    indices = np.asarray(indices, dtype=np.int64)
    sub = store.subset(indices)
    for name in ("targets", "targets_oct"):
        value = getattr(store, name, None)
        setattr(sub, name, None if value is None else value[indices])
    amount = getattr(store, "amount", None)
    if amount is not None:
        rows = np.concatenate([np.arange(store.offsets[i], store.offsets[i + 1]) for i in indices]) if indices.size else np.zeros(0, np.int64)
        sub.amount = amount[rows]
    else:
        sub.amount = None
    return sub


def limit_store(store: EventStore, config: RunConfig2, fraction: float = 1.0) -> EventStore:
    if 0.0 < fraction < 1.0:
        order = np.random.default_rng(2026).permutation(len(store))
        keep = max(1, int(round(fraction * len(store))))
        store = subset_store(store, np.sort(order[:keep]))
    if config.max_train_sequences and len(store) > config.max_train_sequences:
        store = subset_store(store, np.arange(config.max_train_sequences))
    return store


def label_subset(store: EventStore, fraction: float, seed: int) -> EventStore:
    """Stratified label-fraction subsample (keeps the positive rate)."""

    if fraction >= 1.0:
        return store
    rng = np.random.default_rng(10_000 + seed)
    labels = store.labels.astype(np.int64)
    keep = []
    for value in np.unique(labels):
        rows = np.flatnonzero(labels == value)
        count = max(1, int(round(fraction * rows.shape[0])))
        keep.append(rng.choice(rows, size=count, replace=False))
    return subset_store(store, np.sort(np.concatenate(keep)))


def data_statistics(store: EventStore, max_events: int = 3_000_000) -> Dict[str, List[float]]:
    """Adaptive bin edges (log gap, log age, hazard) and amount-bucket centres from a store."""

    times, offsets = store.timestamps, store.offsets
    total = times.shape[0]
    stride = max(1, total // max_events)
    starts = np.zeros(total, dtype=bool)
    starts[offsets[:-1][offsets[:-1] < total]] = True
    gaps = np.diff(times)
    keep = ~starts[1:]
    y_gap = np.log1p(np.maximum(gaps[keep], 0.0))[::stride]
    lengths = np.diff(offsets)
    first = np.repeat(times[offsets[:-1]], lengths)
    y_age = np.log1p(np.maximum(times - first, 0.0))[::stride]
    y_age = y_age[y_age > 0]

    def edges(values, bins):
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        return sorted(set(float(round(value, 4)) for value in quantiles if value > 0.0))

    stats = {
        "gap_bin_edges": edges(y_gap, 64),
        "age_bin_edges": edges(y_age, 64),
        "hazard_edges": [0.0] + edges(y_gap, 32),
    }
    amount = getattr(store, "amount", None)
    fields_ = list(store.vocab.feature_fields)
    if amount is not None and "amount" in fields_:
        column = fields_.index("amount")
        codes = store.features[:, column].astype(np.int64)
        values = amount.astype(np.float64)
        finite = np.isfinite(values)
        signed = np.sign(values[finite]) * np.log1p(np.abs(values[finite]))
        size = store.vocab.field_sizes[column]
        sums = np.bincount(codes[finite], weights=signed, minlength=size)
        counts = np.bincount(codes[finite], minlength=size)
        stats["amount_bucket_centers"] = (sums / np.maximum(counts, 1)).tolist()
    return stats


def build_model2(config: RunConfig2, store: EventStore) -> Etam2Model:
    stats = data_statistics(store)
    fields_ = list(store.vocab.feature_fields)
    base = {
        "num_event_types": store.vocab.num_event_types,
        "field_sizes": store.vocab.field_sizes,
        "amount_field": fields_.index("amount") if "amount" in fields_ else -1,
    }
    base.update(stats)
    base.update(config.model)
    if config.model.get("mixer") in ("flextpp", "kairos"):
        from eventfm.etam.stage2.baselines import InterleavedTPP

        return InterleavedTPP.from_store(Etam2Config(**base), store, config.max_events)
    return Etam2Model(Etam2Config(**base))


def load_checkpoint(path: str) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from_checkpoint(path: str) -> nn.Module:
    """ETAM-2 model, or a Stage 1 EtamModel wrapped in the Stage 2 interface."""

    payload = load_checkpoint(path)
    if payload.get("stage") == 2:
        values = dict(payload["model_config"])
        if values.get("mixer") in ("flextpp", "kairos"):
            from eventfm.etam.stage2.baselines import InterleavedTPP

            extra = {key: values.pop(key) for key in ("time_bins", "tau_max_seconds", "max_events") if key in values}
            model = InterleavedTPP(Etam2Config(**values), **extra)
        else:
            model = Etam2Model(Etam2Config(**values))
        model.load_state_dict(payload["model"])
        return model
    from eventfm.etam.model import EtamConfig, EtamModel, load_model_state

    inner = EtamModel(EtamConfig(**payload["model_config"]))
    load_model_state(inner, payload["model"])
    return E.Stage1Adapter(inner)


def make_collator(store: EventStore, config: RunConfig2, model_config, train: bool) -> Collator2:
    mtp_steps = max(4, getattr(model_config, "mtp_horizon", 0) or 0)
    repr_block = getattr(model_config, "repr_block", 0) if train and getattr(model_config, "repr_dim", 0) else 0
    horizons = getattr(model_config, "count_horizons_days", [1.0, 7.0, 30.0])
    return Collator2(
        store,
        max_events=config.max_events,
        window_time=config.window_time,
        mtp_steps=mtp_steps,
        count_horizons_days=horizons,
        repr_block=repr_block,
    )


def resolve_weights(config: RunConfig2) -> Etam2Weights:
    weights = Etam2Weights(**config.weights)
    if config.stage == "pretrain":
        weights.cls = 0.0
        return weights
    if config.task == "cls":
        return _only_cls(weights)
    weights.count = 0.0
    weights.repr = 0.0
    if config.task == "tpp":
        weights.cls = 0.0
    return weights


def _only_cls(weights: Etam2Weights) -> Etam2Weights:
    zero = {name: 0.0 for name in Etam2Weights().to_dict()}
    zero["cls"] = weights.cls or 1.0
    return Etam2Weights(**zero)


# ---------------------------------------------------------------------------
# LoRA adapters
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, hidden):
        return self.base(hidden) + self.scale * self.up(self.down(hidden))


def add_lora(module: nn.Module, rank: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
            replaced += 1
        else:
            replaced += add_lora(child, rank)
    return replaced


def configure_adaptation(model: nn.Module, config: RunConfig2) -> Dict[str, int]:
    """full: everything trains; lora: frozen backbone + LoRA + heads; head: heads only."""

    if config.adapt == "full":
        return {"lora_layers": 0}
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = 0
    if config.adapt == "lora":
        # Only the mixer: ETAM-2's backbone, or Stage 1's transformer layers.
        target = model.backbone if hasattr(model, "backbone") else model.inner.layers
        replaced = add_lora(target, config.lora_rank)
    for name, parameter in model.named_parameters():
        if name.startswith(HEAD_PREFIXES):
            parameter.requires_grad = True
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.down.weight.requires_grad = True
            module.up.weight.requires_grad = True
    return {"lora_layers": replaced}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def parameter_groups(model: nn.Module, config: RunConfig2):
    groups: Dict[tuple, list] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = 1.0
        if name.startswith(HEAD_PREFIXES):
            multiplier = config.head_lr_mult
        elif ".bridges." in name:
            multiplier = config.bridge_lr_mult
        elif name.startswith(("event_tokenizer", "time_tokenizer")) or "event_layers" in name or "time_layers" in name:
            multiplier = config.encoder_lr_mult
        decay = parameter.ndim >= 2
        groups.setdefault((multiplier, decay), []).append(parameter)
    return [
        {
            "params": params,
            "lr": config.learning_rate * multiplier,
            "weight_decay": config.weight_decay if decay else 0.0,
        }
        for (multiplier, decay), params in groups.items()
    ]


def schedule(optimizer, config: RunConfig2, total_steps: int):
    warmup = max(1, min(config.warmup_steps, total_steps // 10 or 1))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_loader(dataset, collator, batch_size, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collator,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=True,
    )


class Trainer2:
    def __init__(self, config: RunConfig2) -> None:
        self.config = config
        self.device = device()
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.jsonl"
        self.weights = resolve_weights(config)

    def log(self, payload: Dict[str, object]) -> None:
        payload = dict(payload, wall=time.time())
        line = json.dumps(payload, default=float)
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _save(self, model, name, state, optimizer=None, scheduler=None) -> None:
        payload = {
            "stage": 1 if isinstance(model, E.Stage1Adapter) else 2,
            "model": model.state_dict(),
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

    def fit(self, model, dataset, collator, evaluate, higher_is_better: bool) -> Dict[str, object]:
        config = self.config
        model.to(self.device)
        optimizer = torch.optim.AdamW(parameter_groups(model, config), lr=config.learning_rate, betas=(0.9, 0.98))
        accum = max(1, int(config.grad_accum))
        steps_per_epoch = max(1, len(dataset) // (config.batch_size * accum))
        micro = 0
        total_steps = int(round(config.epochs * steps_per_epoch)) if config.epochs > 0 else config.max_steps
        scheduler = schedule(optimizer, config, total_steps)
        eval_every = config.eval_every or steps_per_epoch
        state = {"step": 0, "best": None, "best_step": -1, "best_metrics": {}, "stale": 0, "epoch": 0, "conflict": []}
        last_path = self.run_dir / "last.pt"
        if last_path.exists():
            resumed = load_checkpoint(str(last_path))
            saved, current = dict(resumed.get("run_config", {})), asdict(config)
            for key in ("result_path",):
                saved.pop(key, None)
                current.pop(key, None)
            if saved != current:
                changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
                raise RuntimeError("{} belongs to another run config ({}).".format(last_path, changed))
            model.load_state_dict(resumed["model"])
            optimizer.load_state_dict(resumed["optimizer"])
            scheduler.load_state_dict(resumed["scheduler"])
            state.update(resumed["state"])
            self.log({"event": "resumed", "step": state["step"]})
        self.log(
            {
                "event": "start",
                "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "trainable_parameters": count_parameters(model, trainable_only=True),
                "total_parameters": count_parameters(model),
                "weights": self.weights.to_dict(),
            }
        )
        started = time.time()
        running: Dict[str, float] = {}
        shared = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(HEAD_PREFIXES)]
        done = state["step"] >= total_steps
        while not done:
            loader = train_loader(dataset, collator, config.batch_size, config.num_workers, config.seed * 7919 + state["epoch"])
            for batch in loader:
                model.train()
                batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                with E.autocast(self.device, config.bf16):
                    outputs = model(batch)
                if hasattr(model, "compute_losses"):
                    terms = model.compute_losses(outputs, batch, self.weights)
                else:
                    terms = compute_losses2(model, outputs, batch, self.weights)
                if micro == 0:
                    optimizer.zero_grad(set_to_none=True)
                    if config.log_grad_conflict and "cls" in terms and state["step"] % 50 == 0:
                        state["conflict"].append(self._conflict(terms, shared))
                (terms["loss"] / accum).backward()
                for name, value in terms.items():
                    running[name] = running.get(name, 0.0) + float(value.detach())
                running["_count"] = running.get("_count", 0.0) + 1.0
                micro += 1
                if micro < accum:
                    continue
                micro = 0
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient norm at step {}.".format(state["step"]))
                optimizer.step()
                scheduler.step()
                state["step"] += 1
                if state["step"] % eval_every == 0 or state["step"] >= total_steps:
                    count = running.pop("_count")
                    train_terms = {key: value / count for key, value in running.items()}
                    running = {}
                    score, metrics = evaluate(model)
                    improved = state["best"] is None or (score > state["best"] if higher_is_better else score < state["best"])
                    state["last_metrics"] = metrics
                    if improved:
                        state.update(best=score, best_step=state["step"], best_metrics=metrics, stale=0)
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
                            "elapsed_min": (time.time() - started) / 60.0,
                            "peak_gpu_mb": (
                                torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
                            ),
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
        final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if config.select == "best":
            best = load_checkpoint(str(self.run_dir / "best.pt"))
            model.load_state_dict(best["model"])
        else:
            self._save(model, "best.pt", state)
        conflict = state.get("conflict", [])
        summary = {
            "final_state": final_state,
            "final_step": state["step"],
            "best_step": state["best_step"],
            "best_score": state["best"],
            "valid": state["best_metrics"] if config.select == "best" else state.get("last_metrics", {}),
            "total_steps": total_steps,
            "train_minutes_segment": (time.time() - started) / 60.0,
        }
        if conflict:
            values = np.asarray(conflict, dtype=np.float64)
            summary["grad_conflict"] = {
                "mean_cosine": float(values.mean()),
                "fraction_negative": float((values < 0).mean()),
                "n": int(values.size),
            }
        return summary

    def _conflict(self, terms, shared) -> float:
        task = terms["loss"] - self.weights.cls * terms["cls"]
        grads_cls = torch.autograd.grad(terms["cls"], shared, retain_graph=True, allow_unused=True)
        grads_tpp = torch.autograd.grad(task, shared, retain_graph=True, allow_unused=True)
        pairs = [(a, b) for a, b in zip(grads_cls, grads_tpp) if a is not None and b is not None]
        if not pairs:
            return 0.0
        first = torch.cat([a.flatten() for a, _ in pairs])
        second = torch.cat([b.flatten() for _, b in pairs])
        return float(F.cosine_similarity(first, second, dim=0))

    def finish(self) -> None:
        last = self.run_dir / "last.pt"
        if last.exists():
            last.unlink()


# ---------------------------------------------------------------------------
# Selection metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def quick_tpp(model, store, views, collator, config: RunConfig2) -> Dict[str, float]:
    arrays: Dict[str, list] = {}
    model.eval()
    dev = device()
    for batch in E.loader(store, views, collator, config.eval_batch_size, min(4, config.num_workers)):
        batch = E.to_device(batch, dev)
        with E.autocast(dev, config.bf16):
            out = model(batch)
        dist = model.decoder(out["h"][:, -1].float())
        y, k = batch["next_log_dt"][:, -1].float(), batch["next_type"][:, -1]
        marks = dist.mark_log_probs()
        arrays.setdefault("joint", []).append(dist.log_prob(y, k).cpu())
        arrays.setdefault("time", []).append(dist.time_log_prob(y).cpu())
        arrays.setdefault("type", []).append(marks.gather(-1, k.unsqueeze(-1)).squeeze(-1).cpu())
        arrays.setdefault("hit", []).append((marks.argmax(-1) == k).float().cpu())
    values = {name: torch.cat(parts).double().mean().item() for name, parts in arrays.items()}
    return {
        "joint_nll": -values["joint"],
        "time_nll": -values["time"],
        "type_nll": -values["type"],
        "next_type_accuracy": values["hit"],
    }


@torch.no_grad()
def cls_scores(model, store, config: RunConfig2, collator) -> Dict[str, np.ndarray]:
    model.eval()
    dev = device()
    logits, labels = [], []
    for batch in E.loader(store, full_views(store), collator, config.eval_batch_size, min(4, config.num_workers)):
        batch = E.to_device(batch, dev)
        with E.autocast(dev, config.bf16):
            out = model(batch)
        logits.append(model.classify(out["user"]).float().cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return {"logits": np.concatenate(logits), "labels": np.concatenate(labels).astype(int)}


def cls_report(scores: Dict[str, np.ndarray], train_prior: float, eval_prior: Optional[float] = None) -> Dict[str, float]:
    logits = scores["logits"]
    margin = logits[:, 1] - logits[:, 0]
    report = classification_metrics((logits, scores["labels"]))
    extra = M.classification_report(margin, scores["labels"], train_prior=train_prior, eval_prior=eval_prior or float(scores["labels"].mean()))
    report.update({key: value for key, value in extra.items() if key not in report})
    return report


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_pretrain(config: RunConfig2) -> Dict[str, object]:
    seed_all(config.seed)
    store = limit_store(load_store2(config, config.pretrain_store), config, config.pretrain_fraction)
    valid = load_store2(config, "validation")
    model = build_model2(config, store)
    collator = make_collator(store, config, model.config, train=True)
    eval_collator = make_collator(valid, config, model.config, train=False)
    dataset = Views2(store, p_full=config.p_full, p_between=config.p_between, seed=config.seed)
    users = min(len(valid), config.val_users) if config.val_users else len(valid)
    views = benchmark_tpp_views(valid, seed=config.seed + 1, max_users=users)
    trainer = Trainer2(config)

    def evaluate(current):
        metrics = quick_tpp(current, valid, views, eval_collator, config)
        return metrics["joint_nll"], metrics

    summary = trainer.fit(model, dataset, collator, evaluate, higher_is_better=False)
    summary.pop("final_state")
    results: Dict[str, object] = {
        "valid": summary["valid"],
        "best_step": summary["best_step"],
        "total_steps": summary["total_steps"],
        "train_minutes_segment": summary["train_minutes_segment"],
        "pretrain_sequences": len(store),
        "parameters": count_parameters(model),
    }
    trainer.log({"event": "trained", "summary": results})
    if config.eval_suite:
        results.update(full_evaluation(model, config))
    trainer.finish()
    return results


def full_evaluation(model, config: RunConfig2) -> Dict[str, object]:
    """The Stage 2 evaluation battery for one model (validation, test, probes, efficiency)."""

    dev = device()
    model.to(dev)
    model_config = model.config
    train = load_store2(config, "train")
    frequency = E.type_frequency(train)
    results: Dict[str, object] = {}
    started = time.time()
    splits = ["validation"] + (["test"] if config.test_eval else [])
    for split in splits:
        store = load_store2(config, split)
        collator = make_collator(store, config, model_config, train=False)
        limit = config.val_users if split == "validation" else (config.eval_test_users or None)
        views = benchmark_tpp_views(store, seed=config.seed + 1, max_users=limit)
        arrays = E.collect_next_event(model, store, views, collator, config.eval_batch_size, dev, bf16=config.bf16)
        entry = {"tpp": E.next_event_metrics(arrays, frequency)}
        entry["elapsed"] = E.elapsed_silence_metrics(
            model, store, collator, config.eval_batch_size, dev, seed=config.seed + 7, max_users=limit or 20_000, bf16=config.bf16
        )
        if store.observation_end is not None:
            cut = float(store.observation_end) - 30.0 * SECONDS_PER_DAY
            entry["counts"] = E.count_metrics(
                model, store, collator, config.eval_batch_size, dev, cut, getattr(model_config, "count_horizons_days", [1.0, 7.0, 30.0]), bf16=config.bf16
            )
        results[split] = entry
        if split == "test":
            sample = collator(views[:256])
            results["efficiency"] = E.efficiency(model, sample, dev)
    results["probe"] = probe_suite(model, config)
    results["eval_minutes"] = (time.time() - started) / 60.0
    return results


def probe_suite(model, config: RunConfig2) -> Dict[str, object]:
    """Frozen-embedding logistic probes for the label (standard and strict histories).

    Embeddings are optionally dumped (fp16) for the CPU label-fraction / MLP /
    cross-target probes in ``scripts/etam2_probe.py``.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    dev = device()
    results: Dict[str, object] = {}
    suffixes = [""] + (["_strict"] if has_store(config, "train_strict") else [])
    for suffix in suffixes:
        data = {}
        for split in ("train", "validation", "test"):
            store = load_store2(config, split + suffix)
            collator = make_collator(store, config, model.config, train=False)
            data[split] = E.embeddings(model, store, full_views(store), collator, config.eval_batch_size, dev, config.bf16)
            for name in ("targets", "targets_oct"):
                value = getattr(store, name, None)
                if value is not None:
                    data[split][name] = value
            if config.dump_embeddings:
                directory = config.run_dir() / "embeddings"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / "{}{}.npz".format(split, suffix), **data[split])
        if len(np.unique(data["train"]["labels"])) < 2:
            results["skipped{}".format(suffix)] = "single-class labels"
            continue
        for name in ("user", "joint"):
            scaler = StandardScaler().fit(data["train"][name].astype(np.float32))
            scaled = {split: scaler.transform(data[split][name].astype(np.float32)) for split in data}
            best = None
            for strength in (0.001, 0.01, 0.1, 1.0):
                probe = LogisticRegression(C=strength, max_iter=2000)
                probe.fit(scaled["train"], data["train"]["labels"])
                auc = M.ranking_metrics(probe.decision_function(scaled["validation"]), data["validation"]["labels"])["auc"]
                if best is None or auc > best[0]:
                    best = (auc, strength, probe)
            auc, strength, probe = best
            entry = {"valid_auc": auc, "C": strength}
            if config.test_eval:
                train_prior = float(data["train"]["labels"].mean())
                report = M.classification_report(
                    probe.decision_function(scaled["test"]), data["test"]["labels"], train_prior=train_prior,
                    eval_prior=float(data["validation"]["labels"].mean()),
                )
                entry.update({key: report[key] for key in report if not isinstance(report[key], (list, dict))})
            results["{}{}".format(name, suffix or "_standard")] = entry
    return results


def run_finetune(config: RunConfig2) -> Dict[str, object]:
    seed_all(config.seed)
    suffix = config.cls_suffix
    train_store = limit_store(load_store2(config, "train" + suffix), config)
    if config.label_fraction < 1.0:
        train_store = label_subset(train_store, config.label_fraction, config.seed)
    valid_cls = load_store2(config, "validation" + suffix)
    valid_tpp = load_store2(config, "validation")
    if config.init_checkpoint:
        model = model_from_checkpoint(config.init_checkpoint)
        unsupported = sorted(set(config.model) - {"dropout"})
        if unsupported:
            raise ValueError("Cannot override {} on a checkpoint.".format(unsupported))
        if "dropout" in config.model:
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = float(config.model["dropout"])
    else:
        model = build_model2(config, train_store)
    adaptation = configure_adaptation(model, config)
    model_config = model.config
    collator = make_collator(train_store, config, model_config, train=True)
    task = config.task
    if task == "cls":
        dataset = ListDataset(full_views(train_store))
    else:
        dataset = Views2(train_store, p_full=config.p_full, p_between=config.p_between, seed=config.seed)
    users = min(len(valid_tpp), config.val_users)
    tpp_views = benchmark_tpp_views(valid_tpp, seed=config.seed + 1, max_users=users)
    cls_collator = make_collator(valid_cls, config, model_config, train=False)
    tpp_collator = make_collator(valid_tpp, config, model_config, train=False)
    trainer = Trainer2(config)

    def evaluate(current):
        metrics: Dict[str, float] = {}
        if task in ("cls", "multi"):
            scores = cls_scores(current, valid_cls, config, cls_collator)
            metrics["cls_auc"] = M.ranking_metrics(scores["logits"][:, 1] - scores["logits"][:, 0], scores["labels"])["auc"]
        if task in ("tpp", "multi"):
            metrics.update({"tpp_" + k: v for k, v in quick_tpp(current, valid_tpp, tpp_views, tpp_collator, config).items()})
        if task == "cls":
            return metrics["cls_auc"], metrics
        if task == "tpp":
            return -metrics["tpp_joint_nll"], metrics
        return metrics["cls_auc"] + metrics["tpp_next_type_accuracy"], metrics

    summary = trainer.fit(model, dataset, collator, evaluate, higher_is_better=True)
    final_state = summary.pop("final_state")
    results: Dict[str, object] = {
        "valid": dict(summary["valid"], score=summary["best_score"]),
        "best_step": summary["best_step"],
        "final_step": summary["final_step"],
        "train_minutes_segment": summary["train_minutes_segment"],
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "train_sequences": len(train_store),
        "adaptation": adaptation,
    }
    if "grad_conflict" in summary:
        results["grad_conflict"] = summary["grad_conflict"]
    if config.test_eval:
        results.update(finetune_test(model, config, task, train_store))
        if config.select == "best" and summary["final_step"] != summary["best_step"]:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(final_state)
            results["final_step_scores"] = finetune_test(model, config, task, train_store)
            model.load_state_dict(best_state)
    trainer.log({"event": "done", "results": results})
    trainer.finish()
    return results


def finetune_test(model, config: RunConfig2, task: str, train_store) -> Dict[str, object]:
    dev = device()
    results: Dict[str, object] = {}
    if task in ("cls", "multi"):
        store = load_store2(config, "test" + config.cls_suffix)
        collator = make_collator(store, config, model.config, train=False)
        prior = float((train_store.labels >= 1).mean())
        valid_prior = float((load_store2(config, "validation" + config.cls_suffix).labels >= 1).mean())
        results["classification"] = cls_report(cls_scores(model, store, config, collator), prior, valid_prior)
    if task in ("tpp", "multi"):
        store = load_store2(config, "test")
        collator = make_collator(store, config, model.config, train=False)
        views = benchmark_tpp_views(store, seed=config.seed + 1, max_users=config.eval_test_users or None)
        arrays = E.collect_next_event(model, store, views, collator, config.eval_batch_size, dev, bf16=config.bf16)
        results["tpp"] = E.next_event_metrics(arrays, E.type_frequency(train_store))
    return results


def run_evaluate(config: RunConfig2) -> Dict[str, object]:
    seed_all(config.seed)
    model = model_from_checkpoint(config.init_checkpoint)
    return full_evaluation(model, config)


def run_simulate(config: RunConfig2) -> Dict[str, object]:
    from eventfm.etam.stage2.simulate import simulate_and_score

    seed_all(config.seed)
    model = model_from_checkpoint(config.init_checkpoint).to(device())
    return simulate_and_score(model, config)


STAGES = {
    "pretrain": run_pretrain,
    "finetune": run_finetune,
    "evaluate": run_evaluate,
    "simulate": run_simulate,
}


def run(config: RunConfig2) -> Dict[str, object]:
    if config.stage not in STAGES:
        raise ValueError("Unknown stage {}".format(config.stage))
    config.run_dir().mkdir(parents=True, exist_ok=True)
    with (config.run_dir() / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    results = STAGES[config.stage](config)
    if config.result_path:
        Path(config.result_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(config.result_path).open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=_json_default)
    return results


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return float(value)
