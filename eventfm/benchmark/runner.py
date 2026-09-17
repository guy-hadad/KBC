"""Run one benchmark cell and persist its result.

A *cell* is a (dataset, task, method, training-sample-size, seed) tuple. Cells
are independent, which is what makes the whole grid a SLURM array job: each
array index resolves one cell, writes a single JSON file, and the aggregation
step later reads whatever finished.
"""

import hashlib
import json
import os
import subprocess
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from eventfm.data.io import load_jsonl_sequences, write_jsonl
from eventfm.data.schema import EventSequence
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary
from eventfm.datasets.paths import output_root
from eventfm.datasets.registry import (
    DATASET_REGISTRY,
    PRIMARY_DATASETS,
    DatasetMeta,
    load_dataset_meta,
)
from eventfm.methods import METHOD_REGISTRY, MethodContext, build_method
from eventfm.methods.base import TrainingSpec, seed_all


@dataclass(frozen=True)
class BenchmarkCell:
    dataset: str
    task: str
    method: str
    sample_size: int
    seed: int = 13
    regime: str = "full"
    variant: str = "default"

    @property
    def key(self) -> str:
        base = "{}__{}__{}__n{}__s{}".format(
            self.dataset, self.task, self.method, self.sample_size, self.seed
        )
        if self.regime != "full":
            base = "{}__r{}".format(base, self.regime)
        return base if self.variant == "default" else "{}__v{}".format(base, self.variant)


@dataclass
class RunResult:
    cell: Dict[str, object]
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None
    wall_seconds: float = 0.0
    num_train_sequences: int = 0
    num_test_sequences: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def enumerate_cells(
    datasets: Sequence[str],
    tasks: Sequence[str],
    methods: Sequence[str],
    sample_sizes,
    seeds: Sequence[int] = (13,),
    regimes: Sequence[str] = ("full",),
    variants: Sequence[str] = ("default",),
) -> List[BenchmarkCell]:
    """Expand the grid, skipping method/task pairs a method does not support.

    ``sample_sizes`` is either one sequence shared by every dataset or a mapping
    from dataset name to its own sequence. The mapping form exists because the
    datasets no longer share a training-pool size: clamping a single grid to the
    smallest pool would throw away most of the scaling range on the large ones,
    and letting it run past a pool would silently repeat the largest point.
    """

    cells: List[BenchmarkCell] = []
    for dataset in datasets:
        if isinstance(sample_sizes, Mapping):
            dataset_sizes = sample_sizes.get(dataset, sample_sizes.get("default", ()))
        else:
            dataset_sizes = sample_sizes
        for task in tasks:
            dataset_spec = DATASET_REGISTRY.get(dataset)
            if dataset_spec is not None and task not in dataset_spec.supports:
                continue
            for method in methods:
                spec = METHOD_REGISTRY.get(method)
                if spec is None or task not in spec.supports:
                    continue
                for sample_size in dataset_sizes:
                    for seed in seeds:
                        for regime in regimes:
                            for variant in variants:
                                cells.append(
                                    BenchmarkCell(
                                        dataset=dataset,
                                        task=task,
                                        method=method,
                                        sample_size=int(sample_size),
                                        seed=int(seed),
                                        regime=str(regime),
                                        variant=str(variant),
                                    )
                                )
    return cells


def _stratified_subsample(
    sequences: List[EventSequence], sample_size: int, seed: int
) -> List[EventSequence]:
    """Keep the label balance of the full split when shrinking it.

    Without this, the smallest scaling points would sometimes contain a single
    class and the accuracy curve would be meaningless.
    """

    if sample_size >= len(sequences):
        return list(sequences)
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[EventSequence]] = {}
    for sequence in sequences:
        by_label.setdefault(int(sequence.label or 0), []).append(sequence)

    for bucket in by_label.values():
        rng.shuffle(bucket)

    total = float(len(sequences))
    chosen: List[EventSequence] = []
    for _, bucket in sorted(by_label.items()):
        share = len(bucket) / total
        take = int(round(sample_size * share))
        chosen.extend(bucket[: max(1, take)])

    if len(chosen) > sample_size:
        rng.shuffle(chosen)
        chosen = chosen[:sample_size]
    else:
        # Identity, not equality: `EventSequence` is a dataclass, so `in` would
        # deep-compare every event of every history against every candidate.
        picked = {id(item) for item in chosen}
        remaining = [
            item
            for bucket in by_label.values()
            for item in bucket
            if id(item) not in picked
        ]
        rng.shuffle(remaining)
        chosen.extend(remaining[: sample_size - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def _count_lines(path: Path) -> int:
    """Sequence count of a JSONL split without parsing it."""

    total = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += block.count(b"\n")
    return total


def _publish_once(temporary: Path, path: Path) -> None:
    """Link a finished temp file into place, leaving any existing file alone.

    `os.replace` is atomic but *swaps the inode*, so a worker that already holds
    the old path open gets `OSError: [Errno 116] Stale file handle` on NFS. With
    32 workers converging on the same largest-`n` subset that is a real race: it
    cost three cells on the first pass of the scale-datasets campaign. `os.link`
    instead fails if the target exists, so the file is created exactly once and
    never swapped underneath a reader. Contents are a pure function of
    (sample size, seed, source fingerprint), so first writer wins is safe.
    """

    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def subsampled_train_path(
    meta: DatasetMeta, sample_size: int, seed: int, cache_dir: Optional[Path] = None
) -> tuple:
    """Materialise a stratified training subset, cached on disk for reuse.

    The cache is checked before the split is read. On the large datasets the
    training split is hundreds of megabytes of JSONL, and parsing it only to
    rediscover an already-cached subset was the largest single allocation in a
    small-sample cell.
    """

    path = _subset_path(meta, sample_size, seed, cache_dir)
    if path.exists():
        return str(path), _count_lines(path)

    sequences = load_jsonl_sequences(meta.splits["train"])
    subset = _stratified_subsample(sequences, sample_size, seed)
    del sequences
    temporary = path.with_name(
        "{}.{}.{}.tmp".format(path.name, os.getpid(), uuid.uuid4().hex[:8])
    )
    write_jsonl(str(temporary), subset)
    _publish_once(temporary, path)
    return str(path), len(subset)


def _subset_path(meta: DatasetMeta, sample_size: int, seed: int, cache_dir: Optional[Path] = None) -> Path:
    """Cache path for one training subset. Single source of truth for the name."""

    directory = cache_dir or (output_root() / "subsets" / meta.name)
    directory.mkdir(parents=True, exist_ok=True)
    source_stat = Path(meta.splits["train"]).stat()
    source_identity = "{}:{}".format(source_stat.st_size, source_stat.st_mtime_ns)
    fingerprint = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:10]
    return directory / "train_n{}_s{}_{}.jsonl".format(sample_size, seed, fingerprint)


def materialize_subsets(
    meta: DatasetMeta,
    sample_sizes: Sequence[int],
    seeds: Sequence[int],
    cache_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Write every training subset for one dataset, reading the split once.

    Run this before an array so the workers only ever *read* the cache. Two
    reasons: a worker that has to build a subset parses the whole training split
    (hundreds of megabytes on the large datasets), and concurrent builders of
    the same subset are what produced the `Errno 116` failures that
    `_publish_once` guards against.
    """

    wanted = [
        (int(size), int(seed))
        for size in sample_sizes
        for seed in seeds
        if not _subset_path(meta, int(size), int(seed), cache_dir).exists()
    ]
    written: Dict[str, int] = {}
    if not wanted:
        return written

    sequences = load_jsonl_sequences(meta.splits["train"])
    for size, seed in wanted:
        path = _subset_path(meta, size, seed, cache_dir)
        if path.exists():
            continue
        subset = _stratified_subsample(sequences, size, seed)
        temporary = path.with_name(
            "{}.{}.{}.tmp".format(path.name, os.getpid(), uuid.uuid4().hex[:8])
        )
        write_jsonl(str(temporary), subset)
        _publish_once(temporary, path)
        written[path.name] = len(subset)
    return written


def build_vocabulary(
    meta: DatasetMeta,
    train_path: str,
    additional_paths: Sequence[str] = (),
    event_types: Optional[Sequence[str]] = None,
) -> EventVocabulary:
    """Vocabulary from the *training* subset plus the pinned mark ordering.

    Feature values are learned from the visible training data only; the mark
    list comes from the dataset metadata so the next-event label space is
    identical at every sample size.
    """

    sequences = load_jsonl_sequences(train_path)
    for path in additional_paths:
        sequences.extend(load_jsonl_sequences(path))
    return EventVocabulary.from_sequences(
        sequences, event_types=list(event_types or meta.event_types)
    )


def _cross_dataset_pretrain_path(
    target_dataset: str,
    source_datasets: Sequence[str],
    sample_size_per_dataset: int,
    seed: int,
) -> tuple:
    """Materialize a deterministic selected-dataset unlabeled pretraining pool."""

    sequences: List[EventSequence] = []
    identities = []
    source_metas = []
    for offset, name in enumerate(source_datasets):
        meta = load_dataset_meta(name)
        source_metas.append(meta)
        path = Path(meta.splits["train"])
        stat = path.stat()
        identities.append("{}:{}:{}".format(name, stat.st_size, stat.st_mtime_ns))
        available = load_jsonl_sequences(str(path))
        rng = np.random.default_rng(seed + 1009 * (offset + 1))
        rng.shuffle(available)
        sequences.extend(available[:sample_size_per_dataset])

    if not sequences:
        raise ValueError("Cross-dataset pretraining resolved to an empty source pool.")
    identity = "|".join(identities + [str(sample_size_per_dataset), str(seed)])
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    directory = output_root() / "subsets" / "cross_dataset"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "{}_n{}_s{}_{}.jsonl".format(
        target_dataset, sample_size_per_dataset, seed, fingerprint
    )
    if not path.exists():
        temporary = path.with_name(
            "{}.{}.{}.tmp".format(path.name, os.getpid(), uuid.uuid4().hex[:8])
        )
        write_jsonl(str(temporary), sequences)
        _publish_once(temporary, path)
    return str(path), len(sequences), source_metas


def build_context(
    cell: BenchmarkCell,
    training: Optional[TrainingSpec] = None,
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    extra: Optional[Dict[str, object]] = None,
) -> MethodContext:
    meta = load_dataset_meta(cell.dataset)
    train_path, num_train = subsampled_train_path(meta, cell.sample_size, cell.seed)
    resolved_extra = dict(extra or {})
    pretrain_sources = resolved_extra.get("pretrain_datasets")
    source_metas = []
    additional_paths = []
    model_event_types = list(meta.event_types)
    model_feature_fields = list(meta.feature_fields)
    if pretrain_sources:
        if pretrain_sources == "leave-one-out-primary":
            source_names = [name for name in PRIMARY_DATASETS if name != cell.dataset]
        elif pretrain_sources == "target":
            source_names = [cell.dataset]
        elif isinstance(pretrain_sources, str):
            source_names = [pretrain_sources]
        else:
            source_names = [str(name) for name in pretrain_sources]
        pretrain_path, pretrain_count, source_metas = _cross_dataset_pretrain_path(
            cell.dataset,
            source_names,
            int(resolved_extra.get("pretrain_sample_size_per_dataset", cell.sample_size)),
            cell.seed,
        )
        additional_paths.append(pretrain_path)
        resolved_extra["pretrain_path"] = pretrain_path
        resolved_extra["pretrain_num_sequences"] = pretrain_count
        resolved_extra["pretrain_datasets_resolved"] = [item.name for item in source_metas]
        for source_meta in source_metas:
            for event_type in source_meta.event_types:
                if event_type not in model_event_types:
                    model_event_types.append(event_type)
            for field_name in source_meta.feature_fields:
                if field_name not in model_feature_fields:
                    model_feature_fields.append(field_name)
    resolved_extra["model_feature_fields"] = model_feature_fields
    vocab = build_vocabulary(
        meta,
        train_path,
        additional_paths=additional_paths,
        event_types=model_event_types,
    )
    tokenizer = EventTokenizer(
        vocab=vocab,
        max_features_per_event=max(2, len(model_feature_fields) + 1),
        event_type_to_index={
            event_type: index for index, event_type in enumerate(model_event_types)
        },
    )
    spec = replace(training) if training is not None else TrainingSpec()
    spec.seed = cell.seed
    spec.adaptation = cell.regime
    context = MethodContext(
        dataset=meta,
        task=cell.task,
        train_path=train_path,
        validation_path=meta.splits["validation"],
        test_path=meta.splits["test"],
        vocab=vocab,
        tokenizer=tokenizer,
        training=spec,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=hidden_size * 2,
        extra=resolved_extra,
    )
    context.extra.setdefault("num_train_sequences", num_train)
    return context


def result_path(cell: BenchmarkCell, results_dir: Optional[Path] = None) -> Path:
    directory = results_dir or (output_root() / "results")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{}.json".format(cell.key)


def _git_provenance() -> Dict[str, object]:
    """Best-effort immutable source identity for every result artifact."""

    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        source_digest = hashlib.sha256()
        source_files = [repository / "pyproject.toml"]
        for directory in ("eventfm", "scripts"):
            source_files.extend(sorted((repository / directory).rglob("*.py")))
        for path in source_files:
            if not path.is_file():
                continue
            source_digest.update(str(path.relative_to(repository)).encode("utf-8"))
            source_digest.update(path.read_bytes())
        return {
            "commit": commit,
            "dirty": dirty,
            "source_tree_sha256": source_digest.hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source_tree_sha256": None}


def run_cell(
    cell: BenchmarkCell,
    training: Optional[TrainingSpec] = None,
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    extra: Optional[Dict[str, object]] = None,
    results_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> RunResult:
    target = result_path(cell, results_dir)
    if target.exists() and not overwrite:
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return RunResult(**payload)

    seed_all(cell.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        context = build_context(
            cell,
            training=training,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            extra=extra,
        )
        method = build_method(cell.method, context)
        metrics = method.run()
        if torch.cuda.is_available():
            metrics["peak_gpu_memory_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024.0**2)
            )
        method_spec = METHOD_REGISTRY[cell.method]
        dataset_spec = DATASET_REGISTRY[cell.dataset]
        result = RunResult(
            cell=asdict(cell),
            metrics={
                key: (None if value is None else float(value))
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
            status="ok",
            wall_seconds=time.time() - started,
            num_train_sequences=int(context.extra.get("num_train_sequences", 0)),
            # Counted, not parsed: the method has already read this split, and
            # re-parsing 650 MB of JSONL for a row count was pure overhead.
            num_test_sequences=_count_lines(Path(context.test_path)),
            metadata={
                "method": {
                    "display_name": method_spec.display_name,
                    "family": method_spec.family,
                    "reference": method_spec.reference,
                    "status": method_spec.status,
                    "divergence": method_spec.divergence,
                },
                "dataset": {
                    "display_name": dataset_spec.display_name,
                    "source": dataset_spec.source,
                    "citation": dataset_spec.citation,
                    "supports": list(dataset_spec.supports),
                    "prepared_metadata": json.loads(json.dumps(context.dataset.extra, default=str)),
                },
                "git": _git_provenance(),
                "training": asdict(context.training),
                "model": {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                },
                "extra": json.loads(json.dumps(context.extra, default=str)),
            },
        )
    except Exception as exc:  # noqa: BLE001 - a failed cell must not kill the array job
        result = RunResult(
            cell=asdict(cell),
            metrics={},
            status="failed",
            error="{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc(limit=6)),
            wall_seconds=time.time() - started,
        )

    # Written through a temp file: `open("w")` truncates before the dump, so a
    # worker killed in that window (scancel, node failure, the pre-timeout
    # requeue) would leave a half-written result behind. Distinct cells write
    # distinct names, so there is no concurrent reader to strand here.
    temporary = target.with_name("{}.{}.{}.tmp".format(target.name, os.getpid(), uuid.uuid4().hex[:8]))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result.to_json(), handle, indent=2, sort_keys=True)
    temporary.replace(target)
    return result
