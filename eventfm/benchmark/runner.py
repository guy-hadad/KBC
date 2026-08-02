"""Run one benchmark cell and persist its result.

A *cell* is a (dataset, task, method, training-sample-size, seed) tuple. Cells
are independent, which is what makes the whole grid a SLURM array job: each
array index resolves one cell, writes a single JSON file, and the aggregation
step later reads whatever finished.
"""

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from eventfm.data.io import load_jsonl_sequences, write_jsonl
from eventfm.data.schema import EventSequence
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary
from eventfm.datasets.paths import output_root
from eventfm.datasets.registry import DatasetMeta, load_dataset_meta
from eventfm.methods import METHOD_REGISTRY, MethodContext, build_method
from eventfm.methods.base import TrainingSpec, seed_all


@dataclass(frozen=True)
class BenchmarkCell:
    dataset: str
    task: str
    method: str
    sample_size: int
    seed: int = 13

    @property
    def key(self) -> str:
        return "{}__{}__{}__n{}__s{}".format(
            self.dataset, self.task, self.method, self.sample_size, self.seed
        )


@dataclass
class RunResult:
    cell: Dict[str, object]
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None
    wall_seconds: float = 0.0
    num_train_sequences: int = 0
    num_test_sequences: int = 0

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def enumerate_cells(
    datasets: Sequence[str],
    tasks: Sequence[str],
    methods: Sequence[str],
    sample_sizes: Sequence[int],
    seeds: Sequence[int] = (13,),
) -> List[BenchmarkCell]:
    """Expand the grid, skipping method/task pairs a method does not support."""

    cells: List[BenchmarkCell] = []
    for dataset in datasets:
        for task in tasks:
            for method in methods:
                spec = METHOD_REGISTRY.get(method)
                if spec is None or task not in spec.supports:
                    continue
                for sample_size in sample_sizes:
                    for seed in seeds:
                        cells.append(
                            BenchmarkCell(
                                dataset=dataset,
                                task=task,
                                method=method,
                                sample_size=int(sample_size),
                                seed=int(seed),
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


def subsampled_train_path(
    meta: DatasetMeta, sample_size: int, seed: int, cache_dir: Optional[Path] = None
) -> tuple:
    """Materialise a stratified training subset, cached on disk for reuse."""

    sequences = load_jsonl_sequences(meta.splits["train"])
    subset = _stratified_subsample(sequences, sample_size, seed)
    directory = cache_dir or (output_root() / "subsets" / meta.name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train_n{}_s{}.jsonl".format(sample_size, seed)
    if not path.exists():
        write_jsonl(str(path), subset)
    return str(path), len(subset)


def build_vocabulary(meta: DatasetMeta, train_path: str) -> EventVocabulary:
    """Vocabulary from the *training* subset plus the pinned mark ordering.

    Feature values are learned from the visible training data only; the mark
    list comes from the dataset metadata so the next-event label space is
    identical at every sample size.
    """

    sequences = load_jsonl_sequences(train_path)
    return EventVocabulary.from_sequences(sequences, event_types=meta.event_types)


def build_context(
    cell: BenchmarkCell,
    training: Optional[TrainingSpec] = None,
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    extra: Optional[Dict[str, object]] = None,
) -> MethodContext:
    meta = load_dataset_meta(cell.dataset)
    train_path, num_train = subsampled_train_path(meta, cell.sample_size, cell.seed)
    vocab = build_vocabulary(meta, train_path)
    tokenizer = EventTokenizer(
        vocab=vocab,
        max_features_per_event=max(2, len(meta.feature_fields) + 1),
        event_type_to_index=meta.event_type_to_index(),
    )
    spec = training or TrainingSpec()
    spec.seed = cell.seed
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
        extra=dict(extra or {}),
    )
    context.extra.setdefault("num_train_sequences", num_train)
    return context


def result_path(cell: BenchmarkCell, results_dir: Optional[Path] = None) -> Path:
    directory = results_dir or (output_root() / "results")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{}.json".format(cell.key)


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
            num_test_sequences=len(load_jsonl_sequences(context.test_path)),
        )
    except Exception as exc:  # noqa: BLE001 - a failed cell must not kill the array job
        result = RunResult(
            cell=asdict(cell),
            metrics={},
            status="failed",
            error="{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc(limit=6)),
            wall_seconds=time.time() - started,
        )

    with target.open("w", encoding="utf-8") as handle:
        json.dump(result.to_json(), handle, indent=2, sort_keys=True)
    return result
