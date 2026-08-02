"""Shared conversion from a flat transaction table to ``EventSequence`` JSONL.

All four open benchmarks arrive as one row per transaction. The differences are
only in column names, timestamp encoding and which column carries the target,
so every adapter reduces its source to a :class:`TransactionTable` and this
module does the rest:

1. bucket numeric fields into quantile tokens (type-aware value encoding, as in
   PRAGMA and TabFormer),
2. group rows into per-entity histories ordered by time,
3. hold out the tail of each history to build a leakage-free classification
   label ("does a flagged event occur in the future?"),
4. rebalance negatives so small-sample scaling points still contain positives,
5. split entities into train/validation/test and write JSONL.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eventfm.data.io import write_jsonl
from eventfm.data.schema import Event, EventSequence


@dataclass
class TransactionTable:
    """Column-oriented view of a raw benchmark, ready for conversion."""

    entity_ids: np.ndarray
    timestamps: np.ndarray
    event_types: np.ndarray
    categorical: Dict[str, np.ndarray] = field(default_factory=dict)
    numeric: Dict[str, np.ndarray] = field(default_factory=dict)
    flags: Optional[np.ndarray] = None
    entity_labels: Optional[Dict[str, int]] = None
    profile: Dict[str, np.ndarray] = field(default_factory=dict)
    # When the benchmark ships its own forward-looking target, the history must
    # stop before the label window opens. Rows after this timestamp are dropped.
    time_cutoff: Optional[float] = None

    def __len__(self) -> int:
        return int(self.entity_ids.shape[0])


@dataclass
class ConversionConfig:
    """Knobs that keep the four benchmarks comparable to one another."""

    num_numeric_buckets: int = 16
    max_categorical_cardinality: int = 64
    min_events_per_entity: int = 8
    max_events_per_entity: int = 128
    label_horizon_fraction: float = 0.25
    min_label_horizon_events: int = 1
    target_positive_rate: Optional[float] = 0.25
    max_entities: Optional[int] = 60000
    validation_fraction: float = 0.15
    test_fraction: float = 0.20
    seed: int = 17


def _stable_bucket(entity_id: str, num_buckets: int = 1000) -> int:
    digest = hashlib.md5(str(entity_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def quantile_bucketize(values: np.ndarray, num_buckets: int) -> Tuple[np.ndarray, List[float]]:
    """Map a numeric column to ``b_<idx>`` tokens using quantile edges.

    Quantiles rather than uniform width, because transaction amounts are heavy
    tailed and uniform bins would collapse almost everything into bin 0.
    """

    finite = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(finite)
    if not valid.any():
        return np.full(finite.shape, "na", dtype=object), []
    probs = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    edges = np.unique(np.quantile(finite[valid], probs))
    indices = np.digitize(finite, edges, right=False)
    tokens = np.asarray(["b_{}".format(int(i)) for i in indices], dtype=object)
    tokens[~valid] = "na"
    return tokens, [float(edge) for edge in edges]


def collapse_rare_categories(values: np.ndarray, max_cardinality: int) -> np.ndarray:
    """Keep the most frequent levels, fold the tail into ``other``."""

    tokens = np.asarray([str(value) for value in values], dtype=object)
    unique, counts = np.unique(tokens, return_counts=True)
    if unique.shape[0] <= max_cardinality:
        return tokens
    keep = set(unique[np.argsort(-counts)[:max_cardinality]].tolist())
    return np.asarray([token if token in keep else "other" for token in tokens], dtype=object)


def _group_rows(entity_ids: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """Return ``entity_id -> row indices`` with each group ordered by time."""

    unique_ids, codes = np.unique(entity_ids, return_inverse=True)
    order = np.lexsort((timestamps, codes))
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    return {
        str(unique_ids[codes[chunk[0]]]): chunk
        for chunk in np.split(order, boundaries)
        if chunk.size
    }


def _filter_rows(table: TransactionTable, mask: np.ndarray) -> TransactionTable:
    return TransactionTable(
        entity_ids=table.entity_ids[mask],
        timestamps=np.asarray(table.timestamps)[mask],
        event_types=np.asarray(table.event_types)[mask],
        categorical={name: np.asarray(values)[mask] for name, values in table.categorical.items()},
        numeric={name: np.asarray(values)[mask] for name, values in table.numeric.items()},
        flags=None if table.flags is None else np.asarray(table.flags)[mask],
        entity_labels=table.entity_labels,
        profile={name: np.asarray(values)[mask] for name, values in table.profile.items()},
        time_cutoff=None,
    )


def _rebalance(
    labels: Dict[str, int],
    target_positive_rate: Optional[float],
    rng: np.random.Generator,
) -> List[str]:
    """Downsample negatives so tiny training subsets still contain positives."""

    positives = [key for key, value in labels.items() if value == 1]
    negatives = [key for key, value in labels.items() if value != 1]
    if target_positive_rate is None or not positives or not negatives:
        keep = positives + negatives
        rng.shuffle(keep)
        return keep
    natural = len(positives) / float(len(positives) + len(negatives))
    if natural >= target_positive_rate:
        keep = positives + negatives
        rng.shuffle(keep)
        return keep
    wanted_negatives = int(
        round(len(positives) * (1.0 - target_positive_rate) / target_positive_rate)
    )
    wanted_negatives = min(len(negatives), max(len(positives), wanted_negatives))
    chosen = rng.choice(np.asarray(negatives, dtype=object), size=wanted_negatives, replace=False)
    keep = positives + [str(item) for item in chosen]
    rng.shuffle(keep)
    return keep


def _assign_split(entity_id: str, validation_fraction: float, test_fraction: float) -> str:
    bucket = _stable_bucket(entity_id) / 1000.0
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + validation_fraction:
        return "validation"
    return "train"


def build_sequences(
    table: TransactionTable,
    config: ConversionConfig,
) -> Tuple[Dict[str, List[EventSequence]], Dict[str, object]]:
    """Convert a :class:`TransactionTable` into per-split event histories."""

    rng = np.random.default_rng(config.seed)
    keep_rows = (
        np.ones(len(table), dtype=bool)
        if table.time_cutoff is None
        else np.asarray(table.timestamps, dtype=np.float64) <= float(table.time_cutoff)
    )
    if not keep_rows.all():
        table = _filter_rows(table, keep_rows)

    entity_ids = np.asarray([str(value) for value in table.entity_ids], dtype=object)
    timestamps = np.asarray(table.timestamps, dtype=np.float64)
    event_types = collapse_rare_categories(table.event_types, config.max_categorical_cardinality)

    feature_tokens: Dict[str, np.ndarray] = {}
    bucket_edges: Dict[str, List[float]] = {}
    for name, values in table.categorical.items():
        feature_tokens[name] = collapse_rare_categories(values, config.max_categorical_cardinality)
    for name, values in table.numeric.items():
        tokens, edges = quantile_bucketize(
            np.asarray(values, dtype=np.float64), config.num_numeric_buckets
        )
        feature_tokens[name] = tokens
        bucket_edges[name] = edges

    flags = None if table.flags is None else np.asarray(table.flags, dtype=np.int64)
    groups = _group_rows(entity_ids, timestamps)

    provided_labels = table.entity_labels
    labels: Dict[str, int] = {}
    histories: Dict[str, np.ndarray] = {}
    for entity_id, row_indices in groups.items():
        if row_indices.shape[0] < config.min_events_per_entity:
            continue
        if row_indices.shape[0] > config.max_events_per_entity:
            row_indices = row_indices[-config.max_events_per_entity :]

        if provided_labels is not None:
            if entity_id not in provided_labels:
                continue
            histories[entity_id] = row_indices
            labels[entity_id] = int(provided_labels[entity_id])
            continue

        horizon = max(
            config.min_label_horizon_events,
            int(round(row_indices.shape[0] * config.label_horizon_fraction)),
        )
        observed = row_indices[:-horizon]
        future = row_indices[-horizon:]
        if observed.shape[0] < config.min_events_per_entity - horizon or observed.shape[0] < 2:
            continue
        histories[entity_id] = observed
        labels[entity_id] = int(flags[future].max()) if flags is not None else 0

    natural_positive_rate = (
        float(np.mean(list(labels.values()))) if labels else 0.0
    )
    kept = _rebalance(labels, config.target_positive_rate, rng)
    if config.max_entities is not None and len(kept) > config.max_entities:
        positives = [key for key in kept if labels[key] == 1]
        negatives = [key for key in kept if labels[key] != 1]
        share = config.max_entities / float(len(kept))
        positives = positives[: max(1, int(round(len(positives) * share)))]
        negatives = negatives[: max(1, config.max_entities - len(positives))]
        kept = positives + negatives
        rng.shuffle(kept)

    feature_names = sorted(feature_tokens.keys())
    profile_names = sorted(table.profile.keys())
    splits: Dict[str, List[EventSequence]] = {"train": [], "validation": [], "test": []}
    for entity_id in kept:
        row_indices = histories[entity_id]
        events = [
            Event(
                event_type=str(event_types[row]),
                timestamp=float(timestamps[row]),
                features={name: str(feature_tokens[name][row]) for name in feature_names},
            )
            for row in row_indices
        ]
        first_row = int(row_indices[0])
        sequence = EventSequence(
            user_id=entity_id,
            events=events,
            label=int(labels[entity_id]),
            profile={name: str(table.profile[name][first_row]) for name in profile_names},
        )
        split = _assign_split(entity_id, config.validation_fraction, config.test_fraction)
        splits[split].append(sequence)

    meta = {
        "event_types": sorted({str(value) for value in event_types}),
        "feature_fields": feature_names,
        "profile_fields": profile_names,
        "numeric_bucket_edges": bucket_edges,
        "natural_positive_rate": natural_positive_rate,
        "used_positive_rate": float(np.mean([labels[key] for key in kept])) if kept else 0.0,
        "num_entities_total": len(labels),
        "num_entities_kept": len(kept),
        "num_rows": len(table),
    }
    return splits, meta


def write_splits(
    splits: Dict[str, List[EventSequence]],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for split, sequences in splits.items():
        path = output_dir / "{}.jsonl".format(split)
        write_jsonl(str(path), sequences)
        paths[split] = str(path)
    return paths


def write_meta(output_dir: Path, meta: Dict[str, object]) -> Path:
    path = output_dir / "meta.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, default=str)
    return path


def summarise_splits(splits: Dict[str, Sequence[EventSequence]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for split, sequences in splits.items():
        if not sequences:
            summary[split] = {"num_sequences": 0}
            continue
        lengths = np.asarray([len(sequence.events) for sequence in sequences], dtype=np.float64)
        labels = np.asarray([int(sequence.label or 0) for sequence in sequences], dtype=np.float64)
        summary[split] = {
            "num_sequences": int(lengths.shape[0]),
            "mean_events": float(lengths.mean()),
            "median_events": float(np.median(lengths)),
            "positive_rate": float(labels.mean()),
        }
    return summary
