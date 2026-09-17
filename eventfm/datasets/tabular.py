"""Shared conversion from a flat transaction table to ``EventSequence`` JSONL.

All four open benchmarks arrive as one row per transaction. The differences are
only in column names, timestamp encoding and which column carries the target,
so every adapter reduces its source to a :class:`TransactionTable` and this
module does the rest:

1. split entities, then fit numeric/category transforms on training histories
   only (type-aware value encoding, as in PRAGMA and TabFormer),
2. group rows into per-entity histories ordered by time,
3. hold out the tail of each history to build a leakage-free classification
   label ("does a flagged event occur in the future?"),
4. rebalance training negatives so small-sample scaling points contain positives
   while validation and test retain their natural prevalence,
5. write the disjoint entity splits to JSONL.
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
    preserve_eval_prevalence: bool = True
    split_strategy: str = "hash"
    max_entities: Optional[int] = 60000
    # Caps validation and test only, leaving the training pool untouched. Every
    # cell reads the whole evaluation split, so on a million-entity dataset this
    # is what keeps a single cell inside its memory budget; the subsample is
    # uniform, so the natural prevalence of the split is preserved.
    max_eval_entities: Optional[int] = None
    validation_fraction: float = 0.15
    test_fraction: float = 0.20
    seed: int = 17


def _stable_bucket(entity_id: str, num_buckets: int = 1000) -> int:
    digest = hashlib.md5(str(entity_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def fit_quantile_edges(values: np.ndarray, num_buckets: int) -> List[float]:
    """Fit quantile boundaries without retaining the source values."""

    finite = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(finite)
    if not valid.any():
        return []
    probs = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    return [float(edge) for edge in np.unique(np.quantile(finite[valid], probs))]


def apply_quantile_edges(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Apply training-set quantile boundaries to any split."""

    finite = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(finite)
    indices = np.digitize(finite, np.asarray(edges, dtype=np.float64), right=False)
    tokens = np.asarray(["b_{}".format(int(index)) for index in indices], dtype=object)
    tokens[~valid] = "na"
    return tokens


def quantile_bucketize(values: np.ndarray, num_buckets: int) -> Tuple[np.ndarray, List[float]]:
    """Map a numeric column to ``b_<idx>`` tokens using quantile edges.

    Quantiles rather than uniform width, because transaction amounts are heavy
    tailed and uniform bins would collapse almost everything into bin 0.
    """

    edges = fit_quantile_edges(values, num_buckets)
    return apply_quantile_edges(values, edges), edges


def fit_category_levels(values: np.ndarray, max_cardinality: int) -> List[str]:
    """Select the most frequent training levels with deterministic tie breaks.

    Counted through a dictionary rather than ``np.unique``: sorting an object
    array compares Python strings pairwise, which is minutes per column once a
    dataset reaches tens of millions of events.
    """

    counts: Dict[str, int] = {}
    for value in np.asarray(values).tolist():
        token = value if isinstance(value, str) else str(value)
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return []
    order = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in order[:max_cardinality]]


def apply_category_levels(values: np.ndarray, levels: Sequence[str]) -> np.ndarray:
    """Map categories unseen in training to an explicit ``other`` token."""

    keep = set(levels)
    return np.asarray(
        [str(value) if str(value) in keep else "other" for value in values], dtype=object
    )


def collapse_rare_categories(values: np.ndarray, max_cardinality: int) -> np.ndarray:
    """Keep the most frequent levels, fold the tail into ``other``."""

    levels = fit_category_levels(values, max_cardinality)
    return apply_category_levels(values, levels)


def _positive_rate(entity_ids: Sequence[str], labels: Dict[str, int]) -> float:
    return float(np.mean([labels[entity_id] for entity_id in entity_ids])) if entity_ids else 0.0


def _proportional_caps(lengths: Dict[str, int], total_cap: int) -> Dict[str, int]:
    """Allocate a global entity cap across splits by largest remainder."""

    total = sum(lengths.values())
    if total <= total_cap:
        return dict(lengths)
    exact = {name: total_cap * count / float(total) for name, count in lengths.items()}
    caps = {name: min(lengths[name], int(np.floor(value))) for name, value in exact.items()}
    remaining = total_cap - sum(caps.values())
    order = sorted(lengths, key=lambda name: (-(exact[name] - caps[name]), name))
    for name in order:
        if remaining <= 0:
            break
        if caps[name] < lengths[name]:
            caps[name] += 1
            remaining -= 1
    return caps


def _group_rows(entity_ids: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """Return ``entity_id -> row indices`` with each group ordered by time."""

    # Factorised through a dictionary for the same reason as
    # `fit_category_levels`: `np.unique` on an object array is a string sort.
    # Entity ids are sorted first so the grouping order matches that sort.
    identifiers = np.asarray(entity_ids).tolist()
    unique_ids = sorted({value if isinstance(value, str) else str(value) for value in identifiers})
    code_of = {value: index for index, value in enumerate(unique_ids)}
    codes = np.fromiter(
        (code_of[value if isinstance(value, str) else str(value)] for value in identifiers),
        dtype=np.int64,
        count=len(identifiers),
    )
    del identifiers
    order = np.lexsort((np.asarray(timestamps), codes))
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    return {
        unique_ids[sorted_codes[start]]: chunk
        for start, chunk in zip(
            np.concatenate(([0], boundaries)) if boundaries.size else np.asarray([0]),
            np.split(order, boundaries),
        )
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


def _chronological_entity_splits(
    labels: Dict[str, int],
    histories: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
) -> Dict[str, List[str]]:
    """Assign older whole-entity histories to train and newer ones to evaluation."""

    ordered = sorted(
        labels,
        key=lambda entity_id: (
            float(timestamps[int(histories[entity_id][-1])]),
            entity_id,
        ),
    )
    num_entities = len(ordered)
    num_test = int(round(num_entities * test_fraction))
    num_validation = int(round(num_entities * validation_fraction))
    train_end = max(0, num_entities - num_test - num_validation)
    validation_end = num_entities - num_test
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


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

    if config.split_strategy == "chronological":
        ids_by_split = _chronological_entity_splits(
            labels,
            histories,
            timestamps,
            config.validation_fraction,
            config.test_fraction,
        )
    elif config.split_strategy == "hash":
        ids_by_split = {"train": [], "validation": [], "test": []}
        for entity_id in labels:
            split = _assign_split(entity_id, config.validation_fraction, config.test_fraction)
            ids_by_split[split].append(entity_id)
    else:
        raise ValueError(
            "Unknown split strategy `{}`; expected `hash` or `chronological`.".format(
                config.split_strategy
            )
        )

    split_natural_rates = {
        split: _positive_rate(entity_ids_in_split, labels)
        for split, entity_ids_in_split in ids_by_split.items()
    }
    if config.preserve_eval_prevalence:
        train_labels = {entity_id: labels[entity_id] for entity_id in ids_by_split["train"]}
        kept_by_split = {
            "train": _rebalance(train_labels, config.target_positive_rate, rng),
            "validation": list(ids_by_split["validation"]),
            "test": list(ids_by_split["test"]),
        }
        rng.shuffle(kept_by_split["validation"])
        rng.shuffle(kept_by_split["test"])
    else:
        kept = _rebalance(labels, config.target_positive_rate, rng)
        kept_by_split = {"train": [], "validation": [], "test": []}
        for entity_id in kept:
            split = _assign_split(entity_id, config.validation_fraction, config.test_fraction)
            kept_by_split[split].append(entity_id)

    if config.max_eval_entities is not None:
        for split in ("validation", "test"):
            if len(kept_by_split[split]) > config.max_eval_entities:
                rng.shuffle(kept_by_split[split])
                kept_by_split[split] = kept_by_split[split][: config.max_eval_entities]

    if config.max_entities is not None:
        caps = _proportional_caps(
            {
                split: len(entity_ids_in_split)
                for split, entity_ids_in_split in kept_by_split.items()
            },
            config.max_entities,
        )
        for split, cap in caps.items():
            rng.shuffle(kept_by_split[split])
            kept_by_split[split] = kept_by_split[split][:cap]

    # Every learned transform sees only events in the retained training histories.
    train_rows = np.concatenate(
        [histories[entity_id] for entity_id in kept_by_split["train"]],
        dtype=np.int64,
    ) if kept_by_split["train"] else np.asarray([], dtype=np.int64)
    event_type_levels = fit_category_levels(
        np.asarray(table.event_types)[train_rows], config.max_categorical_cardinality
    )
    event_types = apply_category_levels(table.event_types, event_type_levels)

    feature_tokens: Dict[str, np.ndarray] = {}
    category_levels: Dict[str, List[str]] = {}
    bucket_edges: Dict[str, List[float]] = {}
    for name, values in table.categorical.items():
        levels = fit_category_levels(
            np.asarray(values)[train_rows], config.max_categorical_cardinality
        )
        category_levels[name] = levels
        feature_tokens[name] = apply_category_levels(values, levels)
    for name, values in table.numeric.items():
        numeric_values = np.asarray(values, dtype=np.float64)
        edges = fit_quantile_edges(numeric_values[train_rows], config.num_numeric_buckets)
        feature_tokens[name] = apply_quantile_edges(numeric_values, edges)
        bucket_edges[name] = edges

    feature_names = sorted(feature_tokens.keys())
    profile_names = sorted(table.profile.keys())
    splits: Dict[str, List[EventSequence]] = {"train": [], "validation": [], "test": []}
    for split, kept in kept_by_split.items():
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
            splits[split].append(sequence)

    kept_all = [entity_id for kept in kept_by_split.values() for entity_id in kept]
    split_used_rates = {
        split: _positive_rate(entity_ids_in_split, labels)
        for split, entity_ids_in_split in kept_by_split.items()
    }

    meta = {
        "event_types": sorted({str(value) for value in event_types}),
        "feature_fields": feature_names,
        "profile_fields": profile_names,
        "event_type_levels": event_type_levels,
        "categorical_levels": category_levels,
        "numeric_bucket_edges": bucket_edges,
        "preprocessing_fit_split": "train",
        "split_strategy": config.split_strategy,
        "preserve_eval_prevalence": config.preserve_eval_prevalence,
        "natural_positive_rate": _positive_rate(list(labels), labels),
        # Compatibility: this field now refers to the split that is intentionally
        # rebalanced, rather than pooling altered train and natural evaluation data.
        "used_positive_rate": split_used_rates["train"],
        "split_natural_positive_rate": split_natural_rates,
        "split_used_positive_rate": split_used_rates,
        "split_num_entities_total": {split: len(items) for split, items in ids_by_split.items()},
        "split_num_entities_kept": {split: len(items) for split, items in kept_by_split.items()},
        "num_entities_total": len(labels),
        "num_entities_kept": len(kept_all),
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
