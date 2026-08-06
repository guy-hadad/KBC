"""Tests for the shared transaction-table → EventSequence conversion."""

import numpy as np
import pytest

from eventfm.datasets.tabular import (
    ConversionConfig,
    TransactionTable,
    build_sequences,
    collapse_rare_categories,
    quantile_bucketize,
)


def _table(num_entities: int = 40, events_per_entity: int = 20) -> TransactionTable:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(
        np.asarray(["e{:03d}".format(index) for index in range(num_entities)], dtype=object),
        events_per_entity,
    )
    total = num_entities * events_per_entity
    timestamps = np.tile(np.arange(events_per_entity, dtype=np.float64) * 3600.0, num_entities)
    event_types = np.asarray(
        ["type_{}".format(index % 4) for index in range(total)], dtype=object
    )
    # Only the last events of the first ten entities are flagged.
    flags = np.zeros(total, dtype=np.int64)
    for entity in range(10):
        flags[entity * events_per_entity + events_per_entity - 1] = 1
    return TransactionTable(
        entity_ids=entity_ids,
        timestamps=timestamps,
        event_types=event_types,
        categorical={"channel": np.asarray(["web", "app"] * (total // 2), dtype=object)},
        numeric={"amount": rng.lognormal(3.0, 1.5, size=total)},
        flags=flags,
    )


def test_quantile_bucketize_spreads_a_heavy_tail():
    values = np.concatenate([np.arange(1, 100, dtype=float), np.asarray([1e6, 1e7])])
    tokens, edges = quantile_bucketize(values, num_buckets=8)

    assert len(set(tokens.tolist())) > 1
    assert edges == sorted(edges)
    # A uniform-width binning would put ~all of these in one bucket.
    assert len(set(tokens[:99].tolist())) >= 4


def test_quantile_bucketize_marks_missing_values():
    tokens, _ = quantile_bucketize(np.asarray([1.0, 2.0, np.nan, 4.0]), num_buckets=4)

    assert tokens[2] == "na"


def test_collapse_rare_categories_keeps_the_frequent_levels():
    values = np.asarray(["a"] * 10 + ["b"] * 8 + ["c"] + ["d"], dtype=object)

    collapsed = collapse_rare_categories(values, max_cardinality=2)

    assert set(collapsed.tolist()) == {"a", "b", "other"}


def test_build_sequences_holds_out_the_label_horizon():
    """The flagged events must not survive into the observed history."""

    config = ConversionConfig(
        min_events_per_entity=8,
        max_events_per_entity=32,
        label_horizon_fraction=0.25,
        target_positive_rate=None,
        max_entities=None,
        seed=3,
    )
    splits, meta = build_sequences(_table(), config)

    all_sequences = [item for bucket in splits.values() for item in bucket]
    assert all_sequences, "conversion produced no sequences"

    positives = [item for item in all_sequences if int(item.label) == 1]
    assert positives, "the flagged entities should yield positive labels"

    # 20 events, 25% horizon -> 5 held out, so only 15 remain visible.
    for sequence in all_sequences:
        assert len(sequence.events) == 15
    assert meta["num_entities_kept"] == len(all_sequences)


def test_build_sequences_splits_entities_disjointly():
    config = ConversionConfig(
        min_events_per_entity=8,
        max_events_per_entity=32,
        target_positive_rate=None,
        max_entities=None,
    )
    splits, _ = build_sequences(_table(), config)

    seen = [{sequence.user_id for sequence in bucket} for bucket in splits.values()]
    for index, left in enumerate(seen):
        for right in seen[index + 1 :]:
            assert not (left & right), "an entity leaked across splits"


def test_rebalancing_raises_the_positive_rate():
    config = ConversionConfig(
        min_events_per_entity=8,
        max_events_per_entity=32,
        target_positive_rate=0.5,
        max_entities=None,
        seed=5,
    )
    splits, meta = build_sequences(_table(), config)

    assert meta["natural_positive_rate"] == pytest.approx(0.25)
    assert meta["used_positive_rate"] > meta["natural_positive_rate"]


def test_rebalancing_does_not_change_evaluation_prevalence():
    config = ConversionConfig(
        min_events_per_entity=8,
        max_events_per_entity=32,
        target_positive_rate=0.5,
        max_entities=None,
        seed=5,
    )

    _, meta = build_sequences(_table(), config)

    for split in ("validation", "test"):
        assert meta["split_used_positive_rate"][split] == pytest.approx(
            meta["split_natural_positive_rate"][split]
        )
    assert meta["split_used_positive_rate"]["train"] >= meta[
        "split_natural_positive_rate"
    ]["train"]


def test_categorical_vocabulary_is_fit_on_training_histories_only():
    table = _table()
    table.categorical["entity_code"] = np.asarray(table.entity_ids, dtype=object)
    config = ConversionConfig(
        min_events_per_entity=8,
        max_events_per_entity=32,
        target_positive_rate=None,
        max_categorical_cardinality=100,
        max_entities=None,
    )

    splits, meta = build_sequences(table, config)

    train_ids = {sequence.user_id for sequence in splits["train"]}
    assert set(meta["categorical_levels"]["entity_code"]) == train_ids
    for split in ("validation", "test"):
        for sequence in splits[split]:
            assert {event.features["entity_code"] for event in sequence.events} == {"other"}
    assert meta["preprocessing_fit_split"] == "train"


def test_chronological_split_keeps_whole_entities_and_orders_history_endpoints():
    table = _table()
    entity_offsets = np.repeat(np.arange(40, dtype=np.float64) * 30.0 * 86400.0, 20)
    table.timestamps = table.timestamps + entity_offsets
    config = ConversionConfig(
        min_events_per_entity=8,
        max_events_per_entity=32,
        target_positive_rate=None,
        max_entities=None,
        split_strategy="chronological",
    )

    splits, meta = build_sequences(table, config)

    endpoints = {
        split: [sequence.events[-1].timestamp for sequence in sequences]
        for split, sequences in splits.items()
    }
    assert max(endpoints["train"]) <= min(endpoints["validation"])
    assert max(endpoints["validation"]) <= min(endpoints["test"])
    assert meta["split_strategy"] == "chronological"


def test_time_cutoff_drops_later_rows():
    table = _table()
    table.time_cutoff = 5.0 * 3600.0
    table.entity_labels = {"e{:03d}".format(index): index % 2 for index in range(40)}
    config = ConversionConfig(
        min_events_per_entity=4,
        max_events_per_entity=32,
        target_positive_rate=None,
        max_entities=None,
    )

    splits, _ = build_sequences(table, config)

    for sequence in (item for bucket in splits.values() for item in bucket):
        assert max(event.timestamp for event in sequence.events) <= 5.0 * 3600.0
