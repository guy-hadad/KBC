"""Tests for the three large-scale benchmarks and the per-dataset scaling grid."""

import numpy as np
import pytest

from eventfm.benchmark.experiments import (
    AUTO_SCALING,
    DEFAULT_SIZES,
    SCALE_GRID,
    get_experiment,
    resolve_sample_sizes,
    scaling_sample_sizes,
)
from eventfm.benchmark.runner import enumerate_cells
from eventfm.datasets.adapters import ADAPTERS, _interned
from eventfm.datasets.download import DOWNLOADERS, MBD_PTLS_KEEP_PREFIXES
from eventfm.datasets.registry import DATASET_REGISTRY, SCALE_DATASETS

SCALE_SUITES = ("scale-datasets", "scale-datasets-baselines")


def test_scale_datasets_are_registered_with_an_adapter_and_a_downloader():
    for name in SCALE_DATASETS:
        assert name in DATASET_REGISTRY, name
        assert name in ADAPTERS, name
        assert name in DOWNLOADERS, name
        assert DATASET_REGISTRY[name].supports == ("classification", "tpp"), name


def test_scale_datasets_declare_a_source_and_a_forward_looking_label():
    for name in SCALE_DATASETS:
        spec = DATASET_REGISTRY[name]
        assert spec.source.startswith("hf:"), name
        assert spec.label, name
        assert spec.citation, name


def test_interning_shares_one_string_object_per_distinct_value():
    import polars as pl

    series = pl.Series("mark", ["a", "b", "a", None, "b"])
    values = _interned(series)
    assert values.tolist() == ["a", "b", "a", "na", "b"]
    # Identity, not equality: this is the property that keeps a 64M-row
    # conversion inside its memory budget.
    assert values[0] is values[2]
    assert values[1] is values[4]


def test_mbd_ptls_filter_matches_the_archive_layout():
    # Members are `ptls/<source>/fold=N/part-*.parquet`; a filter of just
    # ("trx",) silently matched nothing and produced an empty extraction.
    assert "ptls/trx" in MBD_PTLS_KEEP_PREFIXES


def test_scaling_grid_is_clamped_to_the_pool_and_ends_on_it():
    sizes = scaling_sample_sizes("unprepared-dataset-name")
    assert sizes == DEFAULT_SIZES


@pytest.mark.parametrize("pool", [3_000, 34_023, 93_305, 500_000])
def test_scaling_grid_ends_at_the_pool_without_a_near_duplicate(pool, monkeypatch):
    import eventfm.benchmark.experiments as experiments

    class _Meta:
        stats = {"train": {"num_sequences": pool}}

    monkeypatch.setattr(experiments, "load_dataset_meta", lambda name: _Meta())
    sizes = experiments.scaling_sample_sizes("anything")
    assert sizes[-1] == pool
    assert sorted(sizes) == list(sizes)
    assert len(set(sizes)) == len(sizes)
    # No point sits within 25 % of the final one, which would double-weight the
    # top of the fitted scaling curve.
    assert sizes[-2] * 1.25 <= pool
    assert all(size in SCALE_GRID for size in sizes[:-1])


def test_scale_suites_resolve_one_grid_per_dataset():
    for name in SCALE_SUITES:
        suite = get_experiment(name)
        assert suite.extra.get("sample_size_mode") == AUTO_SCALING, name
        assert tuple(suite.datasets) == SCALE_DATASETS, name
        sizes = resolve_sample_sizes(suite)
        assert isinstance(sizes, dict), name
        assert set(sizes) == set(SCALE_DATASETS), name


def test_enumerate_cells_honours_a_per_dataset_grid():
    cells = enumerate_cells(
        ["banksim", "paysim"],
        ["classification"],
        ["count-logistic"],
        {"banksim": (64,), "paysim": (1024, 2048)},
    )
    by_dataset = {}
    for cell in cells:
        by_dataset.setdefault(cell.dataset, []).append(cell.sample_size)
    assert by_dataset == {"banksim": [64], "paysim": [1024, 2048]}


def test_enumerate_cells_falls_back_to_the_default_key():
    cells = enumerate_cells(
        ["banksim"], ["classification"], ["count-logistic"], {"default": (128,)}
    )
    assert [cell.sample_size for cell in cells] == [128]


def test_a_dataset_missing_from_the_mapping_contributes_no_cells():
    cells = enumerate_cells(
        ["banksim", "paysim"],
        ["classification"],
        ["count-logistic"],
        {"paysim": (256,)},
    )
    assert {cell.dataset for cell in cells} == {"paysim"}


def test_prepare_name_resolution_keeps_the_flag_scopes_separate():
    from eventfm.datasets.registry import (
        CHRONOLOGICAL_DATASETS,
        GEM_DATASETS,
        PRIMARY_DATASETS,
        resolve_prepare_names,
    )

    assert resolve_prepare_names() == list(PRIMARY_DATASETS)
    # `--scale` alone must not walk the primaries first.
    assert resolve_prepare_names(scale=True) == list(SCALE_DATASETS)
    assert resolve_prepare_names(paper=True) == list(
        PRIMARY_DATASETS + CHRONOLOGICAL_DATASETS + GEM_DATASETS
    )
    combined = resolve_prepare_names(paper=True, scale=True)
    assert set(SCALE_DATASETS) <= set(combined)
    assert set(PRIMARY_DATASETS) <= set(combined)
    assert len(combined) == len(set(combined))
    # Explicit names win over both flags.
    assert resolve_prepare_names(["banksim"], paper=True, scale=True) == ["banksim"]


def test_mbd_target_columns_resolve_across_both_releases():
    from eventfm.datasets.adapters import MBD_FULL_TARGET_COLUMNS, _mbd_label_column

    mini = ["client_id", "mon", "target_1", "target_2", "target_3", "target_4"]
    full = ["client_id", "mon", *MBD_FULL_TARGET_COLUMNS]
    # The same index must select the same product in either release: mini's
    # anonymised order matches the full release's named order.
    assert _mbd_label_column(mini, 1) == "target_1"
    assert _mbd_label_column(full, 1) == "bcard_target"
    assert _mbd_label_column(full, 2) == "cred_target"
    assert _mbd_label_column(full, 3) == "zp_target"
    assert _mbd_label_column(full, 4) == "acquiring_target"
    with pytest.raises(ValueError):
        _mbd_label_column(["client_id", "mon"], 1)


def test_eval_entity_cap_bounds_eval_splits_and_leaves_train_alone():
    from eventfm.datasets.tabular import ConversionConfig, build_sequences

    rng = np.random.default_rng(1)
    entities, events = 900, 20
    ids = np.repeat(
        np.asarray(["e{:04d}".format(i) for i in range(entities)], dtype=object), events
    )
    timestamps = np.tile(np.arange(events, dtype=np.float64) * 3600.0, entities)
    types = np.asarray(["t{}".format(i % 3) for i in range(entities * events)], dtype=object)
    flags = np.zeros(entities * events, dtype=np.int64)
    for entity in range(0, entities, 3):
        flags[entity * events + events - 1] = 1

    from eventfm.datasets.tabular import TransactionTable

    def table():
        return TransactionTable(
            entity_ids=ids,
            timestamps=timestamps,
            event_types=types,
            numeric={"amount": rng.normal(size=entities * events)},
            flags=flags,
        )

    base = ConversionConfig(min_events_per_entity=4, max_entities=None, max_eval_entities=None)
    capped = ConversionConfig(min_events_per_entity=4, max_entities=None, max_eval_entities=25)
    uncapped_splits, _ = build_sequences(table(), base)
    capped_splits, _ = build_sequences(table(), capped)

    assert len(capped_splits["validation"]) <= 25
    assert len(capped_splits["test"]) <= 25
    assert len(uncapped_splits["validation"]) > 25
    assert len(capped_splits["train"]) == len(uncapped_splits["train"])


def test_publish_once_never_swaps_an_existing_file(tmp_path):
    """The concurrency guard behind the three `Errno 116` failures.

    `os.replace` would have swapped the inode, which is what gives a concurrent
    reader a stale NFS handle; `_publish_once` must leave the winner in place.
    """
    from eventfm.benchmark.runner import _publish_once

    target = tmp_path / "subset.jsonl"
    target.write_text("winner\n", encoding="utf-8")
    winner_inode = target.stat().st_ino

    loser = tmp_path / "subset.jsonl.1234.abcd.tmp"
    loser.write_text("loser\n", encoding="utf-8")
    _publish_once(loser, target)

    assert target.read_text(encoding="utf-8") == "winner\n"
    assert target.stat().st_ino == winner_inode
    assert not loser.exists(), "the temp file must be cleaned up either way"


def test_publish_once_creates_the_file_when_absent(tmp_path):
    from eventfm.benchmark.runner import _publish_once

    target = tmp_path / "subset.jsonl"
    temporary = tmp_path / "subset.jsonl.999.ffff.tmp"
    temporary.write_text("payload\n", encoding="utf-8")
    _publish_once(temporary, target)

    assert target.read_text(encoding="utf-8") == "payload\n"
    assert not temporary.exists()
