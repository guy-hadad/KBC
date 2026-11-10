"""Stage-2 MBD data: raw amounts, strict-prospective stores and all four targets.

ETAM-2 needs three things the prepared `mbd` benchmark does not carry. All are
written beside the existing ETAM stores in ``<output>/etam/stores/mbd``:

1. ``amount.npy`` in ``pool``, ``train``, ``validation`` and ``test``: the raw
   transaction amount of every stored event (float32, NaN when missing). Each
   history is replayed from raw MBD exactly as `scripts/build_mbd_pool.py`
   selects it (the last 128 events up to `MBD_HISTORY_CUTOFF`), and it must
   match the store event for event (timestamp, type code, feature codes)
   before anything is written.
2. ``{train,validation,test}_strict``: the same users in the same order, but
   each history is the client's last 128 events strictly before 2022-09-01.
   The benchmark label is the max over Sep 2022 - Jan 2023 while the standard
   history runs to 2022-09-30; cutting at 2022-09-01 means no observed event
   falls inside a label month. Labels are the benchmark's own, and users with
   fewer than two such events are dropped. Each store gets ``amount.npy`` too.
3. ``targets.npy`` (max over Sep 2022 - Jan 2023, the benchmark definition)
   and ``targets_oct.npy`` (Oct 2022 - Jan 2023 only) in the three benchmark
   stores and the three strict ones. Both are int8 ``(N, 4)`` in
   `MBD_FULL_TARGET_COLUMNS` order (column 0 is the benchmark label), with -1
   for a client that has no targets.

It also writes ``<processed>/mbd_strict/{train,validation,test}.jsonl`` and
``meta.json`` (the `mbd_strict` dataset) for the JSONL-based baselines.

`load_mbd` cannot supply the strict histories. It keeps only the last 128
events up to 2022-09-30 inside its scan, so a client with more events than
that has already lost the older ones a pre-September history needs. `_scan` is
`load_mbd`'s scan, file chunk for file chunk and up to the same sort, but it
keeps the union of both per-client tails. The parity check against the four
stores is what proves the standard tail is unchanged.

``--max-clients N`` is a smoke run. It uses `load_mbd`'s client subsample of
that size, writes nothing into the real stores, puts every output under a
``_smoke`` name, reports parity failures instead of raising, and skips the
JSONL.

Imports polars: run on a cpu128/cpu256 node (see the SIGILL note in memory).
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_mbd_pool import _amount_codes, _level_codes, _log  # noqa: E402

from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.datasets.registry import DATASET_REGISTRY, DatasetMeta, processed_dir  # noqa: E402
from eventfm.datasets.tabular import write_meta  # noqa: E402
from eventfm.etam.store import EtamVocab, EventStore, store_from_jsonl  # noqa: E402

STRICT_CUTOFF = 1_661_990_400.0  # 2022-09-01T00:00:00Z
SPLITS = ("train", "validation", "test")
SMOKE_TRAIN_USERS = 2000
# Leading sequences compared after the JSONL round trip. The same number of
# benchmark train.jsonl lines must be reproduced byte for byte by the writer.
JSONL_CHECK_SEQUENCES = 1000
READER_CHECK_SEQUENCES = 32
_STORE_ARRAYS = ("offsets", "types", "features", "timestamps", "labels", "user_ids")


def _save_array(path: Path, array: np.ndarray) -> None:
    """`np.save` under a temporary name, then an atomic rename."""

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(keys: np.ndarray, values) -> np.ndarray:
    """Index of each value in the sorted `keys`, -1 when it is absent."""

    values = np.asarray(values)
    if not keys.shape[0]:
        return np.full(values.shape[0], -1, dtype=np.int64)
    position = np.minimum(np.searchsorted(keys, values), keys.shape[0] - 1)
    return np.where(keys[position] == values, position, -1)


def _client_index(keys: np.ndarray, counts: np.ndarray, user_ids) -> np.ndarray:
    """Client code of each user, -1 when the scan holds none of their events."""

    index = _lookup(keys, user_ids)
    index[(index >= 0) & (counts[np.maximum(index, 0)] == 0)] = -1
    return index


def _scan(config, meta: dict, vocab: EtamVocab, clients: list, files: list) -> dict:
    """`load_mbd`'s transaction scan, keeping two tails per client instead of one.

    Everything up to the sort is `load_mbd`'s (file chunks, filters and sort),
    so rows come out in the order the stores were built from, ties included.
    Where `load_mbd` then keeps ``group_by("client_id").tail(128)``, this keeps
    every row that is in that tail *or* among the client's last 128 rows before
    `STRICT_CUTOFF`, both ranked from the end of the client's sorted rows.
    Columns are encoded chunk by chunk with the pool builder's helpers, so no
    full-size object column is ever built.
    """

    import polars as pl

    from eventfm.datasets.adapters import (
        MBD_EVENT_COLUMNS,
        MBD_FILE_CHUNK,
        MBD_HISTORY_CUTOFF,
        _interned,
    )

    tail = config.max_events_per_entity
    columns = list(MBD_EVENT_COLUMNS)
    keep_series = pl.Series("keep", clients, dtype=pl.Utf8)
    client_codes = pl.Enum(clients)
    edges = meta.get("numeric_bucket_edges", {})
    before = pl.col("event_time").cast(pl.Float64) < STRICT_CUTOFF
    wanted = ["client_id", "event_time", "event_type"] + list(vocab.feature_fields)
    parts = {name: [] for name in ("client", "timestamp", "amount", "type", "features")}
    for start in range(0, len(files), MBD_FILE_CHUNK):
        batch = files[start : start + MBD_FILE_CHUNK]
        frame = (
            pl.scan_parquet(batch)
            .select(["client_id"] + columns)
            .filter(pl.col("client_id").is_in(keep_series.implode()))
            .explode(columns, empty_as_null=True)
            .filter(pl.col("event_time").is_not_null())
            .filter(pl.col("event_time").cast(pl.Float64) <= MBD_HISTORY_CUTOFF)
            .sort("event_time")
            .with_columns(
                _from_end=pl.int_range(pl.len(), 0, -1).over("client_id"),
                _strict_from_end=before.cast(pl.UInt32).cum_sum(reverse=True).over("client_id"),
            )
            .filter(
                (pl.col("_from_end") <= tail) | (before & (pl.col("_strict_from_end") <= tail))
            )
            .select(wanted)
            .collect(engine="streaming")
        )
        _log("files {}-{}: {} rows kept".format(start, start + len(batch), frame.height))
        if not frame.height:
            continue
        codes = frame["client_id"].cast(client_codes).to_physical().to_numpy()
        parts["client"].append(codes.astype(np.uint32))
        parts["timestamp"].append(frame["event_time"].cast(pl.Float64).to_numpy())
        parts["amount"].append(frame["amount"].cast(pl.Float32).to_numpy())
        parts["type"].append(
            _level_codes(
                _interned(frame["event_type"]), meta["event_type_levels"], vocab.encode_type
            )
        )
        encoded = []
        for field_name in vocab.feature_fields:
            if field_name in edges:
                values = frame[field_name].cast(pl.Float64).to_numpy()
                encoded.append(_amount_codes(values, edges[field_name], vocab, field_name))
            else:
                encoded.append(
                    _level_codes(
                        _interned(frame[field_name]),
                        meta["categorical_levels"][field_name],
                        lambda value, name=field_name: vocab.encode_value(name, value),
                    )
                )
        parts["features"].append(np.stack(encoded, axis=1))
        del frame, codes, encoded
    if not parts["client"]:
        raise ValueError("No MBD transactions survived the history cutoff.")
    return {name: np.concatenate(parts.pop(name)) for name in list(parts)}


def _group(client: np.ndarray, timestamps: np.ndarray, num_clients: int, tail: int) -> dict:
    """Per-client row ranges in `_group_rows`' order.

    ``order`` is `_group_rows`' stable sort by client, then time (ties keep scan
    order); client codes index the sorted client list, so groups come in its
    order too. Every scanned row is up to `MBD_HISTORY_CUTOFF`, and the rows
    before `STRICT_CUTOFF` lead each client's range, so both tails are
    contiguous: the standard history is the range's last `tail` rows and the
    strict one is the last `tail` of its leading pre-cutoff rows.
    """

    order = np.lexsort((timestamps, client))
    counts = np.bincount(client, minlength=num_clients)
    ends = np.cumsum(counts)
    early = np.bincount(client[timestamps < STRICT_CUTOFF], minlength=num_clients)
    standard = np.minimum(counts, tail)
    strict = np.minimum(early, tail)
    return {
        "order": order,
        "counts": counts,
        "standard_start": ends - standard,
        "standard_len": standard,
        "strict_start": ends - counts + early - strict,
        "strict_len": strict,
    }


def _amount_report(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "amount_nan": int(np.isnan(values).sum()),
        "amount_negative": int((finite < 0).sum()),
        "amount_zero": int((finite == 0).sum()),
        "amount_min": float(finite.min()) if finite.size else None,
        "amount_max": float(finite.max()) if finite.size else None,
    }


def _align(store: EventStore, index: np.ndarray, groups: dict, data: dict):
    """Raw amounts for every event of `store`, plus its check against the replay."""

    count = len(store)
    offsets = np.asarray(store.offsets)
    lengths = np.diff(offsets)
    present = index >= 0
    safe = np.where(present, index, 0)
    complete = present & (groups["standard_len"][safe] == lengths)
    owner = np.repeat(np.arange(count), lengths)
    position = np.arange(store.num_events) - np.repeat(offsets[:-1], lengths)
    take = complete[owner]
    taken_owner = owner[take]
    rows = groups["order"][groups["standard_start"][safe][taken_owner] + position[take]]
    del owner, position
    same = (
        (data["timestamp"][rows] == np.asarray(store.timestamps)[take])
        & (data["type"][rows] == np.asarray(store.types)[take])
        & (data["features"][rows] == np.asarray(store.features)[take]).all(axis=1)
    )
    differs = np.zeros(count, dtype=bool)
    differs[taken_owner[~same]] = True
    mismatched = (present & ~complete) | differs
    amount = np.full(store.num_events, np.nan, dtype=np.float32)
    amount[take] = data["amount"][rows]
    report = {
        "users": count,
        "events": store.num_events,
        "present_fraction": float(present.mean()) if count else 0.0,
        "missing": int((~present).sum()),
        "mismatched": int(mismatched.sum()),
        "length_mismatched": int((present & ~complete).sum()),
        "matched": int((present & ~mismatched).sum()),
    }
    report.update(_amount_report(amount[take]))
    return amount, report


def _strict_consistency(
    store: EventStore,
    strict: EventStore,
    kept: np.ndarray,
    lengths: np.ndarray,
    present: np.ndarray,
    tail: int,
) -> tuple:
    """Count strict histories that contradict, or extend, their standard ones.

    A standard history's events before `STRICT_CUTOFF` are the client's latest
    pre-cutoff events, so they must end the strict history; when the standard
    history was not truncated (fewer than `tail` events) they must be all of it.
    A strict history longer than that part holds older events the 128 cap had
    removed from the standard store. Returns ``(violations, extended)``.
    """

    offsets = np.asarray(store.offsets)
    standard_lengths = np.diff(offsets)
    timestamps = np.asarray(store.timestamps)
    owner = np.repeat(np.arange(len(store)), standard_lengths)
    early = np.bincount(owner[timestamps < STRICT_CUTOFF], minlength=len(store))
    bad = present & (
        (early > lengths) | ((standard_lengths < tail) & (early != lengths))
    )
    pick = np.flatnonzero(~bad[kept])
    users = kept[pick]
    count = early[users]
    who = np.repeat(np.arange(pick.shape[0]), count)
    step = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    strict_rows = (np.asarray(strict.offsets)[1:][pick] - count)[who] + step
    standard_rows = offsets[:-1][users][who] + step
    same = (
        (strict.timestamps[strict_rows] == timestamps[standard_rows])
        & (strict.types[strict_rows] == np.asarray(store.types)[standard_rows])
        & (strict.features[strict_rows] == np.asarray(store.features)[standard_rows]).all(axis=1)
    )
    bad[users[who[~same]]] = True
    return int(bad.sum()), int((early[kept] < lengths[kept]).sum())


def _strict(store: EventStore, index: np.ndarray, groups: dict, data: dict, tail: int):
    """The strict-prospective copy of `store`, its raw amounts and its report."""

    present = index >= 0
    safe = np.where(present, index, 0)
    lengths = np.where(present, groups["strict_len"][safe], 0)
    keep = lengths >= 2
    kept = np.flatnonzero(keep)
    kept_lengths = lengths[kept]
    offsets = np.zeros(kept.shape[0] + 1, dtype=np.int64)
    np.cumsum(kept_lengths, out=offsets[1:])
    owner = np.repeat(np.arange(kept.shape[0]), kept_lengths)
    position = np.arange(offsets[-1]) - np.repeat(offsets[:-1], kept_lengths)
    rows = groups["order"][groups["strict_start"][safe[kept]][owner] + position]
    labels = np.asarray(store.labels)
    strict = EventStore(
        offsets=offsets,
        types=data["type"][rows],
        features=data["features"][rows],
        timestamps=data["timestamp"][rows],
        labels=labels[kept].astype(np.int8),
        user_ids=np.asarray(store.user_ids)[kept],
        vocab=store.vocab,
        observation_end=STRICT_CUTOFF,
    )
    amount = data["amount"][rows]
    dropped = present & ~keep
    violations, extended = _strict_consistency(store, strict, kept, lengths, present, tail)
    report = {
        "users": len(store),
        "kept": int(kept.shape[0]),
        "dropped": int(dropped.sum()),
        "dropped_positives": int((labels[dropped] == 1).sum()),
        "missing": int((~present).sum()),
        "kept_positives": int((labels[kept] == 1).sum()),
        "events": int(offsets[-1]),
        "mean_events": float(kept_lengths.mean()) if kept.size else 0.0,
        "median_events": float(np.median(kept_lengths)) if kept.size else 0.0,
        "consistency_violations": violations,
        "kept_with_older_events_recovered": extended,
    }
    report.update(_amount_report(amount))
    return strict, amount, report


def _targets(root: Path):
    """Per-client max of the four product targets over both label windows."""

    import polars as pl

    from eventfm.datasets.adapters import MBD_LABEL_MONTHS, _mbd_label_column, _read_parquet_dir

    table = _read_parquet_dir(root / "targets")
    columns = [_mbd_label_column(table.columns, index) for index in range(1, 5)]
    # Month parsing and null handling as in `_mbd_client_labels`.
    months = table["mon"].cast(pl.Utf8).str.slice(0, 10)
    windows = {"targets": MBD_LABEL_MONTHS, "targets_oct": MBD_LABEL_MONTHS[1:]}
    tables = {}
    for name, window in windows.items():
        aggregated = (
            table.filter(months.is_in(list(window)))
            .group_by("client_id")
            .agg([pl.col(column).max().fill_null(0) for column in columns])
        )
        ids = np.asarray([str(client) for client in aggregated["client_id"].to_list()])
        order = np.argsort(ids)
        values = aggregated.select(columns).to_numpy()[order].astype(np.int8)
        tables[name] = (ids[order], values)
    return tables, columns


def _store_targets(ids: np.ndarray, values: np.ndarray, user_ids) -> np.ndarray:
    index = _lookup(ids, user_ids)
    targets = np.full((index.shape[0], values.shape[1]), -1, dtype=np.int8)
    found = index >= 0
    targets[found] = values[index[found]]
    return targets


def _write_labelled_jsonl(path: Path, store: EventStore) -> None:
    """`build_mbd_pool._write_jsonl` plus the integer label.

    Keys keep `write_jsonl`'s sorted order (events, label, metadata, profile,
    user_id), so a benchmark split comes out byte-identical to the original.
    """

    vocab = store.vocab
    fields = vocab.feature_fields
    order = np.argsort(fields)
    sorted_fields = [fields[index] for index in order]
    values = [["<pad>", "<unk>"] + vocab.field_values[name] for name in fields]
    offsets = np.asarray(store.offsets)
    types = np.asarray(store.types)
    features = np.asarray(store.features)
    timestamps = np.asarray(store.timestamps)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for seq in range(len(store)):
            events = []
            for row in range(int(offsets[seq]), int(offsets[seq + 1])):
                feature_codes = features[row]
                described = ", ".join(
                    '"{}": "{}"'.format(name, values[column][int(feature_codes[column])])
                    for name, column in zip(sorted_fields, order)
                )
                events.append(
                    '{{"event_type": "{}", "features": {{{}}}, "timestamp": {}}}'.format(
                        vocab.event_types[int(types[row])],
                        described,
                        repr(float(timestamps[row])),
                    )
                )
            handle.write(
                '{{"events": [{}], "label": {}, "metadata": {{}}, "profile": {{}}, '
                '"user_id": {}}}\n'.format(
                    ", ".join(events),
                    int(store.labels[seq]),
                    json.dumps(str(store.user_ids[seq])),
                )
            )
    temporary.replace(path)


def _strict_jsonl(
    strict_stores: dict, reference: EventStore, meta: dict, spec, out_dir: Path
) -> dict:
    """Write and verify the strict JSONL splits and their ``meta.json``."""

    report = {}
    # The writer must reproduce the benchmark's own lines before it writes new ones.
    count = min(JSONL_CHECK_SEQUENCES, len(reference))
    with tempfile.TemporaryDirectory() as scratch:
        probe = Path(scratch) / "probe.jsonl"
        _write_labelled_jsonl(probe, reference.subset(np.arange(count)))
        source = Path(meta["splits"]["train"])
        with probe.open("r", encoding="utf-8") as mine, source.open(
            "r", encoding="utf-8"
        ) as theirs:
            differing = sum(line != theirs.readline() for line in mine)
    report["contract_lines_checked"] = count
    report["contract_lines_differing"] = differing
    if differing:
        raise RuntimeError(
            "The JSONL writer does not reproduce {} ({} of {} lines differ).".format(
                source, differing, count
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in SPLITS:
        paths[split] = out_dir / "{}.jsonl".format(split)
        _write_labelled_jsonl(paths[split], strict_stores[split])
        _log("wrote {}".format(paths[split]))

    stats = {}
    for split, store in strict_stores.items():
        lengths = np.diff(np.asarray(store.offsets)).astype(np.float64)
        stats[split] = {
            "num_sequences": len(store),
            "mean_events": float(lengths.mean()),
            "median_events": float(np.median(lengths)),
            "positive_rate": float(np.asarray(store.labels, dtype=np.float64).mean()),
        }
    payload = json.loads(json.dumps(meta))
    payload.update(
        name="mbd_strict",
        display_name=spec.display_name,
        label=spec.label,
        splits={split: str(path) for split, path in paths.items()},
        stats=stats,
        split_num_entities_kept={split: len(store) for split, store in strict_stores.items()},
        num_entities_kept=sum(len(store) for store in strict_stores.values()),
        split_used_positive_rate={split: stats[split]["positive_rate"] for split in SPLITS},
        used_positive_rate=stats["train"]["positive_rate"],
        split_sha256={split: _sha256(path) for split, path in paths.items()},
        history_cutoff=STRICT_CUTOFF,
    )
    meta_path = write_meta(out_dir, payload)
    with meta_path.open("r", encoding="utf-8") as handle:
        written = json.load(handle)
    DatasetMeta.from_json(written)
    if EtamVocab.from_meta(written).to_json() != reference.vocab.to_json():
        raise RuntimeError("The strict meta.json does not reproduce the benchmark vocabulary.")
    report["meta"] = str(meta_path)
    report["stats"] = stats

    round_trip = {}
    for split, store in strict_stores.items():
        count = min(JSONL_CHECK_SEQUENCES, len(store))
        back = store_from_jsonl(
            str(paths[split]), store.vocab, observation_end=STRICT_CUTOFF, max_sequences=count
        )
        head = store.subset(np.arange(count))
        if not all(
            np.array_equal(getattr(back, name), getattr(head, name)) for name in _STORE_ARRAYS
        ):
            raise RuntimeError("{} does not round-trip to the strict store.".format(paths[split]))
        round_trip[split] = count
    report["round_trip_sequences"] = round_trip

    # The benchmark's own readers: `JsonlEventDataset` (classification) and
    # `NextEventJsonlDataset` (TPP), on the head of each split.
    from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset

    readers = {}
    with tempfile.TemporaryDirectory() as scratch:
        for split, store in strict_stores.items():
            head_path = Path(scratch) / "{}.jsonl".format(split)
            with paths[split].open("r", encoding="utf-8") as source, head_path.open(
                "w", encoding="utf-8"
            ) as sink:
                for _, line in zip(range(READER_CHECK_SEQUENCES), source):
                    sink.write(line)
            sequences = JsonlEventDataset(str(head_path)).sequences
            examples = NextEventJsonlDataset(str(head_path))
            count = len(sequences)
            ok = (
                [len(sequence.events) for sequence in sequences]
                == np.diff(np.asarray(store.offsets))[:count].tolist()
                and [sequence.label for sequence in sequences]
                == np.asarray(store.labels)[:count].tolist()
                and [sequence.user_id for sequence in sequences]
                == np.asarray(store.user_ids)[:count].tolist()
                and all(sequence.events[-1].timestamp < STRICT_CUTOFF for sequence in sequences)
            )
            if not ok:
                raise RuntimeError("The benchmark readers disagree with {}.".format(paths[split]))
            readers[split] = {"sequences": count, "next_event_examples": len(examples)}
    report["readers"] = readers
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-clients", type=int, default=0, help="smoke run on this many clients")
    parser.add_argument("--result", default="", help="also write the summary here")
    parser.add_argument("--skip-jsonl", action="store_true")
    args = parser.parse_args()

    from eventfm.datasets.adapters import (
        MBD_FULL_DEFAULT_MAX_CLIENTS,
        MBD_HISTORY_CUTOFF,
        _mbd_client_labels,
        _mbd_full_root,
        _select_mbd_clients,
    )

    smoke = bool(args.max_clients)
    suffix = "_smoke" if smoke else ""
    config = DATASET_REGISTRY["mbd"].config
    tail = config.max_events_per_entity
    with (processed_dir("mbd") / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    vocab = EtamVocab.from_meta(meta)
    store_root = output_root() / "etam" / "stores" / "mbd"
    seconds = {}
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        seconds[name] = round(now - clock[0], 1)
        clock[0] = now
        _log("{} took {:.1f}s".format(name, seconds[name]))

    # Client selection exactly as `load_mbd` makes it.
    root = _mbd_full_root()
    trx_dir = next((path for path in [root / "ptls" / "trx", root / "trx"] if path.exists()), None)
    if trx_dir is None:
        raise FileNotFoundError("MBD transaction directory not found under {}".format(root))
    entity_labels, _ = _mbd_client_labels(root, 1)
    max_clients = args.max_clients or int(
        os.environ.get("KBC_MBD_MAX_CLIENTS", MBD_FULL_DEFAULT_MAX_CLIENTS)
    )
    clients = sorted(
        _select_mbd_clients(entity_labels, max_clients if max_clients > 0 else None, config.seed)
    )
    del entity_labels
    _log("scanning MBD transactions of {} clients".format(len(clients)))
    data = _scan(config, meta, vocab, clients, sorted(trx_dir.rglob("*.parquet")))
    keys = np.asarray(clients)
    del clients
    lap("load")
    rows_kept = int(data["timestamp"].shape[0])
    groups = _group(data.pop("client"), data["timestamp"], keys.shape[0], tail)
    clients_with_events = int((groups["counts"] > 0).sum())
    _log("{} rows over {} clients".format(rows_kept, clients_with_events))
    lap("group")

    # 1. Raw amounts aligned to every existing store.
    stores = {
        name: EventStore.load(str(store_root / name), mmap=True) for name in ("pool",) + SPLITS
    }
    for name, store in stores.items():
        if store.vocab.to_json() != vocab.to_json():
            raise RuntimeError("Store `{}` does not use the prepared vocabulary.".format(name))
    index = {
        name: _client_index(keys, groups["counts"], store.user_ids)
        for name, store in stores.items()
    }
    amounts, amount_reports = {}, {}
    for name, store in stores.items():
        amounts[name], amount_reports[name] = _align(store, index[name], groups, data)
        amount_reports[name]["written"] = False
        _log("{}: {}".format(name, json.dumps(amount_reports[name])))
    smoke_store = None
    if smoke:
        # The real stores are only reported on: the subsample misses most users.
        chosen = np.flatnonzero(index["train"] >= 0)[:SMOKE_TRAIN_USERS]
        smoke_store = stores["train"].subset(chosen)
        smoke_dir = smoke_store.save(str(store_root / "train_smoke"))
        amount, report = _align(smoke_store, index["train"][chosen], groups, data)
        _save_array(smoke_dir / "amount.npy", amount)
        amount_reports["train_smoke"] = dict(report, written=True)
        _log("train_smoke: {}".format(json.dumps(report)))
    else:
        failed = {
            name: {"missing": report["missing"], "mismatched": report["mismatched"]}
            for name, report in amount_reports.items()
            if report["missing"] or report["mismatched"]
        }
        if failed:
            raise RuntimeError("The raw replay disagrees with the stores: {}".format(failed))
        for name, amount in amounts.items():
            _save_array(store_root / name / "amount.npy", amount)
            amount_reports[name]["written"] = True
    del amounts
    stores.pop("pool")
    index.pop("pool")
    lap("amounts")

    # 2. Strict-prospective stores: same users and order, history < 2022-09-01.
    strict_stores, strict_reports = {}, {}
    for split in SPLITS:
        name = "{}_strict{}".format(split, suffix)
        strict, amount, report = _strict(stores[split], index[split], groups, data, tail)
        _log("{}: {}".format(name, json.dumps(report)))
        if report["consistency_violations"] and not smoke:
            raise RuntimeError(
                "{} strict histories contradict the {} store.".format(
                    report["consistency_violations"], split
                )
            )
        target = strict.save(str(store_root / name))
        _save_array(target / "amount.npy", amount)
        strict_stores[split] = strict
        strict_reports[name] = report
    del data, groups, index
    lap("strict")

    # 3. All four targets over both label windows.
    tables, target_columns = _targets(root)
    destinations = []
    for split in SPLITS:
        destinations.append((split, stores[split], None if smoke else store_root / split))
        name = "{}_strict{}".format(split, suffix)
        destinations.append((name, strict_stores[split], store_root / name))
    if smoke_store is not None:
        destinations.append(("train_smoke", smoke_store, store_root / "train_smoke"))
    target_reports = {}
    for name, store, directory in destinations:
        report = {"users": len(store), "written": directory is not None}
        for table_name, (ids, values) in tables.items():
            targets = _store_targets(ids, values, store.user_ids)
            found = targets >= 0
            report[table_name] = {
                "missing": int((~found[:, 0]).sum()),
                "positive_rate": [
                    float((targets[:, column] == 1).sum() / max(1, found[:, column].sum()))
                    for column in range(targets.shape[1])
                ],
            }
            if table_name == "targets":
                agreement = int((targets[:, 0] == np.asarray(store.labels)).sum())
                report["label_agreement"] = agreement
            if directory is not None:
                _save_array(directory / "{}.npy".format(table_name), targets)
        target_reports[name] = report
        _log("{}: {}".format(name, json.dumps(report)))
    disagreeing = {
        name: report["users"] - report["label_agreement"]
        for name, report in target_reports.items()
        if report["label_agreement"] != report["users"]
    }
    if disagreeing:
        raise RuntimeError(
            "targets.npy column 0 disagrees with the benchmark labels: {}".format(disagreeing)
        )
    lap("targets")

    # 4. Strict JSONL for the JSONL-based baselines.
    if smoke or args.skip_jsonl:
        jsonl_report = {"skipped": "smoke run" if smoke else "--skip-jsonl"}
    else:
        jsonl_report = _strict_jsonl(
            strict_stores,
            stores["train"],
            meta,
            DATASET_REGISTRY["mbd_strict"],
            processed_dir("mbd_strict"),
        )
    lap("jsonl")

    summary = {
        "mode": "smoke" if smoke else "full",
        "max_clients": args.max_clients,
        "clients_selected": int(keys.shape[0]),
        "clients_with_events": clients_with_events,
        "rows_kept": rows_kept,
        "history_cutoff": MBD_HISTORY_CUTOFF,
        "strict_cutoff": STRICT_CUTOFF,
        "target_columns": target_columns,
        "amount": amount_reports,
        "strict": strict_reports,
        "targets": target_reports,
        "jsonl": jsonl_report,
        "seconds": seconds,
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2, 2),
    }
    summary_path = store_root / "stage2_data_summary{}.json".format(suffix)
    for target in [summary_path] + ([Path(args.result)] if args.result else []):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
