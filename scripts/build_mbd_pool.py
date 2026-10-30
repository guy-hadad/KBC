"""Build the unlabeled MBD pretraining pool: every train-split client.

The prepared `mbd` benchmark keeps 30,665 of the 529,180 train-split clients
(negatives are downsampled to a 20 % positive rate). The other ~498k histories
are unused by every method in the benchmark, and they are exactly what a
foundation model should pretrain on: same hash split, so no validation or test
client can enter, and no labels are written.

The conversion replays `eventfm.datasets.tabular.build_sequences` for the train
split without its rebalancing step, and applies the *prepared* metadata's event
type levels, category levels and amount bucket edges, so the pool shares the
benchmark's token space. It writes

* ``<processed>/mbd/pool_train.jsonl`` (the shared JSONL contract, labels
  omitted) for the baselines' self-supervised stage, and
* ``<output>/etam/stores/mbd/pool`` (columnar ETAM store),

and checks that every one of the 30,665 benchmark training clients comes out
event-for-event identical to `train.jsonl`.

Imports polars: run on a cpu128/cpu256 node (see the SIGILL note in memory).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.datasets.registry import DATASET_REGISTRY, processed_dir  # noqa: E402
from eventfm.datasets.tabular import _assign_split, _group_rows  # noqa: E402
from eventfm.etam.store import EtamVocab, EventStore, store_from_jsonl  # noqa: E402


def _log(message: str) -> None:
    print("[{}] {}".format(time.strftime("%H:%M:%S"), message), flush=True)


def _level_codes(values: np.ndarray, levels, encode) -> np.ndarray:
    """Map an object column through `apply_category_levels` semantics to int16 codes.

    Factorised first so the per-row cost is one integer gather; the column has
    tens of millions of rows but only dozens of distinct values.
    """

    import pandas as pd

    keep = set(str(level) for level in levels)
    # No NA sentinel: a null would otherwise get code -1, i.e. the last level.
    codes, uniques = pd.factorize(values, use_na_sentinel=False)
    lookup = np.asarray(
        [encode(str(value) if str(value) in keep else "other") for value in uniques],
        dtype=np.int16,
    )
    return lookup[codes]


def _amount_codes(values: np.ndarray, edges, vocab: EtamVocab, field_name: str) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    buckets = np.digitize(finite, np.asarray(edges, dtype=np.float64), right=False)
    lookup = np.asarray(
        [vocab.encode_value(field_name, "b_{}".format(index)) for index in range(len(edges) + 1)],
        dtype=np.int16,
    )
    codes = lookup[buckets]
    codes[~np.isfinite(finite)] = vocab.encode_value(field_name, "na")
    return codes


def _write_jsonl(path: Path, store: EventStore) -> None:
    """Serialise the store in the JSONL contract without building event objects."""

    vocab = store.vocab
    fields = vocab.feature_fields
    order = np.argsort(fields)
    sorted_fields = [fields[index] for index in order]
    values = [["<pad>", "<unk>"] + vocab.field_values[name] for name in fields]
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for seq in range(len(store)):
            start, stop = int(store.offsets[seq]), int(store.offsets[seq + 1])
            events = []
            for row in range(start, stop):
                feature_codes = store.features[row]
                features = ", ".join(
                    '"{}": "{}"'.format(name, values[column][int(feature_codes[column])])
                    for name, column in zip(sorted_fields, order)
                )
                events.append(
                    '{{"event_type": "{}", "features": {{{}}}, "timestamp": {}}}'.format(
                        vocab.event_types[int(store.types[row])],
                        features,
                        repr(float(store.timestamps[row])),
                    )
                )
            handle.write(
                '{{"events": [{}], "metadata": {{}}, "profile": {{}}, "user_id": {}}}\n'.format(
                    ", ".join(events), json.dumps(str(store.user_ids[seq]))
                )
            )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-jsonl", action="store_true")
    parser.add_argument("--max-clients", type=int, default=0, help="smoke only")
    parser.add_argument("--result", default="", help="also write the summary here")
    args = parser.parse_args()

    from eventfm.datasets.adapters import MBD_HISTORY_CUTOFF, load_mbd

    spec = DATASET_REGISTRY["mbd"]
    config = spec.config
    meta_path = processed_dir("mbd") / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    vocab = EtamVocab.from_meta(meta)

    _log("loading MBD transactions (all labelled clients)")
    table = load_mbd(config, max_clients=args.max_clients or None)
    keep_rows = np.asarray(table.timestamps, dtype=np.float64) <= MBD_HISTORY_CUTOFF
    _log("rows {} (<= cutoff: {})".format(len(table), int(keep_rows.sum())))

    entity_ids = np.asarray([str(value) for value in table.entity_ids], dtype=object)
    timestamps = np.asarray(table.timestamps, dtype=np.float64)
    if not keep_rows.all():
        entity_ids, timestamps = entity_ids[keep_rows], timestamps[keep_rows]
    row_map = np.flatnonzero(keep_rows)

    _log("encoding columns")
    types = _level_codes(
        np.asarray(table.event_types)[row_map], meta["event_type_levels"], vocab.encode_type
    )
    columns = []
    for field_name in vocab.feature_fields:
        if field_name in meta.get("numeric_bucket_edges", {}):
            column = _amount_codes(
                np.asarray(table.numeric[field_name])[row_map],
                meta["numeric_bucket_edges"][field_name],
                vocab,
                field_name,
            )
        else:
            column = _level_codes(
                np.asarray(table.categorical[field_name])[row_map],
                meta["categorical_levels"][field_name],
                lambda value, name=field_name: vocab.encode_value(name, value),
            )
        columns.append(column)
    features = np.stack(columns, axis=1)
    labelled = table.entity_labels or {}
    del table, columns

    _log("grouping rows by client")
    groups = _group_rows(entity_ids, timestamps)
    selected_ids, selected_rows = [], []
    for entity_id, rows in groups.items():
        if rows.shape[0] < config.min_events_per_entity or entity_id not in labelled:
            continue
        if _assign_split(entity_id, config.validation_fraction, config.test_fraction) != "train":
            continue
        selected_ids.append(entity_id)
        selected_rows.append(rows[-config.max_events_per_entity :])
    del groups
    expected = meta.get("split_num_entities_total", {}).get("train")
    _log("train-split clients: {} (meta says {})".format(len(selected_ids), expected))
    if expected is not None and not args.max_clients and len(selected_ids) != int(expected):
        raise RuntimeError("Pool size disagrees with the prepared benchmark's train split.")

    lengths = np.asarray([rows.shape[0] for rows in selected_rows], dtype=np.int64)
    offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    rows = np.concatenate(selected_rows)
    del selected_rows
    store = EventStore(
        offsets=offsets,
        types=types[rows],
        features=features[rows],
        timestamps=timestamps[rows],
        # Labels are deliberately withheld: the pool is for self-supervision.
        labels=np.full(lengths.shape[0], -1, dtype=np.int8),
        user_ids=np.asarray(selected_ids, dtype=np.str_),
        vocab=vocab,
        observation_end=float(MBD_HISTORY_CUTOFF),
    )
    _log(
        "pool: {} clients, {} events, mean {:.1f}".format(
            len(store), store.num_events, store.num_events / max(1, len(store))
        )
    )

    _log("checking the 30,665 benchmark training clients against train.jsonl")
    benchmark = store_from_jsonl(meta["splits"]["train"], vocab, observation_end=None)
    position = {user: index for index, user in enumerate(store.user_ids.tolist())}
    mismatched = 0
    missing = 0
    for seq, user in enumerate(benchmark.user_ids.tolist()):
        if user not in position:
            missing += 1
            continue
        a, b = benchmark.sequence_slice(seq), store.sequence_slice(position[user])
        same = (
            np.array_equal(benchmark.types[a], store.types[b])
            and np.array_equal(benchmark.features[a], store.features[b])
            and np.array_equal(benchmark.timestamps[a], store.timestamps[b])
        )
        mismatched += int(not same)
    _log("benchmark clients missing from pool: {}, mismatched: {}".format(missing, mismatched))
    if not args.max_clients and (missing or mismatched):
        raise RuntimeError("Pool conversion does not reproduce the benchmark's train histories.")

    store_name = "pool_smoke" if args.max_clients else "pool"
    store_dir = output_root() / "etam" / "stores" / "mbd" / store_name
    store.save(str(store_dir))
    _log("saved store to {}".format(store_dir))
    if not args.skip_jsonl and not args.max_clients:
        target = processed_dir("mbd") / "pool_train.jsonl"
        _write_jsonl(target, store)
        _log("wrote {}".format(target))
    summary = {
        "clients": len(store),
        "events": store.num_events,
        "benchmark_clients_missing": missing,
        "benchmark_clients_mismatched": mismatched,
        "mean_events": store.num_events / max(1, len(store)),
        "observation_end": MBD_HISTORY_CUTOFF,
        "max_clients": args.max_clients,
    }
    for target in [store_dir / "pool_summary.json"] + ([Path(args.result)] if args.result else []):
        with Path(target).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    _log(json.dumps(summary))


if __name__ == "__main__":
    main()
