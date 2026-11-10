"""ETAM stores for the public transfer datasets (Synthea, Stack Overflow).

Synthea: train / validation / test from the prepared JSONL. Stack Overflow: its
``validation.jsonl`` is gone from disk (only its checksum is in meta.json), so
a seeded set of 334 training users (the official validation size) becomes the
validation store and is removed from the training store; the official test
split is untouched, so test prefixes match the benchmark's.

    python scripts/etam2_public_stores.py --result result.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.datasets.registry import processed_dir  # noqa: E402
from eventfm.etam.store import EtamVocab, store_from_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="")
    args = parser.parse_args()
    root = output_root() / "etam" / "stores"
    summary = {}
    for dataset in ("synthea", "stackoverflow"):
        meta = json.loads((processed_dir(dataset) / "meta.json").read_text())
        vocab = EtamVocab.from_meta(meta)
        stores = {}
        for split, path in meta["splits"].items():
            if Path(path).exists():
                stores[split] = store_from_jsonl(path, vocab, observation_end=None)
        if "validation" not in stores:
            train = stores["train"]
            size = int(meta.get("stats", {}).get("validation", {}).get("num_sequences", max(1, len(train) // 8)))
            order = np.random.default_rng(2026).permutation(len(train))
            stores["validation"] = train.subset(np.sort(order[:size]))
            stores["train"] = train.subset(np.sort(order[size:]))
            summary[dataset + "_validation_carved_from_train"] = size
        for split, store in stores.items():
            store.save(str(root / dataset / split))
            summary["{}_{}".format(dataset, split)] = {"sequences": len(store), "events": store.num_events}
        print(json.dumps(summary), flush=True)
    if args.result:
        Path(args.result).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
