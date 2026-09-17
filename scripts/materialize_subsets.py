#!/usr/bin/env python
"""Pre-build every training subset a suite needs, reading each split once.

Run before submitting an array. Workers then only read the subset cache, which
removes both the per-cell cost of parsing a large training split and the
concurrent-builder race that `_publish_once` guards against.

    python scripts/materialize_subsets.py --suite scale-datasets
    python scripts/materialize_subsets.py --suite scale-datasets --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.benchmark.experiments import (  # noqa: E402
    experiment_names,
    get_experiment,
    resolve_sample_sizes,
)
from eventfm.benchmark.runner import _subset_path, materialize_subsets  # noqa: E402
from eventfm.datasets.registry import load_dataset_meta  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", required=True, choices=experiment_names())
    parser.add_argument("--dry-run", action="store_true", help="list what is missing, write nothing")
    args = parser.parse_args()

    wanted = {}
    for name in args.suite:
        suite = get_experiment(name)
        sizes = resolve_sample_sizes(suite)
        for dataset in suite.datasets:
            grid = sizes[dataset] if isinstance(sizes, dict) else sizes
            entry = wanted.setdefault(dataset, {"sizes": set(), "seeds": set()})
            entry["sizes"].update(int(value) for value in grid)
            entry["seeds"].update(int(value) for value in suite.seeds)

    for dataset, entry in sorted(wanted.items()):
        meta = load_dataset_meta(dataset)
        sizes = sorted(entry["sizes"])
        seeds = sorted(entry["seeds"])
        missing = [
            (size, seed)
            for size in sizes
            for seed in seeds
            if not _subset_path(meta, size, seed).exists()
        ]
        print(
            "[subsets] {}: {} sizes x {} seeds, {} missing".format(
                dataset, len(sizes), len(seeds), len(missing)
            ),
            flush=True,
        )
        if args.dry_run or not missing:
            continue
        written = materialize_subsets(meta, sizes, seeds)
        print(json.dumps({dataset: written}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
