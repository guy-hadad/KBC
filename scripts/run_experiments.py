#!/usr/bin/env python
"""Run or list a named paper experiment suite.

Examples:
    python scripts/run_experiments.py --list-suites
    python scripts/run_experiments.py --suite objective-ablation --list
    python scripts/run_experiments.py --suite paper-headline --array-index 0 --array-size 64
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.benchmark.experiments import (  # noqa: E402
    experiment_names,
    get_experiment,
)
from eventfm.benchmark.runner import enumerate_cells, run_cell  # noqa: E402
from eventfm.methods.base import TrainingSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=experiment_names())
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--array-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--results-dir")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--llm-base-model")
    args = parser.parse_args()

    if args.list_suites:
        for name in experiment_names():
            suite = get_experiment(name)
            print("{:<32} {}".format(name, suite.description))
        return
    if not args.suite:
        parser.error("--suite is required unless --list-suites is used")

    suite = get_experiment(args.suite)
    cells = enumerate_cells(
        suite.datasets,
        suite.tasks,
        suite.methods,
        suite.sample_sizes,
        suite.seeds,
        suite.regimes,
        suite.variants,
    )
    if args.list:
        for index, cell in enumerate(cells):
            print("{:5d}  {}".format(index, cell.key))
        print("\n{} cells — {}".format(len(cells), suite.description))
        return
    if args.array_index is not None:
        cells = cells[args.array_index :: args.array_size]

    results_dir = Path(args.results_dir) if args.results_dir else None
    training = TrainingSpec(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        max_events=args.max_events,
        use_cpu=args.cpu,
        bf16=args.bf16 and not args.cpu,
    )
    for cell in cells:
        extra = dict(suite.extra)
        extra.update(suite.variant_extra.get(cell.variant, {}))
        if args.llm_base_model:
            extra["llm_base_model"] = args.llm_base_model
        cell_training = replace(
            training,
            max_events=int(extra.pop("max_events", training.max_events)),
        )
        cell_hidden_size = int(extra.pop("hidden_size", args.hidden_size))
        cell_layers = int(extra.pop("num_hidden_layers", args.layers))
        result = run_cell(
            cell,
            training=cell_training,
            hidden_size=cell_hidden_size,
            num_hidden_layers=cell_layers,
            extra=extra,
            results_dir=results_dir,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "suite": suite.name,
                    "cell": cell.key,
                    "status": result.status,
                    "seconds": round(result.wall_seconds, 1),
                    "error": (result.error or "").splitlines()[0] if result.error else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
