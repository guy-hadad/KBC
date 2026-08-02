"""Run benchmark cells.

Single cell:
    python scripts/run_benchmark.py --dataset banksim --task tpp --method thp --samples 512

Whole grid on one process (slow; prefer the SLURM array):
    python scripts/run_benchmark.py --all

One index of the grid, which is how the SLURM array driver calls it:
    python scripts/run_benchmark.py --all --array-index 37 --array-size 200
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.benchmark.runner import enumerate_cells, run_cell  # noqa: E402
from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.methods import METHOD_REGISTRY, method_names  # noqa: E402
from eventfm.methods.base import TrainingSpec  # noqa: E402

DEFAULT_DATASETS = ["banksim", "paysim", "ibm_aml", "mbd_mini"]
DEFAULT_SAMPLE_SIZES = [64, 128, 256, 512, 1024, 2048]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--task", action="append", default=None, choices=["classification", "tpp"])
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--samples", action="append", type=int, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="run the full grid")
    parser.add_argument("--skip-gpu-only", action="store_true", help="exclude LLM methods")
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--array-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the grid and exit")

    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--llm-learning-rate", type=float, default=1e-4)
    parser.add_argument("--llm-epochs", type=float, default=4.0)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--tpp-max-prefixes", type=int, default=8)
    return parser


def training_spec_for(method: str, args) -> TrainingSpec:
    is_llm = METHOD_REGISTRY[method].family == "llm"
    return TrainingSpec(
        learning_rate=args.llm_learning_rate if is_llm else args.learning_rate,
        num_train_epochs=args.llm_epochs if is_llm else args.epochs,
        batch_size=max(4, args.batch_size // 4) if is_llm else args.batch_size,
        eval_batch_size=max(4, args.batch_size // 2) if is_llm else args.batch_size * 2,
        max_events=args.max_events,
        use_cpu=bool(args.cpu),
        bf16=bool(args.bf16) and not bool(args.cpu),
        dataloader_num_workers=0 if is_llm else 2,
    )


def main() -> None:
    args = build_parser().parse_args()

    datasets = args.dataset or DEFAULT_DATASETS
    tasks = args.task or ["classification", "tpp"]
    methods = args.method or method_names(include_gpu_only=not args.skip_gpu_only)
    if args.skip_gpu_only:
        methods = [name for name in methods if not METHOD_REGISTRY[name].requires_gpu]
    sample_sizes = args.samples or DEFAULT_SAMPLE_SIZES
    seeds = args.seed or [13]

    cells = enumerate_cells(datasets, tasks, methods, sample_sizes, seeds)
    if args.list:
        for index, cell in enumerate(cells):
            print("{:4d}  {}".format(index, cell.key))
        print("\n{} cells".format(len(cells)))
        return

    if args.array_index is not None:
        size = args.array_size or 1
        cells = cells[args.array_index :: size]

    results_dir = Path(args.results_dir) if args.results_dir else (output_root() / "results")
    for cell in cells:
        result = run_cell(
            cell,
            training=training_spec_for(cell.method, args),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            extra={"tpp_max_prefixes_per_user": args.tpp_max_prefixes},
            results_dir=results_dir,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "cell": cell.key,
                    "status": result.status,
                    "seconds": round(result.wall_seconds, 1),
                    "metrics": {
                        key: round(value, 4)
                        for key, value in sorted(result.metrics.items())
                        if value is not None
                    },
                    "error": (result.error or "").splitlines()[0] if result.error else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
