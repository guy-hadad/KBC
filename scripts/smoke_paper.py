#!/usr/bin/env python
"""Run representative one-step gates before submitting the paper suites."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.benchmark.runner import BenchmarkCell, run_cell  # noqa: E402
from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.methods.base import TrainingSpec  # noqa: E402


def _run(cell, mode: str, extra=None):
    settings = dict(extra or {})
    settings.setdefault("pretrain_epochs", 1)
    settings.setdefault("pretrain_max_steps", 1)
    settings.setdefault("contrastive_epochs", 1)
    settings.setdefault("tpp_max_prefixes_per_user", 1)
    settings.setdefault("gem_max_eval_examples", 8)
    settings.setdefault("gem_generation_batch_size", 2)
    result = run_cell(
        cell,
        training=TrainingSpec(
            num_train_epochs=1,
            max_steps=1,
            batch_size=4,
            eval_batch_size=16,
            max_events=12,
            use_cpu=mode == "cpu",
            bf16=mode == "gpu",
            dataloader_num_workers=0,
        ),
        hidden_size=32,
        num_hidden_layers=1,
        extra=settings,
        results_dir=output_root() / "paper_smoke" / mode,
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "cell": cell.key,
                "status": result.status,
                "error": result.error,
                "metrics": result.metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if result.status != "ok":
        raise RuntimeError("Smoke cell failed: {}".format(cell.key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()

    if args.mode == "cpu":
        cases = [
            (BenchmarkCell("banksim", "classification", "engineered-gbdt", 32), {}),
            (BenchmarkCell("banksim", "tpp", "nhp", 16), {}),
            (
                BenchmarkCell("banksim", "classification", "autoencoder", 16, 13, "peft"),
                {},
            ),
            (
                BenchmarkCell(
                    "banksim", "classification", "transaction-mlm", 16, 13, "frozen"
                ),
                {
                    "pretrain_datasets": "leave-one-out-primary",
                    "pretrain_sample_size_per_dataset": 2,
                },
            ),
            (BenchmarkCell("banksim_chrono", "tpp", "cotic", 16), {}),
            # One cell per scale benchmark, so a broken adapter or a missing
            # prepared split fails the gate rather than the whole array.
            (BenchmarkCell("mbd", "classification", "engineered-gbdt", 32), {}),
            (BenchmarkCell("synthea", "classification", "pragma", 16), {}),
            (BenchmarkCell("synthea", "tpp", "ntpp-gru", 16), {}),
            (BenchmarkCell("amazon_beauty", "classification", "pragma", 16), {}),
            (BenchmarkCell("amazon_beauty", "tpp", "ntpp-gru", 16), {}),
        ]
    else:
        cases = [
            (
                BenchmarkCell(
                    "stackoverflow", "tpp", "gem-time-byte-f32", 8, 13, "full", "135m"
                ),
                {"llm_base_model": "HuggingFaceTB/SmolLM2-135M"},
            ),
            (
                BenchmarkCell(
                    "stackoverflow", "tpp", "gem-time-byte-f32", 8, 13, "full", "360m"
                ),
                {"llm_base_model": "HuggingFaceTB/SmolLM2-360M"},
            ),
        ]

    for cell, extra in cases:
        _run(cell, args.mode, extra)


if __name__ == "__main__":
    main()
