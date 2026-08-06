#!/usr/bin/env python
"""Measure interval distributions and tokenizer distortion without training an LLM."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.data.io import load_jsonl_sequences  # noqa: E402
from eventfm.data.temporal_tokenizers import (  # noqa: E402
    build_temporal_tokenizer,
    temporal_tokenizer_names,
)
from eventfm.datasets.registry import PRIMARY_DATASETS, load_dataset_meta  # noqa: E402


def _pairs(sequences):
    rows = []
    for sequence in sequences:
        for previous, following in zip(sequence.events[:-1], sequence.events[1:]):
            rows.append(
                (
                    float(following.timestamp),
                    float(previous.timestamp),
                    max(0.0, float(following.timestamp - previous.timestamp)),
                )
            )
    return rows


def _entropy(values, bins=128):
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / max(1, counts.sum())
    return float(-(probabilities * np.log(probabilities)).sum())


def _distribution(rows):
    delta = np.asarray([row[2] for row in rows], dtype=np.float64)
    log_delta = np.log1p(delta)
    return {
        "count": int(delta.size),
        "zero_rate": float((delta == 0).mean()),
        "mean_seconds": float(delta.mean()),
        "std_seconds": float(delta.std()),
        "p10_seconds": float(np.quantile(delta, 0.10)),
        "p50_seconds": float(np.quantile(delta, 0.50)),
        "p90_seconds": float(np.quantile(delta, 0.90)),
        "p99_seconds": float(np.quantile(delta, 0.99)),
        "linear_histogram_entropy": _entropy(delta),
        "log_histogram_entropy": _entropy(log_delta),
    }


def _evaluate_tokenizer(name, train_rows, validation_rows):
    tokenizer = build_temporal_tokenizer(name)
    train_values = [row[0] if tokenizer.uses_absolute_time else row[2] for row in train_rows]
    tokenizer.fit(train_values)
    residuals = []
    log_residuals = []
    token_counts = []
    invalid = 0
    for timestamp, previous, true_delta in validation_rows:
        value = timestamp if tokenizer.uses_absolute_time else true_delta
        try:
            tokens = tokenizer.encode(value)
            reconstructed = tokenizer.decode(tokens)
            predicted_delta = (
                max(0.0, reconstructed - previous)
                if tokenizer.uses_absolute_time
                else max(0.0, reconstructed)
            )
            residuals.append(predicted_delta - true_delta)
            log_residuals.append(math.log1p(predicted_delta) - math.log1p(true_delta))
            token_counts.append(len(tokens))
        except (ValueError, OverflowError):
            invalid += 1
    residual = np.asarray(residuals, dtype=np.float64)
    log_residual = np.asarray(log_residuals, dtype=np.float64)
    return {
        "rmse_seconds": float(np.sqrt(np.mean(residual**2))),
        "mae_seconds": float(np.mean(np.abs(residual))),
        "log_rmse": float(np.sqrt(np.mean(log_residual**2))),
        "log_mae": float(np.mean(np.abs(log_residual))),
        "tokens_per_time": float(np.mean(token_counts)),
        "invalid_rate": float(invalid / max(1, len(validation_rows))),
        "temporal_vocabulary_size": len(tokenizer.vocabulary()),
    }


def _markdown(results):
    lines = [
        "# Temporal distribution and tokenizer audit",
        "",
        "All tokenizer parameters were fitted on the training split and scored on validation.",
        "This table measures scalar reconstruction distortion before any language-model effects.",
        "",
    ]
    for dataset, payload in results.items():
        distribution = payload["distribution"]
        lines += [
            "## {}".format(dataset),
            "",
            "Intervals: {count:,}; zero rate: {zero_rate:.3f}; median: "
            "{p50_seconds:.1f}s; p90: {p90_seconds:.1f}s; p99: {p99_seconds:.1f}s.".format(
                **distribution
            ),
            "",
            "| Tokenizer | log RMSE | log MAE | tokens/time | vocabulary | invalid |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, metrics in sorted(
            payload["tokenizers"].items(), key=lambda item: item[1]["log_rmse"]
        ):
            lines.append(
                "| `{}` | {:.4f} | {:.4f} | {:.2f} | {} | {:.3f} |".format(
                    name,
                    metrics["log_rmse"],
                    metrics["log_mae"],
                    metrics["tokens_per_time"],
                    metrics["temporal_vocabulary_size"],
                    metrics["invalid_rate"],
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(PRIMARY_DATASETS))
    parser.add_argument("--output-dir", default="outputs/temporal_tokenizers")
    parser.add_argument("--max-values", type=int, default=100000)
    args = parser.parse_args()

    results = {}
    for name in args.datasets:
        meta = load_dataset_meta(name)
        train_rows = _pairs(load_jsonl_sequences(meta.splits["train"]))
        validation_rows = _pairs(load_jsonl_sequences(meta.splits["validation"]))
        rng = np.random.default_rng(13)
        if len(train_rows) > args.max_values:
            selected = rng.choice(len(train_rows), args.max_values, replace=False)
            train_rows = [train_rows[int(index)] for index in selected]
        if len(validation_rows) > args.max_values:
            selected = rng.choice(len(validation_rows), args.max_values, replace=False)
            validation_rows = [validation_rows[int(index)] for index in selected]
        results[name] = {
            "distribution": _distribution(train_rows),
            "tokenizers": {
                tokenizer_name: _evaluate_tokenizer(
                    tokenizer_name, train_rows, validation_rows
                )
                for tokenizer_name in temporal_tokenizer_names()
            },
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "results.md").write_text(_markdown(results), encoding="utf-8")
    print(output_dir / "results.md")


if __name__ == "__main__":
    main()

