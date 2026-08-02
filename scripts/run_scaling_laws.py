"""Run sample-scaling sweeps for downstream event tasks.

The defaults are intentionally CPU-friendly. They are meant to verify the
research harness and produce first-look curves, not to claim a converged law.
Increase `scaling.neural_max_steps`, sample sizes, and model size for serious
experiments.
"""

import csv
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from eventfm.baselines.classic import run_classification_baseline, run_tpp_baseline
from eventfm.data.collator import PragmaDataCollator
from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset
from eventfm.data.schema import EventSequence
from eventfm.data.synthetic import generate_synthetic_sequences, write_jsonl
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary
from eventfm.models import (
    PragmaConfig,
    PragmaForNextEventPrediction,
    PragmaForSequenceClassification,
)
from eventfm.training.metrics import classification_metrics, tpp_metrics
from eventfm.training.utils import prepare_wandb, seed_everything

MetricRow = Dict[str, object]


def _as_int_list(values: Iterable[object]) -> List[int]:
    parsed = sorted({int(value) for value in values})
    if not parsed or min(parsed) <= 0:
        raise ValueError("scaling.sample_sizes must contain positive integers.")
    return parsed


def _model_config(cfg: DictConfig, vocab: EventVocabulary, task_name: str) -> PragmaConfig:
    return PragmaConfig(
        vocab_size=len(vocab),
        hidden_size=int(cfg.model.hidden_size),
        intermediate_size=int(cfg.model.intermediate_size),
        num_attention_heads=int(cfg.model.num_attention_heads),
        profile_num_hidden_layers=int(cfg.model.profile_num_hidden_layers),
        event_num_hidden_layers=int(cfg.model.event_num_hidden_layers),
        history_num_hidden_layers=int(cfg.model.history_num_hidden_layers),
        max_features_per_event=int(cfg.data.max_features_per_event),
        feature_position_vocab_size=int(cfg.model.feature_position_vocab_size),
        dropout=float(cfg.model.dropout),
        classifier_dropout=float(cfg.model.classifier_dropout),
        pad_token_id=vocab.pad_token_id,
        mask_token_id=vocab.mask_token_id,
        unk_token_id=vocab.unk_token_id,
        usr_token_id=vocab.usr_token_id,
        evt_token_id=vocab.evt_token_id,
        num_event_types=int(cfg.data.num_event_types),
        num_labels=2,
        tpp_loss_weight=float(cfg.scaling.tpp_loss_weight) if task_name == "tpp" else 1.0,
        time_encoding_frequencies=int(cfg.model.time_encoding_frequencies),
    )


def _balanced_label_prefix(sequences: List[EventSequence]) -> List[EventSequence]:
    buckets: Dict[int, List[EventSequence]] = {0: [], 1: []}
    for sequence in sequences:
        buckets[int(sequence.label or 0)].append(sequence)
    ordered: List[EventSequence] = []
    while buckets[0] or buckets[1]:
        for label in (0, 1):
            if buckets[label]:
                ordered.append(buckets[label].pop(0))
    return ordered


def _training_args(
    cfg: DictConfig,
    output_dir: Path,
    label_names: Optional[List[str]] = None,
) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(cfg.scaling.neural_num_train_epochs),
        "max_steps": int(cfg.scaling.neural_max_steps),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
        "warmup_ratio": float(cfg.training.warmup_ratio),
        "optim": str(cfg.training.optim),
        "per_device_train_batch_size": int(cfg.scaling.neural_batch_size),
        "per_device_eval_batch_size": int(cfg.scaling.neural_batch_size),
        "gradient_accumulation_steps": 1,
        "logging_strategy": "no",
        "eval_strategy": "no",
        "evaluation_strategy": "no",
        "save_strategy": "steps" if bool(cfg.scaling.save_models) else "no",
        "save_steps": int(cfg.training.save_steps),
        "report_to": ["wandb"] if bool(cfg.logging.wandb.enabled) else [],
        "run_name": "{}_scaling_{}".format(cfg.project_name, output_dir.name),
        "remove_unused_columns": False,
        "fp16": False,
        "bf16": False,
        "use_cpu": bool(cfg.training.use_cpu),
        "disable_tqdm": True,
        "dataloader_num_workers": int(cfg.training.dataloader_num_workers),
        "label_names": label_names,
        "do_train": True,
        "do_eval": True,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" not in signature.parameters:
        kwargs.pop("eval_strategy")
    if "evaluation_strategy" not in signature.parameters:
        kwargs.pop("evaluation_strategy")
    if "label_names" not in signature.parameters:
        kwargs.pop("label_names")
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**kwargs)


def _write_splits(cfg: DictConfig, output_dir: Path, sample_sizes: List[int]) -> Dict[str, object]:
    total_users = max(sample_sizes) + int(cfg.scaling.validation_users) + int(cfg.scaling.test_users)
    sequences = generate_synthetic_sequences(
        num_users=total_users,
        num_event_types=int(cfg.data.num_event_types),
        min_events=int(cfg.data.min_events),
        max_events=int(cfg.data.max_events),
        seed=int(cfg.seed),
    )
    rng = np.random.default_rng(int(cfg.seed))
    order = rng.permutation(len(sequences))
    sequences = [sequences[int(index)] for index in order]

    max_train = max(sample_sizes)
    validation_end = max_train + int(cfg.scaling.validation_users)
    train_pool = sequences[:max_train]
    validation = sequences[max_train:validation_end]
    test = sequences[validation_end:]
    train_pool = _balanced_label_prefix(train_pool)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_path = data_dir / "validation.jsonl"
    test_path = data_dir / "test.jsonl"
    write_jsonl(str(validation_path), validation)
    write_jsonl(str(test_path), test)

    train_paths = {}
    for sample_size in sample_sizes:
        path = data_dir / "train_{}.jsonl".format(sample_size)
        write_jsonl(str(path), train_pool[:sample_size])
        train_paths[sample_size] = str(path)

    vocab = EventVocabulary.synthetic(num_event_types=int(cfg.data.num_event_types))
    vocab.save(str(data_dir / "vocab.json"))
    return {
        "train_paths": train_paths,
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "vocab": vocab,
    }


def _metric_value(metrics: Dict[str, float], key: str) -> Optional[float]:
    for candidate in (key, "eval_{}".format(key), "test_{}".format(key)):
        if candidate in metrics:
            return float(metrics[candidate])
    return None


def _row(
    sample_size: int,
    task: str,
    method: str,
    metrics: Dict[str, float],
    split: str = "test",
) -> MetricRow:
    return {
        "samples": sample_size,
        "task": task,
        "method": method,
        "split": split,
        "accuracy": _metric_value(metrics, "accuracy"),
        "auc": _metric_value(metrics, "auc"),
        "next_type_accuracy": _metric_value(metrics, "next_type_accuracy"),
        "delta_log_rmse": _metric_value(metrics, "delta_log_rmse"),
    }


def _run_neural_classification(
    cfg: DictConfig,
    vocab: EventVocabulary,
    tokenizer: EventTokenizer,
    train_path: str,
    test_path: str,
    output_dir: Path,
) -> Dict[str, float]:
    model = PragmaForSequenceClassification(_model_config(cfg, vocab, "classification"))
    trainer = Trainer(
        model=model,
        args=_training_args(cfg, output_dir),
        train_dataset=JsonlEventDataset(train_path),
        eval_dataset=JsonlEventDataset(test_path),
        data_collator=PragmaDataCollator(
            tokenizer=tokenizer,
            task="classification",
            max_events=int(cfg.data.max_events_per_sequence),
        ),
        compute_metrics=classification_metrics,
    )
    trainer.train()
    return trainer.evaluate()


def _run_neural_tpp(
    cfg: DictConfig,
    vocab: EventVocabulary,
    tokenizer: EventTokenizer,
    train_path: str,
    test_path: str,
    output_dir: Path,
) -> Dict[str, float]:
    model = PragmaForNextEventPrediction(_model_config(cfg, vocab, "tpp"))
    max_prefixes = cfg.scaling.get("tpp_max_prefixes_per_user")
    trainer = Trainer(
        model=model,
        args=_training_args(cfg, output_dir, label_names=["next_event_type_labels", "next_delta_log"]),
        train_dataset=NextEventJsonlDataset(
            train_path,
            min_prefix_events=int(cfg.scaling.tpp_min_prefix_events),
            max_prefixes_per_user=None if max_prefixes is None else int(max_prefixes),
            seed=int(cfg.seed),
        ),
        eval_dataset=NextEventJsonlDataset(
            test_path,
            min_prefix_events=int(cfg.scaling.tpp_min_prefix_events),
            max_prefixes_per_user=None if max_prefixes is None else int(max_prefixes),
            seed=int(cfg.seed) + 1,
        ),
        data_collator=PragmaDataCollator(
            tokenizer=tokenizer,
            task="tpp",
            max_events=int(cfg.data.max_events_per_sequence),
        ),
        compute_metrics=tpp_metrics,
    )
    trainer.train()
    return trainer.evaluate()


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return "{:.4f}".format(value)
    return str(value)


def _write_csv(path: Path, rows: List[MetricRow]) -> None:
    fields = ["samples", "task", "method", "split", "accuracy", "next_type_accuracy", "delta_log_rmse"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[MetricRow], plot_paths: List[Path]) -> None:
    lines = [
        "# Scaling Law Sweep",
        "",
        "CPU-friendly sample-scaling sweep on the synthetic downstream tasks.",
        "",
        "The x-axis in the plots is the number of labeled training users. The default run is CPU-friendly and designed to show directional sample-scaling behavior; increase epochs, sample sizes, and model scale for research-grade curves.",
        "",
        "## Classification",
        "",
    ]
    cls_fields = ["samples", "method", "accuracy"]
    lines.append("| {} |".format(" | ".join(cls_fields)))
    lines.append("| {} |".format(" | ".join(["---"] * len(cls_fields))))
    for row in rows:
        if row["task"] == "classification":
            lines.append("| {} |".format(" | ".join(_fmt(row.get(field)) for field in cls_fields)))

    lines.extend(["", "## Temporal Point Process", ""])
    tpp_fields = ["samples", "method", "next_type_accuracy", "delta_log_rmse"]
    lines.append("| {} |".format(" | ".join(tpp_fields)))
    lines.append("| {} |".format(" | ".join(["---"] * len(tpp_fields))))
    for row in rows:
        if row["task"] == "tpp":
            lines.append("| {} |".format(" | ".join(_fmt(row.get(field)) for field in tpp_fields)))

    lines.extend(["", "## Plots", ""])
    for plot_path in plot_paths:
        lines.append("- [{}]({})".format(plot_path.name, plot_path.name))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _series(rows: List[MetricRow], task: str, metric: str) -> Dict[str, List[tuple]]:
    grouped: Dict[str, List[tuple]] = {}
    for row in rows:
        if row["task"] != task or row.get(metric) is None:
            continue
        grouped.setdefault(str(row["method"]), []).append((int(row["samples"]), float(row[metric])))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def _nice_bounds(values: List[float], lower_is_zero: bool = True) -> tuple:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        pad = max(0.05, abs(high) * 0.1)
        return max(0.0, low - pad) if lower_is_zero else low - pad, high + pad
    pad = 0.08 * (high - low)
    return max(0.0, low - pad) if lower_is_zero else low - pad, high + pad


def _write_svg_plot(
    path: Path,
    title: str,
    y_label: str,
    grouped: Dict[str, List[tuple]],
    lower_is_zero: bool = True,
) -> None:
    width, height = 840, 520
    left, right, top, bottom = 82, 32, 58, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    all_points = [point for values in grouped.values() for point in values]
    if not all_points:
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    min_log, max_log = math.log2(min_x), math.log2(max_x)
    y_min, y_max = _nice_bounds(ys, lower_is_zero=lower_is_zero)
    if abs(max_log - min_log) < 1e-9:
        max_log = min_log + 1.0

    def sx(sample: int) -> float:
        return left + (math.log2(sample) - min_log) / (max_log - min_log) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    y_ticks = np.linspace(y_min, y_max, 5)
    x_ticks = sorted({point[0] for point in all_points})
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{}</text>'.format(
            left, title
        ),
        '<text x="{}" y="{}" font-family="Arial" font-size="13" fill="#4b5563">training users, log2 scale</text>'.format(
            left + plot_w / 2 - 78, height - 24
        ),
        '<text x="20" y="{}" transform="rotate(-90 20,{})" font-family="Arial" font-size="13" fill="#4b5563">{}</text>'.format(
            top + plot_h / 2 + 60, top + plot_h / 2 + 60, y_label
        ),
    ]
    for tick in y_ticks:
        y = sy(float(tick))
        elements.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#e5e7eb" stroke-width="1"/>'.format(
                left, y, left + plot_w, y
            )
        )
        elements.append(
            '<text x="{}" y="{:.2f}" font-family="Arial" font-size="12" text-anchor="end" fill="#6b7280">{:.3g}</text>'.format(
                left - 10, y + 4, tick
            )
        )
    elements.append(
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#9ca3af" stroke-width="1"/>'.format(
            left, top, plot_w, plot_h
        )
    )
    for sample in x_ticks:
        x = sx(sample)
        elements.append(
            '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="#d1d5db" stroke-width="1"/>'.format(
                x, top + plot_h, x, top + plot_h + 6
            )
        )
        elements.append(
            '<text x="{:.2f}" y="{}" font-family="Arial" font-size="12" text-anchor="middle" fill="#374151">{}</text>'.format(
                x, top + plot_h + 23, sample
            )
        )

    legend_x = left + plot_w - 190
    legend_y = top + 8
    for idx, (method, values) in enumerate(sorted(grouped.items())):
        color = palette[idx % len(palette)]
        points = ["{:.2f},{:.2f}".format(sx(sample), sy(value)) for sample, value in values]
        elements.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'.format(
                " ".join(points), color
            )
        )
        for sample, value in values:
            elements.append(
                '<circle cx="{:.2f}" cy="{:.2f}" r="4.5" fill="{}" stroke="#ffffff" stroke-width="1.5"/>'.format(
                    sx(sample), sy(value), color
                )
            )
        y = legend_y + idx * 22
        elements.append(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="3"/>'.format(
                legend_x, y, legend_x + 24, y, color
            )
        )
        elements.append(
            '<text x="{}" y="{}" font-family="Arial" font-size="13" fill="#111827">{}</text>'.format(
                legend_x + 32, y + 4, method
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_plots(output_dir: Path, rows: List[MetricRow]) -> List[Path]:
    plot_specs = [
        ("classification", "accuracy", "Classification Accuracy", "accuracy", True),
        ("tpp", "next_type_accuracy", "TPP Next-Type Accuracy", "accuracy", True),
        ("tpp", "delta_log_rmse", "TPP Log-Time RMSE", "RMSE (lower is better)", False),
    ]
    plot_paths = []
    for task, metric, title, y_label, lower_is_zero in plot_specs:
        grouped = _series(rows, task, metric)
        if not grouped:
            continue
        path = output_dir / "{}_{}.svg".format(task, metric)
        _write_svg_plot(path, title, y_label, grouped, lower_is_zero=lower_is_zero)
        plot_paths.append(path)
    return plot_paths


def _publish_report(
    publish_dir: Optional[str],
    csv_path: Path,
    markdown_path: Path,
    json_path: Path,
    plot_paths: List[Path],
) -> Optional[Path]:
    if not publish_dir:
        return None
    target = Path(to_absolute_path(str(publish_dir)))
    target.mkdir(parents=True, exist_ok=True)
    for source in [csv_path, markdown_path, json_path] + plot_paths:
        shutil.copy2(source, target / source.name)
    return target


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    prepare_wandb(cfg)
    sample_sizes = _as_int_list(cfg.scaling.sample_sizes)
    output_dir = Path(to_absolute_path(cfg.scaling.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    split_info = _write_splits(cfg, output_dir, sample_sizes)
    vocab = split_info["vocab"]
    tokenizer = EventTokenizer(vocab=vocab, max_features_per_event=int(cfg.data.max_features_per_event))
    validation_path = str(split_info["validation_path"])
    test_path = str(split_info["test_path"])
    train_paths: Dict[int, str] = split_info["train_paths"]  # type: ignore[assignment]

    rows: List[MetricRow] = []
    for sample_size in sample_sizes:
        train_path = train_paths[sample_size]
        if bool(cfg.scaling.include_classic):
            cls_baseline = run_classification_baseline(
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                num_event_types=int(cfg.data.num_event_types),
            )
            rows.append(_row(sample_size, "classification", "count-logistic", cls_baseline["test"]))

            tpp_baseline = run_tpp_baseline(
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                num_event_types=int(cfg.data.num_event_types),
            )
            rows.append(_row(sample_size, "tpp", "markov", tpp_baseline["test"]))

        if bool(cfg.scaling.include_neural):
            cls_metrics = _run_neural_classification(
                cfg=cfg,
                vocab=vocab,
                tokenizer=tokenizer,
                train_path=train_path,
                test_path=test_path,
                output_dir=output_dir / "models" / "classification_{}".format(sample_size),
            )
            rows.append(_row(sample_size, "classification", "pragma-tiny", cls_metrics))

            tpp_metrics_ = _run_neural_tpp(
                cfg=cfg,
                vocab=vocab,
                tokenizer=tokenizer,
                train_path=train_path,
                test_path=test_path,
                output_dir=output_dir / "models" / "tpp_{}".format(sample_size),
            )
            rows.append(_row(sample_size, "tpp", "pragma-tiny", tpp_metrics_))

    csv_path = output_dir / "scaling_results.csv"
    markdown_path = output_dir / "scaling_results.md"
    json_path = output_dir / "scaling_results.json"
    _write_csv(csv_path, rows)
    plot_paths = _write_plots(output_dir, rows)
    _write_markdown(markdown_path, rows, plot_paths)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    publish_dir = _publish_report(
        publish_dir=cfg.scaling.get("publish_dir"),
        csv_path=csv_path,
        markdown_path=markdown_path,
        json_path=json_path,
        plot_paths=plot_paths,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "plots": [str(path) for path in plot_paths],
                "publish_dir": None if publish_dir is None else str(publish_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
