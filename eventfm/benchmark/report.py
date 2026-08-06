"""Turn finished benchmark cells into the tables and scaling figures.

Two products come out of here:

* **Result tables** — one per (dataset, task), every method as a row, evaluated
  at the largest training-sample point that finished. This is the "all methods
  per dataset and task" view.
* **Scaling figures** — metric against number of labelled training sequences on
  a log axis. One figure per method (its four datasets overlaid) plus one
  faceted comparison per dataset (small multiples by architecture family, so no
  single axes ever carries more than four series).

Both are written in light and dark variants; the report markdown pairs them in
a ``<picture>`` element so the figures follow the reader's theme.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from eventfm.benchmark.palette import THEMES, Theme
from eventfm.datasets.registry import DATASET_REGISTRY
from eventfm.methods import METHOD_REGISTRY

TASK_METRICS: Dict[str, List[Tuple[str, str, bool]]] = {
    # (metric key, display label, higher-is-better)
    "classification": [
        ("auc", "ROC-AUC", True),
        ("average_precision", "Avg. precision", True),
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro F1", True),
    ],
    "tpp": [
        ("next_type_accuracy", "Next-type acc.", True),
        ("next_type_macro_f1", "Next-type macro F1", True),
        ("delta_log_rmse", "Time RMSE (log1p s)", False),
        ("delta_log_mae", "Time MAE (log1p s)", False),
    ],
}

PRIMARY_METRIC = {"classification": "auc", "tpp": "next_type_accuracy"}

FAMILY_ORDER = [
    "classic",
    "neural-tpp",
    "hierarchical",
    "tabular-transformer",
    "contrastive",
    "state-space",
    "llm",
]

FAMILY_LABEL = {
    "classic": "Non-neural controls",
    "neural-tpp": "Neural TPP",
    "hierarchical": "PRAGMA (hierarchical)",
    "tabular-transformer": "Tabular transformers",
    "contrastive": "Contrastive (CoLES)",
    "state-space": "State space (Mambular)",
    "llm": "LLM-based TPP",
}


@dataclass
class Record:
    dataset: str
    task: str
    method: str
    sample_size: int
    seed: int
    regime: str
    variant: str
    status: str
    metrics: Dict[str, float]
    wall_seconds: float
    num_train_sequences: int


def load_results(results_dir: Path) -> List[Record]:
    records: List[Record] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cell = payload.get("cell", {})
        records.append(
            Record(
                dataset=str(cell.get("dataset", "")),
                task=str(cell.get("task", "")),
                method=str(cell.get("method", "")),
                sample_size=int(cell.get("sample_size", 0)),
                seed=int(cell.get("seed", 0)),
                regime=str(cell.get("regime", "full")),
                variant=str(cell.get("variant", "default")),
                status=str(payload.get("status", "unknown")),
                metrics={
                    key: float(value)
                    for key, value in (payload.get("metrics") or {}).items()
                    if value is not None and not (isinstance(value, float) and math.isnan(value))
                },
                wall_seconds=float(payload.get("wall_seconds", 0.0)),
                num_train_sequences=int(payload.get("num_train_sequences", 0)),
            )
        )
    return records


def write_csv(records: Sequence[Record], path: Path) -> Path:
    import csv

    metric_keys = sorted({key for record in records for key in record.metrics})
    fields = [
        "dataset",
        "task",
        "method",
        "family",
        "fidelity",
        "divergence",
        "sample_size",
        "num_train_sequences",
        "seed",
        "regime",
        "variant",
        "status",
        "wall_seconds",
    ] + metric_keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "dataset": record.dataset,
                "task": record.task,
                "method": record.method,
                "family": _family(record.method),
                # `fidelity` is how faithful the entry is to its paper;
                # `status` is whether this particular run succeeded.
                "fidelity": _status(record.method),
                "divergence": _divergence(record.method),
                "sample_size": record.sample_size,
                "num_train_sequences": record.num_train_sequences,
                "seed": record.seed,
                "regime": record.regime,
                "variant": record.variant,
                "status": record.status,
                "wall_seconds": round(record.wall_seconds, 2),
            }
            row.update({key: record.metrics.get(key) for key in metric_keys})
            writer.writerow(row)
    return path


def _family(method: str) -> str:
    spec = METHOD_REGISTRY.get(method)
    return spec.family if spec else "unknown"


def _display(method: str) -> str:
    spec = METHOD_REGISTRY.get(method)
    return spec.display_name if spec else method


def _status(method: str) -> str:
    """Fidelity label, so a table row named after a paper cannot be read as a
    reproduction of it. See docs/research/README.md for the vocabulary."""

    spec = METHOD_REGISTRY.get(method)
    return spec.status if spec else "unknown"


def _divergence(method: str) -> str:
    spec = METHOD_REGISTRY.get(method)
    return spec.divergence if spec else ""


def _mean_over_seeds(values: Sequence[float]) -> Optional[float]:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return float(np.mean(clean)) if clean else None


def _bootstrap_summary(
    values: Sequence[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Deterministic percentile interval over independent seed-level results."""

    clean = np.asarray(
        [value for value in values if value is not None and not math.isnan(value)],
        dtype=np.float64,
    )
    if not clean.size:
        return None, None, None
    mean = float(clean.mean())
    if clean.size < 2:
        return mean, None, None
    rng = np.random.default_rng(20260806)
    samples = rng.choice(clean, size=(10000, clean.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return mean, float(lower), float(upper)


def _metric_values_at(
    records: Sequence[Record],
    dataset: str,
    task: str,
    method: str,
    metric: str,
    size: int,
    regime: str = "full",
    variant: str = "default",
) -> List[float]:
    return [
        record.metrics[metric]
        for record in records
        if record.dataset == dataset
        and record.task == task
        and record.method == method
        and record.regime == regime
        and record.variant == variant
        and record.status == "ok"
        and (record.num_train_sequences or record.sample_size) == size
        and metric in record.metrics
    ]


def series_for(
    records: Sequence[Record],
    dataset: str,
    task: str,
    method: str,
    metric: str,
    regime: str = "full",
    variant: str = "default",
) -> List[Tuple[int, float]]:
    """Metric against training-set size, averaged over seeds."""

    by_size: Dict[int, List[float]] = {}
    for record in records:
        if (
            record.dataset != dataset
            or record.task != task
            or record.method != method
            or record.regime != regime
            or record.variant != variant
            or record.status != "ok"
            or metric not in record.metrics
        ):
            continue
        size = record.num_train_sequences or record.sample_size
        by_size.setdefault(size, []).append(record.metrics[metric])
    points = [
        (size, value)
        for size, value in ((size, _mean_over_seeds(values)) for size, values in by_size.items())
        if value is not None
    ]
    return sorted(points)


def _format(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return "{:.3f}".format(value)


def _format_summary(summary: Tuple[Optional[float], Optional[float], Optional[float]]) -> str:
    mean, lower, upper = summary
    if mean is None:
        return "—"
    if lower is None or upper is None:
        return "{:.3f}".format(mean)
    return "{:.3f} [{:.3f}, {:.3f}]".format(mean, lower, upper)


def write_result_tables(records: Sequence[Record], path: Path) -> Path:
    """Markdown: all methods per dataset and task, at the largest sample size."""

    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]
    lines: List[str] = [
        "# Benchmark results",
        "",
        "Every architecture surveyed in the project note, evaluated on every open",
        "benchmark it names, for both tasks. Each cell is the mean over seeds at the",
        "largest training-sample point that completed; the per-point curves behind",
        "these numbers are in the scaling figures.",
        "",
        "**Read the `Fidelity` column before citing anything here.** `Implemented`",
        "means the entry is a control with no paper to be faithful to.",
        "`Approximation` means the entry preserves the central comparison but",
        "differs materially from the cited recipe — usually in scale, in the",
        "pretraining corpus, or in the decoding head. The specific divergence for",
        "each method is in `results.csv` and in",
        "[`../research/method_catalog.md`](../research/method_catalog.md). A row",
        "named after a paper is not a reproduction of that paper.",
        "",
    ]

    for task in ("classification", "tpp"):
        metrics = TASK_METRICS[task]
        heading = (
            "Binary sequence classification"
            if task == "classification"
            else "Temporal point process (next event)"
        )
        lines.append("## Task: {}".format(heading))
        lines.append("")
        if task == "classification":
            lines.append(
                "Positive class is the forward-looking label described in each dataset's "
                "`meta.json`. ROC-AUC is the headline metric; accuracy is reported for context "
                "because training may be rebalanced while evaluation keeps natural prevalence."
            )
        else:
            lines.append(
                "Scored separately for *what* and *when*, as the neural-TPP review recommends. "
                "Time metrics are on `log1p(seconds)`, so **lower is better**."
            )
        lines.append("")

        for dataset in datasets:
            spec = DATASET_REGISTRY[dataset]
            best_size = max(
                (
                    record.num_train_sequences or record.sample_size
                    for record in records
                    if record.dataset == dataset
                    and record.task == task
                    and record.regime == "full"
                    and record.variant == "default"
                    and record.status == "ok"
                ),
                default=0,
            )
            if not best_size:
                continue
            lines.append("### {} — {} training sequences".format(spec.display_name, best_size))
            lines.append("")
            lines.append(
                "| Method | Family | Fidelity | {} |".format(
                    " | ".join(label for _, label, _ in metrics)
                )
            )
            lines.append("| --- | --- | --- | {} |".format(" | ".join(["---"] * len(metrics))))

            rows: List[Tuple[float, str]] = []
            for method in sorted(METHOD_REGISTRY):
                summaries = {
                    metric: _bootstrap_summary(
                        _metric_values_at(
                            records, dataset, task, method, metric, best_size, regime="full"
                        )
                    )
                    for metric, _, _ in metrics
                }
                if all(summary[0] is None for summary in summaries.values()):
                    continue
                primary = summaries.get(PRIMARY_METRIC[task], (None, None, None))[0]
                rank = -(primary if primary is not None else -1.0)
                rows.append(
                    (
                        rank,
                        "| {} | {} | {} | {} |".format(
                            _display(method),
                            _family(method),
                            _status(method),
                            " | ".join(
                                _format_summary(summaries[metric]) for metric, _, _ in metrics
                            ),
                        ),
                    )
                )
            for _, row in sorted(rows, key=lambda item: item[0]):
                lines.append(row)
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_adaptation_table(records: Sequence[Record], path: Path) -> Path:
    """Write frozen/PEFT/full comparisons with seed-level uncertainty."""

    adaptation_records = [
        record for record in records if record.regime != "full" and record.variant == "default"
    ]
    lines = [
        "# Adaptation-regime results",
        "",
        "Cells report the mean and deterministic seed-bootstrap 95% interval. Full",
        "fine-tuning is included beside frozen and parameter-efficient conditions",
        "when it was run with the same dataset, task, method, and sample size.",
        "",
    ]
    if not adaptation_records:
        lines.append("No non-default adaptation cells were found.")
        lines.append("")
    coordinates = sorted(
        {
            (record.dataset, record.task, record.method)
            for record in adaptation_records
            if record.status == "ok"
        }
    )
    for dataset, task, method in coordinates:
        metric = PRIMARY_METRIC[task]
        matching = [
            record
            for record in records
            if record.dataset == dataset
            and record.task == task
            and record.method == method
            and record.variant == "default"
            and record.status == "ok"
        ]
        if not matching:
            continue
        size = max(record.num_train_sequences or record.sample_size for record in matching)
        lines.extend(
            [
                "## {} — {} — {}".format(
                    DATASET_REGISTRY[dataset].display_name, task, _display(method)
                ),
                "",
                "| Regime | N | {} | Trainable parameters | Wall seconds |".format(metric),
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for regime in ("frozen", "peft", "full"):
            regime_records = [
                record
                for record in matching
                if record.regime == regime
                and (record.num_train_sequences or record.sample_size) == size
            ]
            if not regime_records:
                continue
            metric_summary = _bootstrap_summary(
                [record.metrics[metric] for record in regime_records if metric in record.metrics]
            )
            parameter_summary = _bootstrap_summary(
                [
                    record.metrics.get(
                        "num_trainable_parameters", record.metrics.get("num_parameters")
                    )
                    for record in regime_records
                    if record.metrics.get(
                        "num_trainable_parameters", record.metrics.get("num_parameters")
                    )
                    is not None
                ]
            )
            wall_summary = _bootstrap_summary(
                [record.wall_seconds for record in regime_records]
            )
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    regime,
                    size,
                    _format_summary(metric_summary),
                    _format_summary(parameter_summary),
                    _format_summary(wall_summary),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_ablation_table(records: Sequence[Record], path: Path) -> Path:
    """Summarize every named non-default experimental variant."""

    variant_records = [record for record in records if record.variant != "default"]
    lines = [
        "# Ablation and robustness variants",
        "",
        "Cells report the mean and deterministic seed-bootstrap 95% interval at",
        "the largest completed sample size for each coordinate.",
        "",
    ]
    if not variant_records:
        lines.extend(["No named variant cells were found.", ""])
    coordinates = sorted(
        {
            (record.dataset, record.task, record.method, record.regime)
            for record in variant_records
            if record.status == "ok"
        }
    )
    for dataset, task, method, regime in coordinates:
        matching = [
            record
            for record in variant_records
            if record.dataset == dataset
            and record.task == task
            and record.method == method
            and record.regime == regime
            and record.status == "ok"
        ]
        if not matching:
            continue
        size = max(record.num_train_sequences or record.sample_size for record in matching)
        metric = PRIMARY_METRIC[task]
        lines.extend(
            [
                "## {} — {} — {} ({})".format(
                    DATASET_REGISTRY[dataset].display_name, task, _display(method), regime
                ),
                "",
                "| Variant | N | {} | Trainable parameters | Wall seconds |".format(metric),
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in sorted({record.variant for record in matching}):
            rows = [
                record
                for record in matching
                if record.variant == variant
                and (record.num_train_sequences or record.sample_size) == size
            ]
            metric_summary = _bootstrap_summary(
                [record.metrics[metric] for record in rows if metric in record.metrics]
            )
            parameter_values = [
                record.metrics.get(
                    "num_trainable_parameters", record.metrics.get("num_parameters")
                )
                for record in rows
            ]
            parameter_summary = _bootstrap_summary(
                [value for value in parameter_values if value is not None]
            )
            wall_summary = _bootstrap_summary([record.wall_seconds for record in rows])
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    variant,
                    size,
                    _format_summary(metric_summary),
                    _format_summary(parameter_summary),
                    _format_summary(wall_summary),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_scaling_exponent_table(records: Sequence[Record], path: Path) -> Path:
    """Fitted data-scaling exponents, per method and dataset."""

    from eventfm.benchmark.scaling import ERROR_METRICS, MIN_R_SQUARED, fit_all

    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]
    methods = sorted(METHOD_REGISTRY)
    fits = fit_all(records, datasets, ("classification", "tpp"), methods)

    lines: List[str] = [
        "# Data-scaling exponents",
        "",
        "Each curve's error is fitted to `E(N) = a · N^-b`, where `N` is the number",
        "of labelled training sequences. **`b` is the data-scaling exponent: how fast",
        "the error falls as data is added.** A larger `b` means the method is still",
        "converting extra sequences into accuracy; `b` near zero means it has",
        "flattened and more labels will not help.",
        "",
        "This ranks architectures by *data efficiency* rather than by score at one",
        "sample size, which is the more relevant question for a foundation model.",
        "",
        "Fitted on six points per curve, so read `b` as a local slope over the",
        "measured range, not an asymptotic claim. Values marked `*` have",
        "`R² < {:.2f}` — the power law does not describe that curve well and the".format(
            MIN_R_SQUARED
        ),
        "exponent should not be trusted.",
        "",
        "Two means are given. **`Mean b (reliable)` is the one to read** — it",
        "averages only the fits that pass the R² gate, with the count in",
        "parentheses, and it is what the rows are sorted by. `Mean b (all)`",
        "includes the flagged fits and is shown so the difference is visible: on",
        "classification, PaySim fails the gate for nearly every method, and its",
        "occasional *negative* exponent (error rising with data) is the signature",
        "of a task with no learnable signal rather than of a bad model.",
        "",
    ]

    for task in ("classification", "tpp"):
        metric, label, _ = ERROR_METRICS[task][0]
        heading = "Classification" if task == "classification" else "Temporal point process"
        lines.append("## {} — fitted on {}".format(heading, label))
        lines.append("")
        lines.append(
            "| Method | Family | {} | Mean b (all) | Mean b (reliable) |".format(
                " | ".join(DATASET_REGISTRY[name].display_name for name in datasets)
            )
        )
        lines.append(
            "| --- | --- | {} | ---: | ---: |".format(" | ".join(["---:"] * len(datasets)))
        )

        rows: List[Tuple[float, str]] = []
        for method in methods:
            if task not in METHOD_REGISTRY[method].supports:
                continue
            cells: List[str] = []
            values: List[float] = []
            reliable: List[float] = []
            for dataset in datasets:
                fit = fits.get((dataset, task, method, metric))
                if fit is None:
                    cells.append("—")
                    continue
                values.append(fit.exponent)
                if fit.is_reliable:
                    reliable.append(fit.exponent)
                cells.append(
                    "{:.3f}{}".format(fit.exponent, "" if fit.is_reliable else "*")
                )
            if not values:
                continue
            mean = float(np.mean(values))
            # Ranked on the reliable-only mean: averaging in a fit the power law
            # does not describe would rank methods on noise.
            reliable_mean = float(np.mean(reliable)) if reliable else float("nan")
            sort_key = reliable_mean if reliable else -1e9
            rows.append(
                (
                    -sort_key,
                    "| {} | {} | {} | {:.3f} | {} |".format(
                        _display(method),
                        _family(method),
                        " | ".join(cells),
                        mean,
                        "**{:.3f}** ({})".format(reliable_mean, len(reliable))
                        if reliable
                        else "— (0)",
                    ),
                )
            )
        for _, row in sorted(rows, key=lambda item: item[0]):
            lines.append(row)
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _new_axes(theme: Theme, figsize: Tuple[float, float]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor(theme.surface)
    axes.set_facecolor(theme.surface)
    return figure, axes


def _style_axes(axes, theme: Theme, x_label: str, y_label: str, title: str) -> None:
    axes.set_title(
        title,
        color=theme.text_primary,
        # Long family names in a 3-up facet grid need a smaller face than a
        # single wide axes does, otherwise adjacent titles run into each other.
        fontsize=13 if "\n" not in title and len(title) < 28 else 11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    axes.set_xlabel(x_label, color=theme.text_secondary, fontsize=10)
    axes.set_ylabel(y_label, color=theme.text_secondary, fontsize=10)
    axes.grid(True, which="major", color=theme.grid, linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme.axis)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=theme.text_secondary, labelsize=9, length=3, width=0.8)


def _plot_series(
    axes,
    theme: Theme,
    grouped: Dict[str, List[Tuple[int, float]]],
    direct_labels: bool = True,
) -> None:
    for index, (label, points) in enumerate(grouped.items()):
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        color = theme.color(index)
        axes.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            linestyle=theme.linestyle(index),
            marker=theme.marker(index),
            markersize=5,
            markeredgecolor=theme.surface,
            markeredgewidth=1.2,
            label=label,
            zorder=3 + index,
        )
        if direct_labels:
            # Identity is never colour-alone: the last point carries the name.
            axes.annotate(
                label,
                xy=(xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=theme.text_primary,
                fontsize=8,
                va="center",
                zorder=10,
            )


def _log_x_axis(axes, sizes: Iterable[int], theme: Theme) -> None:
    sizes = sorted({int(size) for size in sizes})
    if not sizes:
        return
    axes.set_xscale("log", base=2)
    axes.set_xticks(sizes)
    axes.set_xticklabels([str(size) for size in sizes], color=theme.text_secondary)
    axes.minorticks_off()


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=160, facecolor=figure.get_facecolor())
    import matplotlib.pyplot as plt

    plt.close(figure)


def write_method_scaling_figures(
    records: Sequence[Record],
    output_dir: Path,
) -> List[Path]:
    """One figure per method: its scaling curve on each dataset."""

    written: List[Path] = []
    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]
    for method in sorted(METHOD_REGISTRY):
        for task in ("classification", "tpp"):
            for metric, label, higher_is_better in TASK_METRICS[task]:
                grouped = {
                    DATASET_REGISTRY[dataset].display_name: series_for(
                        records, dataset, task, method, metric
                    )
                    for dataset in datasets
                }
                grouped = {key: value for key, value in grouped.items() if value}
                if not grouped:
                    continue
                sizes = {point[0] for points in grouped.values() for point in points}
                if len(sizes) < 2:
                    continue

                for theme_name, theme in THEMES.items():
                    figure, axes = _new_axes(theme, (7.2, 4.4))
                    _plot_series(axes, theme, grouped)
                    _log_x_axis(axes, sizes, theme)
                    _style_axes(
                        axes,
                        theme,
                        "labelled training sequences",
                        "{}{}".format(label, "" if higher_is_better else "  (lower is better)"),
                        "{} — {}".format(_display(method), label),
                    )
                    axes.legend(
                        frameon=False,
                        fontsize=8,
                        labelcolor=theme.text_secondary,
                        loc="best",
                        ncols=2,
                    )
                    axes.margins(x=0.16)
                    suffix = "" if theme_name == "light" else ".dark"
                    path = output_dir / "by_method" / "{}__{}__{}{}.svg".format(
                        method, task, metric, suffix
                    )
                    _save(figure, path)
                    written.append(path)
    return written


def write_dataset_comparison_figures(
    records: Sequence[Record],
    output_dir: Path,
) -> List[Path]:
    """Per dataset and task: small multiples, one panel per architecture family.

    Faceting rather than a single 13-line axes is deliberate — no panel carries
    more than four series, which is what keeps the categorical hues separable.
    """

    import matplotlib.pyplot as plt

    written: List[Path] = []
    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]
    for dataset in datasets:
        for task in ("classification", "tpp"):
            for metric, label, higher_is_better in TASK_METRICS[task]:
                families: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
                for method, spec in sorted(METHOD_REGISTRY.items()):
                    points = series_for(records, dataset, task, method, metric)
                    if len(points) < 2:
                        continue
                    families.setdefault(spec.family, {})[_display(method)] = points
                families = {key: value for key, value in families.items() if value}
                if not families:
                    continue

                ordered = [name for name in FAMILY_ORDER if name in families]
                ordered += [name for name in families if name not in ordered]
                all_points = [
                    point
                    for panels in families.values()
                    for points in panels.values()
                    for point in points
                ]
                sizes = {point[0] for point in all_points}
                y_values = [point[1] for point in all_points]
                y_low, y_high = min(y_values), max(y_values)
                pad = 0.08 * max(1e-6, y_high - y_low)

                for theme_name, theme in THEMES.items():
                    columns = min(3, len(ordered))
                    rows = int(math.ceil(len(ordered) / columns))
                    figure, axes_grid = plt.subplots(
                        rows,
                        columns,
                        figsize=(4.0 * columns, 3.4 * rows),
                        squeeze=False,
                        sharey=True,
                    )
                    figure.patch.set_facecolor(theme.surface)
                    for index, family in enumerate(ordered):
                        panel = families[family]
                        axes = axes_grid[index // columns][index % columns]
                        axes.set_facecolor(theme.surface)
                        _plot_series(axes, theme, panel, direct_labels=False)
                        _log_x_axis(axes, sizes, theme)
                        # A lone series is named in the panel title instead of a
                        # legend box that would just repeat it. The name goes on
                        # its own line: appended inline it overflows the panel
                        # and collides with the neighbouring title.
                        title = FAMILY_LABEL.get(family, family)
                        if len(panel) == 1:
                            title = "{}\n{}".format(title, next(iter(panel)))
                        _style_axes(
                            axes,
                            theme,
                            "training sequences",
                            label if index % columns == 0 else "",
                            title,
                        )
                        axes.set_ylim(y_low - pad, y_high + pad)
                        if len(panel) > 1:
                            axes.legend(
                                frameon=False,
                                fontsize=7.5,
                                labelcolor=theme.text_secondary,
                                loc="best",
                            )
                    for index in range(len(ordered), rows * columns):
                        axes_grid[index // columns][index % columns].axis("off")

                    figure.suptitle(
                        "{} — {} — {}{}".format(
                            DATASET_REGISTRY[dataset].display_name,
                            "classification" if task == "classification" else "TPP",
                            label,
                            "" if higher_is_better else "  (lower is better)",
                        ),
                        color=theme.text_primary,
                        fontsize=14,
                        fontweight="bold",
                        x=0.01,
                        ha="left",
                    )
                    figure.tight_layout(rect=(0, 0, 1, 0.96))
                    suffix = "" if theme_name == "light" else ".dark"
                    path = output_dir / "by_dataset" / "{}__{}__{}{}.svg".format(
                        dataset, task, metric, suffix
                    )
                    _save(figure, path)
                    written.append(path)
    return written
