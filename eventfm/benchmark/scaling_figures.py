"""Scaling-law figures: log-log error curves and fitted exponents.

Three products, each answering a different question:

``by_dataset/``  For one dataset and metric: error against training-set size on
                 log-log axes, faceted by architecture family, with the fitted
                 power law drawn behind each measured curve. A straight line
                 means the power law holds over the measured range.
``by_method/``   For one method: its four datasets overlaid, so a single
                 architecture's data appetite can be read across benchmarks.
``exponents``    One dot plot per task: every method on the y-axis, its fitted
                 exponent per dataset on the x-axis. This is the summary — it
                 ranks architectures by how efficiently they turn labelled
                 sequences into accuracy, rather than by score at one size.
"""

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from eventfm.benchmark.palette import THEMES, Theme
from eventfm.benchmark.report import (
    FAMILY_LABEL,
    FAMILY_ORDER,
    _display,
    _save,
    _style_axes,
    series_for,
)
from eventfm.benchmark.scaling import (
    ERROR_METRICS,
    MIN_R_SQUARED,
    ScalingFit,
    fit_scaling_law,
    to_error,
)
from eventfm.datasets.registry import DATASET_REGISTRY
from eventfm.methods import METHOD_REGISTRY


def _error_series(records, dataset, task, method, metric, mode) -> List[Tuple[int, float]]:
    points = series_for(records, dataset, task, method, metric)
    converted = [(size, to_error(value, mode)) for size, value in points]
    return [(size, error) for size, error in converted if error is not None]


def _log_log_axes(axes, theme: Theme, sizes: Sequence[int]) -> None:
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    ticks = sorted({int(size) for size in sizes})
    axes.set_xticks(ticks)
    axes.set_xticklabels([str(size) for size in ticks], color=theme.text_secondary)
    axes.minorticks_off()


def _plot_fitted(axes, theme: Theme, label: str, index: int, points, fit: ScalingFit) -> None:
    color = theme.color(index)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    if fit is not None:
        grid = np.geomspace(min(xs), max(xs), 64)
        # Dashed = the model, solid+markers = the measurement, so the reader can
        # see where the power law actually fails to describe the curve.
        axes.plot(
            grid,
            fit.predict(grid),
            color=color,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            alpha=0.75 if fit.is_reliable else 0.35,
            zorder=2,
        )
    axes.plot(
        xs,
        ys,
        color=color,
        linewidth=2.0,
        marker=theme.marker(index),
        markersize=5,
        markeredgecolor=theme.surface,
        markeredgewidth=1.2,
        linestyle="none" if fit is not None else "-",
        label=label,
        zorder=4 + index,
    )


def _legend_label(display: str, fit: ScalingFit) -> str:
    if fit is None:
        return display
    flag = "" if fit.is_reliable else "*"
    return "{}  b={:.2f}{}".format(display, fit.exponent, flag)


def write_scaling_law_figures(records, output_dir: Path) -> List[Path]:
    """Log-log error curves with fitted power laws, faceted by family."""

    import matplotlib.pyplot as plt

    written: List[Path] = []
    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]

    for dataset in datasets:
        for task in ("classification", "tpp"):
            for metric, label, mode in ERROR_METRICS[task]:
                families: Dict[str, List[Tuple[str, list, ScalingFit]]] = {}
                for method, spec in sorted(METHOD_REGISTRY.items()):
                    points = _error_series(records, dataset, task, method, metric, mode)
                    if len(points) < 3:
                        continue
                    families.setdefault(spec.family, []).append(
                        (_display(method), points, fit_scaling_law(points))
                    )
                if not families:
                    continue

                ordered = [name for name in FAMILY_ORDER if name in families]
                ordered += [name for name in families if name not in ordered]
                all_points = [
                    point
                    for panels in families.values()
                    for _, panel_points, _ in panels
                    for point in panel_points
                ]
                sizes = {point[0] for point in all_points}
                values = [point[1] for point in all_points]
                y_low, y_high = min(values) * 0.9, max(values) * 1.1

                for theme_name, theme in THEMES.items():
                    columns = min(3, len(ordered))
                    rows = int(math.ceil(len(ordered) / columns))
                    figure, grid = plt.subplots(
                        rows,
                        columns,
                        figsize=(4.2 * columns, 3.6 * rows),
                        squeeze=False,
                        sharey=True,
                    )
                    figure.patch.set_facecolor(theme.surface)
                    for index, family in enumerate(ordered):
                        panel = families[family]
                        axes = grid[index // columns][index % columns]
                        axes.set_facecolor(theme.surface)
                        for series_index, (display, points, fit) in enumerate(panel):
                            _plot_fitted(
                                axes,
                                theme,
                                _legend_label(display, fit),
                                series_index,
                                points,
                                fit,
                            )
                        _log_log_axes(axes, theme, sizes)
                        title = FAMILY_LABEL.get(family, family)
                        if len(panel) == 1:
                            title = "{}\n{}".format(title, panel[0][0])
                        _style_axes(
                            axes,
                            theme,
                            "training sequences",
                            label if index % columns == 0 else "",
                            title,
                        )
                        axes.set_ylim(y_low, y_high)
                        axes.legend(
                            frameon=False,
                            fontsize=7.5,
                            labelcolor=theme.text_secondary,
                            loc="best",
                        )
                    for index in range(len(ordered), rows * columns):
                        grid[index // columns][index % columns].axis("off")

                    figure.suptitle(
                        "{} — {} — scaling law: {} (lower is better)".format(
                            DATASET_REGISTRY[dataset].display_name,
                            "classification" if task == "classification" else "TPP",
                            label,
                        ),
                        color=theme.text_primary,
                        fontsize=14,
                        fontweight="bold",
                        x=0.01,
                        ha="left",
                    )
                    figure.text(
                        0.01,
                        0.005,
                        "Dashed = fitted E(N) = a·N^-b.  b is the data-scaling exponent; "
                        "* marks a fit with R² < {:.2f}.".format(MIN_R_SQUARED),
                        color=theme.text_muted,
                        fontsize=8,
                    )
                    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
                    suffix = "" if theme_name == "light" else ".dark"
                    path = output_dir / "scaling_laws" / "by_dataset" / (
                        "{}__{}__{}{}.svg".format(dataset, task, metric, suffix)
                    )
                    _save(figure, path)
                    written.append(path)
    return written


def write_method_scaling_law_figures(records, output_dir: Path) -> List[Path]:
    """One log-log scaling law per method, its datasets overlaid."""

    written: List[Path] = []
    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]

    for method in sorted(METHOD_REGISTRY):
        for task in ("classification", "tpp"):
            if task not in METHOD_REGISTRY[method].supports:
                continue
            for metric, label, mode in ERROR_METRICS[task]:
                series = []
                for dataset in datasets:
                    points = _error_series(records, dataset, task, method, metric, mode)
                    if len(points) >= 3:
                        series.append(
                            (
                                DATASET_REGISTRY[dataset].display_name,
                                points,
                                fit_scaling_law(points),
                            )
                        )
                if not series:
                    continue
                sizes = {point[0] for _, points, _ in series for point in points}

                for theme_name, theme in THEMES.items():
                    import matplotlib.pyplot as plt

                    figure, axes = plt.subplots(figsize=(7.4, 4.6))
                    figure.patch.set_facecolor(theme.surface)
                    axes.set_facecolor(theme.surface)
                    for index, (display, points, fit) in enumerate(series):
                        _plot_fitted(axes, theme, _legend_label(display, fit), index, points, fit)
                    _log_log_axes(axes, theme, sizes)
                    _style_axes(
                        axes,
                        theme,
                        "labelled training sequences",
                        "{} (lower is better)".format(label),
                        "{} — scaling law".format(_display(method)),
                    )
                    axes.legend(
                        frameon=False, fontsize=8, labelcolor=theme.text_secondary, loc="best"
                    )
                    figure.text(
                        0.01,
                        0.005,
                        "Dashed = fitted E(N) = a·N^-b.  * marks R² < {:.2f}.".format(
                            MIN_R_SQUARED
                        ),
                        color=theme.text_muted,
                        fontsize=8,
                    )
                    suffix = "" if theme_name == "light" else ".dark"
                    path = output_dir / "scaling_laws" / "by_method" / (
                        "{}__{}__{}{}.svg".format(method, task, metric, suffix)
                    )
                    _save(figure, path)
                    written.append(path)
    return written


def write_exponent_summary(records, output_dir: Path) -> List[Path]:
    """Dot plot of the fitted exponent per method, one dot per dataset.

    This is the headline scaling figure: it ranks architectures by how fast
    their error falls with data, which is a different — and for a foundation
    model, more relevant — question than who wins at one sample size.
    """

    import matplotlib.pyplot as plt

    written: List[Path] = []
    datasets = [name for name in DATASET_REGISTRY if any(r.dataset == name for r in records)]

    for task in ("classification", "tpp"):
        metric, label, mode = ERROR_METRICS[task][0]
        rows: List[Tuple[str, str, List[Tuple[str, float, bool]]]] = []
        for method, spec in METHOD_REGISTRY.items():
            if task not in spec.supports:
                continue
            entries = []
            for dataset in datasets:
                points = _error_series(records, dataset, task, method, metric, mode)
                fit = fit_scaling_law(points) if len(points) >= 3 else None
                if fit is not None:
                    entries.append((dataset, fit.exponent, fit.is_reliable))
            if entries:
                rows.append((method, spec.family, entries))
        if not rows:
            continue

        rows.sort(key=lambda item: float(np.mean([value for _, value, _ in item[2]])))

        for theme_name, theme in THEMES.items():
            figure, axes = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 2.2))
            figure.patch.set_facecolor(theme.surface)
            axes.set_facecolor(theme.surface)

            for row_index, (_, _, entries) in enumerate(rows):
                mean = float(np.mean([value for _, value, _ in entries]))
                axes.plot(
                    [0, mean],
                    [row_index, row_index],
                    color=theme.grid,
                    linewidth=1.0,
                    zorder=1,
                )
                for dataset, value, reliable in entries:
                    index = datasets.index(dataset)
                    axes.plot(
                        value,
                        row_index,
                        marker=theme.marker(index),
                        markersize=7 if reliable else 5,
                        color=theme.color(index),
                        markeredgecolor=theme.surface,
                        markeredgewidth=1.2,
                        linestyle="none",
                        alpha=1.0 if reliable else 0.45,
                        zorder=3,
                        label=DATASET_REGISTRY[dataset].display_name if row_index == 0 else None,
                    )

            axes.set_yticks(range(len(rows)))
            axes.set_yticklabels(
                [_display(method) for method, _, _ in rows], color=theme.text_primary, fontsize=9
            )
            axes.axvline(0.0, color=theme.axis, linewidth=0.8)
            _style_axes(
                axes,
                theme,
                "fitted data-scaling exponent b   (higher = error falls faster with data)",
                "",
                "{} — how fast does each method improve with data?".format(
                    "Classification" if task == "classification" else "TPP"
                ),
            )
            axes.grid(axis="y", visible=False)
            handles, labels = axes.get_legend_handles_labels()
            if handles:
                axes.legend(
                    handles,
                    labels,
                    frameon=False,
                    fontsize=8,
                    labelcolor=theme.text_secondary,
                    loc="lower right",
                    ncols=2,
                )
            figure.text(
                0.01,
                0.01,
                "Fitted on {}.  Faded markers are fits with R² < {:.2f}.".format(
                    label, MIN_R_SQUARED
                ),
                color=theme.text_muted,
                fontsize=8,
            )
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            suffix = "" if theme_name == "light" else ".dark"
            path = output_dir / "scaling_laws" / "exponents__{}{}.svg".format(task, suffix)
            _save(figure, path)
            written.append(path)
    return written
