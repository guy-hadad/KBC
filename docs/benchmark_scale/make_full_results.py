"""Write full_results.md: every method, sample size, metric and scaling fit on the scale datasets.

Reads the published ``results.csv`` next to this file, so it needs no access to
the raw result JSON. Kept outside ``scripts/`` on purpose: files there enter the
``source_tree_sha256`` fingerprint that running experiment cells record.

    python docs/benchmark_scale/make_full_results.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from eventfm.benchmark.report import PRIMARY_METRIC, TASK_METRICS  # noqa: E402
from eventfm.benchmark.scaling import (  # noqa: E402
    ERROR_METRICS,
    MIN_R_SQUARED,
    fit_scaling_law,
    to_error,
)
from eventfm.datasets.registry import DATASET_REGISTRY  # noqa: E402
from eventfm.methods import METHOD_REGISTRY  # noqa: E402

DATASETS = ["mbd", "synthea", "amazon_beauty"]
TASKS = ["classification", "tpp"]
EXPECTED_SEEDS = 3


def display(method: str) -> str:
    spec = METHOD_REGISTRY.get(method)
    return spec.display_name if spec else method


def method_order(frame: pd.DataFrame, metric: str, top: int, higher: bool) -> list:
    """Methods ranked by the metric at the full pool; methods absent there go last."""

    at_top = frame[frame.sample_size == top].groupby("method")[metric].mean()
    ranked = list(at_top.sort_values(ascending=not higher).index)
    rest = sorted(set(frame.method) - set(ranked))
    return ranked + rest


def matrix(frame: pd.DataFrame, metric: str, methods: list, sizes: list) -> list:
    grouped = frame.groupby(["method", "sample_size"])[metric].agg(["mean", "count"])
    header = "| Method | Family | " + " | ".join(f"{s:,}".replace(",", " ") for s in sizes) + " |"
    lines = [header, "| --- | --- |" + " ---: |" * len(sizes)]
    for method in methods:
        cells = []
        for size in sizes:
            if (method, size) not in grouped.index or grouped.loc[(method, size), "count"] == 0:
                cells.append("—")
                continue
            mean, count = grouped.loc[(method, size)]
            mark = "" if count >= EXPECTED_SEEDS else f" ({int(count)})"
            cells.append(f"{mean:.3f}{mark}")
        family = frame.loc[frame.method == method, "family"].iloc[0]
        lines.append(f"| {display(method)} | {family} | " + " | ".join(cells) + " |")
    return lines


def full_pool_table(frame: pd.DataFrame, task: str, methods: list, top: int) -> list:
    metrics = TASK_METRICS[task]
    head = "| Method | Fidelity | Seeds | " + " | ".join(label for _, label, _ in metrics) + " | Wall s (median) |"
    lines = [head, "| --- | --- | ---: |" + " ---: |" * (len(metrics) + 1)]
    at_top = frame[frame.sample_size == top]
    for method in methods:
        rows = at_top[at_top.method == method]
        if rows.empty:
            continue
        cells = []
        for key, _, _ in metrics:
            values = rows[key].dropna()
            if values.empty:
                cells.append("—")
            elif len(values) < 2:
                cells.append(f"{values.mean():.3f}")
            else:
                cells.append(f"{values.mean():.3f} ± {values.std(ddof=1):.3f}")
        fidelity = rows.fidelity.iloc[0]
        lines.append(
            f"| {display(method)} | {fidelity} | {len(rows)} | "
            + " | ".join(cells)
            + f" | {rows.wall_seconds.median():,.0f} |"
        )
    return lines


def fit_curve(frame: pd.DataFrame, method: str, metric: str, mode: str):
    """Same fit as scaling_exponents.md: seed means per size, then log-log least squares."""

    rows = frame[(frame.method == method)].dropna(subset=[metric])
    means = rows.groupby("sample_size")[metric].mean()
    points = [(size, to_error(value, mode)) for size, value in means.items()]
    return fit_scaling_law([(size, error) for size, error in points if error is not None])


def seed_exponents(frame: pd.DataFrame, method: str, metric: str, mode: str) -> list:
    """One exponent per seed, from that seed's own curve (at least four points)."""

    exponents = []
    rows = frame[(frame.method == method)].dropna(subset=[metric])
    for _, seed_rows in rows.groupby("seed"):
        points = [(size, to_error(value, mode)) for size, value in zip(seed_rows.sample_size, seed_rows[metric])]
        fit = fit_scaling_law([(size, error) for size, error in points if error is not None])
        if fit is not None and fit.num_points >= 4:
            exponents.append(fit.exponent)
    return exponents


def exponent_cell(fit) -> str:
    if fit is None:
        return "—"
    return f"{fit.exponent:.3f}{'' if fit.is_reliable else '*'}"


def scaling_summary(data: pd.DataFrame) -> list:
    """Method x dataset exponent tables for every fitted error metric, both tasks."""

    names = [DATASET_REGISTRY[name].display_name for name in DATASETS]
    lines = [
        "## Scaling analysis",
        "",
        "Each curve's error is fitted to `E(N) = a · N^-b`, where `N` is the number of training "
        "sequences, using the seed means at each size. **`b` is the slope on log-log axes: how fast "
        "the error falls as data is added.** Larger `b` means the method is still converting data "
        "into accuracy. `b` near zero means it has flattened. Read `b` as a local slope over the "
        "measured range, not an asymptotic claim.",
        "",
        f"`*` marks a fit with `R² < {MIN_R_SQUARED:.2f}` or fewer than four points, which should not "
        "be read as a scaling rate. `Mean b (reliable)` averages only the unflagged fits, with their "
        "count in parentheses, and rows are sorted by it. The primary-metric exponents match "
        "[`scaling_exponents.md`](scaling_exponents.md). Per-dataset fit details, including "
        "`a`, `R²` and the spread of per-seed slopes, are under each dataset below.",
    ]
    for task in TASKS:
        task_label = "Classification" if task == "classification" else "Temporal point process (TPP)"
        for metric, label, mode in ERROR_METRICS[task]:
            rows = []
            for method in sorted(data[data.task == task].method.unique()):
                fits = [
                    fit_curve(data[(data.dataset == name) & (data.task == task)], method, metric, mode)
                    for name in DATASETS
                ]
                present = [fit for fit in fits if fit is not None]
                if not present:
                    continue
                reliable = [fit.exponent for fit in present if fit.is_reliable]
                mean_all = np.mean([fit.exponent for fit in present])
                mean_reliable = f"**{np.mean(reliable):.3f}** ({len(reliable)})" if reliable else "— (0)"
                family = data.loc[data.method == method, "family"].iloc[0]
                key = -np.mean(reliable) if reliable else 1e9
                rows.append((key, f"| {display(method)} | {family} | "
                             + " | ".join(exponent_cell(fit) for fit in fits)
                             + f" | {mean_all:.3f} | {mean_reliable} |"))
            lines += [
                "",
                f"### {task_label} — `b` fitted on {label}",
                "",
                "| Method | Family | " + " | ".join(names) + " | Mean b (all) | Mean b (reliable) |",
                "| --- | --- |" + " ---: |" * (len(names) + 2),
                *[row for _, row in sorted(rows, key=lambda item: item[0])],
            ]
    return lines


def fit_detail_table(frame: pd.DataFrame, methods: list, task: str) -> list:
    metrics = ERROR_METRICS[task]
    head = "| Method | " + " | ".join(
        f"b ({label}) | a | R² | per-seed b (mean ± sd, n)" for _, label, _ in metrics
    ) + " | Points |"
    lines = [head, "| --- |" + " ---: |" * (4 * len(metrics) + 1)]
    for method in methods:
        cells, points = [], 0
        for metric, _, mode in metrics:
            fit = fit_curve(frame, method, metric, mode)
            if fit is None:
                cells += ["—"] * 4
                continue
            points = max(points, fit.num_points)
            seeds = seed_exponents(frame, method, metric, mode)
            if len(seeds) >= 2:
                spread = f"{np.mean(seeds):.3f} ± {np.std(seeds, ddof=1):.3f} ({len(seeds)})"
            elif seeds:
                spread = f"{seeds[0]:.3f} (1)"
            else:
                spread = "—"
            cells += [exponent_cell(fit), f"{fit.coefficient:.3f}", f"{fit.r_squared:.2f}", spread]
        if points:
            lines.append(f"| {display(method)} | " + " | ".join(cells) + f" | {points} |")
    return lines


def main() -> None:
    data = pd.read_csv(HERE / "results.csv")
    data = data[(data.status == "ok") & (data.regime == "full") & (data.variant == "default")]
    data["sample_size"] = data.num_train_sequences.fillna(data.sample_size).astype(int)

    out = [
        "# Full results: large-dataset scaling campaign",
        "",
        f"Every method, sample size, metric and scaling fit on full MBD, Synthea EHR and Amazon Beauty 2014. "
        f"Generated {date.today().isoformat()} from [`results.csv`](results.csv) "
        f"({len(data)} cells) by [`make_full_results.py`](make_full_results.py).",
        "",
        "**Interim.** The campaign is not closed. Cells still missing show as `—`. "
        "See [`../research/campaign_status.md`](../research/campaign_status.md) for what remains.",
        "",
        "How to read the tables:",
        "",
        "* The scaling matrices give the **mean over seeds** at each training-set size. "
        f"A number in parentheses is the seed count when fewer than {EXPECTED_SEEDS} seeds finished.",
        "* Rows are ranked by the metric at the dataset's full training pool (the last column).",
        "* The full-pool table gives **mean ± sample standard deviation** over seeds for every metric.",
        "* Paper-named methods are **approximations**. Read the `Fidelity` column and "
        "[`../research/method_catalog.md`](../research/method_catalog.md) before citing a number.",
        "* The per-seed values behind every cell are in [`results.csv`](results.csv).",
        "",
        "## Coverage",
        "",
        "| Dataset | Task | Methods | Sample sizes | Cells |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for name in DATASETS:
        for task in TASKS:
            frame = data[(data.dataset == name) & (data.task == task)]
            sizes = sorted(frame.sample_size.unique())
            out.append(
                f"| {DATASET_REGISTRY[name].display_name} | {task} | {frame.method.nunique()} | "
                f"{len(sizes)} ({sizes[0]}–{sizes[-1]}) | {len(frame)} |"
            )
    out.append("")
    out.append("Contents: " + " · ".join(
        f"[{DATASET_REGISTRY[n].display_name}](#{DATASET_REGISTRY[n].display_name.lower().replace(' ', '-').replace('(', '').replace(')', '')})"
        for n in DATASETS
    ) + " · [Scaling analysis](#scaling-analysis)")
    out += [""] + scaling_summary(data)

    for name in DATASETS:
        out += ["", f"## {DATASET_REGISTRY[name].display_name}"]
        for task in TASKS:
            frame = data[(data.dataset == name) & (data.task == task)]
            if frame.empty:
                continue
            sizes = sorted(frame.sample_size.unique())
            top = sizes[-1]
            primary = PRIMARY_METRIC[task]
            labels = {key: (label, higher) for key, label, higher in TASK_METRICS[task]}
            methods = method_order(frame, primary, top, labels[primary][1])
            task_label = "Classification" if task == "classification" else "Temporal point process (TPP)"
            out += [
                "",
                f"### {task_label}",
                "",
                f"#### All metrics at the full pool (n = {top})",
                "",
                *full_pool_table(frame, task, methods, top),
                "",
                "#### Scaling fits",
                "",
                "`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits "
                "each seed's own curve separately (four or more points) and gives the spread of `b` "
                "across seeds. `*` flags an unreliable fit.",
                "",
                *fit_detail_table(frame, methods, task),
            ]
            for key, label, higher in TASK_METRICS[task]:
                if frame[key].isna().all():
                    continue
                direction = "higher is better" if higher else "lower is better"
                out += [
                    "",
                    f"#### {label} vs. training sequences ({direction})",
                    "",
                    *matrix(frame.dropna(subset=[key]), key, methods, sizes),
                ]

    out.append("")
    (HERE / "full_results.md").write_text("\n".join(out))
    print("wrote", HERE / "full_results.md")


if __name__ == "__main__":
    main()
