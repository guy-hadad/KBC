"""Write full_results.md: every method, sample size and metric on the scale datasets.

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

from eventfm.benchmark.report import PRIMARY_METRIC, TASK_METRICS  # noqa: E402
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


def main() -> None:
    data = pd.read_csv(HERE / "results.csv")
    data = data[(data.status == "ok") & (data.regime == "full") & (data.variant == "default")]
    data["sample_size"] = data.num_train_sequences.fillna(data.sample_size).astype(int)

    out = [
        "# Full results: large-dataset scaling campaign",
        "",
        f"Every method, sample size and metric on full MBD, Synthea EHR and Amazon Beauty 2014. "
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
    ))

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
