"""Write comparison.md: ETAM on full MBD against the strongest benchmark baselines.

Reads the benchmark's published ``../benchmark_scale/results.csv`` (MBD at
n = 30,665) and the phase-B campaign's run directories, so it can be rerun as
cells finish; arms without results are listed as pending, never dropped.

    conda run -n KBC python docs/etam/make_comparison.py
"""

import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from eventfm.methods import METHOD_REGISTRY  # noqa: E402

CAMPAIGN = Path("/groups/bshapira_group/guyhada/campaigns/etam_mbd_phaseB_20261029")
BENCHMARK = HERE.parent / "benchmark_scale" / "results.csv"
OUT = HERE / "comparison.md"
FULL = "30665"
SEEDS = 3
TOP_BASELINES = 8

# (metric, label, higher is better, decimals)
CLS = [("auc", "AUC", True, 4), ("average_precision", "AP", True, 4)]
TPP = [
    ("next_type_accuracy", "next-type acc.", True, 4),
    ("next_type_macro_f1", "macro-F1", True, 4),
    ("delta_log_rmse", "time RMSE", False, 3),
    ("delta_log_mae", "time MAE", False, 3),
]
HEADLINE = {"classification": "auc", "tpp": "next_type_accuracy"}

# ETAM test arms: (label, pretraining data, run-dir glob). Pretrained arms use the
# 529,180-client unlabeled pool unless stated; all fine-tune on the 30,665
# labeled clients, exactly the baselines' training set.
BOTH, CLS_ONLY, TPP_ONLY = ("classification", "tpp"), ("classification",), ("tpp",)
ARMS = [
    ("ETAM-M multi (M1, headline)", "pool, 20 ep", "ft_main__ft_main__multi_s*", BOTH),
    ("ETAM-M cls (M2)", "pool, 20 ep", "ft_main__ft_main__cls_s*", CLS_ONLY),
    ("ETAM-M tpp (M2)", "pool, 20 ep", "ft_main__ft_main__tpp_s*", TPP_ONLY),
    ("ETAM-M frozen probe (M3)", "pool, 20 ep", "probe_main__probe_main__s*", BOTH),
    ("ETAM-M multi, HPO fine-tune (H2)", "pool, 20 ep", "confirm__confirm__pre_main_s*", BOTH),
    (
        "ETAM-M multi, HPO both stages (H1+H2)",
        "pool, 20 ep, HPO",
        "confirm__confirm__pre_final_s*",
        BOTH,
    ),
    ("ETAM-M multi (M4)", "30,665 labeled only", "ft_data__ft_data__matched_multi_s*", BOTH),
    ("ETAM-M cls (M4)", "30,665 labeled only", "ft_data__ft_data__matched_cls_s*", CLS_ONLY),
    ("ETAM-M tpp (M4)", "30,665 labeled only", "ft_data__ft_data__matched_tpp_s*", TPP_ONLY),
    ("ETAM-M multi, scratch", "none", "ft_scratch__ft_scratch__multi_s*", BOTH),
    ("ETAM-M cls, scratch", "none", "ft_scratch__ft_scratch__cls_s*", CLS_ONLY),
    ("ETAM-M tpp, scratch", "none", "ft_scratch__ft_scratch__tpp_s*", TPP_ONLY),
    ("ETAM Qwen3-0.6B + LoRA multi (L1)", "pool, 2 ep", "ft_llm__ft_llm__s*", BOTH),
]

# Pretraining variants: (label, what changes, pretraining run dir, fine-tune glob).
ABLATIONS = [
    ("M10 (reference)", "deep experts, token layout, abs+rel time, 128 events", "M10"),
    ("shared", "no experts (A1)", "shared"),
    ("ffn", "FFN experts only (A1)", "ffn"),
    ("attn_ffn", "attention + FFN experts (A1)", "attn_ffn"),
    ("additive", "time added to the event token (A2)", "additive"),
    (
        "tokshared_nocond",
        "token layout, shared weights, no cond loss (A2 control)",
        "tokshared_nocond",
    ),
    ("relonly", "relative time only (A3)", "relonly"),
    ("absonly", "absolute time only (A3)", "absonly"),
    ("ropetime", "index + continuous-time RoPE (A4)", "ropetime"),
    ("ev64", "64 events, window-relative time (A5)", "ev64"),
]
SCALES = [
    ("XS", "128 / 2", "pre_scale_small__pre_scale__XS", "ft_scale_small__ft_scale__XS_s*"),
    ("S", "192 / 4", "pre_scale_small__pre_scale__S", "ft_scale_small__ft_scale__S_s*"),
    ("M", "256 / 6", "pre_ablate__pre_ablate__M10", "ft_ablate__ft_ablate__M10_s*"),
    ("L", "512 / 8", "pre_scale_large__pre_scale__L", "ft_scale_large__ft_scale__L_s*"),
    ("XL", "768 / 12", "pre_scale_large__pre_scale__XL", "ft_scale_large__ft_scale__XL_s*"),
]
DATA = [
    ("30,665 labeled", "pre_data__pre_data__matched", "ft_data__ft_data__matched_multi_s*"),
    ("10 % pool (52,918)", "pre_data__pre_data__pool10pct", "ft_data__ft_data__pool10pct_multi_s*"),
    (
        "30 % pool (158,754)",
        "pre_data__pre_data__pool30pct",
        "ft_data__ft_data__pool30pct_multi_s*",
    ),
    ("100 % pool (529,180)", "pre_ablate__pre_ablate__M10", "ft_ablate__ft_ablate__M10_s*"),
]


def display(method: str) -> str:
    spec = METHOD_REGISTRY.get(method)
    return spec.display_name if spec else method


def flat(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, prefix + key + "."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix + key] = float(value)
    return out


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def retries() -> dict:
    """{task id: time of its latest resubmission request} from the driver's log."""

    out = {}
    path = CAMPAIGN / "retries.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip():
            request = json.loads(line)
            out[request["id"]] = max(out.get(request["id"], 0.0), request["t"])
    return out


RETRIES = retries()


def state(run: Path) -> str:
    if (run / "result.json").exists():
        return "done"
    status = read_json(run / "status.json") or {}
    ended = status.get("ended")
    # A failure older than its resubmission request is waiting in the queue
    # (or on its after: groups), not failed.
    if ended and RETRIES.get(run.name, 0) > datetime.fromisoformat(ended).timestamp():
        return "queued"
    return status.get("state", "unsubmitted")


def runs(pattern: str) -> list:
    """(run dir, flat result or None) for every planned task matching the glob."""

    planned = sorted(CAMPAIGN.glob("runs/" + pattern))
    return [(run, flat(read_json(run / "result.json") or {})) for run in planned]


class Stat:
    def __init__(self, values):
        self.values = [v for v in values if v is not None]
        self.n = len(self.values)
        self.mean = statistics.mean(self.values) if self.values else None
        self.sd = statistics.stdev(self.values) if self.n > 1 else 0.0

    def cell(self, decimals: int) -> str:
        if self.mean is None:
            return "–"
        if self.n == 1:
            return f"{self.mean:.{decimals}f}"
        return f"{self.mean:.{decimals}f} ± {self.sd:.{decimals}f}"


def benchmark() -> dict:
    """{(task, method): {metric: [per-seed values]}} at MBD n = 30,665."""

    out = {}
    with BENCHMARK.open() as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"],
                row["sample_size"],
                row["regime"],
                row["variant"],
                row["status"],
            ) != ("mbd", FULL, "full", "default", "ok"):
                continue
            metrics = out.setdefault((row["task"], row["method"]), {})
            for key, value in row.items():
                if value:
                    try:
                        metrics.setdefault(key, []).append(float(value))
                    except ValueError:
                        pass
    return out


def pool_baselines() -> dict:
    """The same {(task, method): {metric: [values]}} for the B1 pool-pretrained cells."""

    out = {}
    for run, result in runs("baselines_pool__baseline__c*"):
        raw = read_json(run / "result.json") or {}
        if "cell_key" not in raw:
            continue
        _, task, method, *_ = raw["cell_key"].split("__")
        metrics = out.setdefault((task, method), {})
        for key, value in result.items():
            metrics.setdefault(key, []).append(value)
    return out


def ranked(table: dict, task: str) -> list:
    metric = HEADLINE[task]
    methods = [m for (t, m) in table if t == task and table[(t, m)].get(metric)]
    return sorted(methods, key=lambda m: -statistics.mean(table[(task, m)][metric]))


def best(table: dict, task: str, metric: str, higher: bool):
    """(method, Stat) of the best benchmark method on one metric."""

    stats = {m: Stat(table[(t, m)].get(metric, [])) for (t, m) in table if t == task}
    stats = {m: s for m, s in stats.items() if s.mean is not None}
    method = (max if higher else min)(stats, key=lambda m: stats[m].mean)
    return method, stats[method]


def verdict(arm: Stat, ref: Stat, higher: bool) -> str:
    """Pre-registered rule: a win needs a gap larger than the seed spread (both arms' sd)."""

    if arm.mean is None:
        return "–"
    delta = (arm.mean - ref.mean) * (1 if higher else -1)
    spread = max(arm.sd, ref.sd)
    sign = f"{arm.mean - ref.mean:+.4f}"
    if arm.n < 2:
        return f"{sign} (n = 1)"
    if delta > spread:
        return f"{sign} **better**"
    if delta < -spread:
        return f"{sign} worse"
    return f"{sign} tie"


def etam_stats(results: list, key: str) -> Stat:
    return Stat([r.get(key) for _, r in results if r])


def n_cell(results: list) -> str:
    done = sum(1 for _, r in results if r)
    if not results:
        return "0/0"
    if done == len(results):
        return str(done)
    states = sorted({state(run) for run, r in results if not r})
    return f"{done}/{len(results)} ({', '.join(states)})"


def task_table(task: str, bench: dict, pool: dict) -> list:
    metrics = CLS if task == "classification" else TPP
    headline = HEADLINE[task]
    ref_method, ref = best(bench, task, headline, True)
    lines = [
        "| Model | Pretraining | "
        + " | ".join(label for _, label, _, _ in metrics)
        + f" | {metrics[0][1]} at final step | n | Δ {metrics[0][1]} vs {display(ref_method)} |",
        "| --- | --- |" + " ---: |" * (len(metrics) + 2) + " --- |",
    ]
    for label, data, pattern, tasks in ARMS:
        if task not in tasks:
            continue
        results = runs(pattern)
        stat = {m: etam_stats(results, f"{task}.{m}") for m, *_ in metrics}
        final = etam_stats(results, f"final_step.{task}.{headline}")
        lines.append(
            f"| {label} | {data} | "
            + " | ".join(stat[m].cell(d) for m, _, _, d in metrics)
            + f" | {final.cell(4)} | {n_cell(results)} | {verdict(stat[headline], ref, True)} |"
        )
    for method in ranked(pool, task):
        values = pool[(task, method)]
        stat = {m: Stat(values.get(m, [])) for m, *_ in metrics}
        lines.append(
            f"| {display(method)} (B1) | pool, 1 ep | "
            + " | ".join(stat[m].cell(d) for m, _, _, d in metrics)
            + f" | {stat[headline].cell(4)} | {stat[headline].n} | "
            + f"{verdict(stat[headline], ref, True)} |"
        )
    for method in ranked(bench, task)[:TOP_BASELINES]:
        values = bench[(task, method)]
        stat = {m: Stat(values.get(m, [])) for m, *_ in metrics}
        lines.append(
            f"| {display(method)} | per method | "
            + " | ".join(stat[m].cell(d) for m, _, _, d in metrics)
            + f" | {stat[headline].cell(4)} | {stat[headline].n} | "
            + ("reference |" if method == ref_method else f"{verdict(stat[headline], ref, True)} |")
        )
    return lines


def best_line(bench: dict) -> list:
    lines = [
        "| Task | Metric | Best benchmark method | Mean ± sd (n = 3) |",
        "| --- | --- | --- | ---: |",
    ]
    for task, metrics in (("classification", CLS[:1]), ("tpp", TPP)):
        for metric, label, higher, decimals in metrics:
            method, stat = best(bench, task, metric, higher)
            lines.append(f"| {task} | {label} | {display(method)} | {stat.cell(decimals)} |")
    return lines


def headline_table(bench: dict) -> list:
    _, ref_auc = best(bench, "classification", "auc", True)
    _, ref_acc = best(bench, "tpp", "next_type_accuracy", True)
    lines = [
        "| ETAM arm (one model for both tasks) | AUC | Δ AUC | next-type acc. | Δ acc. | n "
        "| both better? |",
        "| --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for label, _, pattern, tasks in ARMS:
        if tasks != BOTH:
            continue
        results = runs(pattern)
        auc = etam_stats(results, "classification.auc")
        acc = etam_stats(results, "tpp.next_type_accuracy")
        v_auc, v_acc = verdict(auc, ref_auc, True), verdict(acc, ref_acc, True)
        both = (
            "–"
            if auc.mean is None or acc.mean is None
            else ("**yes**" if "better" in v_auc and "better" in v_acc else "no")
        )
        lines.append(
            f"| {label} | {auc.cell(4)} | {v_auc} | {acc.cell(4)} | {v_acc} | "
            f"{n_cell(results)} | {both} |"
        )
    return lines


PRE_COLS = [
    ("valid.next_type_accuracy", "next-type acc.", 4),
    ("valid.next_type_macro_f1", "macro-F1", 4),
    ("valid.delta_log_rmse", "time RMSE", 3),
    ("valid.delta_log_mae", "time MAE", 3),
    ("valid.joint_nll", "joint NLL", 3),
    ("train_minutes", "GPU min", 0),
]


def pre_cells(run_id: str) -> str:
    run = CAMPAIGN / "runs" / run_id
    result = flat(read_json(run / "result.json") or {})
    if not result:
        return " | ".join(["–"] * len(PRE_COLS)) + f" | {state(run)}"
    return (
        " | ".join(f"{result[k]:.{d}f}" if k in result else "–" for k, _, d in PRE_COLS) + " | done"
    )


def ft_cells(pattern: str) -> str:
    results = runs(pattern)
    auc = etam_stats(results, "classification.auc")
    acc = etam_stats(results, "tpp.next_type_accuracy")
    return f"{auc.cell(4)} | {acc.cell(4)} | {n_cell(results)}"


PRE_HEAD = " | ".join(label for _, label, _ in PRE_COLS) + " | state"
FT_HEAD = "test AUC (multi) | test next-type acc. (multi) | n"


def ablation_table() -> list:
    lines = [
        f"| Variant | Change | {PRE_HEAD} | {FT_HEAD} |",
        "| --- | --- |" + " ---: |" * len(PRE_COLS) + " --- | ---: | ---: | --- |",
    ]
    for label, change, variant in ABLATIONS:
        lines.append(
            f"| {label} | {change} | {pre_cells('pre_ablate__pre_ablate__' + variant)} | "
            f"{ft_cells('ft_ablate__ft_ablate__' + variant + '_s*')} |"
        )
    return lines


def scale_table() -> list:
    lines = [
        f"| Size | Hidden / layers | {PRE_HEAD} | {FT_HEAD} |",
        "| --- | --- |" + " ---: |" * len(PRE_COLS) + " --- | ---: | ---: | --- |",
    ]
    for size, arch, pre, ft in SCALES:
        lines.append(f"| {size} | {arch} | {pre_cells(pre)} | {ft_cells(ft)} |")
    return lines


def data_table() -> list:
    lines = [
        f"| Unlabeled pretraining data | {PRE_HEAD} | {FT_HEAD} |",
        "| --- |" + " ---: |" * len(PRE_COLS) + " --- | ---: | ---: | --- |",
    ]
    for label, pre, ft in DATA:
        lines.append(f"| {label} | {pre_cells(pre)} | {ft_cells(ft)} |")
    return lines


def other_pretraining_table() -> list:
    lines = [
        f"| Run | {PRE_HEAD} |",
        "| --- |" + " ---: |" * len(PRE_COLS) + " --- |",
        f"| ETAM-M, 20 pool epochs (`pre_main`) | {pre_cells('pre_main__pre_main')} |",
        f"| ETAM-M, 20 epochs, HPO recipe (`pre_final`) | {pre_cells('pre_final__pre_final')} |",
        "| Qwen3-0.6B-Base + LoRA r = 64, 2 pool epochs (`pre_llm`) | "
        f"{pre_cells('pre_llm__pre_llm')} |",
    ]
    return lines


def hpo_table() -> list:
    lines = [
        "| Trial | valid probe AUC | valid zero-shot next-type acc. | score (sum) "
        "| hyper-parameters |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    rows = []
    for run, result in runs("hpo_pre__hpo_pre__t*"):
        task = read_json(run / "task.json") or {}
        hp = task.get("params", {}).get("hp", "")
        rows.append((result.get("valid.score"), run.name.rsplit("__", 1)[-1], result, hp))
    rows.sort(key=lambda row: -(row[0] or -1))
    for score, trial, result, hp in rows:
        if score is None:
            lines.append(f"| {trial} | – | – | – | `{hp}` |")
            continue
        lines.append(
            f"| {trial} | {result['valid.probe_auc']:.4f} | "
            f"{result['valid.tpp_next_type_accuracy']:.4f} | "
            f"{score:.4f} | `{hp}` |"
        )
    return lines


def main() -> None:
    bench, pool = benchmark(), pool_baselines()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = [
        "# ETAM vs strong baselines on full MBD",
        "",
        f"Generated {stamp} by `make_comparison.py` from the campaign run directories "
        f"(`{CAMPAIGN}`) and the benchmark's `../benchmark_scale/results.csv`. Do not edit by "
        "hand; rerun the script. Commentary is in [results.md](results.md); the protocol is in "
        "[methodology.md](methodology.md).",
        "",
        "**How to read the tables.**",
        "",
        "- Every test number uses the benchmark's own examples: 40,000 test histories at natural "
        "prevalence for classification, and the benchmark's 320,000 TPP prefixes (seed + 1 replay) "
        "for TPP. All models are trained on the same 30,665 labeled clients.",
        "- Cells are mean ± sample sd over seeds 13, 29 and 43; `n` counts finished seeds.",
        "- ETAM selects its checkpoint on validation (early stopping, patience 3); the baselines "
        "train a fixed 8 epochs with no selection. The *at final step* column is where ETAM "
        "stopped, which is by construction up to 3 non-improving epochs past its best: a "
        "pessimistic bound, **not** a like-for-like number. Like-for-like would need the "
        "baselines rerun with the same validation selection.",
        "- **better** / worse means the gap exceeds the larger of the two arms' seed sd (the "
        "pre-registered rule); otherwise *tie*. With n = 3 this is a screening rule, not a test.",
        "- Every pretrained arm starts from **one** pretraining run (seed 13), so its sd covers "
        "fine-tuning noise only. The frozen probe's AUC is identical across seeds (one "
        "checkpoint, deterministic logistic regression); its seeds vary only the TPP prefixes.",
        "- Time RMSE and MAE are on log1p seconds: lower is better.",
        "- Pretraining-stage numbers are on 10,000 **validation** users with the pretrained "
        "heads and no fine-tuning. Use them to compare pretraining runs with each other, not with "
        "test numbers.",
        "",
        "## Best benchmark baseline per metric (MBD, n = 30,665)",
        "",
        *best_line(bench),
        "",
        "## Headline: one model, best on both tasks?",
        "",
        "Pre-registered pass rule: an arm must beat the best baseline on AUC **and** on next-type "
        "accuracy by more than the seed spread.",
        "",
        *headline_table(bench),
        "",
        "## Classification (test AUC)",
        "",
        f"ETAM arms, then the four baselines pretrained on the same unlabeled pool (B1), then the "
        f"{TOP_BASELINES} best benchmark methods (the full table is in "
        "[`../benchmark_scale/results.md`](../benchmark_scale/results.md)). Arms with no "
        "classification output (`tpp` fine-tunes) are omitted.",
        "",
        *task_table("classification", bench, pool),
        "",
        "## TPP (test next event)",
        "",
        *task_table("tpp", bench, pool),
        "",
        "## Pretraining ablations (A1–A5): 10 pool epochs each",
        "",
        "Pretraining columns are validation, zero-shot. The fine-tune columns are test, `multi` "
        "task, 3 seeds.",
        "",
        *ablation_table(),
        "",
        "## Model scale (S1): 10 pool epochs each",
        "",
        *scale_table(),
        "",
        "## Unlabeled data (S2): fixed 20,670 steps",
        "",
        *data_table(),
        "",
        "## Other pretraining runs",
        "",
        *other_pretraining_table(),
        "",
        "## Pretraining HPO (H1): 10 pool epochs, frozen probe on validation",
        "",
        *hpo_table(),
        "",
    ]
    OUT.write_text("\n".join(doc))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
