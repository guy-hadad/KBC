"""Write Stage 2 tables into stage2/results.md from the run directories.

    conda run -n KBC python docs/etam/make_stage2_tables.py

Every cell is mean ± sample sd over pretraining seeds (``n`` = finished/planned);
missing cells are shown as pending, never dropped. Deltas are against the ETAM-2
reference (``ref``) and are marked * when they exceed the larger of the two arms'
sd (the pre-registered screening rule; single-seed arms are never starred).
"""

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = Path("/groups/bshapira_group/guyhada/campaigns/etam2_mbd_20261108/runs")
TRANSFER = Path("/groups/bshapira_group/guyhada/campaigns/etam2_transfer_20261108/runs")
PHASE_B = Path("/groups/bshapira_group/guyhada/campaigns/etam_mbd_phaseB_20261029/runs")
BENCHMARK = HERE.parent / "benchmark_scale" / "results.csv"
DOC = HERE / "stage2" / "results.md"
CAMPAIGNS = {
    "MBD": ROOT.parent,
    "Transfer": TRANSFER.parent,
    "Data preparation": ROOT.parents[1] / "etam2_cpuprep_20261108",
    "Pilot": ROOT.parents[1] / "etam2_pilot_20261108",
}
START, END = "<!-- AUTO:stage2:START -->", "<!-- AUTO:stage2:END -->"


def get(payload, dotted):
    for part in dotted.split("."):
        if not isinstance(payload, dict) or part not in payload:
            return None
        payload = payload[part]
    return payload if isinstance(payload, (int, float)) and not isinstance(payload, bool) else None


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def task_result(directory):
    """Campaign result_path writes at task root; older layouts use r/result.json."""

    return load_json(directory / "result.json") or load_json(directory / "r" / "result.json")


def collect_runs():
    """name -> result for every bundled run (pre_main, pre, scale, ft) and single tasks."""

    runs = {}
    for group in ("pre_main", "pre", "scale", "ft"):
        for path in glob.glob(str(ROOT / "{}__b*".format(group) / "*" / "result.json")):
            runs[Path(path).parent.name] = load_json(path)
    for directory in ROOT.glob("pre_s1__*"):
        runs[directory.name.replace("pre_s1__", "")] = task_result(directory)
    for directory in ROOT.glob("eval_s1__*"):
        runs["phaseB_" + directory.name.replace("eval_s1__", "")] = task_result(directory)
    return {name: value for name, value in runs.items() if value}


def arm_of(name):
    head, _, seed = name.rpartition("_s")
    return (head, int(seed)) if seed.isdigit() else (name, None)


#: Phase B (seed 13) checkpoints re-scored with the Stage 2 battery are seed 13 of the Stage 1 arms.
ALIASES = {"phaseB_M10": "s1_M10", "phaseB_pool10": "s1_pool10", "phaseB_pool30": "s1_pool30"}


def by_arm(runs):
    arms = defaultdict(dict)
    for name, result in runs.items():
        arm, seed = arm_of(name)
        arms[arm][seed] = result
        if name in ALIASES:
            arms[ALIASES[name]][13] = result
    return arms


def stat(results, metric, scale=1.0):
    values = [get(r, metric) for r in results.values()]
    if metric == "parameters":
        values = [value if value is not None else get(r, "efficiency.parameters")
                  for r, value in zip(results.values(), values)]
    if metric.startswith("valid."):
        values = [value if value is not None else get(r, metric.replace("valid.", "validation.tpp.", 1))
                  for r, value in zip(results.values(), values)]
    values = [v * scale for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd, len(values)


def fmt(cell, digits=4, planned=None):
    if cell is None:
        return "pending"
    mean, sd, n = cell
    text = "{:.{d}f}".format(mean, d=digits)
    if n > 1:
        text += " ± {:.{d}f}".format(sd, d=digits)
    if planned and n < planned:
        text += " ({}/{})".format(n, planned)
    return text


def delta(cell, base, digits=4, lower_better=False):
    if cell is None or base is None:
        return ""
    diff = cell[0] - base[0]
    star = "*" if cell[2] > 1 and base[2] > 1 and abs(diff) > max(cell[1], base[1]) else ""
    return "{:+.{d}f}{}".format(diff, star, d=digits)


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


# Column sets: (metric path, label, digits, lower is better)
ZERO_SHOT = [
    ("valid.joint_nll", "valid joint NLL", 4, True),
    ("test.tpp.joint_nll", "test joint NLL", 4, True),
    ("test.tpp.time_nll", "time NLL", 4, True),
    ("test.tpp.type_nll", "type NLL", 4, True),
    ("test.tpp.next_type_accuracy", "next-type acc.", 4, False),
    ("test.tpp.next_type_macro_f1", "macro-F1", 4, False),
    ("test.tpp.delta_log_rmse", "RMSE", 3, True),
    ("test.tpp.delta_log_mae", "MAE", 3, True),
    ("probe.user_standard.auc", "probe AUC", 4, False),
    ("probe.user_strict.auc", "probe AUC strict", 4, False),
]
CALIBRATION = [
    ("test.tpp.pit_ks", "PIT KS", 4, True),
    ("test.tpp.pit_pce", "PIT PCE", 4, True),
    ("test.tpp.coverage80", "cov. 80 %", 3, False),
    ("test.tpp.survival_ece", "survival ECE", 4, True),
    ("test.tpp.mark_ece", "mark ECE", 4, True),
    ("test.tpp.pmi_mean", "E[PMI]", 4, False),
    ("test.tpp.oracle_time_accuracy", "oracle-time acc.", 4, False),
    ("test.tpp.cheap_accuracy", "cheap-branch acc.", 4, False),
]
PRIVATE = [
    ("test.tpp.last_event_joint_nll", "joint NLL (last event)", 4, True),
    ("test.tpp.unimodal_type_nll", "event-only type NLL", 4, True),
    ("test.tpp.unimodal_time_nll", "time-only time NLL", 4, True),
    ("test.tpp.unimodal_accuracy", "event-only acc.", 4, False),
]
G_TERMS = [
    ("test.elapsed.direct_joint_nll", "silence: direct NLL", 4, True),
    ("test.elapsed.survival_joint_nll", "silence: survival NLL", 4, True),
    ("test.counts.head_30d.crps", "30 d count CRPS (head)", 3, True),
    ("test.counts.history_poisson_30d.crps", "30 d CRPS (history Poisson)", 3, True),
    ("test.counts.head_7d.crps", "7 d CRPS (head)", 3, True),
]
MTP = [
    ("test.tpp.mtp_accuracy_2", "acc. k(n+2)", 4, False),
    ("test.tpp.mtp_accuracy_3", "acc. k(n+3)", 4, False),
    ("test.tpp.mtp_accuracy_4", "acc. k(n+4)", 4, False),
    ("test.tpp.mtp_time_nll_2", "cum-time NLL (n+2)", 4, True),
]
EFFICIENCY = [
    ("parameters", "params (M)", 2, True),
    ("efficiency.gflops_per_sequence", "GFLOPs/seq", 2, True),
    ("efficiency.latency_ms", "run-device ms/256", 1, True),
]

SECTIONS = [
    ("P0: reference and Stage 1 at the same budget", ["ref", "s1_M10", "phaseB_M10", "phaseB_M20", "phaseB_final", "ev64"]),
    ("C: middle architecture", ["ref", "a_shared", "a_shared_wide", "a_stage1", "a_private", "a_private_shared",
                               "a_bridges_all", "a_nobias", "a_lagonly", "a_nogate"]),
    ("C: external baselines in this framework", ["ref", "b_s2p2", "b_itpp", "b_flextpp", "b_kairos"]),
    ("A: time representation and temporal memory", ["ref", "t_stage1", "t_nomem", "t_bins", "t_mem_learned", "t_mem_count",
                                                   "t_mem_ssm", "t_mem_ode", "t_cal_fixed", "t_relonly", "t_vq"]),
    ("D: decoders and time likelihoods", ["ref", "d_indep", "d_time_first", "d_event_first", "d_nocons", "d_cons1",
                                         "d_spline", "d_hazard"]),
    ("G: pretraining-strategy terms", ["ref", "g_nosurv", "g_nocensor", "g_poisson", "g_count_stage1"]),
    ("E/F: multi-event prediction and predictive representation", ["ref", "m_hard4", "m_soft4", "m_soft2", "m_soft8",
                                                                   "m_soft4_time", "r_all", "r_joint", "r_time",
                                                                   "u_event_only", "u_time_only"]),
    ("G 8.1: staged curriculum (cumulative)", ["c0", "c1", "c2", "ref", "c3", "c4", "c5", "c6"]),
    ("G 8.5: data scaling (A: 20,670 steps; B: 10 epochs per user)", ["scaleA_matched", "scaleA_pool10", "scaleA_pool30", "ref",
                                                                       "scaleB_matched", "scaleB_pool10", "scaleB_pool30",
                                                                       "s1_pool10", "s1_pool30", "s1_M10",
                                                                       "phaseB_matched", "phaseB_pool10", "phaseB_pool30"]),
]
PLANNED = defaultdict(lambda: 1)
for arm in ("ref", "a_shared", "a_stage1", "a_private", "a_private_shared", "a_bridges_all", "d_indep", "d_time_first",
            "d_event_first", "t_stage1", "t_nomem", "m_soft4", "m_hard4", "r_all", "c6", "b_s2p2", "b_itpp", "b_flextpp",
            "b_kairos", "scaleA_matched", "scaleA_pool10", "scaleA_pool30", "scaleB_matched", "scaleB_pool10",
            "scaleB_pool30"):
    PLANNED[arm] = 3
PLANNED["s1_M10"] = 3
PLANNED["s1_pool10"] = PLANNED["s1_pool30"] = 3


def metric_table(arms, names, columns, ref="ref"):
    base = arms.get(ref, {})
    rows = []
    for name in names:
        results = arms.get(name, {})
        row = ["`{}`".format(name)]
        for metric, _, digits, lower in columns:
            cell = stat(results, metric, 1e-6 if metric == "parameters" else 1.0)
            text = fmt(cell, digits, PLANNED[name])
            if name != ref and base and metric not in ("parameters",) and not metric.startswith("efficiency"):
                d = delta(cell, stat(base, metric), digits, lower)
                text += " ({})".format(d) if d else ""
            row.append(text)
        rows.append(row)
    return table(rows, ["arm"] + [label for _, label, _, _ in columns])


def finetune_tables(runs):
    lines = []
    groups = defaultdict(dict)
    for name, result in runs.items():
        arm, seed = arm_of(name)
        if arm.split("_")[0] in ("ref", "s1") and arm.count("_") >= 2:
            groups[arm][seed] = result
    cls_cols = [("classification.auc", "AUC", 4, False), ("classification.average_precision", "AP", 4, False),
                ("classification.precision@500", "P@500", 4, False), ("classification.recall@2000", "R@2000", 4, False),
                ("classification.ece", "ECE (prior-corr.)", 4, True)]
    tpp_cols = [("tpp.next_type_accuracy", "next-type acc.", 4, False), ("tpp.next_type_macro_f1", "macro-F1", 4, False),
                ("tpp.joint_nll", "joint NLL", 4, True), ("tpp.delta_log_rmse", "RMSE", 3, True),
                ("tpp.delta_log_mae", "MAE", 3, True)]
    rows = []
    for arm in sorted(groups):
        results = groups[arm]
        row = ["`{}`".format(arm)] + [fmt(stat(results, m), d, 3) for m, _, d, _ in cls_cols + tpp_cols]
        conflict = stat(results, "grad_conflict.mean_cosine")
        row.append(fmt(conflict, 3) if conflict else "")
        rows.append(row)
    if not rows:
        return ["No completed fine-tunes or adaptation runs; all planned comparisons remain pending."]
    lines += table(rows, ["fine-tune (model_task_protocol)"] + [c[1] for c in cls_cols + tpp_cols] + ["grad cosine"])
    return lines


def probe_tables():
    lines = []
    rows = []
    for path in sorted(glob.glob(str(ROOT / "probe__*" / "result.json"))):
        result = load_json(path) or {}
        arm = Path(path).parent.name.replace("probe__", "")
        for key, entry in sorted(result.items()):
            cells = []
            for fraction in ("0.01", "0.05", "0.1", "0.25", "1"):
                lin = entry.get("logistic@" + fraction, {}).get("auc")
                mlp = entry.get("mlp@" + fraction, {}).get("auc")
                cells.append("{} / {}".format("{:.4f}".format(lin) if lin else "–", "{:.4f}".format(mlp) if mlp else "–"))
            targets = " ".join("{}={:.3f}".format(t, entry["target_" + t]["auc"]) for t in ("cred", "zp", "acquiring")
                               if "target_" + t in entry and entry["target_" + t].get("auc"))
            rows.append(["`{}`".format(arm), key] + cells + [targets])
    if not rows:
        return ["No completed label-efficiency or cross-target probe jobs. Full-label logistic probes from the "
                "pretraining evaluations are included in the tables above and the complete metric appendix."]
    lines += table(rows, ["model", "protocol.representation", "1 % (lin / MLP)", "5 %", "10 %", "25 %", "100 %",
                          "other MBD targets (AUC, 100 %)"])
    return lines


def baseline_series():
    cells = defaultdict(dict)
    for name in ("base_strict", "base_pool10"):
        for path in glob.glob(str(ROOT / "{}__c*".format(name) / "cells" / "*.json")):
            payload = load_json(path) or {}
            if payload.get("error") or not payload.get("metrics"):
                continue
            cell = payload.get("cell", {})
            metrics = payload.get("metrics", {})
            key = (name, cell.get("dataset"), cell.get("task"), cell.get("method"), cell.get("variant"))
            cells[key][cell.get("seed")] = metrics
    return cells


def baseline_tables():
    rows = []
    cells = baseline_series()
    for method in ("pragma-mlm", "transaction-mlm", "coles", "autoregressive-transformer"):
        for variant in ("default", "pool"):
            cells.setdefault(("base_strict", "mbd_strict", "classification", method, variant), {})
    for method in ("pragma-mlm", "autoregressive-transformer"):
        for task in ("classification", "tpp"):
            cells.setdefault(("base_pool10", "mbd", task, method, "pool10"), {})
    for key, series in sorted(cells.items()):
        metric = "auc" if key[2] == "classification" else "next_type_accuracy"
        row = [key[0], key[1], key[2], key[3], key[4], fmt(stat(series, metric), 4, 3)]
        if key[2] == "tpp":
            row.append(fmt(stat(series, "delta_log_rmse"), 3))
        else:
            row.append(fmt(stat(series, "average_precision"), 4))
        rows.append(row)
    return table(rows, ["campaign", "dataset", "task", "method", "variant", "AUC / next-type acc.", "AP / RMSE"])


def benchmark_best():
    if not BENCHMARK.exists():
        return {}
    best = {}
    with BENCHMARK.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "mbd" or str(row.get("sample_size")) != "30665":
                continue
            for metric in ("auc", "next_type_accuracy"):
                value = row.get(metric) or row.get("metric_" + metric)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                best.setdefault(metric, []).append((value, row.get("method")))
    return best


def simulation_table():
    rows = []
    for directory in sorted(ROOT.glob("sim__*")):
        result = task_result(directory)
        if not result:
            continue
        model = directory.name.replace("sim__", "")
        row = ["`{}`".format(model)]
        for horizon in ("7d", "30d"):
            sim = result.get("sim_" + horizon, {})
            base = result.get("history_poisson_" + horizon, {})
            row += ["{:.2f} / {:.2f}".format(sim.get("crps", float("nan")), base.get("crps", float("nan"))),
                    "{:.2f} / {:.2f}".format(sim.get("coverage80", float("nan")), base.get("coverage80", float("nan")))]
        row += ["{:.4f}".format(get(result, "type_js.js") or float("nan")),
                "{:.3f}".format(get(result, "log_gap_distance.w1") or float("nan"))]
        rows.append(row)
    if not rows:
        return ["No completed simulation runs; the rollout count and horizon below are planned, not achieved."]
    return table(rows, ["model", "7 d CRPS (sim / Poisson)", "7 d cov. 80 %", "30 d CRPS", "30 d cov. 80 %",
                        "mark-mix JS", "log-gap W1"])


def transfer_series():
    results = defaultdict(dict)
    for path in glob.glob(str(TRANSFER / "etam2__*" / "*" / "result.json")):
        name = Path(path).parent.name
        arm, seed = arm_of(name)
        result = load_json(path)
        if result:
            results["ETAM-2 " + arm][seed] = result
    for directory in TRANSFER.glob("stage1__*"):
        result = task_result(directory)
        if not result:
            continue
        task = directory.name
        dataset = "syn" if "synthea" in task else "so"
        seed = int(task.rsplit("seed-", 1)[-1].split("_")[0]) if "seed-" in task else len(results["Stage 1 " + dataset])
        results["Stage 1 " + dataset][seed] = result
    return results


def transfer_table():
    rows = []
    results = transfer_series()
    for model in ("ETAM-2 syn", "ETAM-2 so", "Stage 1 syn", "Stage 1 so"):
        results.setdefault(model, {})
    for arm in sorted(results):
        series = results[arm]
        rows.append([arm] + [fmt(stat(series, m), d, 3) for m, _, d, _ in ZERO_SHOT[1:8]] +
                    [fmt(stat(series, "probe.user_standard.auc"), 4, 3)])
    return table(rows, ["model / dataset"] + [label for _, label, _, _ in ZERO_SHOT[1:8]] + ["probe AUC"])


def latency_table():
    result = load_json(ROOT / "latency__latency" / "result.json") or {}
    rows = []
    for name, entry in sorted(result.items()):
        if "error" in entry:
            continue
        rows.append([name, "{:.2f}".format(entry.get("parameters", 0) / 1e6), "{:.2f}".format(entry.get("gflops_per_sequence", 0)),
                     "{:.1f}".format(entry.get("latency_ms", float("nan")))])
    if not rows:
        return ["No completed common-device latency job. Per-run parameter and FLOP counts are reported above; "
                "run-device latencies are also retained, but are not a controlled cross-model comparison."]
    return table(rows, ["checkpoint", "params (M)", "GFLOPs/seq", "ms per 256 sequences (RTX 4090)"])


def status_tables():
    """Read recorded task states, without treating an old harvest as live scheduler state."""

    lines = ["### Recorded campaign status", "",
             "Read from each manifest and the latest `runs/<task>/status.json`. These are recorded states, "
             "not a live Slurm query. A task with no status has no recorded completion. "
             "Bundle tasks contain up to three model runs; task counts are not seed counts.", ""]
    for label, campaign in CAMPAIGNS.items():
        manifest = campaign / "manifest.jsonl"
        if not manifest.exists():
            continue
        counts = defaultdict(lambda: defaultdict(int))
        for line in manifest.read_text().splitlines():
            task = json.loads(line)
            status = load_json(campaign / "runs" / task["id"] / "status.json") or {}
            counts[task["group"]][status.get("state", "no status")] += 1
        lines += ["#### {}: `{}`".format(label, campaign.name), ""]
        rows = [[group, sum(states.values())] + [states.get(state, 0)
                for state in ("done", "failed", "terminated", "running", "no status")]
                + [", ".join("{}={}".format(s, n) for s, n in sorted(states.items())
                             if s not in {"done", "failed", "terminated", "running", "no status"})]
                for group, states in counts.items()]
        lines += table(rows, ["group", "planned tasks", "done", "failed", "terminated", "running", "no status", "other"]) + [""]
    return lines


def numeric_leaves(value, prefix=""):
    """Include every numeric metric, including array diagnostics such as PIT histograms."""

    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, "{}.{}".format(prefix, key).lstrip("."))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, "{}[{}]".format(prefix, index))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, value


def complete_metrics(arms):
    lines = ["### Complete recorded metrics", "",
             "Each expandable table includes all finite numeric fields in its result payloads, including "
             "validation, test strata, calibration, count scores, probe ranking metrics and training costs. "
             "`n` counts available seed values for that metric; sd is sample sd and is undefined for n=1. "
             "Stage 1 seed-13 aliases are listed once as `phaseB_*`. `syn` = Synthea; `so` = Stack Overflow.", ""]
    series = [("MBD " + arm, results) for arm, results in sorted(arms.items())
              if not (arm in ALIASES.values() and len(results) == 1 and 13 in results)]
    series += [("Transfer " + arm, results) for arm, results in sorted(transfer_series().items())]
    series += [(" / ".join(key), results) for key, results in sorted(baseline_series().items())]
    for label, campaign in CAMPAIGNS.items():
        if label not in ("Data preparation", "Transfer"):
            continue
        group = "data__*" if label == "Data preparation" else "stores__*"
        for directory in sorted((campaign / "runs").glob(group)):
            result = task_result(directory)
            if result:
                series.append((label + " " + directory.name, {None: result}))
    for label, results in series:
        fields = defaultdict(list)
        for result in results.values():
            for metric, value in numeric_leaves(result):
                fields[metric].append(value)
        lines += ["<details>", "<summary>{} ({} result payloads)</summary>".format(label, len(results)), ""]
        rows = []
        for metric, values in sorted(fields.items()):
            rows.append(["`{}`".format(metric), "{:.9g}".format(statistics.fmean(values)),
                         "{:.9g}".format(statistics.stdev(values)) if len(values) > 1 else "—", len(values)])
        lines += table(rows, ["metric", "mean / single value", "sample sd", "n"]) + ["", "</details>", ""]
    return lines


def main() -> None:
    runs = collect_runs()
    arms = by_arm(runs)
    lines = [START, "", "Generated {} by `make_stage2_tables.py` from `{}`. Cells: mean ± sd over pretraining seeds "
             "(`(n/planned)` while incomplete); `(Δ)` against `ref`, `*` beyond the larger sd. Test numbers use the "
             "benchmark's examples; `valid` is 10,000 validation users. Missing metrics are pending or "
             "not applicable, not zero. Stars indicate magnitude, not significance or improvement; "
             "coverage is assessed against its nominal level. Stage 1 legacy arms have one pretraining seed; "
             "20-epoch and HPO arms have a larger budget than M10. Parameter/FLOP counts alone do not "
             "establish matched compute.".format(datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d %H:%M %Z"), ROOT), ""]
    lines += status_tables()
    for title, names in SECTIONS:
        lines += ["#### {}".format(title), ""]
        lines += metric_table(arms, names, ZERO_SHOT) + [""]
        if title.startswith(("P0", "C:", "A:", "D:")):
            lines += metric_table(arms, names, CALIBRATION + EFFICIENCY) + [""]
        if title.startswith(("C:", "D:", "A:")):
            lines += metric_table(arms, names, PRIVATE) + [""]
        if title.startswith(("G:", "P0")):
            lines += metric_table(arms, names, G_TERMS) + [""]
        if title.startswith("E/F"):
            lines += metric_table(arms, names, MTP) + [""]
    lines += ["#### Fine-tunes and adaptation (test; fine-tune seed = pretraining seed)", ""] + finetune_tables(runs) + [""]
    lines += ["#### Frozen probes: label efficiency and cross-target transfer (test AUC)", ""] + probe_tables() + [""]
    lines += ["#### JSONL baselines: strict labels and matched pool compute (test)", ""] + baseline_tables() + [""]
    lines += ["#### Long-horizon simulation (4,000 test users × 16 rollouts, 30 d from T_cut − 30 d)", ""] + simulation_table() + [""]
    lines += ["#### Transfer: Synthea and Stack Overflow (zero-shot test)", ""] + transfer_table() + [""]
    lines += ["#### Efficiency (one RTX 4090, bf16, batch 256)", ""] + latency_table() + [""]
    lines += complete_metrics(arms) + [END]
    block = "\n".join(lines)
    text = DOC.read_text() if DOC.exists() else ""
    if START in text and END in text:
        text = text[: text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print("runs:", len(runs), "arms:", len(arms))


if __name__ == "__main__":
    main()
