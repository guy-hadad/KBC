"""Generate the ETAM-2 (Stage 2) campaign spec.

    python docs/etam/campaigns/make_stage2.py   # writes etam2_mbd_20261108.yaml

Budget and layout
-----------------
* Every pretraining arm uses Stage 1's M10 budget (10 pool epochs = 20,670
  steps at an effective batch of 256: micro-batch 128 x 2 accumulation), so each
  arm sits in the same table as Stage 1's ablations.
* The user's GPU quota is 5 at a time and was shared with other campaigns, so
  GPU tasks are *bundles* of 3 runs trained side by side on one 48 GB GPU
  (``scripts/etam2_bundle.py``). Bundles are ordered by importance: the
  3-seed comparisons first.
* CPU (training is ~1 min/step there for ETAM-M, so only inference and small
  models): Stage 1 checkpoints re-scored with the Stage 2 battery, frozen
  probes, and the JSONL baselines (strict labels, matched pool compute).

An arm changes exactly one mechanism of the reference configuration
(``Etam2Config()`` defaults), so each ablation is attributable.
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NAME = "etam2_mbd_20261108"
SEEDS = [13, 29, 43]
ONE = [13]
ETAM2 = "python {code_dir}/scripts/etam2.py run"
ETAM1 = "python {code_dir}/scripts/etam.py run"
STEPS = 20_670
PHASE_B = "/groups/bshapira_group/guyhada/campaigns/etam_mbd_phaseB_20261029/runs"
# Parameter values are not template-expanded by the driver, so paths into other tasks are absolute.
ROOT = "/groups/bshapira_group/guyhada/campaigns/" + NAME

PRE = (
    "stage=pretrain pretrain_store=pool batch_size=128 grad_accum=2 learning_rate=6e-4 weight_decay=0.05 "
    "max_steps={} warmup_steps=2000 eval_every=4134 val_users=10000".format(STEPS)
)
S1_PRE = (
    "stage=pretrain pretrain_store=pool batch_size=256 learning_rate=6e-4 warmup_steps=2000 weight_decay=0.05 "
    "eval_every=2000 val_users=10000 num_workers=5 write_results=false max_steps={}".format(STEPS)
)
FT = (
    "stage=finetune batch_size=128 learning_rate=2e-4 warmup_steps=200 weight_decay=0.05 epochs=10 eval_every=0 "
    "patience=3 val_users=10000"
)

SOFT4 = "model.mtp_horizon=4 model.mtp_target=soft weights.mtp=0.5"
TIME4 = "model.mtp_time=true weights.horizon=0.5"
REPR = "model.repr_dim=64 weights.repr=0.1"

#: name -> (overrides vs the reference, seeds). Order = submission priority.
ARMS = {
    # P0 / main
    "ref": ("", SEEDS),
    # P2: middle architecture (C)
    "a_shared": ("model.mixer=shared", SEEDS),
    "a_stage1": ("model.mixer=stage1", SEEDS),
    "a_private": ("'model.bridge_layers=[]'", SEEDS),
    "a_private_shared": ("model.mixer=private_shared", SEEDS),
    "a_bridges_all": ("'model.bridge_layers=[1,2,3,4,5,6]'", SEEDS),
    # P1: decoders (D)
    "d_indep": ("model.decoder=independent", SEEDS),
    "d_time_first": ("model.decoder=time_first", SEEDS),
    "d_event_first": ("model.decoder=event_first", SEEDS),
    # P1: time representation (A)
    "t_stage1": ("model.calendar=fixed model.relative=stage1 model.memory=none", SEEDS),
    "t_nomem": ("model.memory=none", SEEDS),
    # P3: multi-event prediction (E), representation objective (F), full curriculum
    "m_soft4": (SOFT4, SEEDS),
    "m_hard4": ("model.mtp_horizon=4 model.mtp_target=hard weights.mtp=0.5", SEEDS),
    "r_all": (REPR, SEEDS),
    "c6": (SOFT4 + " " + TIME4 + " " + REPR, SEEDS),
    # External baselines in this framework (same data, protocol, evaluation)
    # External baselines rerun in this framework (same data, protocol and evaluation).
    # FlexTPP/Kairos have no "now" token, so their query term is off.
    "b_s2p2": ("model.mixer=s2p2", SEEDS),
    "b_itpp": ("model.mixer=itpp", SEEDS),
    "b_flextpp": ("model.mixer=flextpp model.time_dist=spline weights.query=0", SEEDS),
    "b_kairos": ("model.mixer=kairos weights.query=0", SEEDS),
    # Single-seed mechanism ablations
    "ev64": ("max_events=64 window_time=true", ONE),
    "t_bins": ("model.relative=bins", ONE),
    "t_mem_learned": ("model.memory=exp_learned", ONE),
    "t_mem_count": ("model.memory=count", ONE),
    "t_mem_ssm": ("model.memory=ssm", ONE),
    "t_mem_ode": ("model.memory=ode", ONE),
    "t_cal_fixed": ("model.calendar=fixed", ONE),
    "t_relonly": ("model.calendar=none", ONE),
    "t_vq": ("model.time_vq=256", ONE),
    "d_nocons": ("weights.cons=0", ONE),
    "d_cons1": ("weights.cons=1.0", ONE),
    "d_spline": ("model.time_dist=spline", ONE),
    "d_hazard": ("model.time_dist=hazard", ONE),
    "g_nosurv": ("weights.survival=0", ONE),
    "g_nocensor": ("weights.censor=0", ONE),
    "g_poisson": ("model.count_dist=poisson", ONE),
    "g_count_stage1": ("model.count_dist=stage1", ONE),
    "a_shared_wide": ("model.mixer=shared model.hidden_size=368", ONE),
    "a_nobias": ("model.bridge_bias=none", ONE),
    "a_lagonly": ("model.bridge_bias=lag", ONE),
    "a_nogate": ("model.bridge_gate=false", ONE),
    "m_soft2": ("model.mtp_horizon=2 model.mtp_target=soft weights.mtp=0.5", ONE),
    "m_soft8": ("model.mtp_horizon=8 model.mtp_target=soft weights.mtp=0.5", ONE),
    "m_soft4_time": (SOFT4 + " " + TIME4, ONE),
    "r_joint": (REPR + " 'model.repr_modalities=[\"joint\"]'", ONE),
    "r_time": (REPR + " 'model.repr_modalities=[\"time\"]'", ONE),
    "u_event_only": ("model.inputs=event_only", ONE),
    "u_time_only": ("model.inputs=time_only", ONE),
    # G 8.1 staged curriculum (cumulative; c2 + unimodal = ref, c6 above = full)
    "c0": ("weights.dual=0 weights.cons=0 weights.unimodal=0", ONE),
    "c1": ("weights.cons=0 weights.unimodal=0", ONE),
    "c2": ("weights.unimodal=0", ONE),
    "c3": ("weights.unimodal=0 " + SOFT4, ONE),
    "c4": ("weights.unimodal=0 " + SOFT4 + " " + TIME4, ONE),
    "c5": ("weights.unimodal=0 " + SOFT4 + " " + TIME4 + " " + REPR, ONE),
}
#: Arms whose frozen embeddings are dumped for the CPU probes.
DUMP = {"ref", "u_event_only", "u_time_only", "r_all", "c6", "m_soft4", "a_shared", "a_stage1", "t_stage1"}
#: G 8.5 data scaling (3 seeds): A = fixed 20,670 steps; B = 10 epochs per user.
DATA = {"matched": ("pretrain_store=train", 1198), "pool10": ("pretrain_fraction=0.1", 2067),
        "pool30": ("pretrain_fraction=0.3", 6201)}

GPU_BUNDLE = {"kind": "gpu", "gpus": 1, "constraint": "rtx_6000", "mem": "44G", "time": "2-12:00:00", "gpu_check": "torch"}
GPU_ONE = {"kind": "gpu", "gpus": 1, "constraint": "rtx_4090|rtx_6000", "mem": "32G", "time": "1-00:00:00", "gpu_check": "torch"}
CPU_EVAL = {"kind": "cpu", "cpus": 16, "mem": "48G", "time": "1-12:00:00"}
CPU_PROBE = {"kind": "cpu", "cpus": 8, "mem": "24G", "time": "8:00:00"}


def bundles(runs, size=3):
    rows = []
    for index in range(0, len(runs), size):
        chunk = runs[index : index + size]
        arguments = " ".join("--run '{}::{}'".format(name, args.replace("'", "'\\''")) for name, args in chunk)
        rows.append({"bundle": "{:03d}".format(index // size), "names": "+".join(n for n, _ in chunk), "args": arguments})
    return rows


def pretrain_runs():
    runs = []
    for name, (overrides, seeds) in ARMS.items():
        for seed in seeds:
            dump = " dump_embeddings=true" if name in DUMP and seed == 13 else ""
            runs.append(("{}_s{}".format(name, seed), "{} seed={} {}{}".format(PRE, seed, overrides, dump).strip()))
    return runs


def scaling_runs():
    runs = []
    for data, (choice, steps_b) in DATA.items():
        for seed in SEEDS:
            runs.append(("scaleA_{}_s{}".format(data, seed), "{} seed={} {}".format(PRE, seed, choice)))
            warmup = max(100, min(2000, steps_b // 10))
            runs.append(
                ("scaleB_{}_s{}".format(data, seed),
                 "{} seed={} {} max_steps={} warmup_steps={} eval_every={}".format(
                     PRE, seed, choice, steps_b, warmup, max(200, steps_b // 4)))
            )
    return runs


def finetune_runs(ref_bundle):
    """Downstream adaptation of the reference model (and Stage 1 at the same budget)."""

    ckpt = {
        "ref": {seed: "{}/runs/pre_main__b{}/ref_s{}/best.pt".format(ROOT, ref_bundle.get(seed, "x"), seed) for seed in SEEDS},
        "s1": {seed: ("{}/pre_ablate__pre_ablate__M10/etam/best.pt".format(PHASE_B) if seed == 13
                      else "{}/runs/pre_s1__s1_M10_s{}/etam/best.pt".format(ROOT, seed)) for seed in SEEDS},
    }
    runs = []

    def add(name, init, args):
        runs.append((name, "{} {} init_checkpoint={}".format(FT, args, init)))

    for model in ("ref", "s1"):
        for seed in SEEDS:
            init = ckpt[model][seed]
            for task in ("multi", "cls", "tpp"):
                add("{}_{}_std_s{}".format(model, task, seed), init, "task={} seed={}".format(task, seed))
            for task in ("multi", "cls"):
                add("{}_{}_strict_s{}".format(model, task, seed), init, "task={} cls_suffix=_strict seed={}".format(task, seed))
            add("{}_multi_final8_s{}".format(model, seed), init, "task=multi select=final patience=0 epochs=8 seed={}".format(seed))
            for adapt in ("lora", "full"):
                for fraction in (0.01, 0.05, 0.25):
                    add("{}_cls_{}{}_s{}".format(model, adapt, fraction, seed), init,
                        "task=cls cls_suffix=_strict adapt={} label_fraction={} learning_rate={} seed={}".format(
                            adapt, fraction, "1e-3" if adapt == "lora" else "2e-4", seed))
        if model == "ref":
            for seed in SEEDS:
                init = ckpt[model][seed]
                add("ref_multi_lrgroups_s{}".format(seed), init,
                    "task=multi head_lr_mult=5 encoder_lr_mult=0.3 log_grad_conflict=true seed={}".format(seed))
                for adapt in ("head", "lora"):
                    add("ref_tpp_{}_s{}".format(adapt, seed), init, "task=tpp adapt={} learning_rate=1e-3 seed={}".format(adapt, seed))
    return runs


def run_path(pre_rows, run):
    """Absolute directory of the bundle task that trains ``run`` (first bundle = group pre_main)."""

    for index, row in enumerate(pre_rows):
        if run in row["names"].split("+"):
            return "{}/runs/{}__b{}".format(ROOT, "pre_main" if index == 0 else "pre", row["bundle"])
    raise KeyError(run)


def groups(ref_bundle_of_seed):
    out = []
    pre_rows = bundles(pretrain_runs())
    out.append({"name": "pre_main", "id": "b{bundle}", "resources": GPU_BUNDLE, "grid": pre_rows[:1],
                "command": "python {code_dir}/scripts/etam2_bundle.py --out {run_dir} {args}"})
    out.append({"name": "pre", "id": "b{bundle}", "resources": GPU_BUNDLE, "grid": pre_rows[1:],
                "command": "python {code_dir}/scripts/etam2_bundle.py --out {run_dir} {args}"})
    stage1 = [("s1_M10_s{}".format(seed), seed, "{} seed={}".format(S1_PRE, seed)) for seed in (29, 43)]
    for seed in (29, 43):
        for fraction, tag in ((0.1, "pool10"), (0.3, "pool30")):
            stage1.append(("s1_{}_s{}".format(tag, seed), seed, "{} seed={} pretrain_fraction={}".format(S1_PRE, seed, fraction)))
    out.append({
        "name": "pre_s1", "id": "{name}", "resources": GPU_ONE,
        "grid": [{"name": name, "seed": seed, "args": args} for name, seed, args in stage1],
        "command": (ETAM1 + " {args} run_name=etam output_dir={run_dir} && " + ETAM2 + " stage=evaluate "
                    "init_checkpoint={run_dir}/etam/best.pt seed={seed} dump_embeddings=true run_name=r "
                    "output_dir={run_dir} result_path={run_dir}/result.json"),
    })
    out.append({"name": "scale", "id": "b{bundle}", "resources": GPU_BUNDLE, "grid": bundles(scaling_runs()),
                "command": "python {code_dir}/scripts/etam2_bundle.py --out {run_dir} {args}"})
    resolved = finetune_runs(ref_bundle_of_seed)
    out.append({"name": "ft", "id": "b{bundle}", "after": ["pre_main", "pre_s1"], "resources": GPU_BUNDLE,
                "grid": bundles(resolved), "command": "python {code_dir}/scripts/etam2_bundle.py --out {run_dir} {args}"})
    # CPU: Stage 1 phase-B checkpoints re-scored with the Stage 2 battery.
    s1_old = {"M10": "pre_ablate__pre_ablate__M10", "M20": "pre_main__pre_main", "final": "pre_final__pre_final",
              "pool10": "pre_data__pre_data__pool10pct", "pool30": "pre_data__pre_data__pool30pct",
              "matched": "pre_data__pre_data__matched", "relonly": "pre_ablate__pre_ablate__relonly",
              "absonly": "pre_ablate__pre_ablate__absonly"}
    out.append({
        "name": "eval_s1", "id": "{ckpt}", "resources": CPU_EVAL,
        "grid": [{"ckpt": key, "path": "{}/{}/etam/best.pt".format(PHASE_B, value)} for key, value in s1_old.items()],
        "command": (ETAM2 + " stage=evaluate init_checkpoint={path} seed=13 bf16=false val_users=10000 "
                    "eval_test_users=0 num_workers=6 eval_batch_size=256 dump_embeddings=true run_name=r "
                    "output_dir={run_dir} result_path={run_dir}/result.json"),
    })
    # CPU: frozen probes (label fractions, MLP, cross-target, strict) on dumped embeddings.
    probe_rows = []
    for arm in sorted(DUMP):
        probe_rows.append({"arm": arm, "emb": "%s/%s_s13/embeddings" % (run_path(pre_rows, arm + "_s13"), arm), "extra": ""})
    probe_rows.append({"arm": "s1M10", "emb": ROOT + "/runs/eval_s1__M10/r/embeddings", "extra": ""})
    probe_rows.append({"arm": "s1final", "emb": ROOT + "/runs/eval_s1__final/r/embeddings", "extra": ""})
    probe_rows.append({"arm": "event_plus_time",
                       "emb": "%s/u_event_only_s13/embeddings" % run_path(pre_rows, "u_event_only_s13"),
                       "extra": "--concat %s/u_time_only_s13/embeddings --reps joint" % run_path(pre_rows, "u_time_only_s13")})
    out.append({"name": "probe", "id": "{arm}", "after": ["pre", "eval_s1"], "resources": CPU_PROBE,
                "grid": probe_rows,
                "command": "python {code_dir}/scripts/etam2_probe.py --emb {emb} {extra} --out {run_dir}/result.json"})
    # GPU: long-horizon simulation, then one latency job over every checkpoint.
    sim_rows = []
    for arm in ("ref", "c6", "m_soft4"):
        sim_rows.append({"model": arm, "init": "%s/%s_s13/best.pt" % (run_path(pre_rows, arm + "_s13"), arm)})
    sim_rows.append({"model": "s1M10", "init": "{}/pre_ablate__pre_ablate__M10/etam/best.pt".format(PHASE_B)})
    out.append({"name": "sim", "id": "{model}", "after": ["pre"], "resources": GPU_ONE, "grid": sim_rows,
                "command": (ETAM2 + " stage=simulate init_checkpoint={init} seed=13 simulate_users=4000 simulate_samples=16 "
                            "eval_batch_size=1024 run_name=r output_dir={run_dir} result_path={run_dir}/result.json")})
    out.append({"name": "latency", "id": "latency", "after": ["pre", "pre_s1"],
                "resources": dict(GPU_ONE, constraint="rtx_4090", time="6:00:00"), "grid": {"device": ["rtx_4090"]},
                "command": ("python {code_dir}/scripts/etam2_latency.py --glob '{root}/runs/pre*__b*/*_s13/best.pt' "
                            "--glob '{root}/runs/pre_s1__s1_M10_s29/etam/best.pt' "
                            "--glob '%s/pre_ablate__pre_ablate__M10/etam/best.pt' --out {run_dir}/result.json" % PHASE_B)})
    # CPU: JSONL baselines with strict labels (+/- pool) and with matched pool compute.
    for name, suite, cells, time in (("base_strict", "mbd-strict-baselines", 24, "2-00:00:00"),
                                     ("base_pool10", "mbd-pool-baselines-matched", 12, "5-00:00:00")):
        out.append({
            "name": name, "id": "c{cell}", "resources": {"kind": "cpu", "cpus": 16, "mem": "32G", "time": time},
            "grid": {"cell": list(range(cells))},
            "command": ("python {code_dir}/scripts/run_experiments.py --suite %s --array-index {cell} --array-size %d "
                        "--epochs 8 --batch-size 32 --learning-rate 3e-4 --max-events 64 --hidden-size 128 --layers 2 "
                        "--cpu --results-dir {run_dir}/cells && python {code_dir}/scripts/cell_result.py "
                        "--results-dir {run_dir}/cells --out {run_dir}/result.json") % (suite, cells),
        })
    return out


def main() -> None:
    pre_rows = bundles(pretrain_runs())
    ref_bundle = {}
    for row in pre_rows:
        for seed in SEEDS:
            if "ref_s{}".format(seed) in row["names"].split("+"):
                ref_bundle[seed] = row["bundle"]
    spec = {
        "name": NAME,
        "workdir": "/groups/bshapira_group/guyhada/KBC",
        "snapshot": ["eventfm", "scripts"],
        "setup": (
            'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
            "conda activate KBC\n"
            "export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data\n"
            "export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs\n"
            "export HF_HOME=/groups/bshapira_group/guyhada/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        ),
        "defaults": {"kind": "cpu", "cpus": 8, "mem": "16G", "time": "12:00:00"},
        "groups": groups(ref_bundle),
        "harvest": {"metrics": {"default": ["failed"]}},
    }
    path = HERE / "{}.yaml".format(NAME)
    path.write_text("# Generated by docs/etam/campaigns/make_stage2.py; edit that file, not this one.\n"
                    + yaml.safe_dump(spec, sort_keys=False, width=110))
    total = sum(len(g["grid"]) if isinstance(g["grid"], list) else 1 for g in spec["groups"])
    print(path, "tasks:", total, "pretrain runs:", len(pretrain_runs()), "ft runs:", len(finetune_runs(ref_bundle)))


if __name__ == "__main__":
    main()
