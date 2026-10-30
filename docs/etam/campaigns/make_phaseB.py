"""Generate the ETAM phase-B campaign spec (main arms, ablations, scale, HPO, baselines).

    python docs/etam/campaigns/make_phaseB.py > docs/etam/campaigns/etam_mbd_phaseB_20261029.yaml

Everything that varies between arms is written here once, so the scale ladder,
the ablations and the two HPO stages share one recipe and differ only where
they are meant to.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from etam_hpo import overrides, sample  # noqa: E402

NAME = "etam_mbd_phaseB_20261029"
SEEDS = [13, 29, 43]
ETAM = "python {code_dir}/scripts/etam.py run"
OUT = "run_name=etam output_dir={run_dir} result_path={run_dir}/result.json"
# The driver names each run dir "<group>__<id>", so a path into another task's
# run dir must carry both (e.g. runs/pre_main__pre_main, not runs/pre_main).
CKPT = "{root}/runs/%s/etam/best.pt"

# 10 pool epochs at batch 256 (529,180 // 256 = 2,067 steps per epoch). The
# data ladder holds this step count fixed, so it varies unique data, not compute.
POOL_STEPS_10 = 20_670
PRETRAIN = (
    "stage=pretrain pretrain_store=pool batch_size=256 learning_rate=6e-4 "
    "warmup_steps=2000 weight_decay=0.05 eval_every=2000 val_users=10000 num_workers=5 "
    "seed=13 write_results=false"
)
FINETUNE = (
    "stage=finetune task={task} batch_size=128 learning_rate=2e-4 warmup_steps=200 "
    "weight_decay=0.05 epochs=10 eval_every=0 patience=3 val_users=10000 num_workers=5 "
    "seed={seed} method_name=etam"
)
SCALES = {
    "XS": "model.hidden_size=128 model.num_layers=2 model.num_heads=4",
    "S": "model.hidden_size=192 model.num_layers=4 model.num_heads=6",
    "L": "model.hidden_size=512 model.num_layers=8 model.num_heads=8",
    "XL": "model.hidden_size=768 model.num_layers=12 model.num_heads=12",
}
ABLATIONS = {
    "M10": "",
    "shared": "model.placement=shared",
    "ffn": "model.placement=ffn",
    "attn_ffn": "model.placement=attn_ffn",
    "additive": "model.layout=additive",
    "relonly": "model.time_absolute=false",
    "absonly": "model.time_relative=false",
    "ropetime": "model.rope=index+time",
    # Short-context control: window-relative age/gap, exactly as the baselines.
    "ev64": "max_events=64 window_time=true",
    # Layout control at equal capacity: the additive layout has one expert and
    # no time-conditioned objective, so compare it with token+shared without it.
    "tokshared_nocond": "model.placement=shared 'objectives=[\"tpp\",\"user\",\"future\"]'",
}
# What each ablation changes at fine-tuning time (besides the checkpoint).
FT_EXTRA = {
    "ev64": "max_events=64 window_time=true",
    "tokshared_nocond": "'objectives=[\"cls\",\"tpp\",\"user\"]'",
}
DATA = {
    "matched": "pretrain_store=train",
    "pool10pct": "pretrain_fraction=0.1",
    "pool30pct": "pretrain_fraction=0.3",
}

SMALL_GPU = {
    "kind": "gpu",
    "gpus": 1,
    "constraint": "rtx_6000|rtx_4090",
    "mem": "24G",
    "gpu_check": "torch",
}
LARGE_GPU = {"kind": "gpu", "gpus": 1, "constraint": "rtx_6000", "mem": "32G", "gpu_check": "torch"}


def gpu(base, time):
    return dict(base, time=time)


def ft_command(checkpoint: str, variant: str, extra: str = "") -> str:
    return "{} {} {} init_checkpoint={} variant={} {}".format(
        ETAM, FINETUNE, OUT, checkpoint, variant, extra
    ).strip()


def build():
    groups = []

    # -- headline pretraining (M, 20 epochs) and its fine-tunes -------------
    groups.append(
        {
            "name": "pre_main",
            "id": "pre_main",
            "resources": gpu(SMALL_GPU, "12:00:00"),
            "grid": {"size": ["M"]},
            "command": "{} {} {} epochs=20".format(ETAM, PRETRAIN, OUT),
        }
    )
    # Pretraining HPO: 8 random configurations, 10 epochs each, scored by a
    # frozen probe on validation (probe AUC + zero-shot next-type accuracy).
    pre_trials = [
        {"trial": index, "hp": overrides(config)}
        for index, config in enumerate(sample("pretrain", 8, seed=1))
    ]
    groups.append(
        {
            "name": "hpo_pre",
            "id": "hpo_pre__t{trial}",
            "resources": gpu(SMALL_GPU, "10:00:00"),
            "grid": pre_trials,
            "command": (
                "{etam} {pre} run_name=etam output_dir={{run_dir}} epochs=10 {{hp}} && "
                "{etam} stage=probe init_checkpoint={{run_dir}}/etam/best.pt run_name=probe "
                "output_dir={{run_dir}} test_eval=false write_results=false num_workers=5 "
                "result_path={{run_dir}}/result.json"
            ).format(etam=ETAM, pre=PRETRAIN),
        }
    )
    groups.append(
        {
            "name": "pre_scale_large",
            "id": "pre_scale__{size}",
            "resources": gpu(LARGE_GPU, "2-12:00:00"),
            "grid": [{"size": size, "arch": SCALES[size]} for size in ("L", "XL")],
            "command": "{} {} {} epochs=10 {{arch}}".format(ETAM, PRETRAIN, OUT),
        }
    )
    groups.append(
        {
            "name": "pre_ablate",
            "id": "pre_ablate__{variant}",
            "resources": gpu(SMALL_GPU, "8:00:00"),
            "grid": [{"variant": name, "arch": arch} for name, arch in ABLATIONS.items()],
            "command": "{} {} {} epochs=10 {{arch}}".format(ETAM, PRETRAIN, OUT),
        }
    )
    groups.append(
        {
            "name": "pre_scale_small",
            "id": "pre_scale__{size}",
            "resources": gpu(SMALL_GPU, "8:00:00"),
            "grid": [{"size": size, "arch": SCALES[size]} for size in ("XS", "S")],
            "command": "{} {} {} epochs=10 {{arch}}".format(ETAM, PRETRAIN, OUT),
        }
    )
    groups.append(
        {
            "name": "pre_data",
            "id": "pre_data__{data}",
            "resources": gpu(SMALL_GPU, "8:00:00"),
            "grid": [{"data": name, "arch": arch} for name, arch in DATA.items()],
            "command": "{} {} {} max_steps={} {{arch}}".format(ETAM, PRETRAIN, OUT, POOL_STEPS_10),
        }
    )

    # -- fine-tunes and probes ---------------------------------------------
    groups.append(
        {
            "name": "ft_main",
            "id": "ft_main__{task}_s{seed}",
            "after": ["pre_main"],
            "resources": gpu(SMALL_GPU, "3:00:00"),
            "grid": {"task": ["multi", "cls", "tpp"], "seed": SEEDS},
            "command": ft_command(CKPT % "pre_main__pre_main", "pool-M20-{task}"),
        }
    )
    groups.append(
        {
            "name": "probe_main",
            "id": "probe_main__s{seed}",
            "after": ["pre_main"],
            "resources": gpu(SMALL_GPU, "3:00:00"),
            "grid": {"seed": SEEDS},
            "command": (
                "{} stage=probe init_checkpoint={} seed={{seed}} num_workers=5 method_name=etam "
                "variant=pool-M20-probe {}"
            ).format(ETAM, CKPT % "pre_main__pre_main", OUT),
        }
    )
    ft_trials = [
        {"trial": index, "hp": overrides(config)}
        for index, config in enumerate(sample("finetune", 24, seed=2))
    ]
    groups.append(
        {
            "name": "hpo_ft",
            "id": "hpo_ft__t{trial}",
            "after": ["pre_main"],
            "resources": gpu(SMALL_GPU, "4:00:00"),
            "grid": ft_trials,
            "params": {"task": "multi", "seed": 13},
            "command": ft_command(
                CKPT % "pre_main__pre_main", "hpo", "test_eval=false write_results=false {hp}"
            ),
        }
    )
    select = (
        "python {code_dir}/scripts/etam_hpo.py select --trials '{root}/runs/%s__t*' "
        "--out-config {run_dir}/best_config.json --result {run_dir}/result.json"
    )
    groups.append(
        {
            "name": "hpo_ft_select",
            "id": "hpo_ft_select",
            "after": ["hpo_ft"],
            "resources": {
                "kind": "cpu",
                "cpus": 1,
                "mem": "2G",
                "time": "0:30:00",
                "constraint": None,
            },
            "grid": {"stage": ["finetune"]},
            "command": select % "hpo_ft__hpo_ft",
        }
    )
    groups.append(
        {
            "name": "hpo_pre_select",
            "id": "hpo_pre_select",
            "after": ["hpo_pre"],
            "resources": {
                "kind": "cpu",
                "cpus": 1,
                "mem": "2G",
                "time": "0:30:00",
                "constraint": None,
            },
            "grid": {"stage": ["pretrain"]},
            "command": select % "hpo_pre__hpo_pre",
        }
    )
    groups.append(
        {
            "name": "pre_final",
            "id": "pre_final",
            "after": ["hpo_pre_select"],
            "resources": gpu(SMALL_GPU, "16:00:00"),
            "grid": {"size": ["M"]},
            "command": (
                "{} --config {{root}}/runs/hpo_pre_select__hpo_pre_select/best_config.json {} "
                "seed=13 epochs=20 eval_every=2000 write_results=false"
            ).format(ETAM, OUT),
        }
    )
    groups.append(
        {
            "name": "confirm",
            "id": "confirm__{init}_s{seed}",
            "after": ["pre_final", "hpo_ft_select"],
            "resources": gpu(SMALL_GPU, "4:00:00"),
            "grid": {"init": ["pre_main", "pre_final"], "seed": SEEDS},
            "command": (
                "{} --config {{root}}/runs/hpo_ft_select__hpo_ft_select/best_config.json {} "
                "seed={{seed}} init_checkpoint={{root}}/runs/{{init}}__{{init}}/etam/best.pt "
                "method_name=etam "
                "variant=pool-{{init}}-hpo-multi"
            ).format(ETAM, OUT),
        }
    )
    groups.append(
        {
            "name": "ft_ablate",
            "id": "ft_ablate__{variant}_s{seed}",
            "after": ["pre_ablate"],
            "resources": gpu(SMALL_GPU, "3:00:00"),
            "grid": [
                {"variant": name, "seed": seed, "ftx": FT_EXTRA.get(name, "")}
                for name in ABLATIONS
                for seed in SEEDS
            ],
            "params": {"task": "multi"},
            "command": ft_command(
                CKPT % "pre_ablate__pre_ablate__{variant}",
                "pool-{variant}-multi",
                "{ftx}",
            ),
        }
    )
    for size_group, sizes in (("small", ("XS", "S")), ("large", ("L", "XL"))):
        groups.append(
            {
                "name": "ft_scale_" + size_group,
                "id": "ft_scale__{size}_s{seed}",
                "after": ["pre_scale_" + size_group],
                "resources": gpu(LARGE_GPU if size_group == "large" else SMALL_GPU, "6:00:00"),
                "grid": {"size": list(sizes), "seed": SEEDS},
                "params": {"task": "multi"},
                "command": ft_command(
                    CKPT % ("pre_scale_%s__pre_scale__{size}" % size_group), "pool-{size}10-multi"
                ),
            }
        )
    groups.append(
        {
            "name": "ft_data",
            "id": "ft_data__{data}_{task}_s{seed}",
            "after": ["pre_data"],
            "resources": gpu(SMALL_GPU, "3:00:00"),
            "grid": [
                {"data": data, "task": task, "seed": seed}
                for data in DATA
                for task in (("multi", "cls", "tpp") if data == "matched" else ("multi",))
                for seed in SEEDS
            ],
            "command": ft_command(CKPT % "pre_data__pre_data__{data}", "{data}-M-{task}"),
        }
    )
    groups.append(
        {
            "name": "ft_scratch",
            "id": "ft_scratch__{task}_s{seed}",
            "resources": gpu(SMALL_GPU, "4:00:00"),
            "grid": {"task": ["multi", "cls", "tpp"], "seed": SEEDS},
            "command": (
                "{} {} {} variant=scratch-M-{{task}} epochs=20 patience=4 learning_rate=5e-4 "
                "warmup_steps=500"
            ).format(ETAM, FINETUNE, OUT),
        }
    )

    # -- L1: frozen Qwen3-0.6B-Base with time/event LoRA experts (RAM-style) --
    llm = "model.backbone=Qwen/Qwen3-0.6B-Base model.lora_rank=64"
    groups.append(
        {
            "name": "pre_llm",
            "id": "pre_llm",
            "resources": gpu(LARGE_GPU, "1-00:00:00"),
            "grid": {"size": ["qwen0.6b"]},
            "command": (
                "{} {} {} {} epochs=2 batch_size=32 learning_rate=2e-4 warmup_steps=1000 "
                "eval_every=4000 val_users=5000"
            ).format(ETAM, PRETRAIN, OUT, llm),
        }
    )
    groups.append(
        {
            "name": "ft_llm",
            "id": "ft_llm__s{seed}",
            "after": ["pre_llm"],
            "resources": gpu(LARGE_GPU, "8:00:00"),
            "grid": {"seed": SEEDS},
            "params": {"task": "multi"},
            "command": ft_command(
                CKPT % "pre_llm__pre_llm",
                "pool-qwen0.6b-lora-multi",
                "batch_size=32 learning_rate=1e-4 epochs=5 patience=2",
            ),
        }
    )

    # -- baselines given the same unlabeled pool -----------------------------
    groups.append(
        {
            "name": "baselines_pool",
            "id": "baseline__c{cell}",
            "resources": gpu(dict(SMALL_GPU, mem="40G"), "1-00:00:00"),
            "grid": {"cell": list(range(24))},
            "command": (
                "python {code_dir}/scripts/run_experiments.py --suite mbd-pool-baselines "
                "--array-index {cell} --array-size 24 --epochs 8 --batch-size 32 "
                "--learning-rate 3e-4 --max-events 64 --hidden-size 128 --layers 2 "
                "--results-dir {run_dir}/cells && python {code_dir}/scripts/cell_result.py "
                "--results-dir {run_dir}/cells --out {run_dir}/result.json"
            ),
        }
    )

    return {
        "name": NAME,
        "workdir": "/groups/bshapira_group/guyhada/KBC",
        "snapshot": ["eventfm", "scripts"],
        "setup": (
            'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
            "conda activate KBC\n"
            "export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data\n"
            "export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs\n"
            "export HF_HOME=/groups/bshapira_group/guyhada/hf_cache HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1\n"
        ),
        "defaults": {"kind": "cpu", "cpus": 2, "mem": "8G", "time": "4:00:00"},
        "groups": groups,
        "harvest": {
            "aggregate_over": ["seed"],
            "metrics": {
                "default": [
                    "classification.auc",
                    "tpp.next_type_accuracy",
                    "tpp.next_type_macro_f1",
                    "tpp.delta_log_rmse",
                    "tpp.delta_log_mae",
                    "valid.score",
                ],
                "baselines_pool": [
                    "auc",
                    "next_type_accuracy",
                    "next_type_macro_f1",
                    "delta_log_rmse",
                    "delta_log_mae",
                ],
                "hpo_pre": ["valid.score", "valid.probe_auc", "valid.tpp_next_type_accuracy"],
                "hpo_ft": ["valid.score", "valid.cls_auc", "valid.tpp_next_type_accuracy"],
                # Pretraining runs report zero-shot validation TPP only.
                **{
                    group: [
                        "valid.next_type_accuracy",
                        "valid.next_type_macro_f1",
                        "valid.delta_log_rmse",
                        "valid.delta_log_mae",
                        "valid.joint_nll",
                        "train_minutes",
                    ]
                    for group in (
                        "pre_main",
                        "pre_final",
                        "pre_ablate",
                        "pre_scale_small",
                        "pre_scale_large",
                        "pre_data",
                        "pre_llm",
                    )
                },
            },
            "doc": "docs/etam/results.md",
        },
    }


if __name__ == "__main__":
    spec = build()
    print("# Generated by docs/etam/campaigns/make_phaseB.py; edit that file, not this one.")
    print(yaml.safe_dump(spec, sort_keys=False, width=100))
    json.dumps(spec)
