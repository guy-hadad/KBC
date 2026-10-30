"""Random-search HPO for ETAM: sample trial configs, then pick the winner on validation.

Trials run with ``test_eval=false``, so no test number exists for any trial; the
selected configuration is re-run with three seeds and only then evaluated on
test. Selection reads each trial's ``result.json`` (``valid.score``) and its
saved ``run_config.json``.

    python scripts/etam_hpo.py sample --space finetune --trials 24 --seed 0
    python scripts/etam_hpo.py select --trials '<root>/runs/hpo_ft__*' \\
        --out-config best_config.json --result result.json
"""

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Dict, List

# Keys that describe one run rather than the configuration being selected.
RUN_SPECIFIC = ("run_name", "output_dir", "result_path", "seed", "init_checkpoint", "variant")

SPACES: Dict[str, Dict[str, object]] = {
    # Fine-tuning from a fixed pretrained checkpoint (multitask).
    "finetune": {
        "learning_rate": ("log", 3e-5, 1e-3),
        "batch_size": [64, 128, 256],
        "weight_decay": [0.0, 0.01, 0.05, 0.1],
        "model.dropout": [0.0, 0.1, 0.2, 0.3],
        "epochs": [4, 8, 12, 20],
        "p_full": [0.3, 0.5, 0.7],
        "weights.cls": [0.5, 1.0, 2.0, 4.0],
        "head_learning_rate": [0.0, 1e-3],
        "warmup_steps": [100, 300],
    },
    # Self-supervised pretraining on the pool, scored by a frozen probe.
    "pretrain": {
        "learning_rate": ("log", 2e-4, 2e-3),
        "batch_size": [128, 256, 512],
        "weight_decay": [0.01, 0.05, 0.1],
        "model.dropout": [0.0, 0.1, 0.2],
        "p_full": [0.3, 0.5, 0.7],
        "weights.future": [0.5, 1.0, 2.0],
        "weights.cond_feature": [0.25, 0.5, 1.0],
        "weights.user_next": [0.5, 1.0, 2.0],
    },
}


def sample(space: str, trials: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    configs = []
    for _ in range(trials):
        config: Dict[str, object] = {}
        for key, choices in SPACES[space].items():
            if isinstance(choices, tuple) and choices[0] == "log":
                low, high = math.log(choices[1]), math.log(choices[2])
                config[key] = float("{:.3g}".format(math.exp(rng.uniform(low, high))))
            else:
                config[key] = rng.choice(choices)
        configs.append(config)
    return configs


def overrides(config: Dict[str, object]) -> str:
    """Render a sampled config as `key=value` CLI overrides."""

    return " ".join("{}={}".format(key, json.dumps(value)) for key, value in config.items())


def _get(payload: Dict[str, object], dotted: str):
    for part in dotted.split("."):
        if not isinstance(payload, dict) or part not in payload:
            return None
        payload = payload[part]
    return payload


def select(args) -> None:
    rows = []
    for trial in sorted(glob.glob(args.trials)):
        trial_dir = Path(trial)
        result_path = trial_dir / "result.json"
        configs = list(trial_dir.glob("{}/run_config.json".format(args.config_subdir)))
        if not result_path.exists() or not configs:
            rows.append({"trial": trial_dir.name, "score": None})
            continue
        result = json.loads(result_path.read_text())
        score = _get(result, args.metric)
        config = json.loads(configs[0].read_text())
        rows.append({"trial": trial_dir.name, "score": score, "config": config})
    scored = [row for row in rows if isinstance(row.get("score"), (int, float))]
    if not scored:
        raise SystemExit("No finished trial matched {}.".format(args.trials))
    best = (max if args.mode == "max" else min)(scored, key=lambda row: row["score"])
    chosen = {key: value for key, value in best["config"].items() if key not in RUN_SPECIFIC}
    chosen.update(test_eval=True, write_results=True)
    Path(args.out_config).write_text(json.dumps(chosen, indent=2))
    ordered = sorted(scored, key=lambda row: row["score"], reverse=args.mode == "max")
    summary = {
        "best_trial": best["trial"],
        "best_score": best["score"],
        "trials_scored": len(scored),
        "trials_total": len(rows),
        "second_score": ordered[1]["score"] if len(ordered) > 1 else None,
        "median_score": ordered[len(ordered) // 2]["score"],
        "ranking": [[row["trial"], row["score"]] for row in ordered],
    }
    Path(args.result).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sampler = commands.add_parser("sample")
    sampler.add_argument("--space", choices=sorted(SPACES), required=True)
    sampler.add_argument("--trials", type=int, default=24)
    sampler.add_argument("--seed", type=int, default=0)
    chooser = commands.add_parser("select")
    chooser.add_argument("--trials", required=True, help="glob of trial run directories")
    chooser.add_argument("--metric", default="valid.score")
    chooser.add_argument(
        "--config-subdir", default="etam", help="run directory whose run_config.json is selected"
    )
    chooser.add_argument("--mode", choices=("max", "min"), default="max")
    chooser.add_argument("--out-config", required=True)
    chooser.add_argument("--result", required=True)
    args = parser.parse_args()
    if args.command == "sample":
        for config in sample(args.space, args.trials, args.seed):
            print(overrides(config))
    else:
        select(args)


if __name__ == "__main__":
    main()
