"""Run several ETAM-2 runs concurrently on one GPU, then write one combined result.

The cluster allows only a handful of GPUs per user at a time, while one ETAM-2
run does not saturate a modern GPU, so a campaign task trains a *bundle* of
runs side by side. Each run keeps its own directory, log and full result.json
(``<out>/<name>/``); a run that already has a result is skipped, so a requeued
bundle only redoes what is missing (each run also resumes from its last.pt).

    python scripts/etam2_bundle.py --out RUN_DIR \
        --run 'ref_s13::stage=pretrain seed=13' --run 'ref_s29::stage=pretrain seed=29'
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HEADLINE = (
    ("valid", "joint_nll"),
    ("test", "tpp", "next_type_accuracy"),
    ("test", "tpp", "joint_nll"),
    ("test", "tpp", "delta_log_rmse"),
    ("classification", "auc"),
    ("tpp", "next_type_accuracy"),
)


def _get(payload, path):
    for key in path:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", action="append", required=True, help="name::key=value ...")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "etam2.py"
    processes = []
    launched = []
    for item in args.run:
        name, _, overrides = item.partition("::")
        run_dir = out / name
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_dir / "result.json"
        if result.exists():
            print("[bundle] {} already done".format(name), flush=True)
            processes.append((name, None))
            launched.append(None)
            continue
        command = [sys.executable, str(script), "run"] + shlex.split(overrides) + [
            "run_name={}".format(name),
            "output_dir={}".format(out),
            "result_path={}".format(result),
            "num_workers={}".format(args.workers),
        ]
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        log = (run_dir / "stdout.log").open("a", encoding="utf-8")
        print("[bundle] start {}: {}".format(name, " ".join(command)), flush=True)
        processes.append((name, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)))
        launched.append((command, env))
        time.sleep(20)  # stagger store loading
    failed = []
    for name, process in processes:
        if process is not None and process.wait() != 0:
            failed.append(name)
            print("[bundle] FAILED {} (exit {})".format(name, process.returncode), flush=True)
    # A run that ran out of GPU memory next to its siblings is retried alone (it resumes
    # from its last.pt); anything else stays failed.
    commands = dict(zip([n for n, _ in processes], launched))
    for name in list(failed):
        log_text = (out / name / "stdout.log").read_text(errors="ignore")[-20000:]
        if "out of memory" not in log_text.lower() or name not in commands:
            continue
        print("[bundle] retrying {} alone after an out-of-memory failure".format(name), flush=True)
        command, env = commands[name]
        with (out / name / "stdout.log").open("a", encoding="utf-8") as log:
            if subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env) == 0:
                failed.remove(name)
    combined = {}
    for name, _ in processes:
        result = out / name / "result.json"
        if result.exists():
            payload = json.loads(result.read_text())
            combined[name] = {".".join(path): _get(payload, path) for path in HEADLINE if _get(payload, path) is not None}
    combined["failed"] = len(failed)
    if failed:
        (out / "bundle_failed.json").write_text(json.dumps(failed))
        sys.exit(1)
    (out / "result.json").write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
