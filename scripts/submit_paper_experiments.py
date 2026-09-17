#!/usr/bin/env python
"""Submit every non-empty CPU and GPU component of the paper experiment suites."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.benchmark.experiments import (  # noqa: E402
    experiment_names,
    get_experiment,
    resolve_sample_sizes,
)
from eventfm.benchmark.runner import enumerate_cells  # noqa: E402
from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.methods import METHOD_REGISTRY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_SCRIPT = REPO_ROOT / "slurm" / "run_experiment_cpu_array.sbatch"
GPU_SCRIPT = REPO_ROOT / "slurm" / "run_experiment_gpu_array.sbatch"
DEFAULT_MAX_ARRAY_TASKS = 4
DEFAULT_JOB_CAP = 100


def _count_cells(suite, *, gpu_only: bool) -> int:
    methods = tuple(
        method
        for method in suite.methods
        if METHOD_REGISTRY[method].requires_gpu is gpu_only
    )
    return len(
        enumerate_cells(
            suite.datasets,
            suite.tasks,
            methods,
            resolve_sample_sizes(suite),
            suite.seeds,
            suite.regimes,
            suite.variants,
        )
    )


def _submit(
    suite_name: str,
    kind: str,
    cells: int,
    dependency: str,
    max_array_tasks: int,
) -> dict:
    array_tasks = min(max_array_tasks, cells)
    script = GPU_SCRIPT if kind == "gpu" else CPU_SCRIPT
    command = [
        "sbatch",
        "--parsable",
        "--array=0-{}".format(array_tasks - 1),
        "--dependency=afterok:{}".format(dependency),
        "--export=ALL,KBC_EXPERIMENT_SUITE={}".format(suite_name),
        str(script),
    ]
    completed = subprocess.run(
        command,
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    job_id = completed.stdout.strip().splitlines()[-1].split(";")[0]
    return {
        "suite": suite_name,
        "kind": kind,
        "cells": cells,
        "array_tasks": array_tasks,
        "dependency": dependency,
        "job_id": job_id,
        "script": str(script.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dependency", required=True, help="successful CPU smoke job")
    parser.add_argument("--gpu-dependency", required=True, help="successful GPU smoke job")
    parser.add_argument("--max-array-tasks", type=int, default=DEFAULT_MAX_ARRAY_TASKS)
    parser.add_argument("--job-cap", type=int, default=DEFAULT_JOB_CAP)
    parser.add_argument(
        "--reserved-jobs",
        type=int,
        default=3,
        help="jobs outside the arrays that count toward the campaign cap",
    )
    parser.add_argument("--supersedes", help="manifest path for a cancelled campaign")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=experiment_names(),
        help="submit only these suites (default: every suite with cells)",
    )
    args = parser.parse_args()

    if args.max_array_tasks < 1:
        parser.error("--max-array-tasks must be positive")

    selected = list(args.suites) if args.suites else experiment_names()
    components = []
    for suite_name in selected:
        suite = get_experiment(suite_name)
        for kind, dependency in (
            ("cpu", args.cpu_dependency),
            ("gpu", args.gpu_dependency),
        ):
            cells = _count_cells(suite, gpu_only=kind == "gpu")
            if cells:
                components.append((suite_name, kind, cells, dependency))
    array_elements = sum(min(args.max_array_tasks, cells) for _, _, cells, _ in components)
    campaign_jobs = args.reserved_jobs + array_elements
    if campaign_jobs > args.job_cap:
        parser.error(
            "submission would create {} campaign jobs, above cap {}".format(
                campaign_jobs, args.job_cap
            )
        )

    started_at = datetime.now(timezone.utc).isoformat()
    submissions = []
    total_cells = 0
    for suite_name, kind, cells, dependency in components:
        submission = _submit(
            suite_name,
            kind,
            cells,
            dependency,
            args.max_array_tasks,
        )
        submissions.append(submission)
        total_cells += cells
        print(json.dumps(submission, sort_keys=True), flush=True)

    manifest = {
        "schema_version": 1,
        "submitted_at_utc": started_at,
        "cpu_smoke_dependency": args.cpu_dependency,
        "gpu_smoke_dependency": args.gpu_dependency,
        "job_cap": args.job_cap,
        "reserved_jobs": args.reserved_jobs,
        "array_elements": array_elements,
        "campaign_jobs_at_submission": campaign_jobs,
        "supersedes_cancelled_manifest": args.supersedes,
        "suites_selected": selected,
        "total_cells": total_cells,
        "submissions": submissions,
    }
    manifest_dir = output_root() / "paper_runs" / "submissions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_dir / "campaign_{}.json".format(timestamp)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "total_cells": total_cells}))


if __name__ == "__main__":
    main()
