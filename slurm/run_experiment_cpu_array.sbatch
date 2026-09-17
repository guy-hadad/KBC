#!/bin/bash
#SBATCH --job-name=kbc-exp-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-31
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/exp_cpu_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/exp_cpu_%A_%a.err

# Submit with, for example:
# sbatch --export=ALL,KBC_EXPERIMENT_SUITE=objective-factorial \
#   slurm/run_experiment_cpu_array.sbatch

set -euo pipefail

requeue_before_timeout() {
  trap - USR1
  echo "[$(date --iso-8601=seconds)] approaching time limit; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_before_timeout USR1

REPO=${KBC_REPO:-/groups/bshapira_group/guyhada/KBC}
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Set KBC_ONLY_MISSING=1 for a catch-up pass: run only cells with no result
# file or a result that is not `ok`. Queue one behind the main array to fill any
# gaps it leaves (a transient filesystem error, a node failure) without
# recomputing the cells that already succeeded.
ONLY_MISSING=""
if [[ "${KBC_ONLY_MISSING:-0}" == "1" ]]; then
  ONLY_MISSING="--only-missing"
fi

SUITE=${KBC_EXPERIMENT_SUITE:?set KBC_EXPERIMENT_SUITE to a named non-LLM suite}

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

python scripts/run_experiments.py \
  --suite "${SUITE}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --skip-gpu-only \
  ${ONLY_MISSING} \
  --epochs 8 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --max-events 64 \
  --hidden-size 128 \
  --layers 2 \
  --cpu \
  --results-dir "${KBC_OUTPUT_ROOT}/paper_runs/${SUITE}" &

WORKER_PID=$!
wait "${WORKER_PID}"
