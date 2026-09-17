#!/bin/bash
#SBATCH --job-name=kbc-bench
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/bench_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/bench_%A_%a.err

# One array task walks a strided slice of the benchmark grid. Cells are
# independent and each writes its own result JSON, so a task that dies only
# costs its own slice and can be resubmitted without redoing finished work.

set -euo pipefail

REPO=${KBC_REPO:-/groups/bshapira_group/guyhada/KBC}
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/run_benchmark.py \
  --all \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --epochs 8 \
  --llm-epochs 4 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --llm-learning-rate 1e-4 \
  --max-events 64 \
  --hidden-size 128 \
  --layers 2 \
  --bf16 \
  --tpp-max-prefixes 8
