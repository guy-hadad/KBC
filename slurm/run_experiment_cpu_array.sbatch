#!/bin/bash
#SBATCH --job-name=kbc-exp-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/exp_cpu_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/exp_cpu_%A_%a.err

# Submit with, for example:
# sbatch --export=ALL,KBC_EXPERIMENT_SUITE=objective-factorial \
#   slurm/run_experiment_cpu_array.sbatch

set -euo pipefail

REPO=/home/guyhada/KBC
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

SUITE=${KBC_EXPERIMENT_SUITE:?set KBC_EXPERIMENT_SUITE to a named non-LLM suite}

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

python scripts/run_experiments.py \
  --suite "${SUITE}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --epochs 8 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --max-events 64 \
  --hidden-size 128 \
  --layers 2 \
  --cpu
