#!/bin/bash
#SBATCH --job-name=kbc-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/cpu_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/cpu_%A_%a.err

# Everything except the three LLM entries. These models are small enough that
# CPU is the right resource for them, which also keeps the scarce GPU quota for
# the LoRA-adapted language models that actually need it.

set -euo pipefail

REPO=/home/guyhada/KBC
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

python scripts/run_benchmark.py \
  --all \
  --skip-gpu-only \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --epochs 8 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --max-events 64 \
  --hidden-size 128 \
  --layers 2 \
  --cpu \
  --tpp-max-prefixes 8
