#!/bin/bash
#SBATCH --job-name=kbc-llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/llm_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/llm_%A_%a.err

# The three LoRA-adapted language-model entries only. Everything else runs on
# the CPU array, so this job holds the GPU for exactly the cells that need one.

set -euo pipefail

REPO=/home/guyhada/KBC
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/run_benchmark.py \
  --all \
  --method tpp-llm \
  --method language-tpp \
  --method mm-tpp \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --llm-epochs 4 \
  --llm-learning-rate 1e-4 \
  --batch-size 32 \
  --max-events 64 \
  --bf16 \
  --tpp-max-prefixes 8
