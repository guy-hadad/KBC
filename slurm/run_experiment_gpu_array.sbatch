#!/bin/bash
#SBATCH --job-name=kbc-exp-gpu
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/exp_gpu_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/exp_gpu_%A_%a.err

# Submit with, for example:
# sbatch --export=ALL,KBC_EXPERIMENT_SUITE=gem-full-banking \
#   slurm/run_experiment_gpu_array.sbatch

set -euo pipefail

REPO=/home/guyhada/KBC
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false

SUITE=${KBC_EXPERIMENT_SUITE:?set KBC_EXPERIMENT_SUITE to a named GPU suite}

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python scripts/run_experiments.py \
  --suite "${SUITE}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --max-events 64 \
  --hidden-size 128 \
  --layers 2 \
  --bf16
