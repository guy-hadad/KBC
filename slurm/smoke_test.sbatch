#!/bin/bash
#SBATCH --job-name=kbc-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/smoke_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/smoke_%j.err

# Run one small cell for every method on one dataset, on a GPU. This is the
# gate before submitting the full array: it proves the environment, the GPU
# path and every architecture in the zoo.

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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python scripts/run_benchmark.py \
  --dataset banksim \
  --samples 128 \
  --epochs 2 \
  --llm-epochs 1 \
  --batch-size 32 \
  --max-events 32 \
  --hidden-size 64 \
  --layers 2 \
  --tpp-max-prefixes 2 \
  --overwrite \
  --results-dir "${KBC_OUTPUT_ROOT}/smoke_results"
