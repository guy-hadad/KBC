#!/bin/bash
#SBATCH --job-name=kbc-smoke-paper-gpu
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/smoke_paper_gpu_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/smoke_paper_gpu_%j.err

set -euo pipefail

REPO=${KBC_REPO:-/groups/bshapira_group/guyhada/KBC}
cd "${REPO}"
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/smoke_paper.py --mode gpu
