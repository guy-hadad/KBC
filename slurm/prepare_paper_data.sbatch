#!/bin/bash
#SBATCH --job-name=kbc-data-paper
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/data_paper_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/data_paper_%j.err

set -euo pipefail

REPO=${KBC_REPO:-/groups/bshapira_group/guyhada/KBC}
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

python scripts/prepare_data.py --paper --force
