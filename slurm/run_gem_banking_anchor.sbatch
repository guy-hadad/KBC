#!/bin/bash
#SBATCH --job-name=kbc-gem-bank-anchor
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/gem_bank_anchor_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/gem_bank_anchor_%j.err

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

cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

run_anchor_cells() {
  # Exact gem-main-banking indices for byte-f32, n=2048, seed=13:
  # BankSim, PaySim, IBM AML, and MBD-mini respectively.
  for CELL_INDEX in 5 70 135 200; do
    python scripts/run_experiments.py \
      --suite gem-main-banking \
      --array-index "${CELL_INDEX}" \
      --array-size 260 \
      --only-gpu-only \
      --epochs 5 \
      --batch-size 4 \
      --learning-rate 1e-3 \
      --gradient-accumulation-steps 4 \
      --max-events 64 \
      --hidden-size 128 \
      --layers 2 \
      --bf16 \
      --results-dir "${KBC_OUTPUT_ROOT}/paper_runs/gem-main-banking"
  done

  # Continue every banking-only GEM suite after the four anchor cells pass.
  scontrol release 19972176 19972179 19972181 19972182 19972183
}

run_anchor_cells &
WORKER_PID=$!
wait "${WORKER_PID}"
