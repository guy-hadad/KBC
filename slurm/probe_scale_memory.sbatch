#!/bin/bash
#SBATCH --job-name=kbc-scale-probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/scale_probe_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/scale_probe_%j.err

# Confirms that the largest cell of each large dataset fits the 48 GB budget
# before a full array is submitted. Run one dataset per invocation:
#   sbatch --export=ALL,KBC_PROBE_DATASET=synthea slurm/probe_scale_memory.sbatch

set -euo pipefail

REPO=${KBC_REPO:-/groups/bshapira_group/guyhada/KBC}
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

DATASET=${KBC_PROBE_DATASET:?set KBC_PROBE_DATASET}

mkdir -p "${KBC_OUTPUT_ROOT}/slurm"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KBC

POOL=$(python -c "
from eventfm.benchmark.experiments import scaling_sample_sizes
print(scaling_sample_sizes('${DATASET}')[-1])
")
echo "[probe] ${DATASET} largest sample size = ${POOL}"

for METHOD in pragma ntpp-gru mambular; do
  for TASK in classification tpp; do
    echo "=== ${DATASET} ${TASK} ${METHOD} n=${POOL} ==="
    /usr/bin/time -v python scripts/run_benchmark.py \
      --dataset "${DATASET}" --task "${TASK}" --method "${METHOD}" \
      --samples "${POOL}" --epochs 8 --batch-size 32 --max-events 64 --cpu \
      2>&1 | grep -E '^\{|Maximum resident|Elapsed \(wall' || true
  done
done
