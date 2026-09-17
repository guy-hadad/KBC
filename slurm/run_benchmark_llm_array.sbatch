#!/bin/bash
#SBATCH --job-name=kbc-llm
# The installed torch is a CUDA 13 build, which dropped Pascal (sm_61), so a
# GTX 1080 fails every cell with "no kernel image is available". A node-level
# --constraint is NOT sufficient: the pheno nodes advertise both rtx_3090 and
# gtx_1080 features, match the constraint, and then hand out the 1080. The GPU
# type has to be named in --gres itself.
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/llm_%A_%a.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/llm_%A_%a.err

# The three LoRA-adapted language-model entries only. Everything else runs on
# the CPU array, so this job holds the GPU for exactly the cells that need one.

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

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Preflight: prove this GPU can actually run a kernel before spending hours on
# cells that would each fail the same way and be cached as failures.
python - <<'PREFLIGHT'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("preflight: no CUDA device visible")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
try:
    (torch.zeros(8, device="cuda") + 1).sum().item()
except Exception as exc:  # noqa: BLE001 - fail fast with the real reason
    sys.exit(
        "preflight: {} (sm_{}{}) cannot run kernels from this torch build: {}".format(
            name, capability[0], capability[1], exc
        )
    )
print("preflight ok: {} (sm_{}{})".format(name, capability[0], capability[1]))
PREFLIGHT

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
