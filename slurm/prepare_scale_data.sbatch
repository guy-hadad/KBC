#!/bin/bash
#SBATCH --job-name=kbc-data-scale
#SBATCH --partition=cpu
#SBATCH --constraint=cpu128|cpu256
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/data_scale_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/data_scale_%j.err

# Downloads and converts the three large-scale benchmarks. Full MBD is a ~31 GB
# pull, so this is the long prerequisite of the scale-datasets campaign.
#
# The `cpu128|cpu256` constraint is not about cores. Conversion is the only
# stage that uses polars, whose wheels need avx2/fma/bmi1/bmi2/lzcnt/movbe; the
# older `ise-cpu-intl-*` nodes lack them and the process dies with SIGILL
# several seconds in (job 21067512 on ise-cpu-intl-15). Running a cell needs no
# polars, so the experiment arrays are not constrained.

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

# Fail loudly and immediately rather than core-dumping mid-conversion. Ask
# polars itself in a subprocess: SIGILL kills the child, not this script.
# Grepping /proc/cpuinfo is not a substitute -- AMD reports LZCNT as `abm`, so a
# flag-name check reports false failures on nodes that are actually fine.
if ! python -c "import polars as pl; pl.DataFrame({'a': [1, 2]}).select(pl.col('a').sum())" \
     >/dev/null 2>&1; then
  echo "[prepare] $(hostname) cannot run polars; resubmit elsewhere." >&2
  exit 1
fi
echo "[prepare] $(hostname) runs polars"

python scripts/prepare_data.py --scale
