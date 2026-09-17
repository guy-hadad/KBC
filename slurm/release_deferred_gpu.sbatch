#!/bin/bash
#SBATCH --job-name=kbc-release-deferred
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/groups/bshapira_group/guyhada/KBC_outputs/slurm/release_deferred_%j.out
#SBATCH --error=/groups/bshapira_group/guyhada/KBC_outputs/slurm/release_deferred_%j.err

set -euo pipefail

# Submitted after all banking GEM arrays. Resume the deferred non-banking and
# general GPU suites without creating replacement experiment arrays.
scontrol release 19972174 19972184 19972185 19972187
