#!/bin/bash
#SBATCH --job-name=derm_dif_aggdif
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run script 04 alone (without re-fitting the Rasch model in script 03).
# The 2000-resample bootstrap with 75-level lesion-category dummies trips
# Sherlock login-node OOM, so it has to run via Slurm.

set -euo pipefail
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false

cd "$PROJECT_ROOT"
python scripts/04_aggregate_dif.py --ddi-root "$DDI_ROOT"
echo "----- aggregate_dif.json -----"
cat artifacts/aggregate_dif.json
