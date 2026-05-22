#!/bin/bash
#SBATCH --job-name=derm_dif_configural
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run scripts/07_configural_invariance.py via Slurm. Two amortized-Rasch fits
# (one per FST subset) at J=7 exceeds the login-node memory/CPU cap.

set -euo pipefail
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false

cd "$PROJECT_ROOT"
python scripts/07_configural_invariance.py --ddi-root "$DDI_ROOT"
echo "----- artifacts/configural.json -----"
cat artifacts/configural.json
