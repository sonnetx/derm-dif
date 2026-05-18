#!/bin/bash
#SBATCH --job-name=derm_dif_pilot
#SBATCH --partition=dev
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sherlock wrapper for scripts/01_pilot_calibration.py.
# Login nodes cap CPU + memory; loading BiomedCLIP exceeds those caps, so the
# pilot must run as a batch job. Submit with:
#   sbatch scripts/run_pilot.sh

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1

source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

cd "$PROJECT_ROOT"

python scripts/01_pilot_calibration.py --ddi-root "$DDI_ROOT"

echo "----- artifacts/pilot.json -----"
cat artifacts/pilot.json
