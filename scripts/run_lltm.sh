#!/bin/bash
#SBATCH --job-name=derm_dif_lltm
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run script 05 (LLTM nested-model decomposition). The bootstrap (n=2000)
# over a wide one-hot design matrix (75-level lesion_category dummies +
# malignancy + image features + FST dummies) exceeds login-node memory; has
# to be scheduled.

set -euo pipefail
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false

cd "$PROJECT_ROOT"
python scripts/05_lltm.py --ddi-root "$DDI_ROOT"
echo "----- artifacts/lltm.json -----"
cat artifacts/lltm.json
