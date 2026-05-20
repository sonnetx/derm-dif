#!/bin/bash
#SBATCH --job-name=derm_dif_irt
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build response matrix, embed all 656 images with BiomedCLIP, fit amortized
# Rasch, run the primary-endpoint aggregate-DIF analysis.
#
# Submit with:
#   sbatch scripts/run_irt.sh                 # all currently queried respondents
#   sbatch scripts/run_irt.sh api-openai,api-anthropic   # restrict to a subset
#
# Defaults to the closed-API + contrastive-zeroshot subset (the panel we have
# real data for); pass arg overrides via --source on the command line.

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

# Default source filter: everything we've actually queried.
SOURCES="${1:-api-openai,api-anthropic,api-google,contrastive-zeroshot}"

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false
export HF_HOME=/scratch/users/$USER/huggingface
export TORCH_HOME=/scratch/users/$USER/torch

cd "$PROJECT_ROOT"

echo "----- Fitting amortized Rasch on sources: $SOURCES -----"
python scripts/03_fit_amortized.py \
    --ddi-root "$DDI_ROOT" \
    --source "$SOURCES"

echo ""
echo "----- Aggregate DIF (primary endpoint) -----"
python scripts/04_aggregate_dif.py \
    --ddi-root "$DDI_ROOT"

echo ""
echo "----- Summary -----"
ls -la artifacts/rasch/
