#!/bin/bash
#SBATCH --job-name=derm_dif_query_zeroshot
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Query the three contrastive zero-shot respondents (CLIP / SigLIP / BiomedCLIP)
# against every DDI item. Submit with:
#   sbatch scripts/run_query_zeroshot.sh
#
# CPU-only: contrastive ViT-L forward passes on 656 images are fast enough
# on CPU (~minutes per model). Memory ceiling for SigLIP-L is the binding
# constraint, hence 24G.

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false

export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

cd "$PROJECT_ROOT"
python scripts/02_query_models.py \
    --ddi-root "$DDI_ROOT" \
    --source contrastive-zeroshot \
    --out artifacts/responses.jsonl

echo "----- zeroshot query summary -----"
python scripts/query_summary.py
