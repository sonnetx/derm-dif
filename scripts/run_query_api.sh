#!/bin/bash
#SBATCH --job-name=derm_dif_query_api
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Query the three closed-API respondents (OpenAI / Anthropic / Google) against
# every DDI item under the primary protocol. Submit with:
#   sbatch scripts/run_query_api.sh
#
# Open-weight VLMs and contrastive/SSL respondents are queried by separate
# wrappers (run_query_vllm.sh / run_query_zeroshot.sh) -- not yet implemented.

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
# Exports OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY for the SDKs
source ~/.secrets

export HF_HOME=/scratch/users/$USER/huggingface
export TORCH_HOME=/scratch/users/$USER/torch

cd "$PROJECT_ROOT"
python scripts/02_query_models.py \
    --ddi-root "$DDI_ROOT" \
    --source api-openai,api-anthropic,api-google \
    --out artifacts/responses.jsonl

echo "----- query summary -----"
wc -l artifacts/responses.jsonl
echo "----- error count -----"
grep -c '"error":' artifacts/responses.jsonl || true
