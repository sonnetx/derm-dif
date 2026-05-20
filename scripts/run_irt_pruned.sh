#!/bin/bash
#SBATCH --job-name=derm_dif_irt_pruned
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Robustness counterpart to scripts/run_irt.sh: same panel and fit, but with
# saturated items (all-correct or all-wrong across the panel) pruned BEFORE
# fitting rather than relying on the L2 difficulty prior to bound them.
#
# Fit outputs go to artifacts/rasch_pruned/ and the aggregate-DIF JSON to
# artifacts/aggregate_dif_pruned.json so the canonical (artifacts/rasch/)
# fit is preserved for comparison.

set -euo pipefail
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi
SOURCES="${1:-api-openai,api-anthropic,api-google,contrastive-zeroshot}"

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false
export HF_HOME=/scratch/users/$USER/huggingface
export TORCH_HOME=/scratch/users/$USER/torch

cd "$PROJECT_ROOT"

echo "----- Fitting amortized Rasch on $SOURCES with --prune-saturated -----"
python scripts/03_fit_amortized.py \
    --ddi-root "$DDI_ROOT" \
    --source "$SOURCES" \
    --prune-saturated \
    --out-dir artifacts/rasch_pruned

echo ""
echo "----- Aggregate DIF on pruned fit -----"
python scripts/04_aggregate_dif.py \
    --ddi-root "$DDI_ROOT" \
    --rasch-dir artifacts/rasch_pruned \
    --out artifacts/aggregate_dif_pruned.json

echo ""
echo "----- Summary -----"
ls -la artifacts/rasch_pruned/
echo
echo "--- pruned vs. canonical Delta ---"
echo "Canonical (L2 prior only):"
cat artifacts/aggregate_dif.json | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(f\"  delta={d['delta']:.3f}, CI=[{d['ci_low']:.3f}, {d['ci_high']:.3f}], decision={d['decision']}\")"
echo "Pruned (L2 prior + saturated items removed):"
cat artifacts/aggregate_dif_pruned.json | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(f\"  delta={d['delta']:.3f}, CI=[{d['ci_low']:.3f}, {d['ci_high']:.3f}], decision={d['decision']}\")"
