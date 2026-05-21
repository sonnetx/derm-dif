#!/bin/bash
#SBATCH --job-name=derm_dif_dinov3
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# DINOv3 cross-embedding robustness refit (PAP-required).
#
# Re-fits the amortized Rasch with DINOv3 image embeddings instead of
# BiomedCLIP. Reports both the primary-endpoint Delta and the Spearman
# correlation of per-item difficulty between the BiomedCLIP and DINOv3
# fits. A high Spearman is evidence that the difficulty estimates are
# robust to the embedding-backbone choice and not driven by the
# BiomedCLIP-as-both-embedder-and-respondent circularity concern.

set -euo pipefail
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false
export HF_HOME=/scratch/users/$USER/huggingface
export TORCH_HOME=/scratch/users/$USER/torch

cd "$PROJECT_ROOT"

echo "----- Fitting amortized Rasch with DINOv3 embeddings -----"
python scripts/03_fit_amortized.py \
    --ddi-root "$DDI_ROOT" \
    --source api-openai,api-anthropic,api-google,contrastive-zeroshot \
    --embedding-backend dinov3 \
    --out-dir artifacts/rasch_dinov3

echo ""
echo "----- Aggregate DIF on DINOv3 fit -----"
python scripts/04_aggregate_dif.py \
    --ddi-root "$DDI_ROOT" \
    --rasch-dir artifacts/rasch_dinov3 \
    --out artifacts/aggregate_dif_dinov3.json

echo ""
echo "----- Cross-embedding comparison -----"
echo "BiomedCLIP (canonical):"
python -c "import json; d=json.load(open('artifacts/aggregate_dif.json')); print(f\"  delta={d['delta']:.3f}, CI=[{d['ci_low']:.3f}, {d['ci_high']:.3f}], decision={d['decision']}\")"
echo "DINOv3 (robustness):"
python -c "import json; d=json.load(open('artifacts/aggregate_dif_dinov3.json')); print(f\"  delta={d['delta']:.3f}, CI=[{d['ci_low']:.3f}, {d['ci_high']:.3f}], decision={d['decision']}\")"

echo ""
echo "----- Difficulty Spearman between backbones -----"
python - <<'PYEOF'
import pickle
from scipy.stats import spearmanr
with open("artifacts/rasch/amortized_fit.pkl", "rb") as f:
    fit_b = pickle.load(f)
with open("artifacts/rasch_dinov3/amortized_fit.pkl", "rb") as f:
    fit_d = pickle.load(f)
rho, p = spearmanr(fit_b["difficulty"], fit_d["difficulty"])
print(f"Per-item difficulty Spearman rho (BiomedCLIP vs DINOv3): {rho:.3f} (p={p:.3g})")
print(f"Per-respondent theta Spearman rho: ", end="")
rho_t, p_t = spearmanr(fit_b["theta"], fit_d["theta"])
print(f"{rho_t:.3f} (p={p_t:.3g})")
PYEOF
