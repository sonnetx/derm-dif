#!/bin/bash
#SBATCH --job-name=derm_sensitivity
#SBATCH --partition=roxanad
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi
SIF_STORE="/scratch/users/$USER/simg"
CONTAINER="$SIF_STORE/vllm-v0.11.0.sif"
SCRATCH_TMP=/scratch/users/$USER/tmp

mkdir -p "$SCRATCH_TMP" "$PROJECT_ROOT/logs" \
         "$PROJECT_ROOT/artifacts/rasch_flat" \
         "$PROJECT_ROOT/artifacts/rasch_pruned"

TOOL=$(command -v apptainer || command -v singularity)

cd "$PROJECT_ROOT"

run_in_container() {
    "$TOOL" exec --nv \
        --containall \
        -B "$PROJECT_ROOT":/workspace \
        -B "$DDI_ROOT":/ddi \
        -B "/scratch/users/$USER":/scratch_user \
        -B "$SCRATCH_TMP":/tmp \
        --home /scratch_user \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=/workspace/src:/workspace" \
        --env "HF_HOME=/scratch_user/huggingface" \
        --pwd /workspace \
        "$CONTAINER" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace/src:/workspace && $*"
}

SOURCE_ALL="api-openai,api-anthropic,api-google,contrastive-zeroshot,huggingface"

# Cached embeddings shared across flat/pruned fits
cp artifacts/rasch/embeddings_biomedclip.npy artifacts/rasch_flat/
cp artifacts/rasch/embeddings_biomedclip.npy artifacts/rasch_pruned/

echo "==> (1) Flat-prior fit J=9 (difficulty-l2=0, prune-saturated)"
run_in_container "python scripts/03_fit_amortized.py \
    --ddi-root /ddi \
    --difficulty-l2 0.0 \
    --prune-saturated \
    --out-dir artifacts/rasch_flat"
run_in_container "python scripts/04_aggregate_dif.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch_flat \
    --out artifacts/aggregate_dif_flat.json"

echo "==> (2) Hard-pruning regularized fit J=9 (default difficulty-l2, prune-saturated)"
run_in_container "python scripts/03_fit_amortized.py \
    --ddi-root /ddi \
    --prune-saturated \
    --out-dir artifacts/rasch_pruned"
run_in_container "python scripts/04_aggregate_dif.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch_pruned \
    --out artifacts/aggregate_dif_pruned.json"

echo "==> (3) Seed sweep J=9"
run_in_container "python scripts/seed_sweep.py \
    --ddi-root /ddi \
    --source '$SOURCE_ALL' \
    --out artifacts/seed_sweep.json"

echo "==> (4) Lambda sweep J=9"
run_in_container "python scripts/lambda_sweep.py \
    --ddi-root /ddi \
    --source '$SOURCE_ALL' \
    --out artifacts/lambda_sweep.json"

echo "==> (5) Cross-backbone Spearman J=9"
run_in_container "python3 -c \"
import pickle, numpy as np
from scipy.stats import spearmanr
bc = pickle.load(open('artifacts/rasch/amortized_fit.pkl', 'rb'))
dn = pickle.load(open('artifacts/rasch_dinov3/amortized_fit.pkl', 'rb'))
r_item = spearmanr(bc['difficulty'], dn['difficulty']).statistic
r_resp = spearmanr(bc['theta'], dn['theta']).statistic
import json
out = {'item_difficulty_rho': float(r_item), 'respondent_ability_rho': float(r_resp)}
open('artifacts/backbone_spearman.json', 'w').write(json.dumps(out, indent=2))
print(out)
\""

echo "==> (6) Synthetic validation J=9, 200 reps"
run_in_container "python scripts/10_synthetic_validation.py \
    --n-reps 200 \
    --out artifacts/synthetic_validation.json"

echo "==> All done. Summary:"
run_in_container "python3 -c \"
import json, pathlib
flat   = json.loads(pathlib.Path('artifacts/aggregate_dif_flat.json').read_text())
pruned = json.loads(pathlib.Path('artifacts/aggregate_dif_pruned.json').read_text())
seed   = json.loads(pathlib.Path('artifacts/seed_sweep.json').read_text())
lam    = json.loads(pathlib.Path('artifacts/lambda_sweep.json').read_text())
sp     = json.loads(pathlib.Path('artifacts/backbone_spearman.json').read_text())
print('Flat-prior delta:', round(flat['delta'],3), '  CI:', round(flat['ci_low'],3), round(flat['ci_high'],3))
print('Hard-pruning delta:', round(pruned['delta'],3), '  CI:', round(pruned['ci_low'],3), round(pruned['ci_high'],3))
print('Seed sweep delta range: [', round(seed['delta_min'],3), ',', round(seed['delta_max'],3), ']  all_same:', seed['all_decisions_same'])
lam_deltas = [round(r['delta'],3) for r in lam['per_lambda']]
print('Lambda sweep deltas:', lam_deltas, '  all_no_effect:', lam['all_no_effect'])
print('Backbone Spearman: item rho=', round(sp['item_difficulty_rho'],3), ' resp rho=', round(sp['respondent_ability_rho'],3))
\""
