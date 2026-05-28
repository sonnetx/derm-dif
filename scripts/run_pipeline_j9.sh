#!/bin/bash
#SBATCH --job-name=derm_pipeline_j9
#SBATCH --partition=roxanad
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Full analysis pipeline refit for J=9 panel (adds Qwen2-VL-7B + LLaVA-v1.6 to original J=7).
#
# Usage:
#   sbatch scripts/run_pipeline_j9.sh

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi
SIF_STORE="/scratch/users/$USER/simg"
CONTAINER="$SIF_STORE/vllm-v0.11.0.sif"
SCRATCH_TMP=/scratch/users/$USER/tmp

mkdir -p "$SCRATCH_TMP" /scratch/users/$USER/huggingface "$PROJECT_ROOT/logs"

TOOL=$(command -v apptainer || command -v singularity)

cd "$PROJECT_ROOT"

if [ -f ~/.secrets ]; then set -a; source ~/.secrets; set +a; fi

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
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        --pwd /workspace \
        "$CONTAINER" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace/src:/workspace && $*"
}

echo "==> Step 1: Fit amortized Rasch (J=9, BiomedCLIP)"
run_in_container "python scripts/03_fit_amortized.py \
    --ddi-root /ddi \
    --responses artifacts/responses.jsonl \
    --out-dir artifacts/rasch \
    --embedding-backend biomedclip"

echo "==> Step 2: Primary endpoint Delta"
run_in_container "python scripts/04_aggregate_dif.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch \
    --out artifacts/aggregate_dif.json"

echo "==> Step 3: LLTM nested OLS"
run_in_container "python scripts/05_lltm.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch \
    --out artifacts/lltm.json"

echo "==> Step 4: Mechanism and refusal DIF"
run_in_container "python scripts/06_mechanism_and_refusal.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch \
    --out artifacts/mechanism.json"

echo "==> Step 5: Configural invariance"
run_in_container "python scripts/07_configural_invariance.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch \
    --out artifacts/configural.json"

echo "==> Step 6: Generative-only stratified"
run_in_container "python scripts/08_generative_stratified.py \
    --ddi-root /ddi \
    --responses artifacts/responses.jsonl \
    --out artifacts/generative_stratified.json"

echo "==> Step 7: Per-model FST logistic"
run_in_container "python scripts/09_per_model_fst_logistic.py \
    --ddi-root /ddi \
    --responses artifacts/responses.jsonl \
    --out artifacts/per_model_fst_logistic.json"

echo "==> Step 8: Composition-aware baselines"
run_in_container "python scripts/11_baselines.py \
    --ddi-root /ddi \
    --responses artifacts/responses.jsonl \
    --out artifacts/baselines.json"

echo "==> Step 9: DINOv3 cross-backbone robustness"
run_in_container "python scripts/03_fit_amortized.py \
    --ddi-root /ddi \
    --responses artifacts/responses.jsonl \
    --out-dir artifacts/rasch_dinov3 \
    --embedding-backend dinov3"

run_in_container "python scripts/04_aggregate_dif.py \
    --ddi-root /ddi \
    --rasch-dir artifacts/rasch_dinov3 \
    --out artifacts/aggregate_dif_dinov3.json"

echo "==> All steps done. Key results:"
run_in_container "python -c \"
import json, pathlib
dif = json.loads(pathlib.Path('artifacts/aggregate_dif.json').read_text())
lltm = json.loads(pathlib.Path('artifacts/lltm.json').read_text())
bl = json.loads(pathlib.Path('artifacts/baselines.json').read_text())
dif9 = json.loads(pathlib.Path('artifacts/aggregate_dif_dinov3.json').read_text())
print('=== Primary (BiomedCLIP, J=9) ===')
print(f'  Delta: {dif[\\\"delta\\\"]:.4f}  CI: [{dif[\\\"ci_low\\\"]:.4f}, {dif[\\\"ci_high\\\"]:.4f}]  decision: {dif[\\\"decision\\\"]}')
print('=== LLTM ===')
for k, v in lltm.items():
    print(f'  {k}: {v}')
print('=== DINOv3 backbone ===')
print(f'  Delta: {dif9[\\\"delta\\\"]:.4f}  CI: [{dif9[\\\"ci_low\\\"]:.4f}, {dif9[\\\"ci_high\\\"]:.4f}]  decision: {dif9[\\\"decision\\\"]}')
print('=== Baselines ===')
pm = bl[\\\"propensity_match\\\"]
pl = bl[\\\"pooled_logistic\\\"]
print(f'  Propensity match gap: {pm[\\\"gap_matched\\\"]:+.4f}  [{pm[\\\"ci_low\\\"]:+.4f}, {pm[\\\"ci_high\\\"]:+.4f}]')
print(f'  Pooled logistic beta_FST: {pl[\\\"fst_vvi_beta\\\"]:+.4f}  [{pl[\\\"ci_low\\\"]:+.4f}, {pl[\\\"ci_high\\\"]:+.4f}]')
\""

echo "==> Done."
