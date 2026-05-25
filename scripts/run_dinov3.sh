#!/bin/bash
#SBATCH --job-name=derm_dinov3
#SBATCH --partition=roxanad
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage:
#   sbatch scripts/run_dinov3.sh

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi
SIF_STORE="/scratch/users/$USER/simg"
CONTAINER="$SIF_STORE/vllm-v0.11.0.sif"
SCRATCH_TMP=/scratch/users/$USER/tmp

mkdir -p "$SCRATCH_TMP" /scratch/users/$USER/huggingface logs

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

echo "==> Step 1: extracting DINOv3 embeddings"
run_in_container "python scripts/dinov3_extract_embeddings.py \
    --ddi-root /ddi \
    --out artifacts/dinov3_embeddings.npz"

echo "==> Step 2: cross-fit logistic probe"
run_in_container "python scripts/dinov3_crossfit_probe.py \
    --ddi-root /ddi \
    --embeddings artifacts/dinov3_embeddings.npz \
    --out artifacts/responses.jsonl"

echo "==> DINOv3 done."
