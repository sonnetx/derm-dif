#!/bin/bash
#SBATCH --job-name=derm_vllm
#SBATCH --partition=roxanad
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Serve one open-weight VLM via vLLM (Apptainer container) and query it against DDI.
# Both the vLLM server and the query script run inside the container.
#
# Usage:
#   sbatch scripts/run_vllm_container.sh llava-hf/llava-v1.6-mistral-7b-hf
#   sbatch scripts/run_vllm_container.sh Qwen/Qwen2-VL-7B-Instruct
#   sbatch scripts/run_vllm_container.sh OpenGVLab/InternVL2-8B
#
# Prerequisites:
#   sbatch scripts/setup_container.sh   # pull .sif + create scratch venv once

set -euo pipefail

MODEL_ID="${1:?usage: sbatch run_vllm_container.sh <model_id>}"

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi
SIF_STORE="/scratch/users/$USER/simg"
CONTAINER="$SIF_STORE/vllm-v0.11.0.sif"
SCRATCH_VENV="/scratch/users/$USER/container_env"
VLLM_PORT=8766
LOG_DIR="$PROJECT_ROOT/logs/vllm"
SCRATCH_TMP=/scratch/users/$USER/tmp

mkdir -p "$LOG_DIR" "$SCRATCH_TMP" /scratch/users/$USER/huggingface

TOOL=$(command -v apptainer || command -v singularity)

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    echo "Run:  sbatch $PROJECT_ROOT/scripts/setup_container.sh"
    exit 1
fi

if [ ! -d "$SCRATCH_VENV" ]; then
    echo "ERROR: Scratch venv not found at $SCRATCH_VENV"
    echo "Run:  sbatch $PROJECT_ROOT/scripts/setup_container.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

# Helper: run a command inside the container (no GPU needed)
run_in_container() {
    "$TOOL" exec \
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

# ── Start vLLM server inside the container (with GPU) ───────────────────────
echo "==> Starting vLLM server for $MODEL_ID"
"$TOOL" exec --nv \
    --containall \
    -B "$PROJECT_ROOT":/workspace \
    -B "/scratch/users/$USER":/scratch_user \
    -B "$SCRATCH_TMP":/tmp \
    --home /scratch_user \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=/workspace/src:/workspace" \
    --env "HF_HOME=/scratch_user/huggingface" \
    --env "VLLM_ATTENTION_BACKEND=TORCH_SDPA" \
    --pwd /workspace \
    "$CONTAINER" \
    bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace/src:/workspace && vllm serve '$MODEL_ID' --port $VLLM_PORT --max-model-len 4096 --dtype auto --trust-remote-code" \
    > "$LOG_DIR/server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

trap 'echo "==> Stopping vLLM (pid=$VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

# ── Wait for server to be ready ──────────────────────────────────────────────
echo "==> Waiting for vLLM to come up (max 15 min)..."
for i in $(seq 1 90); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "==> vLLM ready after ${i}0s"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "ERROR: vLLM did not start within 15 minutes. Log tail:"
        tail -50 "$LOG_DIR/server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
done

# ── Run queries inside the container (hits vLLM via localhost HTTP) ──────────
echo "==> Querying $MODEL_ID against DDI"
VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1" \
    run_in_container \
    "VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1 python scripts/02_query_models.py \
        --ddi-root /ddi \
        --source huggingface \
        --model-id '$MODEL_ID' \
        --out artifacts/responses.jsonl"

echo ""
echo "==> Done. Response summary:"
run_in_container "python scripts/query_summary.py" \
    | grep -E "(model_id|$(echo "$MODEL_ID" | sed 's|[/.-]|.|g'))" || true
