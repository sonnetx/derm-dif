#!/bin/bash
#SBATCH --job-name=derm_dif_vllm
#SBATCH --partition=roxanad
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Serve a single open-weight VLM via vLLM and query it against DDI.
#
# Usage:
#   sbatch scripts/run_query_vllm.sh llava-hf/llava-1.5-13b-hf
#   sbatch scripts/run_query_vllm.sh Qwen/Qwen2-VL-7B-Instruct
#   sbatch scripts/run_query_vllm.sh OpenGVLab/InternVL2-8B
#
# Requires vllm to be installed in the project venv (see setup_sherlock.sh).

set -euo pipefail

MODEL_ID="${1:?usage: sbatch run_query_vllm.sh <model_id>}"
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi
VLLM_PORT=8765
LOG_DIR=$PROJECT_ROOT/logs/vllm
mkdir -p "$LOG_DIR"

ml python/3.12.1 cuda/11.7.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

cd "$PROJECT_ROOT"

echo "==> Starting vLLM for $MODEL_ID on port $VLLM_PORT"
vllm serve "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --max-model-len 4096 \
    --dtype auto \
    > "$LOG_DIR/vllm_$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!

# Tear down vLLM on exit (success or failure)
trap 'echo "==> Stopping vLLM (pid=$VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

echo "==> Waiting for vLLM server to come up (max 15 minutes for first-load model download + warmup)..."
for i in $(seq 1 90); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "==> vLLM ready after ${i}0s"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "==> vLLM failed to come up. Tail of vLLM log:"
        tail -50 "$LOG_DIR/vllm_$SLURM_JOB_ID.log"
        exit 1
    fi
    sleep 10
done

echo "==> Running script 02 with --model-id $MODEL_ID"
export VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1"
python scripts/02_query_models.py \
    --ddi-root "$DDI_ROOT" \
    --source huggingface \
    --model-id "$MODEL_ID" \
    --out artifacts/responses.jsonl

echo ""
echo "==> Done. Per-model summary:"
python scripts/query_summary.py | grep -E "(model_id|$(echo "$MODEL_ID" | sed 's|[/.-]|.|g'))" || true
