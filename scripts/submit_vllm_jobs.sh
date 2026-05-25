#!/bin/bash
# Submit the full vLLM pipeline: pull container, then query all three open-weight
# models as dependent jobs. Each model job starts only after the pull succeeds.
#
# Usage (from project root on login node):
#   bash scripts/submit_vllm_jobs.sh

set -euo pipefail

cd /home/groups/roxanad/sonnet/derm-dif
mkdir -p logs

# Step 1: submit the container pull job
PULL_JID=$(sbatch --parsable scripts/setup_container.sh)
echo "==> Pull job: $PULL_JID"

# Step 2: submit each model query job, dependent on the pull succeeding
for MODEL in \
    "llava-hf/llava-v1.6-mistral-7b-hf" \
    "Qwen/Qwen2-VL-7B-Instruct" \
    "OpenGVLab/InternVL2-8B"
do
    JID=$(sbatch --parsable \
        --dependency=afterok:$PULL_JID \
        scripts/run_vllm_container.sh "$MODEL")
    echo "==> Model job: $JID  ($MODEL)"
done

echo ""
echo "==> Submitted. Monitor with: squeue -u \$USER"
