#!/bin/bash
#SBATCH --job-name=derm_setup
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# ONE-TIME SETUP: Pull vLLM container and create scratch venv with project deps.
# Run once before using run_vllm_container.sh / submit_vllm_jobs.sh.
#
# Usage:
#   sbatch scripts/setup_container.sh

set -e

PROJECT_DIR="/home/groups/roxanad/sonnet/derm-dif"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="vllm-v0.11.0.sif"
PIP_CACHE="/scratch/users/$USER/pip_cache"

mkdir -p "$SIF_STORE" "$PIP_CACHE" \
    /scratch/users/$USER/tmp \
    /scratch/users/$USER/huggingface \
    logs

export TMPDIR="/scratch/users/$USER/tmp"

TOOL=$(command -v apptainer || command -v singularity)
echo "INFO: Using container tool: $TOOL"

# --- Pull vLLM image (one-time, ~15 GB) ---
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "INFO: Pulling vLLM v0.11.0 container (~15 GB)..."
    cd "$SIF_STORE"
    $TOOL pull "$SIF_IMAGE" docker://vllm/vllm-openai:v0.11.0
    cd "$PROJECT_DIR"
else
    echo "INFO: Container image already exists at $SIF_STORE/$SIF_IMAGE"
fi

# --- Create venv inside container with project deps ---
echo "INFO: Setting up virtual environment inside container..."

"$TOOL" exec \
    --containall \
    -B "$PROJECT_DIR:/workspace" \
    -B "/scratch/users/$USER:/scratch_user" \
    -B "$PIP_CACHE:/root/.cache/pip" \
    -B "/scratch/users/$USER/tmp:/tmp" \
    --home /scratch_user \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=/workspace/src:/workspace" \
    --pwd /workspace \
    "$SIF_STORE/$SIF_IMAGE" \
    bash -c "
    set -e

    echo 'INFO: Python: '\$(which python3)' ('\$(python3 --version)')'
    echo 'INFO: PyTorch: '\$(python3 -c 'import torch; print(torch.__version__)')
    echo 'INFO: vLLM:    '\$(python3 -c 'import vllm; print(vllm.__version__)')

    VENV=/scratch_user/container_env

    if [ ! -d \$VENV ]; then
        echo 'INFO: Creating virtual environment...'
        python3 -m venv --system-site-packages \$VENV
    else
        echo 'INFO: Virtual environment already exists'
    fi

    source \$VENV/bin/activate
    export PATH=\$VIRTUAL_ENV/bin:\$PATH
    export PYTHONPATH=/workspace/src:/workspace

    pip3 install --no-deps -e /workspace

    pip3 install --upgrade pip

    # Deps not already in the container (torch, vllm, transformers handled below)
    pip3 install --no-cache-dir \
        'numpy>=1.24' \
        'pandas>=2.0' \
        'scipy>=1.11' \
        'statsmodels>=0.14' \
        'scikit-learn>=1.3' \
        'matplotlib>=3.7' \
        'Pillow>=10.0' \
        'PyYAML>=6.0' \
        'openai>=1.0' \
        'anthropic>=0.20' \
        'tqdm>=4.60'

    # Pin transformers to version compatible with vLLM 0.11.0
    pip3 install --no-cache-dir 'transformers==4.57.1'

    echo ''
    echo '=========================================='
    echo 'VERIFICATION'
    echo '=========================================='
    python3 -c \"
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import vllm
print(f'vLLM:           {vllm.__version__}')
import transformers
print(f'Transformers:   {transformers.__version__}')
from derm_dif.data.ddi import load_ddi
from derm_dif.dif.aggregate import aggregate_fst_shift
print('Project imports: OK')
\"
    echo ''
    echo 'INFO: Setup complete!'
    echo 'INFO: Venv location: '\$VENV
"
