#!/bin/bash
#SBATCH --job-name=derm_dif_setup
#SBATCH --partition=roxanad
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-time environment setup on Sherlock for derm-dif.
#
# Submit with:   sbatch scripts/setup_sherlock.sh
# Or run directly on a login/interactive node (skips Slurm).
#
# Assumes the repo has been cloned to:
#   /home/groups/roxanad/sonnet/derm-dif
# Creates venv at:
#   /home/groups/roxanad/sonnet/derm-dif/.venv
# Expects DDI dataset at:
#   /home/groups/roxanad/datasets/ddi   (must contain ddi_metadata.csv + images/)
#
# After this completes, run the pilot with:
#   source /home/groups/roxanad/sonnet/derm-dif/.venv/bin/activate
#   python scripts/01_pilot_calibration.py --ddi-root /home/groups/roxanad/datasets/ddi

set -euo pipefail

PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
VENV=$PROJECT_ROOT/.venv
DDI_ROOT=/home/groups/roxanad/datasets/ddi

ml gcc/10.3.0
ml python/3.12.1
ml cuda/11.7.1

# Wipe a partial/failed install so this script is idempotent.
rm -rf "$VENV"

python3 -m venv "$VENV"
source "$VENV/bin/activate"

# Sherlock's default pip config sets user=true, which causes pip to install
# into ~/.local even when a venv is active — uninstalls from the venv and
# reinstalls outside it, silently breaking the venv. Disable for this script.
export PIP_USER=false

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" logs

which python
python --version

pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Numpy first, before torch picks its own index.
pip3 install --no-cache-dir numpy==1.26.4

# Torch + cu118 wheels. Pytorch bundles its own CUDA runtime so cu118 wheels
# work fine against the cuda/11.7.1 module loaded above.
pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Pin pandas/scipy/sklearn to versions with prebuilt Py3.12 wheels.
pip3 install --no-cache-dir --only-binary :all: pandas==2.2.3
pip3 install --no-cache-dir --only-binary :all: scipy==1.12.0
pip3 install --no-cache-dir --only-binary :all: scikit-learn==1.4.2

# Imaging / IRT / config / tests
# transformers is required by open_clip_torch for HF-backed text towers
# (BiomedCLIP uses PubMedBERT via the HF integration).
pip3 install --no-cache-dir Pillow PyYAML einops open_clip_torch transformers pytest

# Editable install of derm-dif itself.
pip3 install --no-cache-dir -e "$PROJECT_ROOT"

# Smoke tests on synthetic Rasch data. No DDI required.
cd "$PROJECT_ROOT"
pytest -q

echo ""
echo "derm-dif environment setup complete."
echo "Pilot run:"
echo "  source $VENV/bin/activate"
echo "  python scripts/01_pilot_calibration.py --ddi-root $DDI_ROOT"
