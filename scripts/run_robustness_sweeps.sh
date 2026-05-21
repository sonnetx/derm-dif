#!/bin/bash
#SBATCH --job-name=derm_dif_robustness
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs the three reviewer-credibility robustness sweeps:
#   1. Power simulation (null Spearman distribution) -- fast, ~1 minute.
#   2. Seed sweep -- 5 seeds x (fit + bootstrap), ~15-25 minutes per seed.
#   3. Lambda_b sensitivity sweep -- 5 lambdas x (fit + bootstrap), ~15-25 min each.

set -euo pipefail
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi

ml python/3.12.1
source "$PROJECT_ROOT/.venv/bin/activate"
export PIP_USER=false

cd "$PROJECT_ROOT"

echo "===== 1/3 Power simulation ====="
python scripts/power_simulation.py --n-iter 5000

echo ""
echo "===== 2/3 Multi-seed sweep ====="
python scripts/seed_sweep.py --ddi-root "$DDI_ROOT" --seeds 11 13 17 19 23

echo ""
echo "===== 3/3 Lambda_b sensitivity sweep ====="
python scripts/lambda_sweep.py --ddi-root "$DDI_ROOT" --lambdas 1e-3 3e-3 1e-2 3e-2 1e-1

echo ""
echo "===== Summaries ====="
echo "--- power_simulation.json ---"
cat artifacts/power_simulation.json
echo ""
echo "--- seed_sweep.json (top-level) ---"
python -c "import json; d=json.load(open('artifacts/seed_sweep.json')); d.pop('per_seed'); print(json.dumps(d, indent=2))"
echo ""
echo "--- lambda_sweep.json (per-lambda) ---"
python -c "
import json
d = json.load(open('artifacts/lambda_sweep.json'))
for r in d['per_lambda']:
    print(f\"  lambda_b={r['lambda_b']:.0e}: delta={r['delta']:+.3f} CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] decision={r['decision']} AUC={r['holdout_auc']:.3f}\")
"
