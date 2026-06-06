# Beyond Stratified Accuracy: An Item Response Theory Audit of Dermatology AI Fairness

Item-level measurement-invariance analysis for the DDI dermatology benchmark, applied to foundation vision-language models.

Stratified accuracy across Fitzpatrick Skin Types (FST) is the dominant fairness metric for dermatology AI. It conflates three things: model-capability differences, item-difficulty differences, and measurement non-invariance. This project disentangles them by combining amortized Rasch calibration of vision foundation models with measurement-invariance analysis on the [DDI benchmark](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965).

Code is available at https://github.com/sonnetx/derm-dif.

---

## Requirements

- Python ≥ 3.10
- GPU with ≥ 12 GB VRAM (required for open-weight VLM queries via vLLM; not needed for analysis scripts)
- [DDI dataset](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) — download and unpack to a local directory (referred to as `$DDI_ROOT` below)
- API keys for OpenAI, Anthropic, and Google (only needed to re-run the closed-API model queries; analysis scripts work from the pre-queried `responses.jsonl`)

---

## Installation

```bash
git clone https://github.com/sonnetx/derm-dif
cd derm-dif
pip install -e .
pip install -r requirements.txt
```

For exact reproducibility, a frozen requirements snapshot from the Linux GPU cluster is provided at `requirements-frozen.txt`:

```bash
pip install -r requirements-frozen.txt
```

---

## Repository structure

| Path | Contents |
|------|----------|
| `src/derm_dif/` | Core library: IRT fitting (`irt/`), DIF aggregation (`dif/`), data loading (`data/`), response parsing |
| `scripts/` | Numbered analysis pipeline (`02_` → `11_`) plus robustness and sensitivity scripts |
| `config/` | Pre-registered analysis decisions: `models.yaml`, `analysis.yaml`, `protocol.yaml` |
| `paper/` | Manuscript (`final_report.tex`), pre-analysis plan, figures |
| `artifacts/` | Generated outputs — gitignored; recreated by the pipeline |

---

## Pre-computed artifacts

To reproduce analysis results without re-querying models (which requires API keys and GPU access), pre-computed artifacts are available on request: `artifacts/responses.jsonl` (all model responses, ~50 MB) and `artifacts/rasch/amortized_fit.pkl` (fitted IRT model). Email sonnet@stanford.edu or open a GitHub issue. 

---

## Reproducing results

All analysis scripts accept `--ddi-root /path/to/ddi`. Run from the repo root.

**Step 1 — Query models** (skip if using pre-computed `responses.jsonl`):
```bash
python scripts/02_query_models.py --ddi-root $DDI_ROOT
# outputs: artifacts/responses.jsonl
```
Requires API keys in environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Re-querying open-weight models (LLaVA, Qwen2-VL) requires the Slurm/vLLM scripts under `scripts/run_query_vllm.sh`.

**Step 2 — Fit amortized Rasch IRT**:
```bash
python scripts/03_fit_amortized.py --ddi-root $DDI_ROOT
# outputs: artifacts/rasch/amortized_fit.pkl, artifacts/rasch/embeddings_biomedclip.npy
```

**Step 3 — Primary endpoint** (aggregate FST difficulty shift δ):
```bash
python scripts/04_aggregate_dif.py --ddi-root $DDI_ROOT
# outputs: artifacts/aggregate_dif.json
```

**Step 4 — LLTM mechanism decomposition**:
```bash
python scripts/05_lltm.py --ddi-root $DDI_ROOT
# outputs: artifacts/lltm.json
```

**Step 5 — Per-model FST logistic regression**:
```bash
python scripts/09_per_model_fst_logistic.py --ddi-root $DDI_ROOT
# outputs: artifacts/per_model_fst.json
```

**Step 6 — Non-IRT baselines**:
```bash
python scripts/11_baselines.py --ddi-root $DDI_ROOT
# outputs: artifacts/baselines.json
```

**Step 7 — All-model accuracy table**:
```bash
python scripts/all_model_fst_accuracy.py --ddi-root $DDI_ROOT
# outputs: artifacts/all_model_fst_accuracy.json
```

**Step 8 — Prediction distribution figure**:
```bash
python scripts/prediction_distribution.py --ddi-root $DDI_ROOT
# outputs: paper/figures/prediction_distribution.pdf
```

**Step 9 — Robustness and sensitivity sweeps** (requires Slurm GPU node):
```bash
bash scripts/run_sensitivity_reruns.sh
# outputs: artifacts/seed_sweep.json, artifacts/lambda_sweep.json,
#          artifacts/backbone_spearman.json, artifacts/synthetic_validation.json
```

---

## Script → paper result mapping

| Script | Output artifact | Paper location |
|--------|----------------|---------------|
| `scripts/03_fit_amortized.py` | `artifacts/rasch/amortized_fit.pkl` | Methods: IRT calibration |
| `scripts/04_aggregate_dif.py` | `artifacts/aggregate_dif.json` | Results: primary endpoint (δ = +0.051) |
| `scripts/05_lltm.py` | `artifacts/lltm.json` | Results: LLTM table (M0–M3, M_raw) |
| `scripts/09_per_model_fst_logistic.py` | `artifacts/per_model_fst.json` | Results: per-model heterogeneity table |
| `scripts/11_baselines.py` | `artifacts/baselines.json` | Results: baselines paragraph |
| `scripts/all_model_fst_accuracy.py` | `artifacts/all_model_fst_accuracy.json` | Results: all-model accuracy table |
| `scripts/prediction_distribution.py` | `paper/figures/prediction_distribution.pdf` | Figure 1 |
| `scripts/seed_sweep.py` | `artifacts/seed_sweep.json` | Results: robustness (seed sweep) |
| `scripts/lambda_sweep.py` | `artifacts/lambda_sweep.json` | Results: robustness (λ sweep) |
| `scripts/10_synthetic_validation.py` | `artifacts/synthetic_validation.json` | Robustness: synthetic validation |

---

## Expected runtime

| Task | Hardware | Time |
|------|----------|------|
| Closed-API model queries (GPT-4o, Claude, Gemini) | CPU / API | ~2 hrs |
| Open-weight VLM queries (LLaVA, Qwen2-VL) | GPU ≥ 12 GB | ~4 hrs per model |
| IRT fit (`03_fit_amortized.py`) | GPU ≥ 8 GB | ~10 min |
| Individual analysis scripts (04–11) | CPU | < 1 min each |
| Sensitivity sweeps (`run_sensitivity_reruns.sh`) | Slurm GPU node | ~4 hrs total |

---

## Notes on cluster scripts

The `scripts/run_*.sh` shell scripts are Slurm batch scripts configured for the Stanford Sherlock cluster. Before running elsewhere, edit the site-specific variables at the top of each script:

```bash
PROJECT_ROOT=/home/groups/roxanad/sonnet/derm-dif   # change to your repo path
DDI_ROOT=/home/groups/roxanad/sonnet/datasets/ddi   # change to your DDI path
```

---

## Pre-registration

Analysis decisions are frozen in `config/analysis.yaml` (focal group: FST V-VI; reference: FST I-II; controls: lesion category + malignancy; threshold: 0.5 logits; bootstrap B = 2000). The pre-analysis plan is in `paper/pre_analysis_plan.tex`. Three deviations from the pre-registration are documented in the paper's Results section.

---

## Citation

```bibtex
@techreport{derm-dif-2026,
  title  = {Beyond Stratified Accuracy: An Item Response Theory Audit of Dermatology AI Fairness},
  author = {Xu, Sonnet},
  year   = {2026},
  url    = {https://github.com/sonnetx/derm-dif}
}
```
