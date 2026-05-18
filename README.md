# derm-dif

Item-level measurement-invariance analysis for the DDI dermatology benchmark, applied to foundation vision-language models. Companion code to the pre-analysis plan in [paper/pre_analysis_plan.tex](paper/pre_analysis_plan.tex).

## What this is

Stratified accuracy across Fitzpatrick Skin Types (FST) is the dominant fairness metric for dermatology AI. It conflates three things: model-capability differences, item-difficulty differences, and measurement non-invariance. This project disentangles them by combining amortized Rasch calibration of vision foundation models with measurement-invariance analysis on the [DDI benchmark](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965).

## Primary endpoint (pre-registered)

Aggregate residualized FST difficulty shift, $\Delta = \overline{\hat b_i^{\text{resid}}}_{V\text{-}VI} - \overline{\hat b_i^{\text{resid}}}_{I\text{-}II}$, with bootstrap 95% CI. Decision rule: $|\Delta| \geq 0.5$ logits AND CI excludes zero $\Rightarrow$ meaningful non-invariance.

## Layout

```
paper/                # pre-analysis plan (NeurIPS registered-report hybrid)
config/               # pre-registered model list, query protocol, analysis decisions
src/derm_dif/
  data/               # DDI loader, BiomedCLIP / DINOv3 embeddings
  query.py            # multi-backend model querying (OpenAI, Anthropic, Google, vLLM)
  parsing.py          # response parsing with refusal as a separate category
  irt/                # amortized Rasch, traditional Rasch, LLTM
  dif/                # aggregate (primary), per-item (exploratory), mechanism, refusal
scripts/              # 01-pilot, 02-query, 03-fit, 04-aggregate, 05-lltm, 06-mechanism
tests/                # smoke tests on synthetic Rasch data
```

## Pipeline

```bash
# Week 1
python scripts/01_pilot_calibration.py --ddi-root /path/to/ddi
git tag prereg-v1                    # freeze config/ + paper/pre_analysis_plan.tex

# Week 2
python scripts/02_query_models.py --ddi-root /path/to/ddi
python scripts/03_fit_amortized.py --ddi-root /path/to/ddi

# Week 3
python scripts/04_aggregate_dif.py --ddi-root /path/to/ddi
python scripts/05_lltm.py --ddi-root /path/to/ddi
python scripts/06_mechanism_and_refusal.py --ddi-root /path/to/ddi
```

## Pre-registration

Tag `prereg-v1` freezes:
- [config/models.yaml](config/models.yaml) — respondent model list with family annotations
- [config/protocol.yaml](config/protocol.yaml) — prompt, decoding, parsing, refusal handling
- [config/analysis.yaml](config/analysis.yaml) — primary endpoint, decision rule, bootstrap counts, FDR threshold, LLTM nesting, mechanism features
- [paper/pre_analysis_plan.tex](paper/pre_analysis_plan.tex) — full plan with anticipated discussion

Any deviations from these files post-tag must be documented as such in the final paper.

## Tests

```bash
pip install -e .
pytest -q
```

The smoke tests verify amortized Rasch on synthetic Rasch-distributed data and that the aggregate-DIF decision rule fires under injected signal but not under null data.
