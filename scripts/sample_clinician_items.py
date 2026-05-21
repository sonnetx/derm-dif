"""Sample 100 DDI items for clinician validation, stratified by
(FST group x lesion-category bucket x IRT-difficulty quartile).

The output is a CSV of (item_id, fst_group, lesion_category, malignant_truth,
IRT_difficulty, IRT_difficulty_quartile, stratum_key) plus a small JSON manifest
describing the stratification. The same script also writes a rater-ready CSV
template that maps item_id -> image_path so the rating UI can ingest it.

Design rationale: clinician time is the binding constraint, so 100 items is
the sweet spot for ~30-60 minutes per rater. We want representation across
FST groups, lesion-category buckets, and IRT-difficulty quartiles so the
construct-validity test (clinician-derived vs. model-derived difficulty)
covers the same heterogeneity the rest of the analysis sees.

  python scripts/sample_clinician_items.py --ddi-root /path/to/ddi
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from derm_dif.data.ddi import load_ddi

# Top-N lesion categories to keep as their own stratum; everything else
# gets collapsed into "other". 10 is enough to capture the modal categories
# in DDI (melanocytic-nevi, seborrheic-keratosis, verruca-vulgaris, BCC, etc.)
# without over-stratifying small cells.
TOP_LESION_CATEGORIES = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/clinician_sample"))
    ap.add_argument("--seed", type=int, default=0xC1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    items = load_ddi(args.ddi_root)
    with (args.rasch_dir / "amortized_fit.pkl").open("rb") as f:
        fit = pickle.load(f)

    difficulty = fit["difficulty"]
    if len(difficulty) != len(items):
        raise SystemExit(
            f"fit difficulty length {len(difficulty)} != number of items {len(items)}"
        )

    df = pd.DataFrame({
        "item_id": [it.item_id for it in items],
        "image_path": [str(it.image_path) for it in items],
        "fst_group": [it.fst_group for it in items],
        "lesion_category": [it.lesion_category for it in items],
        "malignant": [it.malignant for it in items],
        "IRT_difficulty": difficulty,
    })

    # Lesion-category bucket: keep top-N, lump rest into "other".
    top_n = df["lesion_category"].value_counts().head(TOP_LESION_CATEGORIES).index.tolist()
    df["lesion_bucket"] = df["lesion_category"].where(
        df["lesion_category"].isin(top_n), other="other"
    )

    # IRT-difficulty quartile.
    quartile_edges = np.quantile(difficulty, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartile_edges[-1] += 1e-9
    df["IRT_difficulty_quartile"] = pd.cut(
        df["IRT_difficulty"],
        bins=quartile_edges,
        labels=["Q1_easiest", "Q2", "Q3", "Q4_hardest"],
        include_lowest=True,
    )

    # Stratum key: (fst_group, lesion_bucket, quartile). Sample
    # proportionally across strata, with floor of 1 per non-empty stratum.
    df["stratum_key"] = (
        df["fst_group"].astype(str)
        + "|" + df["lesion_bucket"].astype(str)
        + "|" + df["IRT_difficulty_quartile"].astype(str)
    )
    stratum_counts = df["stratum_key"].value_counts()
    print(f"{len(stratum_counts)} non-empty strata; total items {len(df)}")

    # Proportional allocation: target_n_per_stratum = round(n_items * size / total)
    # with floor of 1 (so we always sample at least one item per non-empty stratum
    # to keep coverage), and a cap so we don't overshoot.
    total = len(df)
    raw_alloc = (stratum_counts / total * args.n_items).round().astype(int)
    raw_alloc = raw_alloc.clip(lower=1)
    # Adjust to hit target n_items by trimming the largest allocations.
    while raw_alloc.sum() > args.n_items:
        largest = raw_alloc.idxmax()
        raw_alloc.loc[largest] -= 1
    while raw_alloc.sum() < args.n_items:
        # Add to the stratum with the most remaining un-sampled items
        deficit = stratum_counts - raw_alloc
        candidate = deficit.idxmax()
        raw_alloc.loc[candidate] += 1

    sampled_rows = []
    for stratum, target in raw_alloc.items():
        candidates = df[df["stratum_key"] == stratum]
        n_take = min(target, len(candidates))
        if n_take == 0:
            continue
        chosen = candidates.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))
        sampled_rows.append(chosen)
    sample = pd.concat(sampled_rows, ignore_index=True)

    # Shuffle the final order so raters don't see items in stratum order.
    sample = sample.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    sample["rater_order"] = np.arange(1, len(sample) + 1)

    print(f"sampled {len(sample)} items")
    print(sample["fst_group"].value_counts().to_string())
    print(sample["lesion_bucket"].value_counts().to_string())
    print(sample["IRT_difficulty_quartile"].value_counts().to_string())

    # The clinician sample with ground-truth (kept INTERNAL; not shown to raters).
    sample.to_csv(args.out_dir / "sample_with_truth.csv", index=False)

    # Rater-facing CSV: item_id + image_path + rater_order ONLY. No truth,
    # no FST, no lesion category, no model-derived difficulty -- those would
    # bias the rater. Add a blank `rater_label` column for the answer.
    rater_template = sample[["rater_order", "item_id", "image_path"]].copy()
    rater_template["rater_label"] = ""              # benign | malignant
    rater_template["rater_uncertain"] = ""          # 0 | 1
    rater_template.to_csv(args.out_dir / "rater_template.csv", index=False)

    manifest = {
        "n_items": int(len(sample)),
        "n_strata": int((raw_alloc > 0).sum()),
        "stratification": {
            "fst_groups": sample["fst_group"].value_counts().to_dict(),
            "lesion_buckets": sample["lesion_bucket"].value_counts().to_dict(),
            "irt_difficulty_quartiles": {
                k: int(v) for k, v in sample["IRT_difficulty_quartile"].value_counts().to_dict().items()
            },
        },
        "top_n_lesion_categories_kept_distinct": top_n,
        "seed": int(args.seed),
        "sample_with_truth_path": str(args.out_dir / "sample_with_truth.csv"),
        "rater_template_path": str(args.out_dir / "rater_template.csv"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
