"""Baseline DIF analyses compared against the IRT primary endpoint.

Two composition-aware baselines that do not require a latent-trait model:

(1) Propensity-matched accuracy gap.
    Restrict to lesion categories present in both FST-I/II and FST-V/VI subsets
    (eliminates category-selection as a confound by construction). Compute the
    pooled accuracy gap in this matched subset across all respondents.

(2) Pooled logistic regression.
    logit P(X_ij = 1) ~ fst_vvi + lesion_category + malignant + model_id (all
    fixed effects), pooled over the full (J x I) response matrix. Reports the
    FST V-VI coefficient with item-level bootstrap CI.

Run:
    python scripts/11_baselines.py \\
        --ddi-root /path/to/ddi \\
        --responses artifacts/responses.jsonl \\
        --out artifacts/baselines.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from derm_dif.data.ddi import load_ddi
from derm_dif.parsing import parse_primary


def load_long_responses(
    responses_path: Path, refusal_markers: list[str], items: list
) -> pd.DataFrame:
    """Long-format (model_id, item_id, correct, fst_group, lesion_category, malignant)."""
    item_meta = {
        it.item_id: {
            "fst_group": it.fst_group,
            "lesion_category": it.lesion_category,
            "malignant": it.malignant,
        }
        for it in items
    }
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with responses_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("error"):
                continue
            key = (d["model_id"], d["item_id"])
            if key in seen:
                continue
            seen.add(key)
            parsed = parse_primary(d["raw_text"], refusal_markers)
            if parsed.label not in ("benign", "malignant"):
                continue
            meta = item_meta.get(d["item_id"])
            if meta is None:
                continue
            rows.append(
                {
                    "model_id": d["model_id"],
                    "item_id": d["item_id"],
                    "label": parsed.label,
                    "correct": int(
                        parsed.label == ("malignant" if meta["malignant"] else "benign")
                    ),
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def propensity_match(
    df: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int = 2000
) -> dict:
    """Restrict to lesion categories present in both FST-I/II and V-VI; compute gap."""
    sub = df[df["fst_group"].isin(["I-II", "V-VI"])].copy()

    item_to_lc: dict[str, str] = (
        sub.drop_duplicates("item_id")
        .set_index("item_id")["lesion_category"]
        .to_dict()
    )
    items_ii = set(sub[sub["fst_group"] == "I-II"]["item_id"].unique())
    items_vvi = set(sub[sub["fst_group"] == "V-VI"]["item_id"].unique())
    cats_ii = {item_to_lc[i] for i in items_ii if i in item_to_lc}
    cats_vvi = {item_to_lc[i] for i in items_vvi if i in item_to_lc}
    shared_cats = cats_ii & cats_vvi

    sub_m = sub[sub["lesion_category"].isin(shared_cats)].copy()
    matched_items = sub_m["item_id"].unique()

    acc = sub_m.groupby("fst_group")["correct"].mean()
    gap_matched = float(acc.get("V-VI", 0.0) - acc.get("I-II", 0.0))

    acc_all = sub.groupby("fst_group")["correct"].mean()
    gap_unmatched = float(acc_all.get("V-VI", 0.0) - acc_all.get("I-II", 0.0))

    # Item-level bootstrap
    boot_gaps: list[float] = []
    for _ in range(n_bootstrap):
        boot_ids = set(rng.choice(matched_items, size=len(matched_items), replace=True))
        b = sub_m[sub_m["item_id"].isin(boot_ids)]
        b_acc = b.groupby("fst_group")["correct"].mean()
        boot_gaps.append(float(b_acc.get("V-VI", 0.0) - b_acc.get("I-II", 0.0)))

    ci = np.quantile(boot_gaps, [0.025, 0.975])

    return {
        "n_shared_categories": len(shared_cats),
        "n_ii_items_matched": int(
            sub_m[sub_m["fst_group"] == "I-II"]["item_id"].nunique()
        ),
        "n_vvi_items_matched": int(
            sub_m[sub_m["fst_group"] == "V-VI"]["item_id"].nunique()
        ),
        "gap_matched": gap_matched,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "gap_unmatched": gap_unmatched,
        "acc_ii_matched": float(acc.get("I-II", 0.0)),
        "acc_vvi_matched": float(acc.get("V-VI", 0.0)),
        "acc_ii_all": float(acc_all.get("I-II", 0.0)),
        "acc_vvi_all": float(acc_all.get("V-VI", 0.0)),
    }


def pooled_logistic(
    df: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int = 2000
) -> dict:
    """Pooled logistic: logit P(correct) ~ fst_vvi + lesion_category + malignant + model_id."""
    sub = df[df["fst_group"].isin(["I-II", "V-VI"])].copy().reset_index(drop=True)
    sub["fst_vvi"] = (sub["fst_group"] == "V-VI").astype(float)

    lc_ref = sub["lesion_category"].value_counts().index[0]
    model_ref = sub["model_id"].value_counts().index[0]

    lc_dummies = (
        pd.get_dummies(sub["lesion_category"], prefix="lc")
        .drop(columns=[f"lc_{lc_ref}"], errors="ignore")
        .astype(float)
    )
    model_dummies = (
        pd.get_dummies(sub["model_id"], prefix="mid")
        .drop(columns=[f"mid_{model_ref}"], errors="ignore")
        .astype(float)
    )

    X = pd.concat(
        [sub[["fst_vvi", "malignant"]].astype(float), lc_dummies, model_dummies],
        axis=1,
    ).values
    y = sub["correct"].values
    item_ids = sub["item_id"].values
    unique_items = pd.unique(item_ids)

    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    clf.fit(X, y)
    fst_beta = float(clf.coef_.ravel()[0])

    # Item-level bootstrap via sample weights
    boot_betas: list[float] = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(len(unique_items), size=len(unique_items), replace=True)
        boot_selected = unique_items[boot_idx]
        counts = Counter(boot_selected.tolist())
        weights = np.array([counts.get(iid, 0) for iid in item_ids], dtype=float)
        if y[weights > 0].mean() in (0.0, 1.0):
            continue
        try:
            clf_b = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
            clf_b.fit(X, y, sample_weight=weights)
            boot_betas.append(float(clf_b.coef_.ravel()[0]))
        except Exception:
            continue

    if len(boot_betas) >= 100:
        ci = np.quantile(boot_betas, [0.025, 0.975])
        ci_low, ci_high = float(ci[0]), float(ci[1])
    else:
        ci_low = ci_high = float("nan")

    return {
        "fst_vvi_beta": fst_beta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap_success": len(boot_betas),
        "n_obs": int(len(sub)),
        "n_items": int(len(unique_items)),
        "n_models": int(sub["model_id"].nunique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--protocol", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/baselines.json"))
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0xDD11)
    args = ap.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("parsing", {}).get("refusal_markers", [])

    items = load_ddi(args.ddi_root)
    df = load_long_responses(args.responses, refusal_markers, items)
    print(
        f"Loaded {len(df)} parseable responses  "
        f"({df['model_id'].nunique()} models × {df['item_id'].nunique()} items)"
    )

    rng = np.random.default_rng(args.seed)

    print("\n── Propensity-matched accuracy gap ──────────────────────────────")
    pm = propensity_match(df, rng, n_bootstrap=args.n_bootstrap)
    print(f"  Shared lesion categories:  {pm['n_shared_categories']}")
    print(
        f"  Matched items:  I-II={pm['n_ii_items_matched']}  V-VI={pm['n_vvi_items_matched']}"
    )
    print(
        f"  Accuracy (matched):   I-II={pm['acc_ii_matched']:.3f}  V-VI={pm['acc_vvi_matched']:.3f}"
    )
    print(
        f"  Gap (matched):   {pm['gap_matched']:+.4f}  [{pm['ci_low']:+.4f}, {pm['ci_high']:+.4f}]"
    )
    print(f"  Gap (unmatched): {pm['gap_unmatched']:+.4f}")
    print(
        f"  Δgap = {pm['gap_matched'] - pm['gap_unmatched']:+.4f}  "
        f"(composition-explained share of gap)"
    )

    print("\n── Pooled logistic FST coefficient ──────────────────────────────")
    pl = pooled_logistic(df, rng, n_bootstrap=args.n_bootstrap)
    print(
        f"  N = {pl['n_obs']} obs  ({pl['n_items']} items × {pl['n_models']} models)"
    )
    print(
        f"  β_FST(V-VI) = {pl['fst_vvi_beta']:+.4f}  [{pl['ci_low']:+.4f}, {pl['ci_high']:+.4f}]"
    )
    print(f"  Bootstrap success: {pl['n_bootstrap_success']}/{args.n_bootstrap}")

    results = {"propensity_match": pm, "pooled_logistic": pl}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {args.out}")

    print("\n── Paper-ready numbers ───────────────────────────────────────────")
    print(
        f"Propensity match ({pm['n_shared_categories']} shared categories, "
        f"I-II n={pm['n_ii_items_matched']}, V-VI n={pm['n_vvi_items_matched']}): "
        f"gap = {pm['gap_matched']:+.3f} [{pm['ci_low']:+.3f}, {pm['ci_high']:+.3f}] pp "
        f"(unmatched: {pm['gap_unmatched']:+.3f} pp)"
    )
    print(
        f"Pooled logistic: β_FST = {pl['fst_vvi_beta']:+.3f} "
        f"[{pl['ci_low']:+.3f}, {pl['ci_high']:+.3f}]"
    )


if __name__ == "__main__":
    main()
