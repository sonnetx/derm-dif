"""Compute the primary endpoint: aggregate residualized FST difficulty shift with bootstrap CI.

Sensitivity flags:
  --exclude-iii-iv   Drop FST-III/IV items before the residualization fit.
                     The primary endpoint uses only I-II vs V-VI items for Δ
                     but III-IV items participate in the OLS fit; this flag
                     tests whether excluding them changes the result.

Family-leave-one-out sensitivity: supply a different --rasch-dir pointing to
a Rasch fit trained on a respondent subset (e.g., generative-only or
contrastive-only). The script then recomputes Δ from that difficulty vector.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.dif.aggregate import aggregate_fst_shift, configural_invariance_spearman


def items_to_attrs(items) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [it.item_id for it in items],
            "fst_group": [it.fst_group for it in items],
            "lesion_category": [it.lesion_category for it in items],
            "malignant": [it.malignant for it in items],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--analysis-config", type=Path, default=Path("config/analysis.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/aggregate_dif.json"))
    ap.add_argument(
        "--exclude-iii-iv",
        action="store_true",
        help="Drop FST-III/IV items from the residualization fit (sensitivity check).",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.analysis_config.read_text())["aggregate_dif"]
    with (args.rasch_dir / "amortized_fit.pkl").open("rb") as f:
        fit = pickle.load(f)
    items = load_ddi(args.ddi_root)
    attrs = items_to_attrs(items)

    difficulty = fit["difficulty"]
    if args.exclude_iii_iv:
        keep = attrs["fst_group"] != "III-IV"
        difficulty = difficulty[keep.values]
        attrs = attrs[keep].reset_index(drop=True)
        print(f"[--exclude-iii-iv] Dropped III-IV: {keep.sum()} items retained "
              f"({(~keep).sum()} dropped)")

    result = aggregate_fst_shift(
        difficulty=difficulty,
        item_attrs=attrs,
        focal=tuple(cfg["comparison_groups"]["focal"]),
        reference=tuple(cfg["comparison_groups"]["reference"]),
        controls=tuple(cfg["residualize_against"]),
        threshold_logits=cfg["decision_rule"]["threshold_logits"],
        n_bootstrap=cfg["bootstrap"]["n_resamples"],
        seed=cfg["bootstrap"]["seed"],
    )

    summary = {
        "delta": result.delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "threshold": result.threshold,
        "decision": result.decision,
        "n_focal": result.n_focal,
        "n_reference": result.n_reference,
        "sensitivity_flags": {
            "exclude_iii_iv": args.exclude_iii_iv,
            "rasch_dir": str(args.rasch_dir),
        },
    }

    # Sensitivity thresholds.
    summary["sensitivity"] = []
    for t in cfg.get("sensitivity_thresholds", []):
        excludes_zero = (result.ci_low > 0) or (result.ci_high < 0)
        meets = abs(result.delta) >= t and excludes_zero
        summary["sensitivity"].append({"threshold": t, "would_conclude_dif": bool(meets)})

    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
