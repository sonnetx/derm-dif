"""Compute the primary endpoint: aggregate residualized FST difficulty shift with bootstrap CI."""

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
    args = ap.parse_args()

    cfg = yaml.safe_load(args.analysis_config.read_text())["aggregate_dif"]
    with (args.rasch_dir / "amortized_fit.pkl").open("rb") as f:
        fit = pickle.load(f)
    items = load_ddi(args.ddi_root)
    attrs = items_to_attrs(items)

    result = aggregate_fst_shift(
        difficulty=fit["difficulty"],
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
