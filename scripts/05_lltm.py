"""LLTM nested-model decomposition; primary coefficient is FST after lesion/malignancy/acquisition controls."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.dif.mechanism import image_properties
from derm_dif.irt.lltm import fit_lltm, nested_variance_explained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--analysis-config", type=Path, default=Path("config/analysis.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/lltm.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.analysis_config.read_text())["lltm"]
    with (args.rasch_dir / "amortized_fit.pkl").open("rb") as f:
        fit = pickle.load(f)
    items = load_ddi(args.ddi_root)

    img_feats = image_properties([it.image_path for it in items])
    attrs = pd.DataFrame(
        {
            "lesion_category": [it.lesion_category for it in items],
            "malignancy": [it.malignant for it in items],
            "fst_group": [it.fst_group for it in items],
        }
    )
    # Map config feature names to actual columns.
    attrs = pd.concat([attrs, img_feats], axis=1)
    rename = {
        "image_resolution": "image_resolution",
        "mean_luminance": "mean_luminance",
        "color_channel_stats": "luminance_std",  # one summary stat for variance budget
    }
    # Build M0..M4 sequentially.
    summaries = []
    for spec in cfg["nesting"]:
        feats = [rename.get(f, f) for f in spec["features"]]
        if not feats:
            summaries.append({"model": spec["name"], "r2": 0.0, "coefficients": {}})
            continue
        lltm_fit = fit_lltm(
            difficulty=fit["difficulty"],
            item_attributes=attrs,
            features=feats,
            n_bootstrap=cfg["bootstrap_n"],
        )
        coefs = {}
        for name in lltm_fit.feature_names:
            coefs[name] = {
                "beta": float(lltm_fit.coef[name]),
                "ci_low": float(lltm_fit.ci_low[name]),
                "ci_high": float(lltm_fit.ci_high[name]),
            }
        summaries.append(
            {"model": spec["name"], "r2": lltm_fit.r2, "coefficients": coefs}
        )

    args.out.write_text(json.dumps(summaries, indent=2))
    for s in summaries:
        fst_keys = [k for k in s["coefficients"] if k.startswith("fst_group_")]
        if fst_keys:
            print(f"  {s['model']}: r2 = {s['r2']:.3f}; FST coefs:")
            for k in fst_keys:
                c = s["coefficients"][k]
                print(f"    {k}: beta = {c['beta']:.3f}  [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
        else:
            print(f"  {s['model']}: r2 = {s['r2']:.3f}")


if __name__ == "__main__":
    main()
