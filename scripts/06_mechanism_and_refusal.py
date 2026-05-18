"""Mechanism analysis (per-item DIF magnitude vs image properties) and refusal-rate DIF."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from derm_dif.data.ddi import load_ddi
from derm_dif.dif.mechanism import (
    correlate_dif_with_features,
    embedding_distance_to_centroid,
    image_properties,
    refusal_logit_fst,
    refusal_rate_by_fst,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/mechanism.json"))
    args = ap.parse_args()

    items = load_ddi(args.ddi_root)
    with (args.rasch_dir / "amortized_fit.pkl").open("rb") as f:
        fit = pickle.load(f)
    R = np.load(args.rasch_dir / "refusal_matrix.npy")
    embeddings = np.load(args.rasch_dir / f"embeddings_{fit['embedding_backend']}.npy")

    fst = pd.Series([it.fst_group for it in items])
    in_I_II = (fst == "I-II").values
    distance_feature = embedding_distance_to_centroid(embeddings, in_I_II)

    img_feats = image_properties([it.image_path for it in items])
    img_feats["embedding_distance_to_fst_I_II_centroid"] = distance_feature

    # |delta_b| vs features: use a per-item FST-V/VI dummy proxy of group difficulty
    # contribution as a stand-in for |delta_b| in the absence of stratified-fit estimates.
    # Concretely, partial out lesion/malignancy and use abs residual as the per-item DIF magnitude.
    from derm_dif.dif.aggregate import residualize

    attrs = pd.DataFrame(
        {
            "fst_group": [it.fst_group for it in items],
            "lesion_category": [it.lesion_category for it in items],
            "malignant": [it.malignant for it in items],
        }
    )
    b_resid = residualize(fit["difficulty"], attrs, ["lesion_category", "malignant"])
    abs_dif = np.abs(b_resid)

    correlations = correlate_dif_with_features(abs_dif, img_feats)
    refusal_table = refusal_rate_by_fst(R, fst)
    refusal_model = refusal_logit_fst(R, fst)

    out = {
        "feature_correlations": correlations.to_dict(orient="records"),
        "refusal_rate_by_fst": refusal_table.to_dict(orient="records"),
        "refusal_logit_coefficients": [float(x) for x in refusal_model["coef"]],
        "refusal_logit_ci_low": [float(x) for x in refusal_model["ci_low"]],
        "refusal_logit_ci_high": [float(x) for x in refusal_model["ci_high"]],
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
