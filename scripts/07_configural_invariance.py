"""Configural-invariance check: re-fit the amortized Rasch on the FST-I/II and
FST-V/VI item subsets independently, then Spearman-correlate the respondent
ability vectors across the two fits.

A high Spearman rho (> 0.9) is evidence that the two subsets rank respondents
the same way, i.e., the construct being measured is stable across FST strata
even if item-difficulty scaling differs. A low Spearman would indicate that
whatever the test is measuring on FST-I/II items is a different construct from
what it measures on FST-V/VI items, which would undermine the comparability
assumption of the primary endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from derm_dif.data.ddi import load_ddi
from derm_dif.irt.amortized import AmortizedRaschConfig, fit_amortized_rasch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--embedding-backend", default="biomedclip")
    ap.add_argument("--out", type=Path, default=Path("artifacts/configural.json"))
    ap.add_argument("--cutoff", type=float, default=0.9)
    args = ap.parse_args()

    Y = np.load(args.rasch_dir / "responses_matrix.npy")
    embeddings = np.load(args.rasch_dir / "embeddings_{0}.npy".format(args.embedding_backend))
    items = load_ddi(args.ddi_root)
    if len(items) != Y.shape[1]:
        raise SystemExit(
            f"Y has {Y.shape[1]} items but DDI loader returned {len(items)}; "
            "the loader and the saved matrix are out of sync."
        )

    fst = np.array([it.fst_group for it in items])
    mask_ref = fst == "I-II"
    mask_focal = fst == "V-VI"
    print(f"FST I-II items: {int(mask_ref.sum())}; FST V-VI items: {int(mask_focal.sum())}")

    cfg = AmortizedRaschConfig(embedding_dim=embeddings.shape[1], n_models=Y.shape[0])
    fit_ref = fit_amortized_rasch(Y[:, mask_ref], embeddings[mask_ref], cfg)
    fit_focal = fit_amortized_rasch(Y[:, mask_focal], embeddings[mask_focal], cfg)

    rho, p = spearmanr(fit_ref.theta, fit_focal.theta)
    summary = {
        "n_reference": int(mask_ref.sum()),
        "n_focal": int(mask_focal.sum()),
        "n_models": int(Y.shape[0]),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "high_invariance_cutoff": float(args.cutoff),
        "configural_invariant": bool(rho > args.cutoff) if not np.isnan(rho) else None,
        "theta_I_II": fit_ref.theta.tolist(),
        "theta_V_VI": fit_focal.theta.tolist(),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
