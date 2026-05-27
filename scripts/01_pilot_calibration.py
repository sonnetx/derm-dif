"""Pilot end-to-end pipeline check on a 50-100 item subset.

Run this in week 1 of the project, BEFORE the full benchmark query, to make sure
the data loader, embedding pipeline, and amortized Rasch fit produce sane outputs
on a small slice of DDI. No DIF analysis here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from derm_dif.data.ddi import load_ddi
from derm_dif.data.embeddings import embed_images
from derm_dif.irt.amortized import (
    AmortizedRaschConfig,
    fit_amortized_rasch,
    held_out_auc,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pilot.json"))
    ap.add_argument("--n-items", type=int, default=80)
    ap.add_argument("--n-fake-models", type=int, default=12,
                    help="Generate synthetic responses for the pilot if real model queries are not yet collected.")
    ap.add_argument("--seed", type=int, default=0xDD)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    items = load_ddi(args.ddi_root)
    pilot = list(rng.choice(items, size=args.n_items, replace=False))

    embeddings = embed_images([it.image_path for it in pilot])
    # Synthetic responses: simulate Rasch-distributed responses to verify the fitting code.
    true_theta = rng.normal(size=args.n_fake_models)
    true_b = embeddings @ rng.normal(scale=0.05, size=embeddings.shape[1])
    logit = true_theta[:, None] - true_b[None, :]
    probs = 1.0 / (1.0 + np.exp(-logit))
    Y = (rng.uniform(size=probs.shape) < probs).astype(float)

    fit = fit_amortized_rasch(
        Y,
        embeddings,
        AmortizedRaschConfig(
            embedding_dim=embeddings.shape[1], n_models=args.n_fake_models, n_epochs=1500
        ),
    )

    n_h = int(0.2 * args.n_items)
    auc = held_out_auc(Y[:, -n_h:], fit.theta, fit.difficulty[-n_h:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n_items": args.n_items,
                "n_models": args.n_fake_models,
                "final_loss": fit.history[-1],
                "held_out_auc": auc,
                "theta_mean": float(fit.theta.mean()),
                "theta_sd": float(fit.theta.std()),
                "difficulty_sd": float(fit.difficulty.std()),
            },
            indent=2,
        )
    )
    print(f"pilot summary -> {args.out}")


if __name__ == "__main__":
    main()
