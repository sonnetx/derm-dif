"""Build the (J x I) response matrix from the JSONL log, embed images, fit amortized Rasch."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.data.embeddings import assert_no_circularity, embed_images
from derm_dif.irt.amortized import (
    AmortizedRaschConfig,
    fit_amortized_rasch,
    held_out_auc,
    infit_outfit,
)
from derm_dif.parsing import parse_primary, to_correctness


def build_response_matrix(
    responses_path: Path,
    items,
    model_ids: list[str],
    refusal_markers: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (Y, refused_mask, item_attrs_dict). NaN in Y for missing/refusal/unparseable."""
    item_id_to_idx = {it.item_id: i for i, it in enumerate(items)}
    model_id_to_idx = {m: j for j, m in enumerate(model_ids)}
    J, I = len(model_ids), len(items)
    Y = np.full((J, I), np.nan)
    R = np.zeros((J, I), dtype=bool)
    truth = np.array([it.malignant for it in items])

    with responses_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d["error"] is not None:
                continue
            if d["model_id"] not in model_id_to_idx or d["item_id"] not in item_id_to_idx:
                continue
            j = model_id_to_idx[d["model_id"]]
            i = item_id_to_idx[d["item_id"]]
            parsed = parse_primary(d["raw_text"], refusal_markers)
            if parsed.label == "refusal":
                R[j, i] = True
                continue
            corr = to_correctness(parsed, bool(truth[i]))
            if corr is not None:
                Y[j, i] = float(corr)
    return Y, R, {"truth": truth}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    ap.add_argument("--protocol-config", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--embedding-backend", choices=["biomedclip", "dinov3"], default="biomedclip")
    ap.add_argument("--seed", type=int, default=0xDD1F)
    ap.add_argument(
        "--source",
        default=None,
        help="Comma-separated list of model `source` values to include "
        "(e.g., api-openai,api-anthropic,contrastive-zeroshot). "
        "Default: all non-optional models in the config.",
    )
    ap.add_argument(
        "--min-coverage",
        type=int,
        default=600,
        help="Drop respondents with fewer than this many parseable "
        "responses (default: 600 of 656 DDI items).",
    )
    ap.add_argument(
        "--prune-saturated",
        action="store_true",
        help="Drop items at the response-pattern floor (all-wrong) or ceiling "
        "(all-correct) across the panel before fitting. Robustness check on "
        "the L2 difficulty prior: if results agree with the unpruned fit, the "
        "prior is doing its job; if they differ, the prior is over-shrinking.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = load_ddi(args.ddi_root)
    models_cfg = yaml.safe_load(args.models_config.read_text())
    protocol = yaml.safe_load(args.protocol_config.read_text())["primary_protocol"]
    eligible = [m for m in models_cfg["models"] if not m.get("optional", False)]
    if args.source:
        allowed = {s.strip() for s in args.source.split(",")}
        eligible = [m for m in eligible if m["source"] in allowed]
    model_ids = [m["id"] for m in eligible]
    families = [m["family"] for m in eligible]
    if not model_ids:
        raise SystemExit("no models match the requested --source filter")
    assert_no_circularity(args.embedding_backend, families)

    Y, R, _ = build_response_matrix(
        args.responses, items, model_ids, protocol["parsing"]["refusal_markers"]
    )

    # Drop respondents with too few parsed responses. Otherwise SigLIP-failed
    # (all-NaN) or Gemini-partial (e.g., 246/656) rows pollute the fit with
    # uninformative anchor mass and bias the difficulty estimates.
    n_valid = (~np.isnan(Y)).sum(axis=1)
    keep = n_valid >= args.min_coverage
    dropped = [model_ids[j] for j in range(len(model_ids)) if not keep[j]]
    if dropped:
        print(f"Dropping {len(dropped)} respondents below --min-coverage={args.min_coverage}: {dropped}")
    Y = Y[keep]
    R = R[keep]
    model_ids = [model_ids[j] for j in range(len(model_ids)) if keep[j]]
    families = [families[j] for j in range(len(families)) if keep[j]]
    if not model_ids:
        raise SystemExit("no respondents meet --min-coverage threshold")
    print(f"Fitting on {len(model_ids)} respondents x {Y.shape[1]} items")

    np.save(args.out_dir / "responses_matrix.npy", Y)
    np.save(args.out_dir / "refusal_matrix.npy", R)

    embeddings_all = embed_images([it.image_path for it in items], backend=args.embedding_backend)
    np.save(args.out_dir / f"embeddings_{args.embedding_backend}.npy", embeddings_all)

    # Determine which items participate in the likelihood. With --prune-saturated,
    # items where the panel response pattern is all-wrong (sum=0) or all-correct
    # (sum=n_valid) are excluded from the fit because the masked likelihood has
    # no gradient on them. Difficulty for those items is still computed below
    # via the trained MLP so the saved difficulty vector covers all 656 items
    # (downstream scripts 04-06 work unchanged).
    if args.prune_saturated:
        n_valid_per_item = (~np.isnan(Y)).sum(axis=0)
        n_correct_per_item = np.nansum(Y, axis=0).astype(int)
        saturated = (n_correct_per_item == 0) | (n_correct_per_item == n_valid_per_item)
        n_pruned = int(saturated.sum())
        print(f"Pruning {n_pruned} saturated items (all-wrong or all-correct across panel); "
              f"fitting on {len(items) - n_pruned} of {len(items)} items")
        fit_mask = ~saturated
    else:
        fit_mask = np.ones(len(items), dtype=bool)

    Y_fit = Y[:, fit_mask]
    embeddings_fit = embeddings_all[fit_mask]

    rng = np.random.default_rng(args.seed)
    n_fit = int(fit_mask.sum())
    holdout = rng.choice(n_fit, size=int(0.2 * n_fit), replace=False)
    train_mask = np.ones(n_fit, dtype=bool)
    train_mask[holdout] = False

    fit = fit_amortized_rasch(
        Y_fit[:, train_mask],
        embeddings_fit[train_mask],
        AmortizedRaschConfig(embedding_dim=embeddings_fit.shape[1], n_models=len(model_ids)),
    )

    # Re-evaluate difficulty on held-out items using the trained MLP.
    import torch

    from derm_dif.irt.amortized import _DifficultyMLP

    mlp = _DifficultyMLP(embeddings_fit.shape[1], 128)
    mlp.load_state_dict({k: torch.as_tensor(v) for k, v in fit.mlp_state_dict.items()})
    mlp.eval()
    with torch.inference_mode():
        b_holdout = mlp(torch.tensor(embeddings_fit[holdout], dtype=torch.float32)).numpy()
    auc = held_out_auc(Y_fit[:, holdout], fit.theta, b_holdout)

    # Re-fit on full likelihood items for downstream analyses.
    full_fit = fit_amortized_rasch(
        Y_fit,
        embeddings_fit,
        AmortizedRaschConfig(embedding_dim=embeddings_fit.shape[1], n_models=len(model_ids)),
    )

    # Predict difficulty for ALL items (including saturated ones, which get the
    # MLP-extrapolated value) so the saved difficulty vector is length-656.
    mlp.load_state_dict({k: torch.as_tensor(v) for k, v in full_fit.mlp_state_dict.items()})
    mlp.eval()
    with torch.inference_mode():
        difficulty_all = mlp(torch.tensor(embeddings_all, dtype=torch.float32)).numpy()

    # Infit/outfit on items in the likelihood only (saturated items have no
    # meaningful residual under the model).
    Y_full = np.full_like(Y, np.nan)
    Y_full[:, fit_mask] = Y_fit
    fits = infit_outfit(Y_full, full_fit.theta, difficulty_all)

    with (args.out_dir / "amortized_fit.pkl").open("wb") as f:
        pickle.dump(
            {
                "model_ids": model_ids,
                "items": [it.item_id for it in items],
                "theta": full_fit.theta,
                "difficulty": difficulty_all,
                "infit": fits["infit"],
                "outfit": fits["outfit"],
                "holdout_auc": auc,
                "embedding_backend": args.embedding_backend,
                "fit_item_mask": fit_mask,
                "n_pruned": int((~fit_mask).sum()),
                "prune_saturated": bool(args.prune_saturated),
            },
            f,
        )
    print(f"holdout AUC = {auc:.3f}; theta sd = {full_fit.theta.std():.3f}; "
          f"n_items_in_fit = {int(fit_mask.sum())} / {len(items)}")


if __name__ == "__main__":
    main()
