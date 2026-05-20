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
    np.save(args.out_dir / "responses_matrix.npy", Y)
    np.save(args.out_dir / "refusal_matrix.npy", R)

    embeddings = embed_images([it.image_path for it in items], backend=args.embedding_backend)
    np.save(args.out_dir / f"embeddings_{args.embedding_backend}.npy", embeddings)

    rng = np.random.default_rng(args.seed)
    holdout = rng.choice(len(items), size=int(0.2 * len(items)), replace=False)
    train_mask = np.ones(len(items), dtype=bool)
    train_mask[holdout] = False

    fit = fit_amortized_rasch(
        Y[:, train_mask],
        embeddings[train_mask],
        AmortizedRaschConfig(embedding_dim=embeddings.shape[1], n_models=len(model_ids)),
    )

    # Re-evaluate difficulty on held-out items using the trained MLP.
    import torch

    from derm_dif.irt.amortized import _DifficultyMLP

    mlp = _DifficultyMLP(embeddings.shape[1], 128)
    mlp.load_state_dict({k: torch.as_tensor(v) for k, v in fit.mlp_state_dict.items()})
    mlp.eval()
    with torch.inference_mode():
        b_holdout = mlp(torch.tensor(embeddings[holdout], dtype=torch.float32)).numpy()
    auc = held_out_auc(Y[:, holdout], fit.theta, b_holdout)

    # Re-fit on full data for downstream analyses.
    full_fit = fit_amortized_rasch(
        Y,
        embeddings,
        AmortizedRaschConfig(embedding_dim=embeddings.shape[1], n_models=len(model_ids)),
    )
    fits = infit_outfit(Y, full_fit.theta, full_fit.difficulty)

    with (args.out_dir / "amortized_fit.pkl").open("wb") as f:
        pickle.dump(
            {
                "model_ids": model_ids,
                "items": [it.item_id for it in items],
                "theta": full_fit.theta,
                "difficulty": full_fit.difficulty,
                "infit": fits["infit"],
                "outfit": fits["outfit"],
                "holdout_auc": auc,
                "embedding_backend": args.embedding_backend,
            },
            f,
        )
    print(f"holdout AUC = {auc:.3f}; theta sd = {full_fit.theta.std():.3f}")


if __name__ == "__main__":
    main()
