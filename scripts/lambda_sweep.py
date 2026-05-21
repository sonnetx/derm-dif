"""Sensitivity sweep over the difficulty-prior strength lambda_b.

The canonical fit uses lambda_b = 1e-2 (chosen as the smallest prior that
bounded the floor/ceiling divergence). A reviewer's natural question is
whether the headline $\\hat\\Delta = +0.13$ no_effect decision survives a
range of prior strengths. This script refits at lambda_b values spanning
two orders of magnitude and reports $\\hat\\Delta$, the 95% CI, and the
held-out AUC at each.

  python scripts/lambda_sweep.py --ddi-root /path/to/ddi --lambdas 1e-3 3e-3 1e-2 3e-2 1e-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.data.embeddings import embed_images
from derm_dif.dif.aggregate import aggregate_fst_shift
from derm_dif.irt.amortized import (
    AmortizedRaschConfig,
    fit_amortized_rasch,
    held_out_auc,
)
from derm_dif.parsing import parse_primary, to_correctness


def build_response_matrix(jsonl_path: Path, items, model_ids: list[str], refusal_markers: list[str]):
    import json as _json
    item_id_to_idx = {it.item_id: i for i, it in enumerate(items)}
    model_id_to_idx = {m: j for j, m in enumerate(model_ids)}
    J, I = len(model_ids), len(items)
    Y = np.full((J, I), np.nan)
    truth = np.array([it.malignant for it in items])
    with jsonl_path.open() as f:
        for line in f:
            d = _json.loads(line)
            if d.get("error") is not None:
                continue
            if d["model_id"] not in model_id_to_idx or d["item_id"] not in item_id_to_idx:
                continue
            j = model_id_to_idx[d["model_id"]]
            i = item_id_to_idx[d["item_id"]]
            parsed = parse_primary(d["raw_text"], refusal_markers)
            if parsed.label == "refusal":
                continue
            corr = to_correctness(parsed, bool(truth[i]))
            if corr is not None:
                Y[j, i] = float(corr)
    return Y


def one_lambda(
    lam: float,
    Y: np.ndarray,
    embeddings: np.ndarray,
    attrs: pd.DataFrame,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    cfg = AmortizedRaschConfig(
        embedding_dim=embeddings.shape[1],
        n_models=Y.shape[0],
        difficulty_l2=lam,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    n = Y.shape[1]
    holdout = rng.choice(n, size=int(0.2 * n), replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[holdout] = False
    train_fit = fit_amortized_rasch(Y[:, train_mask], embeddings[train_mask], cfg)

    import torch
    from derm_dif.irt.amortized import _DifficultyMLP
    mlp = _DifficultyMLP(embeddings.shape[1], 128)
    mlp.load_state_dict({k: torch.as_tensor(v) for k, v in train_fit.mlp_state_dict.items()})
    mlp.eval()
    with torch.inference_mode():
        b_holdout = mlp(torch.tensor(embeddings[holdout], dtype=torch.float32)).numpy()
    auc = held_out_auc(Y[:, holdout], train_fit.theta, b_holdout)

    full_fit = fit_amortized_rasch(Y, embeddings, cfg)
    agg = aggregate_fst_shift(
        difficulty=full_fit.difficulty,
        item_attrs=attrs,
        threshold_logits=threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "lambda_b": float(lam),
        "delta": float(agg.delta),
        "ci_low": float(agg.ci_low),
        "ci_high": float(agg.ci_high),
        "decision": agg.decision,
        "holdout_auc": float(auc),
        "difficulty_min": float(full_fit.difficulty.min()),
        "difficulty_max": float(full_fit.difficulty.max()),
        "difficulty_sd": float(full_fit.difficulty.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    ap.add_argument("--protocol-config", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"))
    ap.add_argument("--source", default="api-openai,api-anthropic,api-google,contrastive-zeroshot")
    ap.add_argument("--min-coverage", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--seed", type=int, default=0xDD1F)
    ap.add_argument("--out", type=Path, default=Path("artifacts/lambda_sweep.json"))
    args = ap.parse_args()

    items = load_ddi(args.ddi_root)
    models_cfg = yaml.safe_load(args.models_config.read_text())
    protocol = yaml.safe_load(args.protocol_config.read_text())["primary_protocol"]
    eligible = [m for m in models_cfg["models"] if not m.get("optional", False)]
    if args.source:
        allowed = {s.strip() for s in args.source.split(",")}
        eligible = [m for m in eligible if m["source"] in allowed]
    model_ids = [m["id"] for m in eligible]

    Y = build_response_matrix(args.responses, items, model_ids, protocol["parsing"]["refusal_markers"])
    n_valid = (~np.isnan(Y)).sum(axis=1)
    keep = n_valid >= args.min_coverage
    Y = Y[keep]
    print(f"Sweeping over {len(args.lambdas)} lambda values on J={Y.shape[0]} x I={Y.shape[1]}")

    embeddings_path = args.rasch_dir / "embeddings_biomedclip.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"Loaded cached embeddings from {embeddings_path}")
    else:
        print("Embedding from scratch...")
        embeddings = embed_images([it.image_path for it in items])

    attrs = pd.DataFrame({
        "item_id": [it.item_id for it in items],
        "fst_group": [it.fst_group for it in items],
        "lesion_category": [it.lesion_category for it in items],
        "malignant": [it.malignant for it in items],
    })

    rows = []
    for lam in args.lambdas:
        print(f"\n--- lambda_b = {lam} ---")
        r = one_lambda(lam, Y, embeddings, attrs, args.threshold, args.n_bootstrap, args.seed)
        print(json.dumps(r, indent=2))
        rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_lambdas": len(args.lambdas),
        "seed": int(args.seed),
        "per_lambda": rows,
        "all_no_effect": all(r["decision"] == "no_effect" for r in rows),
        "decisions": [r["decision"] for r in rows],
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Decisions across lambda: {summary['decisions']}")
    print(f"All no_effect across the sweep: {summary['all_no_effect']}")


if __name__ == "__main__":
    main()
