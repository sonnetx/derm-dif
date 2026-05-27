"""Multi-seed robustness sweep for the headline IRT primary endpoint.

Re-runs the amortized Rasch fit + aggregate-DIF with K different seeds
(controlling MLP initialization, Adam, train/holdout split, and bootstrap
resampling) and reports the median + range of $\\hat\\Delta$, the CI bounds,
and the LLTM $\\beta_{FST}$ at M4. A reviewer-credibility move: shows the
headline numbers don't depend on a particular random seed.

  python scripts/seed_sweep.py --ddi-root /path/to/ddi --seeds 11 13 17 19 23
"""

from __future__ import annotations

import argparse
import json
import pickle
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
    infit_outfit,
)
from derm_dif.parsing import parse_primary, to_correctness


def build_response_matrix(jsonl_path: Path, items, model_ids: list[str], refusal_markers: list[str]):
    """Mirror of scripts/03's build_response_matrix; inlined so we don't depend on script imports."""
    import json
    item_id_to_idx = {it.item_id: i for i, it in enumerate(items)}
    model_id_to_idx = {m: j for j, m in enumerate(model_ids)}
    J, I = len(model_ids), len(items)
    Y = np.full((J, I), np.nan)
    truth = np.array([it.malignant for it in items])
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
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


def one_seed(
    seed: int,
    Y: np.ndarray,
    embeddings: np.ndarray,
    items_list,
    attrs: pd.DataFrame,
    threshold: float,
    n_bootstrap: int,
) -> dict[str, Any]:
    cfg = AmortizedRaschConfig(
        embedding_dim=embeddings.shape[1], n_models=Y.shape[0], seed=seed
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
        "seed": int(seed),
        "delta": float(agg.delta),
        "ci_low": float(agg.ci_low),
        "ci_high": float(agg.ci_high),
        "decision": agg.decision,
        "holdout_auc": float(auc),
        "theta": full_fit.theta.tolist(),
        "difficulty_sd": float(full_fit.difficulty.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    ap.add_argument("--protocol-config", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--rasch-dir", type=Path, default=Path("artifacts/rasch"),
                    help="Used to load the precomputed BiomedCLIP embeddings; faster than re-embedding per seed.")
    ap.add_argument("--source", default="api-openai,api-anthropic,api-google,contrastive-zeroshot")
    ap.add_argument("--min-coverage", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    ap.add_argument("--out", type=Path, default=Path("artifacts/seed_sweep.json"))
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
    model_ids = [model_ids[j] for j in range(len(model_ids)) if keep[j]]
    print(f"Sweeping over {len(args.seeds)} seeds on J={Y.shape[0]} respondents x I={Y.shape[1]} items")

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
    for s in args.seeds:
        print(f"\n--- seed = {s} ---")
        r = one_seed(s, Y, embeddings, items, attrs, args.threshold, args.n_bootstrap)
        print(json.dumps({k: v for k, v in r.items() if k != "theta"}, indent=2))
        rows.append(r)

    deltas = np.array([r["delta"] for r in rows])
    ci_widths = np.array([r["ci_high"] - r["ci_low"] for r in rows])
    aucs = np.array([r["holdout_auc"] for r in rows])

    summary = {
        "n_seeds": len(args.seeds),
        "delta_median": float(np.median(deltas)),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "delta_iqr": [float(np.quantile(deltas, 0.25)), float(np.quantile(deltas, 0.75))],
        "ci_width_median": float(np.median(ci_widths)),
        "holdout_auc_median": float(np.median(aucs)),
        "all_decisions_same": len(set(r["decision"] for r in rows)) == 1,
        "decisions": [r["decision"] for r in rows],
        "per_seed": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Across {len(args.seeds)} seeds: delta median = {summary['delta_median']:+.3f}, "
          f"min/max = [{summary['delta_min']:+.3f}, {summary['delta_max']:+.3f}], "
          f"decisions = {summary['decisions']}")


if __name__ == "__main__":
    main()
