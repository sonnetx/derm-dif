"""Mechanism analysis: per-item DIF magnitude vs image properties, and refusal-rate DIF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr


def image_properties(image_paths: list) -> pd.DataFrame:
    """Cheap-to-compute image-level features used as candidate DIF mechanisms."""
    rows = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        h, w, _ = arr.shape
        rows.append(
            {
                "image_resolution": h * w,
                "mean_luminance": float(arr.mean()),
                "r_channel_mean": float(arr[..., 0].mean()),
                "g_channel_mean": float(arr[..., 1].mean()),
                "b_channel_mean": float(arr[..., 2].mean()),
                "luminance_std": float(arr.mean(axis=-1).std()),
            }
        )
    return pd.DataFrame(rows)


def correlate_dif_with_features(
    abs_dif: np.ndarray, features: pd.DataFrame
) -> pd.DataFrame:
    """Spearman correlation of |delta_b| against each candidate mechanism feature."""
    out = []
    for col in features.columns:
        rho, p = spearmanr(abs_dif, features[col].values)
        out.append({"feature": col, "spearman_rho": float(rho), "p": float(p)})
    return pd.DataFrame(out).sort_values("spearman_rho", key=np.abs, ascending=False)


def embedding_distance_to_centroid(
    embeddings: np.ndarray, in_centroid_mask: np.ndarray
) -> np.ndarray:
    """L2 distance from each item's embedding to the centroid of `in_centroid_mask` items."""
    centroid = embeddings[in_centroid_mask].mean(axis=0)
    return np.linalg.norm(embeddings - centroid, axis=1)


def refusal_rate_by_fst(
    refused: np.ndarray,           # (J, I) bool
    fst_group: pd.Series,          # length I
) -> pd.DataFrame:
    """Refusal rate per (model, FST group), and overall by FST."""
    long = []
    J, I = refused.shape
    for j in range(J):
        for g in fst_group.unique():
            mask = (fst_group == g).values
            n = int(mask.sum())
            if n == 0:
                continue
            r = float(refused[j, mask].mean())
            long.append({"model_idx": j, "fst_group": g, "refusal_rate": r, "n_items": n})
    return pd.DataFrame(long)


def refusal_logit_fst(
    refused: np.ndarray, fst_group: pd.Series, n_bootstrap: int = 2000, seed: int = 0xDD6
) -> dict:
    """Cluster-bootstrap (clustered by model) logistic regression of refusal on FST."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    J, I = refused.shape
    fst_dummies = pd.get_dummies(fst_group, drop_first=True).astype(float).values  # (I, k)

    def _fit(model_idx_subset: np.ndarray) -> np.ndarray:
        # Stack (model x item) into long format on the chosen model subset.
        ys = []
        Xs = []
        for j in model_idx_subset:
            ys.append(refused[j])
            Xs.append(fst_dummies)
        y = np.concatenate(ys)
        X = np.concatenate(Xs, axis=0)
        if y.sum() == 0 or y.sum() == len(y):
            return np.full(X.shape[1] + 1, np.nan)
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        clf.fit(X, y)
        return np.concatenate([clf.intercept_, clf.coef_.ravel()])

    point = _fit(np.arange(J))
    boot = np.zeros((n_bootstrap, len(point)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, J, size=J)  # cluster-bootstrap on models
        boot[b] = _fit(idx)
    return {
        "coef": point,
        "ci_low": np.nanquantile(boot, 0.025, axis=0),
        "ci_high": np.nanquantile(boot, 0.975, axis=0),
    }
