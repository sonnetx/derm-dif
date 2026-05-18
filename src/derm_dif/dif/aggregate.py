"""Primary endpoint: aggregate FST difficulty shift on the residualized Rasch logit scale.

Decision rule (pre-registered in config/analysis.yaml):
  Conclude meaningful FST measurement non-invariance iff
    |Delta| >= 0.5 logits AND bootstrap 95% CI of Delta excludes zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class AggregateDIFResult:
    delta: float
    ci_low: float
    ci_high: float
    threshold: float
    decision: str            # "non_invariance", "no_effect", "indeterminate"
    n_focal: int
    n_reference: int
    bootstrap: np.ndarray    # raw bootstrap distribution for plotting


def residualize(difficulty: np.ndarray, item_attrs: pd.DataFrame, controls: list[str]) -> np.ndarray:
    """OLS-residualize fitted difficulty against control variables (lesion category, malignancy)."""
    pieces = [np.ones((len(item_attrs), 1))]
    for c in controls:
        col = item_attrs[c]
        if (not pd.api.types.is_numeric_dtype(col)) or pd.api.types.is_bool_dtype(col):
            dummies = pd.get_dummies(col, prefix=c, drop_first=True).astype(float)
            pieces.append(dummies.values)
        else:
            z = (col - col.mean()) / col.std(ddof=0).clip(1e-8)
            pieces.append(z.values.reshape(-1, 1))
    X = np.concatenate(pieces, axis=1)
    reg = LinearRegression(fit_intercept=False).fit(X, difficulty)
    return difficulty - reg.predict(X)


def _delta(b_resid: np.ndarray, fst: pd.Series, focal: list[str], reference: list[str]) -> tuple[float, int, int]:
    in_focal = fst.isin(focal).values
    in_ref = fst.isin(reference).values
    return float(b_resid[in_focal].mean() - b_resid[in_ref].mean()), int(in_focal.sum()), int(in_ref.sum())


def aggregate_fst_shift(
    difficulty: np.ndarray,
    item_attrs: pd.DataFrame,
    *,
    fst_column: str = "fst_group",
    focal: tuple[str, ...] = ("V-VI",),
    reference: tuple[str, ...] = ("I-II",),
    controls: tuple[str, ...] = ("lesion_category", "malignant"),
    threshold_logits: float = 0.5,
    n_bootstrap: int = 2000,
    seed: int = 0xDD4,
    paired_resample: bool = True,
) -> AggregateDIFResult:
    """Compute the primary endpoint with paired (item, model) bootstrap.

    `paired_resample` resamples items only here because difficulty is a per-item
    estimate; full (item, model) bootstrap requires re-fitting the Rasch model
    per resample and is exposed in `aggregate_fst_shift_full_bootstrap`.
    """
    rng = np.random.default_rng(seed)
    fst = item_attrs[fst_column]

    b_resid = residualize(difficulty, item_attrs, list(controls))
    delta_hat, n_f, n_r = _delta(b_resid, fst, list(focal), list(reference))

    I = len(difficulty)
    boot = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, I, size=I)
        d_b = difficulty[idx]
        a_b = item_attrs.iloc[idx].reset_index(drop=True)
        b_resid_b = residualize(d_b, a_b, list(controls))
        boot[b], _, _ = _delta(b_resid_b, a_b[fst_column], list(focal), list(reference))

    ci_low, ci_high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    excludes_zero = (ci_low > 0) or (ci_high < 0)
    meaningful = abs(delta_hat) >= threshold_logits
    if meaningful and excludes_zero:
        decision = "non_invariance"
    elif not excludes_zero and abs(delta_hat) < threshold_logits:
        decision = "no_effect"
    else:
        decision = "indeterminate"

    return AggregateDIFResult(
        delta=delta_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        threshold=threshold_logits,
        decision=decision,
        n_focal=n_f,
        n_reference=n_r,
        bootstrap=boot,
    )


def configural_invariance_spearman(
    responses: np.ndarray,
    embeddings: np.ndarray,
    item_attrs: pd.DataFrame,
    fst_column: str,
    subset_a: list[str],
    subset_b: list[str],
    fit_amortized,
    config,
) -> float:
    """Refit the Rasch model on each FST subset; report Spearman of resulting model abilities.

    `fit_amortized` and `config` injected to avoid a hard dependency in this module.
    """
    from scipy.stats import spearmanr

    mask_a = item_attrs[fst_column].isin(subset_a).values
    mask_b = item_attrs[fst_column].isin(subset_b).values

    fit_a = fit_amortized(responses[:, mask_a], embeddings[mask_a], config)
    fit_b = fit_amortized(responses[:, mask_b], embeddings[mask_b], config)
    rho, _ = spearmanr(fit_a.theta, fit_b.theta)
    return float(rho)
