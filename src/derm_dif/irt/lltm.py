"""Linear Logistic Test Model: decompose item difficulty into linear effects.

We fit b_i = X_i @ beta directly against fitted Rasch difficulties via OLS, with
bootstrap CIs over (item, model) resamples. The original LLTM formulation refits
the Rasch likelihood with the constraint b = X beta; in our two-stage setup we
bake the difficulties from the amortized fit and regress, which is computationally
cheaper and gives the same point estimate up to noise. The bootstrap accounts for
the two-stage uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LLTMFit:
    coef: pd.Series           # name -> coefficient
    ci_low: pd.Series
    ci_high: pd.Series
    r2: float
    feature_names: list[str]


def design_matrix(item_attributes: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build a one-hot / dummy design matrix for the requested features.

    Categorical columns are dummy-coded with the most-frequent level as reference.
    Continuous columns are z-scored.
    """
    pieces: list[np.ndarray] = []
    names: list[str] = ["(intercept)"]
    pieces.append(np.ones((len(item_attributes), 1)))
    for f in features:
        col = item_attributes[f]
        if col.dtype == "O" or str(col.dtype).startswith("category") or col.dtype == bool:
            dummies = pd.get_dummies(col, prefix=f, drop_first=True).astype(float)
            pieces.append(dummies.values)
            names.extend(dummies.columns.tolist())
        else:
            z = (col - col.mean()) / col.std(ddof=0).clip(1e-8)
            pieces.append(z.values.reshape(-1, 1))
            names.append(f)
    X = np.concatenate(pieces, axis=1)
    return X, names


def fit_lltm(
    difficulty: np.ndarray,
    item_attributes: pd.DataFrame,
    features: list[str],
    n_bootstrap: int = 2000,
    seed: int = 0xDD3,
) -> LLTMFit:
    """OLS regression of fitted item difficulty on attribute features, with bootstrap CIs."""
    rng = np.random.default_rng(seed)
    X, names = design_matrix(item_attributes, features)
    I = X.shape[0]

    beta_hat, *_ = np.linalg.lstsq(X, difficulty, rcond=None)
    yhat = X @ beta_hat
    ss_res = float(((difficulty - yhat) ** 2).sum())
    ss_tot = float(((difficulty - difficulty.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    boot = np.zeros((n_bootstrap, X.shape[1]))
    for b in range(n_bootstrap):
        idx = rng.integers(0, I, size=I)
        Xb = X[idx]
        yb = difficulty[idx]
        beta_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        boot[b] = beta_b
    ci_low = np.quantile(boot, 0.025, axis=0)
    ci_high = np.quantile(boot, 0.975, axis=0)

    return LLTMFit(
        coef=pd.Series(beta_hat, index=names),
        ci_low=pd.Series(ci_low, index=names),
        ci_high=pd.Series(ci_high, index=names),
        r2=r2,
        feature_names=names,
    )


def nested_variance_explained(
    difficulty: np.ndarray,
    item_attributes: pd.DataFrame,
    nesting: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    """Variance explained at each nesting level. `nesting` is a list of (label, features)."""
    rows = []
    for label, features in nesting:
        if not features:
            rows.append({"model": label, "r2": 0.0, "n_features": 0})
            continue
        fit = fit_lltm(difficulty, item_attributes, features, n_bootstrap=200)
        rows.append({"model": label, "r2": fit.r2, "n_features": len(fit.feature_names)})
    return pd.DataFrame(rows)
