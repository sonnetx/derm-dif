"""Exploratory secondary endpoints: per-item DIF.

Three methods are reported, all with FDR control. With J = 12-20 respondents per
item these tests are severely power-limited; we report effect-size scatter and
cross-method agreement rather than treating individual discoveries as confirmatory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PerItemDIFResult:
    item_id: str
    lord_chi2: float
    lord_p: float
    mh_delta: float           # Mantel-Haenszel delta-difficulty in logits
    mh_p: float
    permutation_p: float


def _matched_subset(theta: np.ndarray, scores_focal: np.ndarray, scores_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For matched-ability conditioning, bin respondents by their theta and pair within bins.

    Returns 2x2 contingency table summed across matching bins (focal vs reference).
    """
    bins = np.quantile(theta, np.linspace(0, 1, 6))
    bins[-1] += 1e-6
    bin_idx = np.digitize(theta, bins) - 1
    a = b = c = d = 0
    for k in range(5):
        in_bin = bin_idx == k
        if not in_bin.any():
            continue
        a += int(scores_focal[in_bin].sum())
        b += int((1 - scores_focal[in_bin]).sum())
        c += int(scores_ref[in_bin].sum())
        d += int((1 - scores_ref[in_bin]).sum())
    return np.array([[a, b], [c, d]]), bin_idx


def lord_chi_square(
    theta: np.ndarray,
    item_responses: np.ndarray,   # (J,) for one item
    is_focal_item: bool,
    focal_difficulty: float,
    ref_difficulty: float,
) -> tuple[float, float]:
    """Approximate Lord chi-square for a single item.

    With one b parameter per group, Lord's test reduces to (b_focal - b_ref)^2 / Var.
    Variance is estimated from the inverse Fisher information at the joint MLE.
    """
    p_focal = 1.0 / (1.0 + np.exp(-(theta - focal_difficulty)))
    p_ref = 1.0 / (1.0 + np.exp(-(theta - ref_difficulty)))
    var_focal = 1.0 / np.clip((p_focal * (1 - p_focal)).sum(), 1e-6, None)
    var_ref = 1.0 / np.clip((p_ref * (1 - p_ref)).sum(), 1e-6, None)
    diff = focal_difficulty - ref_difficulty
    chi2 = diff**2 / (var_focal + var_ref)
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def mantel_haenszel(
    theta: np.ndarray,
    scores_focal: np.ndarray,    # (J,) responses on this item from focal-FST respondents
    scores_ref: np.ndarray,      # (J,) from reference-FST respondents (matched-respondent design)
) -> tuple[float, float]:
    """MH delta-difficulty in logits with one-sided chi-square p-value.

    For our items-grouped design, scores_focal/scores_ref refer to responses from
    the same respondent pool, computed against items in the focal/reference FST
    subsets matched on lesion type. The interpretation is the standard MH alpha
    transformed to delta = -2.35 * ln(alpha).
    """
    table, _ = _matched_subset(theta, scores_focal, scores_ref)
    a, b = table[0]
    c, d = table[1]
    if a * d == 0 or b * c == 0:
        return float("nan"), float("nan")
    alpha = (a * d) / (b * c)
    delta = -2.35 * np.log(alpha)
    n = a + b + c + d
    e = (a + b) * (a + c) / n
    v = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1)) if n > 1 else 1.0
    chi2 = (a - e) ** 2 / max(v, 1e-9)
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(delta), float(p)


def permutation_residual_test(
    item_responses: np.ndarray,   # (J,) responses on this item
    expected: np.ndarray,         # (J,) Rasch-predicted P(correct) under null (no DIF)
    fst_focal_mask: np.ndarray,   # (J,) bool, which respondents are "focal-aligned"
    n_perm: int = 10000,
    seed: int = 0xDD5,
) -> float:
    """Two-sided p-value for the difference in mean residual between focal/ref strata.

    NOTE: in the items-grouped design this test is run with respondents standing in
    for groups only when an alignment metric (e.g., model performance ratio
    on FST-V/VI vs FST-I/II) is available; otherwise it is skipped.
    """
    rng = np.random.default_rng(seed)
    obs_resid = item_responses - expected
    if fst_focal_mask.sum() in (0, len(item_responses)):
        return float("nan")
    obs_diff = obs_resid[fst_focal_mask].mean() - obs_resid[~fst_focal_mask].mean()
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(fst_focal_mask)
        diff = obs_resid[perm].mean() - obs_resid[~perm].mean()
        if abs(diff) >= abs(obs_diff):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Returns boolean mask of items declared significant under BH FDR."""
    p = np.asarray(p_values)
    valid = ~np.isnan(p)
    out = np.zeros_like(p, dtype=bool)
    if not valid.any():
        return out
    p_v = p[valid]
    order = np.argsort(p_v)
    n = len(p_v)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p_v[order] <= thresh
    if passed.any():
        cutoff = np.where(passed)[0].max()
        sig_idx_in_valid = order[: cutoff + 1]
        valid_idx = np.where(valid)[0]
        out[valid_idx[sig_idx_in_valid]] = True
    return out
