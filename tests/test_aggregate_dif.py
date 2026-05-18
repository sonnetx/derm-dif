"""Aggregate DIF: with no FST signal injected the CI should cover zero;
with strong FST signal the CI should exclude zero and the threshold should fire."""

from __future__ import annotations

import numpy as np
import pandas as pd

from derm_dif.dif.aggregate import aggregate_fst_shift


def _attrs(I: int, rng) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fst_group": rng.choice(["I-II", "III-IV", "V-VI"], size=I, p=[0.4, 0.3, 0.3]),
            "lesion_category": rng.choice(["A", "B", "C"], size=I),
            "malignant": rng.choice([True, False], size=I),
        }
    )


def test_null_does_not_fire():
    rng = np.random.default_rng(7)
    I = 600
    attrs = _attrs(I, rng)
    b = rng.normal(scale=1.0, size=I)
    res = aggregate_fst_shift(b, attrs, n_bootstrap=400)
    assert res.decision in {"no_effect", "indeterminate"}


def test_injected_signal_fires():
    rng = np.random.default_rng(11)
    I = 600
    attrs = _attrs(I, rng)
    b = rng.normal(scale=0.5, size=I)
    boost = (attrs["fst_group"] == "V-VI").astype(float).values * 1.0
    b = b + boost
    res = aggregate_fst_shift(b, attrs, n_bootstrap=400)
    assert res.decision == "non_invariance"
    assert res.delta > 0.5
