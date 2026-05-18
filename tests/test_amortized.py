"""Smoke test: amortized Rasch on synthetic Rasch-distributed data recovers theta order."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from derm_dif.irt.amortized import (
    AmortizedRaschConfig,
    fit_amortized_rasch,
    predict_response_prob,
)


def test_recovers_theta_rank_on_synthetic():
    rng = np.random.default_rng(0)
    J, I, D = 15, 200, 32
    theta = rng.normal(size=J)
    embeds = rng.normal(size=(I, D))
    w = rng.normal(scale=0.3, size=D)
    b = embeds @ w
    probs = 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    Y = (rng.uniform(size=probs.shape) < probs).astype(float)

    fit = fit_amortized_rasch(
        Y, embeds, AmortizedRaschConfig(embedding_dim=D, n_models=J, n_epochs=1500)
    )
    rho, _ = spearmanr(fit.theta, theta)
    assert rho > 0.85, f"theta rank correlation too low: {rho}"

    rho_b, _ = spearmanr(fit.difficulty, b)
    assert rho_b > 0.6, f"difficulty rank correlation too low: {rho_b}"


def test_predict_response_prob_shape():
    theta = np.array([0.0, 1.0])
    b = np.array([-0.5, 0.0, 0.5])
    P = predict_response_prob(theta, b)
    assert P.shape == (2, 3)
    assert (P > 0).all() and (P < 1).all()
