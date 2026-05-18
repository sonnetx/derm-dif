"""Traditional Rasch fit (joint MLE) for amortized-calibration validation.

Used only on the subset of items where responses are dense enough for the
joint MLE to converge. The amortized fit is the primary instrument; this is
a sanity check.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def fit_traditional_rasch(
    responses: np.ndarray,
    n_epochs: int = 3000,
    lr: float = 5e-2,
    seed: int = 0xDD2,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint MLE for theta_j and b_i. Returns (theta, b)."""
    torch.manual_seed(seed)
    J, I = responses.shape
    Y = torch.tensor(responses, dtype=torch.float32)
    M = torch.tensor(~np.isnan(responses), dtype=torch.float32)
    Y_clean = torch.nan_to_num(Y, nan=0.0)

    theta = nn.Parameter(torch.zeros(J))
    b = nn.Parameter(torch.zeros(I))
    opt = torch.optim.Adam([theta, b], lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()
        logit = theta[:, None] - b[None, :]
        ll = -nn.functional.binary_cross_entropy_with_logits(
            logit, Y_clean, weight=M, reduction="sum"
        )
        anchor = (theta.mean()) ** 2 + (b.mean()) ** 2  # identifiability: zero-mean
        loss = -ll + 1e3 * anchor
        loss.backward()
        opt.step()

    return theta.detach().numpy(), b.detach().numpy()


def spearman_against_amortized(b_traditional: np.ndarray, b_amortized: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(b_traditional, b_amortized)
    return float(rho)
