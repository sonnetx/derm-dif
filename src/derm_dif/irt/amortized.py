"""Amortized Rasch calibration (Truong et al. 2025, ported to image embeddings).

We jointly estimate per-respondent ability theta_j and a difficulty MLP
b_i = f_phi(e_i) by maximizing the marginal log-likelihood of observed responses.

Missing responses (refusals, unparseable, parser-failed) are masked out of the
likelihood; the missing-mask is itself analyzed downstream as a refusal-DIF signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AmortizedRaschConfig:
    embedding_dim: int
    n_models: int
    hidden_dim: int = 128
    ability_l2: float = 1e-2
    # Soft prior on item difficulty to keep b_i bounded on items where the
    # response pattern is saturated (all-correct or all-wrong across the
    # panel). Without it, the masked likelihood has no gradient on those
    # items and the MLP diverges to extreme values (we observed b_i in
    # [-36, +13] on a J=5 fit, vs. the typical Rasch range of about pm 3).
    difficulty_l2: float = 1e-2
    weight_decay: float = 1e-4
    lr: float = 5e-3
    n_epochs: int = 2000
    seed: int = 0xDD1F


class _DifficultyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:  # (N, D) -> (N,)
        return self.net(e).squeeze(-1)


@dataclass(frozen=True)
class AmortizedRaschFit:
    theta: np.ndarray            # (J,) model abilities
    difficulty: np.ndarray       # (I,) item difficulties b_i = f_phi(e_i)
    mlp_state_dict: dict         # for refit / inference on new items
    history: list[float]         # neg log-likelihood per epoch


def fit_amortized_rasch(
    responses: np.ndarray,       # (J, I) in {0, 1}, NaN for missing
    embeddings: np.ndarray,      # (I, D)
    config: AmortizedRaschConfig,
    device: str = "cpu",
) -> AmortizedRaschFit:
    """Joint MLE under the Rasch model with embedding-amortized item difficulty.

    Returns a fit object with point estimates; uncertainty is obtained by
    bootstrapping in the calling code.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    J, I = responses.shape
    assert embeddings.shape[0] == I
    assert embeddings.shape[1] == config.embedding_dim
    assert J == config.n_models

    Y = torch.tensor(responses, dtype=torch.float32, device=device)
    M = torch.tensor(~np.isnan(responses), dtype=torch.float32, device=device)
    Y_clean = torch.nan_to_num(Y, nan=0.0)
    E = torch.tensor(embeddings, dtype=torch.float32, device=device)

    theta = nn.Parameter(torch.zeros(J, device=device))
    mlp = _DifficultyMLP(config.embedding_dim, config.hidden_dim).to(device)

    opt = torch.optim.Adam(
        [{"params": [theta], "weight_decay": 0.0}, {"params": mlp.parameters(), "weight_decay": config.weight_decay}],
        lr=config.lr,
    )

    history: list[float] = []
    for _ in range(config.n_epochs):
        opt.zero_grad()
        b = mlp(E)                                 # (I,)
        # Rasch log-likelihood on observed entries.
        logit = theta[:, None] - b[None, :]        # (J, I)
        ll = -nn.functional.binary_cross_entropy_with_logits(
            logit, Y_clean, weight=M, reduction="sum"
        )
        # Anchor identifiability: center theta to mean zero (Rasch is invariant
        # to constant shift; pinning the centroid removes the indeterminacy).
        anchor = (theta.mean()) ** 2
        prior = (
            config.ability_l2 * (theta**2).sum()
            + config.difficulty_l2 * (b**2).sum()
        )
        loss = -ll + prior + 1e3 * anchor
        loss.backward()
        opt.step()
        history.append(float(loss.item()))

    with torch.inference_mode():
        b_final = mlp(E).cpu().numpy()
    return AmortizedRaschFit(
        theta=theta.detach().cpu().numpy(),
        difficulty=b_final,
        mlp_state_dict={k: v.detach().cpu() for k, v in mlp.state_dict().items()},
        history=history,
    )


def predict_response_prob(theta: np.ndarray, difficulty: np.ndarray) -> np.ndarray:
    """Posterior-predictive correctness probability under the fitted Rasch model."""
    logit = theta[:, None] - difficulty[None, :]
    return 1.0 / (1.0 + np.exp(-logit))


def held_out_auc(
    responses_holdout: np.ndarray,   # (J, I_h)
    theta: np.ndarray,
    difficulty_holdout: np.ndarray,  # (I_h,)
) -> float:
    """Macro-averaged AUC of predicted correctness on held-out items."""
    from sklearn.metrics import roc_auc_score

    probs = predict_response_prob(theta, difficulty_holdout)  # (J, I_h)
    y_flat: list[int] = []
    p_flat: list[float] = []
    for j in range(responses_holdout.shape[0]):
        for i in range(responses_holdout.shape[1]):
            if not np.isnan(responses_holdout[j, i]):
                y_flat.append(int(responses_holdout[j, i]))
                p_flat.append(float(probs[j, i]))
    if len(set(y_flat)) < 2:
        return float("nan")
    return float(roc_auc_score(y_flat, p_flat))


def infit_outfit(responses: np.ndarray, theta: np.ndarray, difficulty: np.ndarray) -> dict:
    """Standard Rasch infit/outfit mean-square statistics, item-wise."""
    P = predict_response_prob(theta, difficulty)
    W = P * (1.0 - P)                              # variance under the model
    M = ~np.isnan(responses)
    Y = np.where(M, responses, P)                  # neutral for masked entries
    Z2 = (Y - P) ** 2 / np.clip(W, 1e-6, None)
    # outfit: unweighted mean of standardized residuals squared, per item.
    outfit = np.where(M.any(axis=0), (Z2 * M).sum(axis=0) / np.clip(M.sum(axis=0), 1, None), np.nan)
    # infit: variance-weighted mean of squared raw residuals, per item.
    raw_sq = (Y - P) ** 2
    infit_num = (raw_sq * M).sum(axis=0)
    infit_den = (W * M).sum(axis=0)
    infit = np.where(infit_den > 0, infit_num / infit_den, np.nan)
    return {"infit": infit, "outfit": outfit}
