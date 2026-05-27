"""Synthetic validation: does the IRT residualization recover ground-truth FST effects?

Three scenarios (500 replications each):

  Scenario 1 — Composition confound only (ground truth δ = 0)
    V-VI items drawn from a harder lesion mix (shift by +0.5 per lesion).
    No FST-specific item shift. Shows: naive accuracy reports a gap;
    IRT residualization recovers δ ≈ 0.

  Scenario 2 — True FST DIF (ground truth δ = 0.5)
    V-VI items have b_i shifted by +0.5 over lesion composition.
    Shows: IRT residualization recovers δ ≈ 0.5.

  Scenario 3 — Combined (composition + FST DIF, ground truth δ = 0.5)
    Both composition confound and FST shift present.
    Shows: IRT correctly partials out composition and recovers the FST residual.

  Scenario 4 — Flat-prior (unregularized) inflation at δ = 0
    Same as Scenario 1 but fits with a flat prior.
    Shows: unregularized IRT inflates Δ on saturated items.

Output: prints a table and writes artifacts/synthetic_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from derm_dif.dif.aggregate import aggregate_fst_shift, residualize
from derm_dif.irt.traditional import fit_traditional_rasch

# ---------------------------------------------------------------------------
# Simulation parameters (matching DDI structure)
# ---------------------------------------------------------------------------
I = 656          # items
J = 9            # respondents
P_BENIGN = 0.74  # DDI marginal benign prevalence

N_LESIONS = 10   # simplified: 10 lesion categories
N_REPLICATIONS = 200

# FST group sizes matching DDI
FST_I_II = 208
FST_III_IV = 241
FST_V_VI = 207

# Lesion distribution by FST: V-VI has shifted lesion mix (harder categories)
# encoded as P(drawn from "hard" lesion pool) for each FST group
P_HARD_LESION = {"I-II": 0.25, "III-IV": 0.30, "V-VI": 0.60}
HARD_LESION_DIFFICULTY_SHIFT = +0.5   # b_i += 0.5 for items from hard lesion pool


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def simulate_items(
    rng: np.random.Generator,
    fst_dif_delta: float = 0.0,
    composition_confound: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate I items with known difficulty and FST/lesion attributes.

    Args:
        fst_dif_delta: FST-specific difficulty shift added to V-VI items
                       *over and above* any lesion composition effect.
        composition_confound: if False, lesion categories are drawn from the
                              same distribution for all FST groups (no confound).

    Returns:
        b_true: (I,) true item difficulties
        attrs:  DataFrame with columns fst_group, lesion_category, malignant
    """
    fst_groups = (
        ["I-II"] * FST_I_II
        + ["III-IV"] * FST_III_IV
        + ["V-VI"] * FST_V_VI
    )
    attrs = pd.DataFrame({"fst_group": fst_groups})

    # Assign lesion category
    lesion_cats = []
    for fst in fst_groups:
        if composition_confound:
            p_hard = P_HARD_LESION[fst]
        else:
            p_hard = 0.375  # uniform mix (weighted avg of I-II/III-IV/V-VI shares)
        pool = "hard" if rng.random() < p_hard else "easy"
        # 10 lesion categories: 0-4 easy, 5-9 hard
        if pool == "hard":
            cat = int(rng.integers(5, 10))
        else:
            cat = int(rng.integers(0, 5))
        lesion_cats.append(cat)
    attrs["lesion_category"] = lesion_cats

    # Malignancy: correlated with lesion (hard lesions more often malignant)
    malignant = (attrs["lesion_category"] >= 5) & (rng.random(I) < 0.50)
    malignant |= (attrs["lesion_category"] < 5) & (rng.random(I) < 0.10)
    attrs["malignant"] = malignant.astype(bool)

    # Base difficulty: lesion-category effect (hard categories are harder)
    lesion_effect = np.where(attrs["lesion_category"] >= 5,
                             HARD_LESION_DIFFICULTY_SHIFT, 0.0)
    b_base = rng.normal(0.0, 1.5, size=I) + lesion_effect

    # FST DIF: add delta to V-VI items (independent of lesion)
    fst_shift = np.where(attrs["fst_group"] == "V-VI", fst_dif_delta, 0.0)
    b_true = b_base + fst_shift

    return b_true, attrs


def simulate_responses(
    rng: np.random.Generator,
    b_true: np.ndarray,
    n_saturated_injection: int = 0,
) -> np.ndarray:
    """Generate (J, I) response matrix from true difficulty.

    n_saturated_injection: if > 0, randomly zero out responses for this many
    items for all respondents (simulating saturated floor items), used to
    test unregularized IRT inflation.
    """
    # Respondent abilities: 2 degenerate (low), 5 varying
    theta_true = np.array([-1.5, -0.8, -0.5, -0.2, 0.0, 0.3, 0.6, 0.9, 1.2])
    responses = np.zeros((J, I), dtype=float)
    for j in range(J):
        p = sigmoid(theta_true[j] - b_true)
        responses[j] = (rng.random(I) < p).astype(float)

    if n_saturated_injection > 0:
        sat_items = rng.choice(I, size=n_saturated_injection, replace=False)
        responses[:, sat_items] = 0.0  # all-wrong (floor)

    return responses


def fit_rasch_regularized(
    responses: np.ndarray,
    lambda_b: float = 1e-2,
    seed: int = 0,
) -> np.ndarray:
    """Fit Rasch with L2 prior on b (MAP) using PyTorch; returns b estimates."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    Y = torch.tensor(responses, dtype=torch.float32)
    M = torch.ones_like(Y)

    J_local, I_local = Y.shape
    theta = nn.Parameter(torch.zeros(J_local))
    b = nn.Parameter(torch.zeros(I_local))
    opt = torch.optim.Adam([theta, b], lr=5e-2)

    for _ in range(2000):
        opt.zero_grad()
        logit = theta[:, None] - b[None, :]
        ll = -nn.functional.binary_cross_entropy_with_logits(
            logit, Y, weight=M, reduction="sum"
        )
        reg = lambda_b * (b ** 2).sum()
        anchor = 1e3 * (theta.mean() ** 2 + b.mean() ** 2)
        loss = -ll + reg + anchor
        loss.backward()
        opt.step()

    return b.detach().numpy()


def stratified_accuracy_gap(responses: np.ndarray, attrs: pd.DataFrame) -> float:
    """Naive V-VI minus I-II accuracy gap averaged across respondents."""
    correct = responses  # 1 = correct (note: for synthetic data b_true is ground truth;
    # responses encode correct/wrong directly since we generate from P(correct|θ,b))
    fst = attrs["fst_group"].values
    acc_vvi = correct[:, fst == "V-VI"].mean()
    acc_iii = correct[:, fst == "I-II"].mean()
    return float(acc_vvi - acc_iii)


def run_scenario(
    name: str,
    fst_dif_delta: float,
    true_delta: float,
    n_reps: int,
    composition_confound: bool = True,
    lambda_b_reg: float = 1e-2,
    lambda_b_unreg: float = 0.0,
    n_saturated_injection: int = 0,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    irt_deltas, naive_gaps = [], []
    ci_covers = []

    for rep in range(n_reps):
        b_true, attrs = simulate_items(
            rng, fst_dif_delta=fst_dif_delta,
            composition_confound=composition_confound,
        )
        responses = simulate_responses(rng, b_true, n_saturated_injection=n_saturated_injection)

        lambda_b = lambda_b_unreg if lambda_b_unreg > 0 else lambda_b_reg
        if lambda_b == 0.0:
            # flat-prior MLE via traditional fit
            _, b_hat = fit_traditional_rasch(responses, n_epochs=1500, seed=int(rng.integers(1 << 30)))
        else:
            b_hat = fit_rasch_regularized(responses, lambda_b=lambda_b, seed=int(rng.integers(1 << 30)))

        result = aggregate_fst_shift(
            difficulty=b_hat,
            item_attrs=attrs,
            focal=("V-VI",),
            reference=("I-II",),
            controls=("lesion_category", "malignant"),
            threshold_logits=0.5,
            n_bootstrap=500,  # fewer for speed in simulation
            seed=int(rng.integers(1 << 30)),
        )
        irt_deltas.append(result.delta)
        ci_covers.append(int(result.ci_low <= true_delta <= result.ci_high))

        # Naive stratified accuracy gap
        naive_gaps.append(stratified_accuracy_gap(responses, attrs))

    irt_arr = np.array(irt_deltas)
    naive_arr = np.array(naive_gaps)
    return {
        "scenario": name,
        "true_delta": true_delta,
        "n_reps": n_reps,
        "irt": {
            "mean_delta": float(irt_arr.mean()),
            "rmse": float(np.sqrt(((irt_arr - true_delta) ** 2).mean())),
            "bias": float(irt_arr.mean() - true_delta),
            "ci_coverage": float(np.mean(ci_covers)),
        },
        "naive_accuracy": {
            "mean_gap": float(naive_arr.mean()),
            "rmse": float(np.sqrt(((naive_arr - true_delta) ** 2).mean())),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=N_REPLICATIONS)
    ap.add_argument("--out", type=Path, default=Path("artifacts/synthetic_validation.json"))
    ap.add_argument("--seed", type=int, default=0xDD10)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    scenarios = [
        # S1: composition confound only — naive gap exists, IRT should return δ≈0
        dict(name="S1_composition_only",
             fst_dif_delta=0.0, true_delta=0.0,
             composition_confound=True),
        # S2: composition confound + true FST DIF — IRT partials out composition,
        # recovers residual δ=0.5
        dict(name="S2_composition_plus_fst",
             fst_dif_delta=0.5, true_delta=0.5,
             composition_confound=True),
        # S3: no composition confound, true FST DIF only — IRT detects the effect
        # cleanly (sensitivity: method works even without confound structure)
        dict(name="S3_fst_only_no_confound",
             fst_dif_delta=0.5, true_delta=0.5,
             composition_confound=False),
        # S4: flat-prior (unregularized) IRT under null with saturated items injected —
        # shows false-positive inflation even when true δ=0, motivating the L2 prior
        dict(name="S4_unregularized_null",
             fst_dif_delta=0.0, true_delta=0.0,
             composition_confound=True,
             lambda_b_unreg=0.0, lambda_b_reg=0.0,
             n_saturated_injection=80),
    ]

    for i, sc in enumerate(scenarios):
        print(f"Running {sc['name']} ({args.n_reps} reps)...")
        seed = args.seed + i * 1000
        r = run_scenario(n_reps=args.n_reps, seed=seed, **sc)
        results.append(r)
        irt = r["irt"]
        naive = r["naive_accuracy"]
        print(
            f"  IRT:   mean_Δ={irt['mean_delta']:+.3f}  RMSE={irt['rmse']:.3f}"
            f"  bias={irt['bias']:+.3f}  CI_cov={irt['ci_coverage']:.2f}"
        )
        print(
            f"  Naive: mean_gap={naive['mean_gap']:+.3f}  RMSE={naive['rmse']:.3f}"
        )

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}")

    # Print summary table
    print(f"\n{'Scenario':<30}  {'True δ':>7}  {'IRT Δ̂':>8}  {'IRT RMSE':>9}  {'CI cov':>7}  {'Naive gap':>10}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['scenario']:<30}  {r['true_delta']:>7.2f}"
            f"  {r['irt']['mean_delta']:>+8.3f}  {r['irt']['rmse']:>9.3f}"
            f"  {r['irt']['ci_coverage']:>7.2f}  {r['naive_accuracy']['mean_gap']:>+10.3f}"
        )


if __name__ == "__main__":
    main()
