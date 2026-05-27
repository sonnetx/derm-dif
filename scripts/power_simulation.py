"""Null-distribution simulation for the J=5 configural-invariance Spearman.

The empirical configural Spearman (Section sec:prelim_results) is rho ~ 1.000
between FST-I/II and FST-V/VI subset refits. With J=5 respondents, that's a
Spearman over only 5 points -- mechanically constrained -- so a reviewer's
natural question is "what is the chance of seeing rho > 0.9 under no real
construct-stability signal?" This script runs the corresponding null sim.

Method:
- For each of n_iter null replications:
  - Draw J "true" abilities theta from N(0, sigma_theta).
  - Draw I=656 "true" item difficulties b from N(0, sigma_b). No FST signal.
  - Sample Rasch responses: Y_{ji} ~ Bernoulli(sigmoid(theta_j - b_i)).
  - Split items randomly into two subsets of size 208 and 207
    (matching the empirical FST-I/II and FST-V/VI splits).
  - Compute theta_hat per respondent in each subset by per-respondent
    logit-mean correctness (the joint-MLE limit when difficulties are
    integrated out and embeddings are non-informative).
  - Compute Spearman rho(theta_hat_a, theta_hat_b).
- Report the empirical CDF of rho, with markers at the cutoff (0.9)
  and at the empirically-observed rho (1.0).

The simulation does NOT use the amortized MLP -- the embedding-amortized
fit only adds prior shrinkage and would make the rho even higher than the
per-respondent logit-mean baseline. So this is a conservative upper bound
on the null distribution: if rho > 0.9 is rare even under this minimal
baseline, the empirical rho is a real signal.

  python scripts/power_simulation.py --n-iter 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def logit_mean(y: np.ndarray) -> float:
    """logit(mean(y)) with shrinkage for items at floor/ceiling."""
    n = len(y)
    s = y.sum()
    # Laplace smoothing so logit is finite when s == 0 or s == n.
    p = (s + 0.5) / (n + 1.0)
    return float(np.log(p / (1.0 - p)))


def one_null(J: int, n_items: int, n_a: int, sigma_theta: float, sigma_b: float, rng) -> float:
    theta = rng.normal(0.0, sigma_theta, size=J)
    b = rng.normal(0.0, sigma_b, size=n_items)
    logits = theta[:, None] - b[None, :]
    p = 1.0 / (1.0 + np.exp(-logits))
    Y = (rng.uniform(size=p.shape) < p).astype(float)

    perm = rng.permutation(n_items)
    idx_a, idx_b = perm[:n_a], perm[n_a : 2 * n_a + 0]  # 2 disjoint subsets
    Y_a = Y[:, idx_a]
    Y_b = Y[:, idx_b]
    theta_a = np.array([logit_mean(Y_a[j]) for j in range(J)])
    theta_b = np.array([logit_mean(Y_b[j]) for j in range(J)])
    rho, _ = spearmanr(theta_a, theta_b)
    return float(rho)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=5000)
    ap.add_argument("--J", type=int, default=5, help="Respondent panel size")
    ap.add_argument("--n-items", type=int, default=656, help="Total items per replication")
    ap.add_argument("--n-per-subset", type=int, default=208, help="Items per FST subset (208 ~ I-II/V-VI)")
    ap.add_argument("--sigma-theta", type=float, default=0.4, help="True ability sd (matches observed)")
    ap.add_argument("--sigma-b", type=float, default=2.0, help="True difficulty sd (matches observed)")
    ap.add_argument("--cutoff", type=float, default=0.9, help="configural-invariance cutoff")
    ap.add_argument("--observed-rho", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/power_simulation.json"))
    ap.add_argument("--seed", type=int, default=0xC1F1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rhos = np.array([
        one_null(args.J, args.n_items, args.n_per_subset,
                 args.sigma_theta, args.sigma_b, rng)
        for _ in range(args.n_iter)
    ])

    p_above_cutoff = float((rhos >= args.cutoff).mean())
    p_above_observed = float((rhos >= args.observed_rho).mean())
    quantiles = {f"q{int(q*100)}": float(np.quantile(rhos, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}

    summary = {
        "n_iter": int(args.n_iter),
        "J": int(args.J),
        "n_items": int(args.n_items),
        "n_per_subset": int(args.n_per_subset),
        "sigma_theta_true": float(args.sigma_theta),
        "sigma_b_true": float(args.sigma_b),
        "rho_quantiles": quantiles,
        "rho_mean": float(rhos.mean()),
        "rho_sd": float(rhos.std()),
        "pap_cutoff": float(args.cutoff),
        "p_rho_above_cutoff": p_above_cutoff,
        "observed_rho": float(args.observed_rho),
        "p_rho_above_observed": p_above_observed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print()
    print(f"Under H0 (no construct-stability signal, J={args.J}, item sd matching observed),")
    print(f"  P(rho >= {args.cutoff}) = {p_above_cutoff:.3f}")
    print(f"  P(rho >= {args.observed_rho}) = {p_above_observed:.3f}")
    print(f"  Empirical rho ({args.observed_rho}) is in the top "
          f"{p_above_observed*100:.1f}% of the null distribution.")


if __name__ == "__main__":
    main()
