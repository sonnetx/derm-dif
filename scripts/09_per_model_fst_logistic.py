"""Per-model logistic regression: P(correct) ~ lesion_category + malignant + fst_group.

The IRT framework estimates one difficulty parameter per item pooled across all
respondents. If GPT-4o is disproportionately worse on V-VI items, that signal is
averaged out by the contrastive zero-shot models (CLIP, SigLIP) that never
discriminate. This script tests the user's claim directly: for each model
individually, does FST predict accuracy after controlling for lesion category?

Output: artifacts/per_model_fst_logistic.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from derm_dif.data.ddi import load_ddi
from derm_dif.parsing import parse_primary


def load_response_matrix(
    responses_path: Path, refusal_markers: list[str], items: list
) -> pd.DataFrame:
    """Return long-format DataFrame: model_id, item_id, label, correct."""
    item_meta = {
        it.item_id: {
            "fst_group": it.fst_group,
            "lesion_category": it.lesion_category,
            "malignant": it.malignant,
        }
        for it in items
    }
    rows, seen = [], set()
    with responses_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("error"):
                continue
            key = (d["model_id"], d["item_id"])
            if key in seen:
                continue
            seen.add(key)
            parsed = parse_primary(d["raw_text"], refusal_markers)
            if parsed.label not in ("benign", "malignant"):
                continue  # exclude refusals / unparseable from accuracy denominator
            meta = item_meta.get(d["item_id"])
            if meta is None:
                continue
            rows.append(
                {
                    "model_id": d["model_id"],
                    "item_id": d["item_id"],
                    "label": parsed.label,
                    "correct": int(parsed.label == ("malignant" if meta["malignant"] else "benign")),
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def fit_per_model(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 0xDD9,
) -> dict:
    """For each model, fit logistic(correct ~ lesion_category + malignant + fst_group)."""
    rng = np.random.default_rng(seed)
    results = {}

    for model_id in sorted(df["model_id"].unique()):
        sub = df[df["model_id"] == model_id].reset_index(drop=True)
        n = len(sub)
        if n < 20:
            continue

        lc_ref = sub["lesion_category"].value_counts().index[0]
        lc_dummies = pd.get_dummies(sub["lesion_category"], prefix="lc").astype(float)
        lc_dummies = lc_dummies.drop(
            columns=[c for c in lc_dummies.columns if c == f"lc_{lc_ref}"],
            errors="ignore",
        )
        fst_dummies = pd.get_dummies(sub["fst_group"], prefix="fst").astype(float)
        fst_dummies = fst_dummies.drop(columns=["fst_I-II"], errors="ignore")

        X = pd.concat(
            [
                pd.DataFrame({"malignant": sub["malignant"].astype(float)}),
                lc_dummies,
                fst_dummies,
            ],
            axis=1,
        ).values
        y = sub["correct"].values

        if y.mean() in (0.0, 1.0):
            # Degenerate model — all correct or all wrong, skip logistic.
            results[model_id] = {"skipped": "degenerate_accuracy", "n": n}
            continue

        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        try:
            clf.fit(X, y)
        except Exception as e:
            results[model_id] = {"skipped": str(e), "n": n}
            continue

        feature_names = (
            ["malignant"]
            + lc_dummies.columns.tolist()
            + fst_dummies.columns.tolist()
        )
        coef = dict(zip(feature_names, clf.coef_.ravel()))

        boot = {name: [] for name in feature_names}
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            Xb, yb = X[idx], y[idx]
            if yb.mean() in (0.0, 1.0):
                continue
            clf_b = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
            try:
                clf_b.fit(Xb, yb)
            except Exception:
                continue
            for i, name in enumerate(feature_names):
                boot[name].append(clf_b.coef_.ravel()[i])

        ci = {}
        for name in feature_names:
            samples = np.array(boot[name])
            if len(samples) < 50:
                ci[name] = {"ci_low": float("nan"), "ci_high": float("nan")}
            else:
                ci[name] = {
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                }

        fst_keys = [k for k in feature_names if k.startswith("fst_")]
        results[model_id] = {
            "n": n,
            "base_accuracy": float(y.mean()),
            "fst_coefs": {
                k: {"beta": coef[k], **ci[k]} for k in fst_keys
            },
            "all_coefs": {
                k: {"beta": coef[k], **ci[k]} for k in feature_names
            },
        }

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--protocol", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/per_model_fst_logistic.json"))
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("parsing", {}).get("refusal_markers", [])

    items = load_ddi(args.ddi_root)
    df = load_response_matrix(args.responses, refusal_markers, items)

    results = fit_per_model(df, n_bootstrap=args.n_bootstrap)
    args.out.write_text(json.dumps(results, indent=2))

    # Print summary: FST V-VI coefficient per model.
    print(f"\n{'model':<45}  {'n':>5}  {'acc':>5}  {'fst_V-VI beta':>14}  {'95% CI':>22}")
    for model_id in sorted(results):
        r = results[model_id]
        if "skipped" in r:
            print(f"{model_id:<45}  {'skipped: ' + r['skipped']}")
            continue
        fst = r["fst_coefs"].get("fst_V-VI", {})
        beta = fst.get("beta", float("nan"))
        lo = fst.get("ci_low", float("nan"))
        hi = fst.get("ci_high", float("nan"))
        print(
            f"{model_id:<45}  {r['n']:>5}  {r['base_accuracy']:>5.3f}"
            f"  {beta:>+14.3f}  [{lo:>+.3f}, {hi:>+.3f}]"
        )


if __name__ == "__main__":
    main()
