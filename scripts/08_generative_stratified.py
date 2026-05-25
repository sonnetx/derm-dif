"""Per-FST accuracy restricted to non-degenerate generative respondents.
Run:
  python scripts/08_generative_stratified.py --ddi-root <DDI_ROOT>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.parsing import parse_primary

# Models included in the restricted panel.  Update if the model list changes.
GENERATIVE_FULL_SIZE = {
    "openai/gpt-4o-2024-11-20",
    "anthropic/claude-sonnet-4-5",
}


def load_responses(path: Path, refusal_markers: list[str]) -> pd.DataFrame:
    rows = []
    seen: set[tuple[str, str]] = set()
    with path.open() as f:
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
            rows.append(
                {
                    "model_id": d["model_id"],
                    "item_id": d["item_id"],
                    "label": parsed.label,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument(
        "--responses", type=Path, default=Path("artifacts/responses.jsonl")
    )
    ap.add_argument(
        "--protocol", type=Path, default=Path("config/protocol.yaml")
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/generative_stratified.json"))
    args = ap.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("parsing", {}).get("refusal_markers", [])

    items = load_ddi(args.ddi_root)
    meta = pd.DataFrame(
        {
            "item_id": [it.item_id for it in items],
            "fst_group": [it.fst_group for it in items],
            "malignant": [it.malignant for it in items],
            "lesion_category": [it.lesion_category for it in items],
        }
    )

    df = load_responses(args.responses, refusal_markers)
    df = df[df["model_id"].isin(GENERATIVE_FULL_SIZE)].copy()
    df = df.merge(meta, on="item_id", how="left")

    # Correct = prediction matches ground truth label.
    df["predicted_malignant"] = df["label"] == "malignant"
    df["correct"] = df["predicted_malignant"] == df["malignant"]

    fst_groups = ["I-II", "III-IV", "V-VI"]
    results: dict = {"per_model": {}, "pooled": {}}

    # Per-model breakdown.
    for model_id in sorted(df["model_id"].unique()):
        sub = df[df["model_id"] == model_id]
        results["per_model"][model_id] = {}
        for g in fst_groups:
            g_df = sub[sub["fst_group"] == g]
            if len(g_df) == 0:
                continue
            # Exclude refusals from accuracy denominator.
            answered = g_df[g_df["label"].isin(["benign", "malignant"])]
            acc = float(answered["correct"].mean()) if len(answered) > 0 else float("nan")
            results["per_model"][model_id][g] = {
                "accuracy": acc,
                "n_answered": int(len(answered)),
                "n_refused": int((g_df["label"] == "refusal").sum()),
                "n_total": int(len(g_df)),
            }
        # Gap: V-VI minus I-II.
        acc_vvi = results["per_model"][model_id].get("V-VI", {}).get("accuracy", float("nan"))
        acc_iii = results["per_model"][model_id].get("I-II", {}).get("accuracy", float("nan"))
        results["per_model"][model_id]["gap_vvi_minus_iii"] = acc_vvi - acc_iii

    # Pooled across generative models (macro-average over models, then groups).
    results["pooled"] = {}
    for g in fst_groups:
        accs = []
        for model_id in sorted(df["model_id"].unique()):
            v = results["per_model"][model_id].get(g, {}).get("accuracy", float("nan"))
            if not np.isnan(v):
                accs.append(v)
        results["pooled"][g] = float(np.mean(accs)) if accs else float("nan")
    results["pooled"]["gap_vvi_minus_iii"] = (
        results["pooled"].get("V-VI", float("nan"))
        - results["pooled"].get("I-II", float("nan"))
    )

    args.out.write_text(json.dumps(results, indent=2))

    print(f"{'model':<45}  {'I-II':>6}  {'III-IV':>7}  {'V-VI':>6}  {'gap':>6}")
    for model_id in sorted(results["per_model"]):
        r = results["per_model"][model_id]
        print(
            f"{model_id:<45}  "
            f"{r.get('I-II', {}).get('accuracy', float('nan')):>6.3f}  "
            f"{r.get('III-IV', {}).get('accuracy', float('nan')):>7.3f}  "
            f"{r.get('V-VI', {}).get('accuracy', float('nan')):>6.3f}  "
            f"{r.get('gap_vvi_minus_iii', float('nan')):>+6.3f}"
        )
    p = results["pooled"]
    print(
        f"{'[pooled generative]':<45}  "
        f"{p.get('I-II', float('nan')):>6.3f}  "
        f"{p.get('III-IV', float('nan')):>7.3f}  "
        f"{p.get('V-VI', float('nan')):>6.3f}  "
        f"{p.get('gap_vvi_minus_iii', float('nan')):>+6.3f}"
    )


if __name__ == "__main__":
    main()
