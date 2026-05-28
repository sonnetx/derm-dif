"""Per-model FST accuracy + prediction-distribution for all 9 respondents.

Run:
  python scripts/all_model_fst_accuracy.py --ddi-root /ddi
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

MODELS_J9 = [
    "openai/gpt-4o-2024-11-20",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5-20251001",
    "google/gemini-3.1-flash-lite",
    "openai/clip-vit-large-patch14",
    "google/siglip-large-patch16-384",
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "Qwen/Qwen2-VL-7B-Instruct",
    "llava-hf/llava-v1.6-mistral-7b-hf",
]

SHORT = {
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "anthropic/claude-sonnet-4-5": "Claude Sonnet 4.5",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "google/gemini-3.1-flash-lite": "Gemini-3.1-flash-lite",
    "openai/clip-vit-large-patch14": "CLIP ViT-L/14",
    "google/siglip-large-patch16-384": "SigLIP-L/16",
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "BiomedCLIP",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-v1.6-Mistral-7B",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--protocol", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/all_model_fst_accuracy.json"))
    args = ap.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("parsing", {}).get("refusal_markers", [])

    items = load_ddi(args.ddi_root)
    meta = pd.DataFrame({
        "item_id": [it.item_id for it in items],
        "fst_group": [it.fst_group for it in items],
        "malignant": [it.malignant for it in items],
    })
    true_prev = {g: meta[meta.fst_group == g]["malignant"].mean()
                 for g in ["I-II", "III-IV", "V-VI"]}

    rows = []
    seen: set = set()
    with args.responses.open() as f:
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
            rows.append({"model_id": d["model_id"], "item_id": d["item_id"], "label": parsed.label})

    df = pd.DataFrame(rows)
    df = df[df["model_id"].isin(MODELS_J9)]
    df = df.merge(meta, on="item_id", how="left")
    df["predicted_malignant"] = df["label"] == "malignant"
    df["correct"] = df["predicted_malignant"] == df["malignant"]

    fst_groups = ["I-II", "III-IV", "V-VI"]
    results = {}

    for model_id in MODELS_J9:
        sub = df[df["model_id"] == model_id]
        answered = sub[sub["label"].isin(["benign", "malignant"])]
        row = {"model": SHORT[model_id]}
        for g in fst_groups:
            g_ans = answered[answered["fst_group"] == g]
            acc = float(g_ans["correct"].mean()) if len(g_ans) > 0 else float("nan")
            pred_mal = float(g_ans["predicted_malignant"].mean()) if len(g_ans) > 0 else float("nan")
            row[f"acc_{g}"] = round(acc, 3)
            row[f"pred_mal_{g}"] = round(pred_mal, 3)
            row[f"n_{g}"] = len(g_ans)
        row["gap"] = round(row["acc_V-VI"] - row["acc_I-II"], 3)
        results[model_id] = row

    # Print accuracy table
    print(f"\n{'Model':<30}  {'I-II':>6}  {'III-IV':>7}  {'V-VI':>6}  {'Gap':>6}  {'N':>5}")
    print("-" * 65)
    for mid in MODELS_J9:
        r = results[mid]
        print(f"{r['model']:<30}  {r['acc_I-II']:>6.3f}  {r['acc_III-IV']:>7.3f}  {r['acc_V-VI']:>6.3f}  {r['gap']:>+6.3f}  {r['n_I-II']:>5}")

    # Print prediction-distribution table
    print(f"\n{'Model':<30}  {'pred_mal I-II':>13}  {'pred_mal III-IV':>15}  {'pred_mal V-VI':>13}  |  {'true I-II':>9}  {'true III-IV':>11}  {'true V-VI':>9}")
    print("-" * 110)
    for mid in MODELS_J9:
        r = results[mid]
        print(f"{r['model']:<30}  {r['pred_mal_I-II']:>13.3f}  {r['pred_mal_III-IV']:>15.3f}  {r['pred_mal_V-VI']:>13.3f}  |  {true_prev['I-II']:>9.3f}  {true_prev['III-IV']:>11.3f}  {true_prev['V-VI']:>9.3f}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
