"""Sanity-check responses.jsonl: parse each row with the project parser and
report the label distribution per model, plus a sample raw response per
parsed category.

Run after scripts/02_query_models.py:
  python scripts/check_responses.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml

from derm_dif.parsing import parse_primary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--protocol", type=Path, default=Path("config/protocol.yaml"))
    args = ap.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("refusal_markers", [])

    # Per-model: label counts and one example per label
    label_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: dict[tuple[str, str], str] = {}
    seen_keys: set[tuple[str, str]] = set()

    with args.responses.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("error"):
                continue
            model = d["model_id"]
            key = (model, d["item_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            parsed = parse_primary(d["raw_text"], refusal_markers)
            label_counts[model][parsed.label] += 1
            ex_key = (model, parsed.label)
            if ex_key not in examples:
                examples[ex_key] = d["raw_text"][:300]

    print(f"{'model_id':<45}  {'benign':>7}  {'malig':>6}  {'refuse':>7}  {'unparse':>8}")
    for model in sorted(label_counts):
        c = label_counts[model]
        print(
            f"{model:<45}  {c.get('benign', 0):>7}  {c.get('malignant', 0):>6}  "
            f"{c.get('refusal', 0):>7}  {c.get('unparseable', 0):>8}"
        )

    print("\n--- one sample raw response per (model, parsed_label) ---")
    for model in sorted(label_counts):
        for label in ("benign", "malignant", "refusal", "unparseable"):
            ex = examples.get((model, label))
            if ex is not None:
                print(f"\n[{model} -> {label}]")
                print("  " + ex.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
