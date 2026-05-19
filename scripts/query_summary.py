"""Print per-model totals, error counts, unique successful items, and a sample
successful response per model from artifacts/responses.jsonl.

Run after script 02 to sanity-check the query log:
  python scripts/query_summary.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    args = ap.parse_args()

    totals: collections.Counter = collections.Counter()
    successes: collections.Counter = collections.Counter()
    errs: collections.Counter = collections.Counter()
    unique_success: dict[str, set[str]] = collections.defaultdict(set)
    err_example: dict[str, str] = {}

    with args.responses.open() as f:
        for line in f:
            d = json.loads(line)
            m = d["model_id"]
            totals[m] += 1
            if d.get("error"):
                errs[m] += 1
                err_example.setdefault(m, d["error"])
            else:
                successes[m] += 1
                unique_success[m].add(d["item_id"])

    print(f"{'model_id':<45}  {'rows':>6}  {'ok':>6}  {'err':>6}  {'uniq_ok_items':>13}")
    for m in sorted(totals):
        print(
            f"{m:<45}  {totals[m]:>6}  {successes[m]:>6}  {errs[m]:>6}  {len(unique_success[m]):>13}"
        )

    if err_example:
        print()
        for m, e in err_example.items():
            print(f"first error for {m}:")
            print(f"  {e[:240]}")

    print("\n--- sample successful response per model ---")
    seen: set[str] = set()
    with args.responses.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("error") is None and d["model_id"] not in seen:
                print(f"{d['model_id']} on {d['item_id']}: {d['raw_text'][:200]!r}")
                seen.add(d["model_id"])


if __name__ == "__main__":
    main()
