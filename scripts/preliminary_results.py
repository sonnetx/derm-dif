"""Compute preliminary descriptive statistics for the closed-API respondent
subpanel and emit LaTeX-ready tables.

Used to populate the preliminary-results section of the paper while the full
respondent panel is still being collected. Reports per-model accuracy
stratified by FST group and by lesion category, inter-model agreement
(Cohen's kappa and raw concordance), and parse/refusal rates.

Run after scripts/02_query_models.py:
  python scripts/preliminary_results.py --ddi-root /path/to/ddi
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.parsing import parse_primary, to_correctness


def latest_success_by_key(responses_path: Path) -> dict[tuple[str, str], dict]:
    """Walk the JSONL and keep, per (model_id, item_id), the most recent successful row."""
    out: dict[tuple[str, str], dict] = {}
    with responses_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("error"):
                continue
            key = (d["model_id"], d["item_id"])
            prior = out.get(key)
            if prior is None or d["timestamp"] >= prior["timestamp"]:
                out[key] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--protocol", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/prelim"))
    ap.add_argument("--min-coverage", type=int, default=600,
                    help="Skip models with fewer than this many successful unique items.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("refusal_markers", [])

    items = load_ddi(args.ddi_root)
    item_by_id = {it.item_id: it for it in items}
    truth = {it.item_id: it.malignant for it in items}
    fst = {it.item_id: it.fst_group for it in items}
    lesion = {it.item_id: it.lesion_category for it in items}

    success_rows = latest_success_by_key(args.responses)

    # Bucket parsed predictions by model
    by_model: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for (model_id, item_id), row in success_rows.items():
        parsed = parse_primary(row["raw_text"], refusal_markers)
        by_model[model_id][item_id] = {
            "label": parsed.label,
            "correct": to_correctness(parsed, truth.get(item_id, False)),
        }

    # Filter to high-coverage models
    kept_models = [m for m, items_dict in by_model.items()
                   if sum(1 for v in items_dict.values() if v["correct"] is not None) >= args.min_coverage]
    kept_models.sort()
    print(f"Models with >= {args.min_coverage} parseable answers: {kept_models}")
    print()

    # --- Per-model accuracy, overall and by FST group ---
    rows = []
    for m in kept_models:
        per_item = by_model[m]
        records = []
        for item_id, v in per_item.items():
            if v["correct"] is None:
                continue
            records.append({
                "item_id": item_id,
                "correct": v["correct"],
                "label": v["label"],
                "fst_group": fst[item_id],
                "lesion": lesion[item_id],
                "malignant": truth[item_id],
            })
        df = pd.DataFrame(records)
        overall = df["correct"].mean()
        by_fst = df.groupby("fst_group")["correct"].agg(["mean", "count"])
        n_refusal = sum(1 for v in per_item.values() if v["label"] == "refusal")
        n_unparseable = sum(1 for v in per_item.values() if v["label"] == "unparseable")
        rows.append({
            "model": m,
            "n_parsed": len(records),
            "overall_acc": overall,
            "fst_i_ii_acc": by_fst.loc["I-II", "mean"] if "I-II" in by_fst.index else float("nan"),
            "fst_i_ii_n":   int(by_fst.loc["I-II", "count"]) if "I-II" in by_fst.index else 0,
            "fst_iii_iv_acc": by_fst.loc["III-IV", "mean"] if "III-IV" in by_fst.index else float("nan"),
            "fst_iii_iv_n":   int(by_fst.loc["III-IV", "count"]) if "III-IV" in by_fst.index else 0,
            "fst_v_vi_acc": by_fst.loc["V-VI", "mean"] if "V-VI" in by_fst.index else float("nan"),
            "fst_v_vi_n":   int(by_fst.loc["V-VI", "count"]) if "V-VI" in by_fst.index else 0,
            "gap_v_vi_vs_i_ii": (by_fst.loc["V-VI", "mean"] if "V-VI" in by_fst.index else float("nan"))
                              - (by_fst.loc["I-II", "mean"] if "I-II" in by_fst.index else float("nan")),
            "refusals": n_refusal,
            "unparseable": n_unparseable,
        })

    summary = pd.DataFrame(rows)
    print("=== Per-model accuracy, stratified by FST ===")
    print(summary.to_string(index=False, float_format="%.3f"))
    print()

    # --- Inter-model agreement (only among models with common parseable items) ---
    print("=== Inter-model agreement (raw concordance and Cohen's kappa) ===")
    print(f"{'pair':<70}  {'n':>5}  {'agree':>6}  {'kappa':>6}")
    for i, m1 in enumerate(kept_models):
        for m2 in kept_models[i + 1:]:
            common = set(by_model[m1].keys()) & set(by_model[m2].keys())
            pairs = []
            for item_id in common:
                l1 = by_model[m1][item_id]["label"]
                l2 = by_model[m2][item_id]["label"]
                if l1 in ("benign", "malignant") and l2 in ("benign", "malignant"):
                    pairs.append((l1 == "malignant", l2 == "malignant"))
            if not pairs:
                continue
            y1 = np.array([p[0] for p in pairs])
            y2 = np.array([p[1] for p in pairs])
            agree = (y1 == y2).mean()
            # Cohen's kappa
            p_o = agree
            p1 = y1.mean()
            p2 = y2.mean()
            p_e = p1 * p2 + (1 - p1) * (1 - p2)
            kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan")
            print(f"{m1[:32]} vs {m2[:32]}".ljust(70) +
                  f"  {len(pairs):>5}  {agree:>6.3f}  {kappa:>6.3f}")

    # --- LaTeX-ready FST-stratified accuracy table ---
    tex_path = args.out_dir / "fst_stratified_accuracy.tex"
    with tex_path.open("w") as f:
        f.write("% Auto-generated by scripts/preliminary_results.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline FST-stratified accuracy (closed-API respondent subpanel). "
                "Per-model proportion of items answered correctly under the binary "
                "benign-vs-malignant protocol, stratified by Fitzpatrick Skin Type group. "
                "The right-most column gives the V--VI minus I--II accuracy gap that "
                "conventional fairness audits would report; the pre-registered Rasch-difficulty "
                "analysis (\\S\\ref{sec:aggregate}) is what tests whether that gap reflects "
                "differential capability or differential item difficulty.}\n")
        f.write("\\label{tab:prelim-fst-accuracy}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & $N$ & FST I--II & FST III--IV & FST V--VI & Gap (V--VI $-$ I--II) \\\\\n")
        f.write("\\midrule\n")
        for _, r in summary.iterrows():
            f.write(
                f"{r['model'].replace('_', r'\\_')} & "
                f"{int(r['n_parsed'])} & "
                f"{r['fst_i_ii_acc']:.3f} ({int(r['fst_i_ii_n'])}) & "
                f"{r['fst_iii_iv_acc']:.3f} ({int(r['fst_iii_iv_n'])}) & "
                f"{r['fst_v_vi_acc']:.3f} ({int(r['fst_v_vi_n'])}) & "
                f"{r['gap_v_vi_vs_i_ii']:+.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"\nWrote {tex_path}")
    summary.to_csv(args.out_dir / "fst_stratified_accuracy.csv", index=False)
    print(f"Wrote {args.out_dir / 'fst_stratified_accuracy.csv'}")


if __name__ == "__main__":
    main()
