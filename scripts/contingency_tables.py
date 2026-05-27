"""Lesion-category x FST and malignant x FST contingency tables.

These come from DDI metadata only (no model responses required), and they
empirically demonstrate the entanglement structure that motivates the LLTM
analysis: lesion categories are not balanced across FST strata, so a per-FST
mean accuracy compares partially non-overlapping mixtures of lesions.

Emits printed tables, CSVs, and LaTeX-ready tables for the paper.

  python scripts/contingency_tables.py --ddi-root /path/to/ddi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from derm_dif.data.ddi import load_ddi


def _crosstab_with_totals(s_row: pd.Series, s_col: pd.Series) -> pd.DataFrame:
    ct = pd.crosstab(s_row, s_col, margins=True, margins_name="Total")
    return ct


def _latex_crosstab(
    ct: pd.DataFrame, caption: str, label: str, row_name: str
) -> str:
    cols = list(ct.columns)
    ncols = len(cols)
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\begin{tabular}{l" + "r" * ncols + "}")
    out.append("\\toprule")
    out.append(row_name + " & " + " & ".join(str(c) for c in cols) + " \\\\")
    out.append("\\midrule")
    for idx, row in ct.iterrows():
        label_cell = str(idx).replace("_", r"\_")
        cells = " & ".join(str(int(v)) for v in row)
        if idx == "Total":
            out.append("\\midrule")
        out.append(f"{label_cell} & {cells} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/contingency"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = load_ddi(args.ddi_root)
    df = pd.DataFrame(
        {
            "item_id": [it.item_id for it in items],
            "fst_group": [it.fst_group for it in items],
            "lesion_category": [it.lesion_category for it in items],
            "malignant": [it.malignant for it in items],
        }
    )
    fst_order = ["I-II", "III-IV", "V-VI"]
    df["fst_group"] = pd.Categorical(df["fst_group"], categories=fst_order, ordered=True)

    lesion_x_fst = _crosstab_with_totals(df["lesion_category"], df["fst_group"])
    print("=== Lesion category x FST group ===")
    print(lesion_x_fst.to_string())
    lesion_x_fst.to_csv(args.out_dir / "lesion_x_fst.csv")
    (args.out_dir / "lesion_x_fst.tex").write_text(
        _latex_crosstab(
            lesion_x_fst,
            caption=(
                "Lesion category $\\times$ Fitzpatrick Skin Type contingency table on the "
                "full DDI release. The non-uniform marginals across rows are the structural "
                "entanglement that motivates the LLTM decomposition (\\S\\ref{sec:lltm}): "
                "a per-FST stratified accuracy compares partially non-overlapping mixtures "
                "of lesion categories, not a paired comparison on a common item pool."
            ),
            label="tab:contingency-lesion-fst",
            row_name="Lesion category",
        )
    )
    print(f"  -> {args.out_dir / 'lesion_x_fst.tex'}")
    print()

    mal_x_fst = _crosstab_with_totals(df["malignant"], df["fst_group"])
    print("=== Malignant x FST group ===")
    print(mal_x_fst.to_string())
    mal_x_fst.to_csv(args.out_dir / "malignant_x_fst.csv")
    (args.out_dir / "malignant_x_fst.tex").write_text(
        _latex_crosstab(
            mal_x_fst,
            caption=(
                "Malignancy $\\times$ FST group contingency table. The malignancy base rate "
                "differs across FST strata in DDI; this prevalence shift is a candidate "
                "non-capability explanation for the FST-stratified accuracy gap reported "
                "in Section~\\ref{sec:prelim_results}."
            ),
            label="tab:contingency-malignant-fst",
            row_name="Malignant",
        )
    )
    print(f"  -> {args.out_dir / 'malignant_x_fst.tex'}")
    print()

    print("=== Malignancy base rate by FST group ===")
    base_rate = df.groupby("fst_group", observed=True)["malignant"].mean()
    print(base_rate.to_string())


if __name__ == "__main__":
    main()
