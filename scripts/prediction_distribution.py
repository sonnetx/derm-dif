"""Per-model prediction-label distribution stratified by FST group.

Produces a stacked bar chart that makes per-model class-prediction patterns
(and degenerate "always benign" behavior like CLIP's) visually obvious.

  python scripts/prediction_distribution.py --ddi-root /path/to/ddi
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.parsing import parse_primary


def latest_success_by_key(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("error"):
                continue
            key = (d["model_id"], d["item_id"])
            prior = out.get(key)
            if prior is None or d["timestamp"] >= prior["timestamp"]:
                out[key] = d
    return out


def short_name(model_id: str) -> str:
    """Shorten HF-style ids for plot labels."""
    last = model_id.rsplit("/", 1)[-1]
    if "BiomedCLIP" in last:
        return "BiomedCLIP"
    if "clip-vit" in last.lower():
        return "CLIP-ViT-L/14"
    if "siglip" in last.lower():
        return "SigLIP-L/16"
    if "gpt-4o" in last.lower():
        return "GPT-4o"
    if "claude" in last.lower():
        return "Claude Sonnet 4.5"
    if "gemini" in last.lower():
        return "Gemini Flash Lite"
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--protocol", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/prelim"))
    ap.add_argument("--min-coverage", type=int, default=600)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    protocol = yaml.safe_load(args.protocol.read_text())["primary_protocol"]
    refusal_markers = protocol.get("refusal_markers", [])

    items = load_ddi(args.ddi_root)
    fst_of = {it.item_id: it.fst_group for it in items}
    truth_of = {it.item_id: it.malignant for it in items}

    success = latest_success_by_key(args.responses)

    # Bucket: model -> fst_group -> label -> count
    counts: dict = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    coverage = collections.Counter()
    for (model, item), row in success.items():
        parsed = parse_primary(row["raw_text"], refusal_markers)
        counts[model][fst_of[item]][parsed.label] += 1
        coverage[model] += 1

    kept = sorted([m for m in counts if coverage[m] >= args.min_coverage])
    if not kept:
        print("No models meet min coverage; nothing to plot.")
        return

    fst_order = ["I-II", "III-IV", "V-VI"]
    label_order = ["benign", "malignant", "refusal", "unparseable"]
    label_colors = {
        "benign": "#4C72B0",
        "malignant": "#DD8452",
        "refusal": "#8C8C8C",
        "unparseable": "#000000",
    }

    n_models = len(kept)
    n_fst = len(fst_order)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 4.2), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Compute the per-FST truth-malignancy rate, plot as a horizontal line
    truth_rate = {g: 0 for g in fst_order}
    truth_n = {g: 0 for g in fst_order}
    for it in items:
        truth_rate[it.fst_group] += int(it.malignant)
        truth_n[it.fst_group] += 1
    truth_share = {g: (truth_rate[g] / truth_n[g] if truth_n[g] else 0.0) for g in fst_order}

    for ax, model in zip(axes, kept):
        # Build proportions per FST stratum
        props = np.zeros((n_fst, len(label_order)))
        for i, g in enumerate(fst_order):
            total = sum(counts[model][g].values())
            if total == 0:
                continue
            for j, lab in enumerate(label_order):
                props[i, j] = counts[model][g].get(lab, 0) / total

        x = np.arange(n_fst)
        bottom = np.zeros(n_fst)
        for j, lab in enumerate(label_order):
            heights = props[:, j]
            if heights.sum() == 0:
                continue
            ax.bar(
                x,
                heights,
                bottom=bottom,
                color=label_colors[lab],
                edgecolor="white",
                linewidth=0.6,
                label=lab if model == kept[0] else None,
            )
            bottom += heights

        # Overlay the true-malignancy share per FST as a small marker
        for i, g in enumerate(fst_order):
            ax.scatter(
                x[i],
                truth_share[g],
                marker="_",
                s=180,
                color="black",
                linewidth=2.0,
                zorder=10,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(fst_order)
        ax.set_xlabel("FST group")
        ax.set_ylim(0, 1)
        ax.set_title(short_name(model), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Proportion of parsed responses")

    # Single legend at top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=label_colors[lab]) for lab in ("benign", "malignant")
    ]
    handles.append(
        plt.Line2D([0], [0], marker="_", color="black", linewidth=2, linestyle="None", markersize=14)
    )
    fig.legend(
        handles,
        ["predicted: benign", "predicted: malignant", "true malignancy share"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Per-model prediction distribution by FST group "
        "(black tick = true malignancy share in that stratum)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    png_path = args.out_dir / "prediction_distribution.png"
    pdf_path = args.out_dir / "prediction_distribution.pdf"
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")

    # Also print a text summary
    print("\n=== Prediction shares per model x FST ===")
    for model in kept:
        print(f"\n{short_name(model)} ({model}):")
        for g in fst_order:
            total = sum(counts[model][g].values())
            if total == 0:
                continue
            mal = counts[model][g].get("malignant", 0)
            print(f"  FST {g}: n={total}, predicted malignant={mal} ({mal/total:.2%}), true malignancy={truth_share[g]:.2%}")


if __name__ == "__main__":
    main()
