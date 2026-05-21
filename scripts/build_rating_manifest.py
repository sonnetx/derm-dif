"""Build the rating-ui sample.json manifest and copy item images into
rating-ui/images/ for the clinician-validation study.

Reads artifacts/clinician_sample/sample_with_truth.csv (from
scripts/sample_clinician_items.py), copies each item's image into
rating-ui/images/<item_id>, and writes rating-ui/sample.json as the
rater-facing manifest (blinded -- no ground truth, no FST, no IRT difficulty).

  python scripts/build_rating_manifest.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample-csv",
        type=Path,
        default=Path("artifacts/clinician_sample/sample_with_truth.csv"),
        help="Output of scripts/sample_clinician_items.py.",
    )
    ap.add_argument("--ui-dir", type=Path, default=Path("rating-ui"))
    args = ap.parse_args()

    if not args.sample_csv.exists():
        raise SystemExit(
            f"{args.sample_csv} not found. Run scripts/sample_clinician_items.py first."
        )

    images_dir = args.ui_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.sample_csv)
    if "image_path" not in df.columns or "item_id" not in df.columns:
        raise SystemExit(
            "sample_with_truth.csv must have item_id and image_path columns."
        )

    n_copied = 0
    items_for_manifest = []
    for _, row in df.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            print(f"WARNING: source image missing for {row['item_id']}: {src}")
            continue
        dest = images_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            n_copied += 1
        items_for_manifest.append({
            "item_id": str(row["item_id"]),
            "image_path": f"images/{src.name}",
        })

    manifest = {
        "_comment": (
            "Rater-facing item manifest. The image_path values are relative to "
            "the rating-ui directory. Truth labels, FST groups, lesion categories, "
            "and IRT difficulties are deliberately omitted so raters are blind."
        ),
        "items": items_for_manifest,
    }
    (args.ui_dir / "sample.json").write_text(json.dumps(manifest, indent=2))

    print(f"Copied {n_copied} new images into {images_dir}")
    print(f"Wrote manifest of {len(items_for_manifest)} items to {args.ui_dir / 'sample.json'}")
    print(
        f"Total bytes in {images_dir}: "
        f"{sum(p.stat().st_size for p in images_dir.iterdir() if p.is_file())} "
        "(Vercel free-tier limit is 1 GB per project)."
    )


if __name__ == "__main__":
    main()
