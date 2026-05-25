"""K-fold cross-fitting of a logistic regression probe on DINOv3 embeddings.

Usage:
  python scripts/dinov3_crossfit_probe.py \
      --ddi-root /path/to/ddi \
      --embeddings artifacts/dinov3_embeddings.npz \
      --out artifacts/responses.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from derm_dif.data.ddi import load_ddi

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, default=Path("artifacts/dinov3_embeddings.npz"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    embeddings = data["embeddings"]          # (N, D)
    emb_item_ids = list(data["item_ids"])    # N strings

    items = load_ddi(args.ddi_root)
    item_id_to_item = {it.item_id: it for it in items}

    # Align embeddings to item order; skip any items not in embeddings
    aligned_ids, X, y = [], [], []
    for iid, emb in zip(emb_item_ids, embeddings):
        if iid not in item_id_to_item:
            continue
        it = item_id_to_item[iid]
        aligned_ids.append(iid)
        X.append(emb)
        y.append(int(it.malignant))

    X = np.array(X)
    y = np.array(y)

    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.sum() > 0:
        print(f"Warning: {nan_rows.sum()} items have NaN embeddings (float16 overflow), replacing with zeros")
        X[nan_rows] = 0.0

    print(f"Items: {len(aligned_ids)}, malignant rate: {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    predictions = np.full(len(y), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_val)
        predictions[val_idx] = preds

        acc = (preds == y[val_idx]).mean()
        print(f"  Fold {fold+1}/{args.k_folds}: val_acc={acc:.3f} (n={len(val_idx)})")

    assert (predictions >= 0).all(), "Some items have no prediction"

    overall_acc = (predictions == y).mean()
    print(f"Cross-fit accuracy: {overall_acc:.3f}")

    # Write to responses.jsonl (append mode — respects already_done dedup)
    existing_done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("model_id") == MODEL_ID and d.get("error") is None:
                existing_done.add(d["item_id"])

    written = 0
    with args.out.open("a") as f:
        for iid, pred in zip(aligned_ids, predictions):
            if iid in existing_done:
                continue
            label = "malignant" if pred == 1 else "benign"
            row = {
                "model_id": MODEL_ID,
                "item_id": iid,
                "raw_text": label,
                "error": None,
            }
            f.write(json.dumps(row) + "\n")
            written += 1

    print(f"Written {written} new responses to {args.out}")


if __name__ == "__main__":
    main()
