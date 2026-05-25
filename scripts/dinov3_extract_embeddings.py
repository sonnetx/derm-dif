"""Extract DINOv3 CLS-token embeddings for all DDI items and save to disk.

Run on a GPU node:
  python scripts/dinov3_extract_embeddings.py --ddi-root /path/to/ddi \
      --out artifacts/dinov3_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from derm_dif.data.ddi import load_ddi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--model-id", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--out", type=Path, default=Path("artifacts/dinov3_embeddings.npz"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.eval().to(device)

    items = load_ddi(args.ddi_root)
    print(f"DDI items: {len(items)}")

    item_ids = [it.item_id for it in items]
    image_paths = [it.image_path for it in items]

    embeddings = []
    for batch_start in range(0, len(items), args.batch_size):
        batch_paths = image_paths[batch_start : batch_start + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token from last hidden state
        cls = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        embeddings.append(cls)
        done = min(batch_start + args.batch_size, len(items))
        print(f"  {done}/{len(items)}")

    embeddings = np.concatenate(embeddings, axis=0)  # (N, D)
    print(f"Embedding shape: {embeddings.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, embeddings=embeddings, item_ids=np.array(item_ids))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
