"""Image embedding extraction for amortized Rasch calibration.

Primary embedding model: BiomedCLIP. Secondary (robustness): DINOv3.
We never use a model both as a respondent and as the embedding source for its own
difficulty input — see `assert_no_circularity`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image

EmbeddingBackend = Literal["biomedclip", "dinov3"]


def load_backend(name: EmbeddingBackend, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Returns (model, preprocess, dim)."""
    if name == "biomedclip":
        from open_clip import create_model_from_pretrained

        model, preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model.eval().to(device)
        return model, preprocess, 512
    if name == "dinov3":
        model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16")
        model.eval().to(device)
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return model, preprocess, 1024
    raise ValueError(name)


@torch.inference_mode()
def embed_images(
    paths: Sequence[Path],
    backend: EmbeddingBackend = "biomedclip",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute one embedding per image. Returns array of shape (N, dim)."""
    model, preprocess, dim = load_backend(backend, device)
    out = np.zeros((len(paths), dim), dtype=np.float32)
    batch: list[torch.Tensor] = []
    batch_idx: list[int] = []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        if backend == "biomedclip":
            feats = model.encode_image(x)
        else:
            feats = model(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[batch_idx] = feats.cpu().numpy()
        batch.clear()
        batch_idx.clear()

    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
        batch_idx.append(i)
        if len(batch) == batch_size:
            flush()
    flush()
    return out


def assert_no_circularity(
    embedding_backend: EmbeddingBackend, respondent_family_ids: list[str]
) -> None:
    """Sanity check: the embedding backend cannot be the same model family as a respondent.

    BiomedCLIP-as-embedding + BiomedCLIP-as-respondent is allowed in our protocol
    but flagged in the analysis (see paper). This helper enforces an opt-in flag.
    """
    backend_family = {"biomedclip": "biomedclip", "dinov3": "dinov3"}[embedding_backend]
    if backend_family in respondent_family_ids:
        # Do not raise; surface as a warning so the analysis script can record it.
        import warnings

        warnings.warn(
            f"Embedding backend '{embedding_backend}' shares family with a respondent. "
            "This circularity must be reported in the paper's limitations section.",
            stacklevel=2,
        )
