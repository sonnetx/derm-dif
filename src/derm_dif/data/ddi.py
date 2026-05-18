"""DDI benchmark loader.

The DDI release ships as a CSV of metadata plus a directory of JPEGs. We expose a
typed record per item with the fields needed downstream by the IRT and DIF code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd

FSTGroup = Literal["I-II", "III-IV", "V-VI"]


@dataclass(frozen=True)
class DDIItem:
    item_id: str
    image_path: Path
    fst_group: FSTGroup
    fst_raw: int  # 12 / 34 / 56 (paired-group encoding, as shipped by DDI)
    lesion_category: str
    malignant: bool


def _fst_group(fst_raw: int) -> FSTGroup:
    if fst_raw == 12:
        return "I-II"
    if fst_raw == 34:
        return "III-IV"
    if fst_raw == 56:
        return "V-VI"
    raise ValueError(f"unexpected FST value: {fst_raw}")


def load_ddi(root: Path) -> list[DDIItem]:
    """Load DDI items from `root`, which must contain `ddi_metadata.csv` and the image files (e.g., `000001.png`) directly at the top level.

    The metadata schema is the one shipped by Daneshjou et al. (2022). Column names
    are normalized here so downstream code is insulated from minor release changes.
    """
    meta = pd.read_csv(root / "ddi_metadata.csv")
    rename = {
        "DDI_file": "item_id",
        "skin_tone": "fst_raw",
        "disease": "lesion_category",
        "malignant": "malignant",
    }
    meta = meta.rename(columns=rename)
    items: list[DDIItem] = []
    for row in meta.itertuples(index=False):
        items.append(
            DDIItem(
                item_id=str(row.item_id),
                image_path=root / f"{row.item_id}",
                fst_group=_fst_group(int(row.fst_raw)),
                fst_raw=int(row.fst_raw),
                lesion_category=str(row.lesion_category),
                malignant=bool(row.malignant),
            )
        )
    return items


def fst_group_index(items: list[DDIItem]) -> dict[FSTGroup, list[int]]:
    """Return positional indices of items in each FST group, in input order."""
    out: dict[FSTGroup, list[int]] = {"I-II": [], "III-IV": [], "V-VI": []}
    for i, it in enumerate(items):
        out[it.fst_group].append(i)
    return out


def iter_image_paths(items: list[DDIItem]) -> Iterator[Path]:
    for it in items:
        yield it.image_path
