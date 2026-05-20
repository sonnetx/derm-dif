"""Query every (model, item) under the primary protocol and append to a JSONL log.

Resumable: skips (model, item) pairs already present in the output file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from derm_dif.data.ddi import load_ddi
from derm_dif.query import append_jsonl, load_model_specs, query_one, set_zero_shot_config


def already_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("error") is None:
                done.add((d["model_id"], d["item_id"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi-root", type=Path, required=True)
    ap.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    ap.add_argument("--protocol-config", type=Path, default=Path("config/protocol.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/responses.jsonl"))
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument(
        "--source",
        default=None,
        help="Comma-separated list of model `source` values to include "
        "(e.g., api-openai,api-anthropic,api-google). Default: all sources.",
    )
    ap.add_argument(
        "--model-id",
        default=None,
        help="Restrict to a single model.id from models.yaml (e.g., "
        "llava-hf/llava-1.5-13b-hf). Useful when serving one open-weight VLM "
        "at a time via vLLM. Applies after --source filtering.",
    )
    args = ap.parse_args()

    items = load_ddi(args.ddi_root)
    specs = load_model_specs(args.models_config)
    full_protocol = yaml.safe_load(args.protocol_config.read_text())
    protocol = full_protocol["primary_protocol"]
    if "zero_shot_protocol" in full_protocol:
        set_zero_shot_config(full_protocol["zero_shot_protocol"])

    if not args.include_optional:
        specs = [s for s in specs if not s.optional]

    if args.source:
        allowed = {s.strip() for s in args.source.split(",")}
        specs = [s for s in specs if s.source in allowed]
    if args.model_id:
        specs = [s for s in specs if s.id == args.model_id]
        if not specs:
            raise SystemExit(f"--model-id {args.model_id!r} did not match any model in the config")

    done = already_done(args.out)
    total = len(specs) * len(items)
    progress = 0

    for spec in specs:
        for it in items:
            progress += 1
            key = (spec.id, it.item_id)
            if key in done:
                continue
            result = query_one(
                spec,
                it.image_path,
                it.item_id,
                protocol["prompt_template"],
                protocol["decoding"],
            )
            append_jsonl(args.out, result)
            if progress % 50 == 0:
                print(f"  {progress}/{total}")


if __name__ == "__main__":
    main()
