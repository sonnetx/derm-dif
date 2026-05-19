"""Model querying. Backend dispatch by `source` field in the model registry.

This module owns the I/O contract: given a (model_spec, image_path, prompt) tuple,
return a raw text response and a small provenance record. Parsing happens downstream.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    modality: str
    source: str
    deterministic_decoding: bool
    version: str
    optional: bool = False
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class QueryResult:
    model_id: str
    item_id: str
    raw_text: str
    elapsed_s: float
    timestamp: float
    error: str | None


def load_model_specs(path: Path) -> list[ModelSpec]:
    cfg = yaml.safe_load(path.read_text())
    specs: list[ModelSpec] = []
    for entry in cfg["models"]:
        specs.append(
            ModelSpec(
                id=entry["id"],
                family=entry["family"],
                modality=entry["modality"],
                source=entry["source"],
                deterministic_decoding=entry["deterministic_decoding"],
                version=entry["version"],
                optional=entry.get("optional", False),
                notes=entry.get("notes", ""),
            )
        )
    return specs


def _b64(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode("ascii")


def query_openai(spec: ModelSpec, image_path: Path, prompt: str, decoding: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    model_name = spec.id.split("/", 1)[1]
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"},
                    },
                ],
            }
        ],
        temperature=decoding["temperature"],
        max_tokens=decoding["max_tokens"],
    )
    return resp.choices[0].message.content or ""


def query_anthropic(spec: ModelSpec, image_path: Path, prompt: str, decoding: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    model_name = spec.id.split("/", 1)[1]
    resp = client.messages.create(
        model=model_name,
        max_tokens=decoding["max_tokens"],
        temperature=decoding["temperature"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _b64(image_path),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def query_google(spec: ModelSpec, image_path: Path, prompt: str, decoding: dict) -> str:
    from google import genai

    client = genai.Client()
    model_name = spec.id.split("/", 1)[1]
    img = client.files.upload(file=str(image_path))
    resp = client.models.generate_content(
        model=model_name,
        contents=[img, prompt],
        config={"temperature": decoding["temperature"], "max_output_tokens": decoding["max_tokens"]},
    )
    return resp.text or ""


def query_huggingface(spec: ModelSpec, image_path: Path, prompt: str, decoding: dict) -> str:
    """Open-weights models served locally. We assume a vLLM endpoint at $VLLM_BASE_URL.

    Concrete vLLM serving is set up per-model in scripts/02_query_models.py; this
    function only constructs the request.
    """
    import os

    import requests

    base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = spec.id
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_path)}"},
                    },
                ],
            }
        ],
        "temperature": decoding["temperature"],
        "max_tokens": decoding["max_tokens"],
    }
    r = requests.post(f"{base}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_BACKENDS = {
    "api-openai": query_openai,
    "api-anthropic": query_anthropic,
    "api-google": query_google,
    "huggingface": query_huggingface,
}


_RATE_LIMIT_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "rate_limit",
    "rate limit",
    "overloaded",
    "quota",
)


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc)
    return any(m.lower() in msg.lower() for m in _RATE_LIMIT_MARKERS)


def _with_backoff(call, max_attempts: int = 8, base_delay: float = 2.0):
    """Retry `call` on rate-limit-style exceptions with exponential backoff
    capped at 60 seconds. Non-rate-limit exceptions raise immediately so
    they're recorded as errors in the JSONL log."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as e:  # noqa: BLE001 -- we re-raise unless rate-limited
            if not _is_rate_limit(e):
                raise
            last_exc = e
            delay = min(60.0, base_delay * (2 ** attempt))
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def query_one(
    spec: ModelSpec, image_path: Path, item_id: str, prompt: str, decoding: dict
) -> QueryResult:
    fn = _BACKENDS.get(spec.source)
    if fn is None:
        raise ValueError(f"unknown backend: {spec.source}")
    t0 = time.time()
    try:
        text = _with_backoff(lambda: fn(spec, image_path, prompt, decoding))
        return QueryResult(
            model_id=spec.id,
            item_id=item_id,
            raw_text=text,
            elapsed_s=time.time() - t0,
            timestamp=t0,
            error=None,
        )
    except Exception as e:  # noqa: BLE001 -- we record the error string as data
        return QueryResult(
            model_id=spec.id,
            item_id=item_id,
            raw_text="",
            elapsed_s=time.time() - t0,
            timestamp=t0,
            error=f"{type(e).__name__}: {e}",
        )


def append_jsonl(path: Path, result: QueryResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(result)) + "\n")
