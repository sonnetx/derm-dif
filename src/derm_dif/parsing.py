"""Response parsing: classify each free-text model response as benign / malignant / refusal.

Refusals are NOT coerced into a default label. They are an analytic outcome
category in their own right (see paper §3.2 and §3.6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ResponseLabel = Literal["benign", "malignant", "refusal", "unparseable"]


@dataclass(frozen=True)
class ParsedResponse:
    label: ResponseLabel
    raw: str
    confidence: str | None
    reasoning: str | None
    used_fallback: bool


_ANSWER_RE = re.compile(r"Answer:\s*(benign|malignant)", re.IGNORECASE)
_CONF_RE = re.compile(r"Confidence:\s*(low|medium|high)", re.IGNORECASE)
_REASON_RE = re.compile(r"Reasoning:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _looks_like_refusal(text: str, markers: list[str]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


def parse_primary(text: str, refusal_markers: list[str]) -> ParsedResponse:
    """Apply the primary regex parser. Does not invoke the fallback parser."""
    if _looks_like_refusal(text, refusal_markers) and not _ANSWER_RE.search(text):
        return ParsedResponse("refusal", text, None, None, False)
    m = _ANSWER_RE.search(text)
    if m is None:
        return ParsedResponse("unparseable", text, None, None, False)
    conf_m = _CONF_RE.search(text)
    reason_m = _REASON_RE.search(text)
    return ParsedResponse(
        label=m.group(1).lower(),  # type: ignore[arg-type]
        raw=text,
        confidence=conf_m.group(1).lower() if conf_m else None,
        reasoning=reason_m.group(1).strip() if reason_m else None,
        used_fallback=False,
    )


def to_correctness(parsed: ParsedResponse, malignant_truth: bool) -> int | None:
    """Map (parsed label, truth) -> {0, 1, None}.

    Returns None for refusals and unparseable responses; downstream IRT code
    treats those as missing-at-random for the Rasch fit but they are
    separately analyzed (refusal-rate DIF).
    """
    if parsed.label == "refusal" or parsed.label == "unparseable":
        return None
    predicted_malignant = parsed.label == "malignant"
    return int(predicted_malignant == malignant_truth)
