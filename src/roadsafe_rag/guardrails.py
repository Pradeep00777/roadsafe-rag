"""Input and output guardrails for the RAG pipeline.

Input:  length limit and a small prompt-injection screen.
Output: every answer must cite retrieved passages, and every citation must
        point at a passage that was actually supplied to the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

REFUSAL_TEXT = "I could not find this in the provided report."
MAX_QUESTION_CHARS = 500

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"(reveal|show|print|repeat)\s+.{0,30}(system\s+prompt|instructions)",
    r"you\s+are\s+now\s+",
    r"\bjailbreak\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def check_input(question: str) -> tuple[bool, str | None]:
    """Return (ok, reason). `reason` is set only when the input is rejected."""
    text = (question or "").strip()
    if not text:
        return False, "empty_question"
    if len(text) > MAX_QUESTION_CHARS:
        return False, "question_too_long"
    if _INJECTION_RE.search(text):
        return False, "unsafe_input"
    return True, None


def extract_citations(answer: str) -> set[int]:
    return {int(m) for m in _CITATION_RE.findall(answer)}


@dataclass(frozen=True)
class OutputCheck:
    ok: bool
    reason: str | None = None
    is_refusal: bool = False
    cited: tuple[int, ...] = ()


def check_output(answer: str, n_sources: int) -> OutputCheck:
    text = (answer or "").strip()
    if not text:
        return OutputCheck(False, "empty_answer")
    if text.startswith(REFUSAL_TEXT):
        return OutputCheck(True, is_refusal=True)
    cited = extract_citations(text)
    if not cited:
        return OutputCheck(False, "uncited_answer")
    if any(c < 1 or c > n_sources for c in cited):
        return OutputCheck(False, "invalid_citation")
    return OutputCheck(True, cited=tuple(sorted(cited)))
