"""Helpers for turning emails into tokens that the classifier understands."""
from __future__ import annotations

import re
from typing import Iterable, Iterator

from .data_models import EmailMessage

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "is",
    "in",
    "to",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "this",
    "that",
    "it",
    "be",
}


def normalise_text(text: str) -> str:
    """Lowercase *text* and collapse internal whitespace."""

    cleaned = re.sub(r"\s+", " ", text.lower())
    return cleaned.strip()


def tokenise(text: str) -> Iterator[str]:
    """Yield token strings from *text* using a conservative tokenizer."""

    normalised = normalise_text(text)
    for match in _TOKEN_RE.finditer(normalised):
        token = match.group(0)
        if token not in _STOP_WORDS:
            yield token


def tokenize_email(email: EmailMessage) -> Iterable[str]:
    """Return an iterable of tokens extracted from the important email fields."""

    combined = email.as_text()
    return list(tokenise(combined))
