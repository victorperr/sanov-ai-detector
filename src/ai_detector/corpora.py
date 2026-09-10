"""Tokenisation and deterministic offline reference distributions."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[.!?,;:]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def ngrams(tokens: Sequence[str], order: int = 1) -> list[str]:
    if order < 1:
        raise ValueError("order must be positive")
    return [" ".join(tokens[index: index + order]) for index in range(len(tokens) - order + 1)]


def vocabulary_from_sequences(sequences: Iterable[Sequence[str]], *, min_count: int = 1) -> tuple[str, ...]:
    counts = Counter(symbol for sequence in sequences for symbol in sequence)
    return tuple(sorted(symbol for symbol, count in counts.items() if count >= min_count))


def demo_references() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return a small interpretable reference pair for offline experiments."""
    alphabet = ("the", "and", "of", "to", "in", "a", "is",
                "for", "that", "model", "data", "research")
    human = np.array([0.18, 0.10, 0.11, 0.09, 0.09, 0.10,
                     0.08, 0.06, 0.06, 0.04, 0.05, 0.04])
    llm = np.array([0.13, 0.12, 0.08, 0.12, 0.07, 0.08,
                   0.10, 0.06, 0.08, 0.07, 0.05, 0.04])
    return alphabet, human / human.sum(), llm / llm.sum()


def project_to_alphabet(tokens: Iterable[str], alphabet: Sequence[str], *, unknown: str = "<unk>") -> list[str]:
    """Map out-of-vocabulary tokens to a stable bucket when it exists."""
    allowed = set(alphabet)
    if unknown not in allowed:
        return [token for token in tokens if token in allowed]
    return [token if token in allowed else unknown for token in tokens]
