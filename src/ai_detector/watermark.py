"""Kirchenbauer-style green-list watermarking and large-deviation analysis.

This module is a framework-compatible toy implementation: a keyed hash partitions
candidate token IDs into green and red lists at every position. Production use
must reproduce the generator's exact tokenizer, hashing scheme, and logits bias.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import erfc, log, sqrt
from multiprocessing import context
from typing import Iterable, Sequence

import numpy as np

from .types import kl_divergence


@dataclass(frozen=True)
class WatermarkResult:
    green_hits: int
    token_count: int
    green_rate: float
    z_score: float
    p_value: float
    sanov_exponent: float
    binomial_ld_bound: float
    is_watermarked: bool


class GreenlistWatermarker:
    """Keyed green-list detector with a configurable expected green rate."""

    def __init__(self, vocabulary_size: int, *, key: str = "ai-detector-demo", gamma: float = 0.25, delta: float = 2.0, context_width: int = 1) -> None:
        if vocabulary_size < 2 or not 0 < gamma < 1 or delta < 0:
            raise ValueError("invalid vocabulary_size, gamma, or delta")
        self.vocabulary_size = vocabulary_size
        self.key = key.encode("utf-8")
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.context_size = max(1, int(context_width))

    def _seed(self, context: Sequence[int], position: int) -> int:
        recent = context[-self.context_width:]
        ctx = ":".join(str(t) for t in recent)
        digest = hashlib.blake2b(
            self.key + f":{ctx}:{position}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big")

    def greenlist(self, context: Sequence[int], position: int) -> set[int]:
        rng = np.random.default_rng(self._seed(context, position))
        size = max(1, int(round(self.gamma * self.vocabulary_size)))
        return set(rng.choice(self.vocabulary_size, size=size, replace=False).tolist())

    def is_green(self, token: int, context: Sequence[int], position: int) -> bool:
        return int(token) in self.greenlist(context, position)

    def generate_toy(self, length: int, *, seed: int = 0, watermarked: bool = True, start_token: int = 0) -> list[int]:
        if length < 1:
            raise ValueError("length must be positive")
        rng = np.random.default_rng(seed)
        tokens = [int(start_token) % self.vocabulary_size]
        for position in range(1, length):
            candidates = self.greenlist(tokens[-1], position)
            pool = sorted(candidates if watermarked else set(
                range(self.vocabulary_size)) - candidates)
            tokens.append(int(rng.choice(pool)))
        return tokens

    def analyze(self, tokens: Sequence[int]) -> WatermarkResult:
        if len(tokens) < 2:
            raise ValueError("at least two tokens are required")
        green_hits = sum(self.is_green(
            token, tokens[position - 1], position) for position, token in enumerate(tokens[1:], 1))
        token_count = len(tokens) - 1
        green_rate = green_hits / token_count
        standard_deviation = sqrt(self.gamma * (1 - self.gamma) * token_count)
        z_score = (green_hits - self.gamma * token_count) / standard_deviation
        p_value = 0.5 * erfc(z_score / sqrt(2))
        exponent = kl_divergence(
            [green_rate, 1 - green_rate], [self.gamma, 1 - self.gamma])
        bound = min(1.0, (token_count + 1) * np.exp(-token_count * exponent))
        return WatermarkResult(green_hits, token_count, green_rate, z_score, p_value, exponent, float(bound), z_score >= self.delta)


def watermark_ld_exponent(green_rate: float, gamma: float) -> float:
    """Bernoulli Sanov exponent D((r,1-r)||(gamma,1-gamma))."""
    if not 0 <= green_rate <= 1 or not 0 < gamma < 1:
        raise ValueError("rates must be in [0, 1] and gamma must be in (0, 1)")
    return kl_divergence([green_rate, 1 - green_rate], [gamma, 1 - gamma])
