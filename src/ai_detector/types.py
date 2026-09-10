"""Empirical types and information-theoretic distances."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence

import numpy as np


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    total = probabilities.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("probabilities must have a positive finite sum")
    return probabilities / total


def kl_divergence(p: Sequence[float], q: Sequence[float], *, eps: float = 1e-12) -> float:
    """Return D(P || Q), using a tiny floor only for numerical stability."""
    p_array = _normalise(np.asarray(p, dtype=float))
    q_array = _normalise(np.asarray(q, dtype=float))
    if p_array.shape != q_array.shape:
        raise ValueError("p and q must have the same shape")
    positive = p_array > 0
    return float(np.sum(p_array[positive] * np.log(p_array[positive] / np.maximum(q_array[positive], eps))))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Return the symmetric Jensen-Shannon divergence."""
    p_array = _normalise(np.asarray(p, dtype=float))
    q_array = _normalise(np.asarray(q, dtype=float))
    midpoint = 0.5 * (p_array + q_array)
    return 0.5 * kl_divergence(p_array, midpoint) + 0.5 * kl_divergence(q_array, midpoint)


@dataclass(frozen=True)
class EmpiricalType:
    """A type (histogram and normalised frequencies) over a fixed alphabet."""

    alphabet: tuple[str, ...]
    counts: np.ndarray
    probabilities: np.ndarray

    @property
    def sample_size(self) -> int:
        return int(self.counts.sum())

    @property
    def support_size(self) -> int:
        return int(np.count_nonzero(self.counts))

    def kl_to(self, reference: Sequence[float]) -> float:
        return kl_divergence(self.probabilities, reference)

    def as_dict(self) -> dict[str, float]:
        return {token: float(probability) for token, probability in zip(self.alphabet, self.probabilities)}


class TypeEstimator:
    """Estimate empirical types with additive (Dirichlet/Laplace) smoothing."""

    def __init__(self, alphabet: Iterable[str], *, alpha: float = 0.5) -> None:
        self.alphabet = tuple(alphabet)
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet must contain unique, non-empty symbols")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self._indices = {symbol: index for index,
                         symbol in enumerate(self.alphabet)}

    def fit(self, observations: Iterable[str]) -> EmpiricalType:
        counts = np.zeros(len(self.alphabet), dtype=float)
        unknown = 0
        for observation in observations:
            index = self._indices.get(observation)
            if index is None:
                unknown += 1
            else:
                counts[index] += 1
        if unknown:
            raise ValueError(
                f"{unknown} observations are outside the estimator alphabet")
        if counts.sum() == 0:
            raise ValueError("at least one observation is required")
        smoothed = counts + self.alpha
        return EmpiricalType(self.alphabet, counts.astype(int), _normalise(smoothed))

    def fit_counts(self, counts: Sequence[int | float]) -> EmpiricalType:
        array = np.asarray(counts, dtype=float)
        if array.shape != (len(self.alphabet),) or np.any(array < 0):
            raise ValueError(
                "counts must be non-negative and match the alphabet")
        if array.sum() == 0:
            raise ValueError("at least one count is required")
        return EmpiricalType(self.alphabet, array.astype(int), _normalise(array + self.alpha))


def sanov_exponent(type_distribution: Sequence[float], reference: Sequence[float]) -> float:
    """The Sanov rate D(R||P) for an observed type R and reference P."""
    return kl_divergence(type_distribution, reference)


def sanov_probability_bound(exponent: float, sample_size: int, *, alphabet_size: int) -> float:
    """Method-of-types polynomial prefactor times exp(-n D(R||P))."""
    if exponent < 0 or sample_size <= 0 or alphabet_size <= 0:
        raise ValueError(
            "exponent must be non-negative and sizes must be positive")
    number_of_types = (sample_size + 1) ** alphabet_size
    return min(1.0, number_of_types * np.exp(-sample_size * exponent))
