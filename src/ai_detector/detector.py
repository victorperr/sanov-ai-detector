"""Hypothesis testing with empirical types and Sanov rate functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.special import expit

from .types import EmpiricalType, TypeEstimator, kl_divergence, sanov_probability_bound


@dataclass(frozen=True)
class DetectionResult:
    label: str
    score: float
    log_likelihood_ratio: float
    kl_to_human: float
    kl_to_ai: float
    sanov_exponent_human: float
    sanov_exponent_ai: float
    finite_sample_bound_human: float
    finite_sample_bound_ai: float
    sample_size: int


@dataclass(frozen=True)
class Calibration:
    threshold: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float


class SanovDetector:
    """A plug-in type test for H0=human versus H1=AI.

    For an observed type R, the per-symbol log likelihood ratio is
    sum_x R(x) log(Q(x)/P(x)). This is equivalent to the difference
    D(R||P)-D(R||Q), up to the sign convention used below.
    """

    def __init__(self, alphabet: Sequence[str], human_distribution: Sequence[float], ai_distribution: Sequence[float], *, alpha: float = 0.5) -> None:
        self.estimator = TypeEstimator(alphabet, alpha=alpha)
        self.human = np.asarray(human_distribution, dtype=float)
        self.ai = np.asarray(ai_distribution, dtype=float)
        if self.human.shape != self.ai.shape or self.human.shape != (len(self.estimator.alphabet),):
            raise ValueError("reference distributions must match the alphabet")
        self.human = self.human / self.human.sum()
        self.ai = self.ai / self.ai.sum()
        self.threshold = 0.0

    def score_type(self, empirical: Sequence[float]) -> float:
        """Return a positive score when the type is more AI-like."""
        return kl_divergence(empirical, self.human) - kl_divergence(empirical, self.ai)

    def score(self, observations: Iterable[str]) -> DetectionResult:
        empirical = self.estimator.fit(observations)
        score = self.score_type(empirical.probabilities)
        kl_human = empirical.kl_to(self.human)
        kl_ai = empirical.kl_to(self.ai)
        return DetectionResult(
            label="ai" if score >= self.threshold else "human",
            score=score,
            log_likelihood_ratio=score * empirical.sample_size,
            kl_to_human=kl_human,
            kl_to_ai=kl_ai,
            sanov_exponent_human=kl_human,
            sanov_exponent_ai=kl_ai,
            finite_sample_bound_human=sanov_probability_bound(
                kl_human, empirical.sample_size, alphabet_size=len(self.estimator.alphabet)),
            finite_sample_bound_ai=sanov_probability_bound(
                kl_ai, empirical.sample_size, alphabet_size=len(self.estimator.alphabet)),
            sample_size=empirical.sample_size,
        )

    def calibrate(self, human_samples: Sequence[Sequence[str]], ai_samples: Sequence[Sequence[str]], *, target_fpr: float | None = None) -> Calibration:
        human_scores = np.array([self.score_type(self.estimator.fit(
            sample).probabilities) for sample in human_samples])
        ai_scores = np.array([self.score_type(self.estimator.fit(
            sample).probabilities) for sample in ai_samples])
        if target_fpr is None:
            threshold = float(
                (np.median(human_scores) + np.median(ai_scores)) / 2)
        else:
            if not 0 < target_fpr < 1:
                raise ValueError("target_fpr must be between 0 and 1")
            threshold = float(np.quantile(human_scores, 1 - target_fpr))
        self.threshold = threshold
        fpr = float(np.mean(human_scores >= threshold))
        fnr = float(np.mean(ai_scores < threshold))
        accuracy = float((np.mean(human_scores < threshold) +
                         np.mean(ai_scores >= threshold)) / 2)
        return Calibration(threshold, fpr, fnr, accuracy)

    def posterior_ai(self, observations: Iterable[str], *, prior_ai: float = 0.5) -> float:
        """Bayes posterior from the type likelihood ratio, useful for ranking."""
        if not 0 < prior_ai < 1:
            raise ValueError("prior_ai must be between 0 and 1")
        result = self.score(observations)
        prior_log_odds = np.log(prior_ai / (1 - prior_ai))
        return float(expit(prior_log_odds + result.log_likelihood_ratio))
