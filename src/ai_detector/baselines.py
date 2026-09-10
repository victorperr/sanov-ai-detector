"""Optional comparison baselines for empirical evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    labels: np.ndarray
    scores: np.ndarray
    accuracy: float | None
    roc_auc: float | None


class TfidfLogisticBaseline:
    """A lightweight supervised baseline.

    It intentionally keeps the dependency optional so the statistical experiment
    remains runnable in a clean environment.
    """

    def __init__(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:
            raise ImportError(
                "Install the 'ml' extra to use TfidfLogisticBaseline") from error
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=1, sublinear_tf=True)
        self.model = LogisticRegression(max_iter=1000, random_state=0)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfLogisticBaseline":
        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, labels)
        return self

    def predict(self, texts: Sequence[str]) -> BaselineResult:
        features = self.vectorizer.transform(texts)
        scores = self.model.predict_proba(features)[:, 1]
        labels = (scores >= 0.5).astype(int)
        return BaselineResult(labels, scores, None, None)

    def evaluate(self, texts: Sequence[str], labels: Sequence[int]) -> BaselineResult:
        result = self.predict(texts)
        truth = np.asarray(labels, dtype=int)
        if truth.shape != result.labels.shape:
            raise ValueError("labels must have one entry per text")
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score
        except ImportError as error:
            raise ImportError(
                "Install the 'ml' extra to evaluate the baseline") from error
        return BaselineResult(
            result.labels,
            result.scores,
            float(accuracy_score(truth, result.labels)),
            float(roc_auc_score(truth, result.scores)) if len(
                np.unique(truth)) > 1 else None,
        )
