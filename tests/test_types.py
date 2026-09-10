import numpy as np
import pytest

from ai_detector.types import TypeEstimator, kl_divergence, sanov_probability_bound


def test_type_counts_and_smoothing():
    result = TypeEstimator(["a", "b"], alpha=0.5).fit(["a", "a", "b"])
    assert result.counts.tolist() == [2, 1]
    assert np.isclose(result.probabilities.sum(), 1.0)


def test_kl_is_zero_for_same_distribution():
    assert np.isclose(kl_divergence([0.2, 0.8], [0.2, 0.8]), 0.0)


def test_unknown_symbols_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        TypeEstimator(["a"]).fit(["b"])


def test_sanov_bound_is_a_probability():
    assert 0 < sanov_probability_bound(0.5, 20, alphabet_size=3) <= 1
