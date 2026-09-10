import numpy as np

from ai_detector.detector import SanovDetector


def make_detector():
    return SanovDetector(["a", "b"], [0.8, 0.2], [0.2, 0.8], alpha=0.01)


def test_detector_prefers_matching_reference():
    detector = make_detector()
    human = detector.score(["a"] * 80 + ["b"] * 20)
    ai = detector.score(["a"] * 20 + ["b"] * 80)
    assert human.label == "human"
    assert ai.label == "ai"
    assert ai.score > human.score


def test_calibration_sets_threshold():
    detector = make_detector()
    calibration = detector.calibrate(
        [["a"] * 80 + ["b"] * 20, ["a"] * 70 + ["b"] * 30],
        [["a"] * 20 + ["b"] * 80, ["a"] * 30 + ["b"] * 70],
    )
    assert np.isfinite(calibration.threshold)
    assert calibration.accuracy >= 0.5
