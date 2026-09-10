from ai_detector.watermark import GreenlistWatermarker, watermark_ld_exponent


def test_watermarked_sequence_has_large_green_rate():
    detector = GreenlistWatermarker(64, gamma=0.25)
    result = detector.analyze(
        detector.generate_toy(200, seed=4, watermarked=True))
    assert result.green_rate > 0.8
    assert result.z_score > 2
    assert result.sanov_exponent > 0


def test_unmarked_sequence_is_not_systematically_green():
    detector = GreenlistWatermarker(64, gamma=0.25)
    result = detector.analyze(
        detector.generate_toy(200, seed=4, watermarked=False))
    assert result.green_rate < 0.4


def test_watermark_exponent_zero_at_null_rate():
    assert watermark_ld_exponent(0.25, 0.25) == 0
