import numpy as np

from src.media_forensics.heuristics import analyze_image, analyze_frames


def _random_natural_like_image(size=256):
    """Low-frequency-dominant synthetic image to loosely stand in for a natural photo."""
    rng = np.random.default_rng(42)
    base = rng.normal(128, 20, (size, size)).astype(np.float32)
    # smooth it to concentrate energy at low frequencies, like real photos
    import cv2
    base = cv2.GaussianBlur(base, (15, 15), 0)
    img = np.stack([base, base, base], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_analyze_image_returns_bounded_scores():
    img = _random_natural_like_image()
    report = analyze_image(img)

    assert 0.0 <= report.frequency_anomaly <= 1.0
    assert 0.0 <= report.noise_inconsistency <= 1.0
    assert 0.0 <= report.compression_anomaly <= 1.0
    assert 0.0 <= report.overall_score <= 1.0


def test_analyze_frames_averages_across_frames():
    frames = [_random_natural_like_image() for _ in range(3)]
    report = analyze_frames(frames)
    assert 0.0 <= report.overall_score <= 1.0


def test_analyze_frames_raises_on_empty_list():
    try:
        analyze_frames([])
        assert False, "expected ValueError"
    except ValueError:
        pass
