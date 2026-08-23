"""Model-free manipulation heuristics.

These are weak, classical signals — useful as a fast pre-filter and for
explainability ("why was this flagged"), but they are NOT a substitute for a
trained deepfake classifier. Modern generators increasingly defeat naive
frequency/noise checks. Combine with `deepfake_classifier.py` for anything
beyond a coarse triage pass, and validate thresholds on labeled data
(FaceForensics++, DFDC, Celeb-DF) before trusting scores in production.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class HeuristicReport:
    frequency_anomaly: float   # 0-1, higher = more GAN-like periodic spectral energy
    noise_inconsistency: float  # 0-1, higher = more spatially inconsistent noise (splicing signal)
    compression_anomaly: float  # 0-1, higher = signs of double/inconsistent JPEG compression
    overall_score: float        # weighted aggregate, 0-1

    def as_dict(self) -> dict:
        return self.__dict__


def _frequency_anomaly_score(gray: np.ndarray) -> float:
    """GAN upsampling often leaves periodic checkerboard energy in the FFT
    high-frequency band. Score = normalized high-freq energy concentration."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8

    # mask out the low-frequency center (natural image content)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_freq_mask = dist > radius

    high_energy = magnitude[high_freq_mask].mean() if high_freq_mask.any() else 0.0
    total_energy = magnitude.mean() + 1e-6
    ratio = float(high_energy / total_energy)

    # empirically, natural photos sit ~0.6-0.9; GAN artifacts push this higher.
    # squash into 0-1 with a soft threshold around 1.0
    return float(np.clip((ratio - 0.85) / 0.5, 0.0, 1.0))


def _noise_inconsistency_score(gray: np.ndarray, tile: int = 32) -> float:
    """Splicing/blending often stitches regions with different sensor-noise
    statistics. Estimate local noise variance per tile via a high-pass filter
    and score how inconsistent tile-to-tile variance is."""
    hp = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    h, w = hp.shape
    variances = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            patch = hp[y : y + tile, x : x + tile]
            variances.append(patch.var())

    if len(variances) < 4:
        return 0.0

    variances = np.array(variances)
    # coefficient of variation across tiles: natural images vary too, so we
    # look for unusually high dispersion relative to typical CoV.
    mean_v = variances.mean() + 1e-6
    cov = variances.std() / mean_v
    return float(np.clip((cov - 1.0) / 2.0, 0.0, 1.0))


def _compression_anomaly_score(bgr: np.ndarray, quality: int = 90) -> float:
    """Simple Error Level Analysis: recompress at a fixed JPEG quality and
    measure residual energy. Regions/images that were already compressed at a
    different quality (typical of copy-move or spliced-in content) show
    elevated, spatially uneven residuals."""
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return 0.0
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if recompressed is None or recompressed.shape != bgr.shape:
        return 0.0

    diff = cv2.absdiff(bgr, recompressed).astype(np.float32)
    residual = diff.mean()
    # heuristic scale — natural single-generation JPEGs show low, even residual
    return float(np.clip((residual - 3.0) / 12.0, 0.0, 1.0))


def analyze_image(bgr: np.ndarray) -> HeuristicReport:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    freq = _frequency_anomaly_score(gray)
    noise = _noise_inconsistency_score(gray)
    compression = _compression_anomaly_score(bgr)

    overall = 0.4 * freq + 0.35 * noise + 0.25 * compression
    return HeuristicReport(
        frequency_anomaly=round(freq, 3),
        noise_inconsistency=round(noise, 3),
        compression_anomaly=round(compression, 3),
        overall_score=round(float(overall), 3),
    )


def analyze_frames(frames: list[np.ndarray]) -> HeuristicReport:
    """Average per-frame heuristic scores across a sampled video frame set."""
    if not frames:
        raise ValueError("No frames to analyze")

    reports = [analyze_image(f) for f in frames]
    freq = float(np.mean([r.frequency_anomaly for r in reports]))
    noise = float(np.mean([r.noise_inconsistency for r in reports]))
    compression = float(np.mean([r.compression_anomaly for r in reports]))
    overall = 0.4 * freq + 0.35 * noise + 0.25 * compression

    return HeuristicReport(
        frequency_anomaly=round(freq, 3),
        noise_inconsistency=round(noise, 3),
        compression_anomaly=round(compression, 3),
        overall_score=round(overall, 3),
    )
