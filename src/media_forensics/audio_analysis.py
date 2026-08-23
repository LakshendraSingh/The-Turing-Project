"""Spectral heuristics plus an optional pretrained anti-spoofing model for
flagging likely synthetic/cloned speech.

The heuristics (spectral flatness, pitch-contour naturalness) are a coarse,
model-free pre-filter — useful for explainability but easy for a good TTS
system to defeat. By default this module also loads a pretrained audio
anti-spoofing classifier (see VOICE_MODEL_NAME in .env) and blends it in,
the same pattern used in deepfake_classifier.py for images/video.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import librosa

from src.utils.config import VOICE_MODEL_NAME, MEDIA_FLAG_THRESHOLD, VOICE_HYBRID_MODEL_PATH
from src.media_forensics.hybrid_model import HybridClassifier


@dataclass
class AudioReport:
    spectral_flatness_anomaly: float   # TTS/vocoder output tends to be spectrally "too clean"
    pitch_variance_anomaly: float      # unnaturally smooth/robotic pitch contour
    overall_score: float


def analyze_audio(path: str) -> AudioReport:
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio signal")

    # Spectral flatness: real speech has natural formant peaks (low flatness);
    # some vocoder artifacts push this unnaturally high or show odd periodicity.
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(flatness))
    flatness_score = float(np.clip((flatness_mean - 0.15) / 0.25, 0.0, 1.0))

    # Pitch (f0) contour naturalness via librosa.pyin; synthetic speech often
    # has an unusually smooth/low-variance pitch trajectory.
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if f0_voiced.size > 5:
        pitch_cv = float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-6))
        # natural speech typically has pitch_cv roughly in the 0.1-0.4 range;
        # very low variance is the synthetic-flagging signal here
        pitch_score = float(np.clip((0.12 - pitch_cv) / 0.12, 0.0, 1.0))
    else:
        pitch_score = 0.0  # not enough voiced signal to judge

    overall = 0.5 * flatness_score + 0.5 * pitch_score
    return AudioReport(
        spectral_flatness_anomaly=round(flatness_score, 3),
        pitch_variance_anomaly=round(pitch_score, 3),
        overall_score=round(float(overall), 3),
    )


_pipeline_cache: dict = {}


def _get_cached_pipeline(model_name: str):
    if model_name not in _pipeline_cache:
        from transformers import pipeline
        _pipeline_cache[model_name] = pipeline("audio-classification", model=model_name)
    return _pipeline_cache[model_name]


@lru_cache(maxsize=1)
def _load_hybrid_classifier() -> HybridClassifier | None:
    """Load a trained hybrid fusion model if one has been saved (see
    scripts/train_hybrid_model.py --domain audio). Returns None if no trained
    model exists, in which case assess_voice() falls back to the fixed-weight blend."""
    if not VOICE_HYBRID_MODEL_PATH.exists():
        return None
    try:
        return HybridClassifier.load(VOICE_HYBRID_MODEL_PATH)
    except Exception:
        return None


def _maybe_run_model(path: str) -> float | None:
    """Lazily load and run a pretrained anti-spoofing model if VOICE_MODEL_NAME
    is set. Default model (MelodyMachine/Deepfake-audio-detection-V2) is a
    wav2vec2-based binary classifier. Label matching is loose ("fake"/"spoof"/
    "synthetic" substring) to work across differently-labeled models — if your
    chosen model uses numeric labels (e.g. "LABEL_0"/"LABEL_1"), check its
    id2label config and map explicitly instead."""
    if not VOICE_MODEL_NAME or VOICE_MODEL_NAME.strip().lower() == "none":
        return None

    try:
        import transformers  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "transformers not installed but VOICE_MODEL_NAME is set. "
            "pip install transformers torch"
        )

    clf = _get_cached_pipeline(VOICE_MODEL_NAME)
    y, sr = librosa.load(path, sr=16000, mono=True)
    results = clf({"array": y, "sampling_rate": sr})

    fake_score = 0.0
    for r in results:
        label = str(r.get("label", "")).lower()
        if "fake" in label or "spoof" in label or "synthetic" in label:
            fake_score = max(fake_score, float(r.get("score", 0.0)))
    return fake_score


def assess_voice(path: str) -> dict:
    """Full audio assessment: heuristics + optional model + optional trained
    hybrid fusion classifier, combined score, flag."""
    heuristic_report = analyze_audio(path)
    model_score = _maybe_run_model(path)
    hybrid = _load_hybrid_classifier()

    if hybrid is not None:
        features = {**asdict(heuristic_report), "model_score": model_score if model_score is not None else 0.0}
        combined = hybrid.predict_proba(features)
        note = (
            "Score from a trained hybrid fusion classifier (logistic regression over "
            "heuristics + model score) — see data/models/voice_hybrid.joblib. "
            "Retrain with scripts/train_hybrid_model.py as you collect more labeled data."
        )
    elif model_score is not None:
        combined = 0.7 * model_score + 0.3 * heuristic_report.overall_score
        note = (
            "No trained hybrid model found — using a fixed 70% model / 30% heuristics "
            "blend. Train a hybrid classifier for a data-driven weighting: "
            "scripts/train_hybrid_model.py"
        )
    else:
        combined = heuristic_report.overall_score
        note = (
            "No trained model configured (VOICE_MODEL_NAME unset) — score is "
            "heuristics-only and should be treated as a weak pre-filter, not a verdict."
        )

    return {
        "heuristics": asdict(heuristic_report),
        "model_score": model_score,
        "combined_score": round(float(combined), 3),
        "flagged": combined >= MEDIA_FLAG_THRESHOLD,
        "note": note,
    }


def analyze_audio_dict(path: str) -> dict:
    """Kept for backward compatibility — heuristics only, no model."""
    return asdict(analyze_audio(path))
