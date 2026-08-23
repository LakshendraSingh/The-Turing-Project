"""Combines the model-free heuristics with an optional pretrained deepfake
classifier. Ships working with heuristics only; set DEEPFAKE_MODEL_NAME in
.env to plug in a real model (e.g. a HuggingFace image-classification
checkpoint trained on FaceForensics++/DFDC) without changing calling code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

from src.utils.config import DEEPFAKE_MODEL_NAME, MEDIA_FLAG_THRESHOLD, IMAGE_HYBRID_MODEL_PATH
from src.media_forensics.heuristics import analyze_image, analyze_frames, HeuristicReport
from src.media_forensics.hybrid_model import HybridClassifier


@dataclass
class DeepfakeAssessment:
    heuristics: dict
    model_score: float | None   # None if no model configured
    combined_score: float
    flagged: bool
    note: str


def _maybe_run_model(image_or_frames) -> float | None:
    """Lazily load and run a pretrained classifier if DEEPFAKE_MODEL_NAME is set.
    Kept as a narrow integration point so swapping models means editing only
    this function."""
    if not DEEPFAKE_MODEL_NAME or DEEPFAKE_MODEL_NAME.strip().lower() == "none":
        return None

    try:
        from transformers import pipeline  # imported lazily; heavy dependency
    except ImportError:
        raise RuntimeError(
            "transformers not installed but DEEPFAKE_MODEL_NAME is set. "
            "pip install transformers torch"
        )

    clf = _get_cached_pipeline(DEEPFAKE_MODEL_NAME)
    # Default model (prithivMLmods/Deep-Fake-Detector-v2-Model) outputs labels
    # "Realism" / "Deepfake". Label matching below is deliberately loose
    # ("fake"/"deepfake"/"synthetic" substring) so other binary classifiers with
    # differently-cased or -worded labels work without code changes. If you swap
    # in a model with numeric labels (e.g. "LABEL_0"/"LABEL_1"), you'll need to
    # map those explicitly using the model's id2label config.
    frame = image_or_frames[0] if isinstance(image_or_frames, list) else image_or_frames
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    results = clf(pil_img)

    fake_score = 0.0
    for r in results:
        label = str(r.get("label", "")).lower()
        if "fake" in label or "deepfake" in label or "synthetic" in label:
            fake_score = max(fake_score, float(r.get("score", 0.0)))
    return fake_score


_pipeline_cache: dict = {}


def _get_cached_pipeline(model_name: str):
    if model_name not in _pipeline_cache:
        from transformers import pipeline
        _pipeline_cache[model_name] = pipeline("image-classification", model=model_name)
    return _pipeline_cache[model_name]


@lru_cache(maxsize=1)
def _load_hybrid_classifier() -> HybridClassifier | None:
    """Load a trained hybrid fusion model if one has been saved (see
    scripts/train_hybrid_model.py). Returns None if no trained model exists,
    in which case _combine() falls back to the fixed-weight blend."""
    if not IMAGE_HYBRID_MODEL_PATH.exists():
        return None
    try:
        return HybridClassifier.load(IMAGE_HYBRID_MODEL_PATH)
    except Exception:
        return None


def assess_image(bgr: np.ndarray) -> DeepfakeAssessment:
    heuristic_report: HeuristicReport = analyze_image(bgr)
    model_score = _maybe_run_model(bgr)
    return _combine(heuristic_report, model_score)


def assess_video_frames(frames: list[np.ndarray]) -> DeepfakeAssessment:
    heuristic_report: HeuristicReport = analyze_frames(frames)
    model_score = _maybe_run_model(frames)
    return _combine(heuristic_report, model_score)


def _combine(heuristic_report: HeuristicReport, model_score: float | None) -> DeepfakeAssessment:
    hybrid = _load_hybrid_classifier()

    if hybrid is not None:
        features = {**asdict(heuristic_report), "model_score": model_score if model_score is not None else 0.0}
        combined = hybrid.predict_proba(features)
        note = (
            "Score from a trained hybrid fusion classifier (logistic regression over "
            "heuristics + model score) — see data/models/image_hybrid.joblib. "
            "Retrain with scripts/train_hybrid_model.py as you collect more labeled data."
        )
    elif model_score is not None:
        combined = 0.7 * model_score + 0.3 * heuristic_report.overall_score
        note = (
            "No trained hybrid model found — using a fixed 70% model / 30% heuristics "
            "blend. Train a hybrid classifier on labeled examples for a data-driven "
            "weighting: scripts/train_hybrid_model.py"
        )
    else:
        combined = heuristic_report.overall_score
        note = (
            "No trained model configured (DEEPFAKE_MODEL_NAME unset) — score is "
            "heuristics-only and should be treated as a weak pre-filter, not a verdict."
        )

    return DeepfakeAssessment(
        heuristics=asdict(heuristic_report),
        model_score=model_score,
        combined_score=round(float(combined), 3),
        flagged=combined >= MEDIA_FLAG_THRESHOLD,
        note=note,
    )
