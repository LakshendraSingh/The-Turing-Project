import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.media_forensics.hybrid_model import HybridClassifier, IMAGE_FEATURE_NAMES


def _synthetic_dataset(n_per_class: int = 30, seed: int = 0):
    """Build a linearly-separable synthetic feature set so the classifier has
    something real to learn without needing actual images/models."""
    rng = np.random.default_rng(seed)
    feature_dicts, labels = [], []

    for _ in range(n_per_class):
        # "real" examples: low anomaly scores, low model score
        feature_dicts.append({
            "frequency_anomaly": float(rng.uniform(0.0, 0.3)),
            "noise_inconsistency": float(rng.uniform(0.0, 0.3)),
            "compression_anomaly": float(rng.uniform(0.0, 0.3)),
            "model_score": float(rng.uniform(0.0, 0.3)),
        })
        labels.append(0)

    for _ in range(n_per_class):
        # "fake" examples: high anomaly scores, high model score
        feature_dicts.append({
            "frequency_anomaly": float(rng.uniform(0.6, 1.0)),
            "noise_inconsistency": float(rng.uniform(0.6, 1.0)),
            "compression_anomaly": float(rng.uniform(0.6, 1.0)),
            "model_score": float(rng.uniform(0.6, 1.0)),
        })
        labels.append(1)

    return feature_dicts, labels


def test_fit_and_predict_separates_classes():
    feature_dicts, labels = _synthetic_dataset()
    clf = HybridClassifier(IMAGE_FEATURE_NAMES)
    result = clf.fit(feature_dicts, labels)

    assert result.n_train > 0
    assert result.n_test > 0
    assert result.test_auc is not None
    # on this obviously-separable synthetic data, AUC should be very high
    assert result.test_auc > 0.9

    clearly_real = clf.predict_proba({
        "frequency_anomaly": 0.05, "noise_inconsistency": 0.05,
        "compression_anomaly": 0.05, "model_score": 0.05,
    })
    clearly_fake = clf.predict_proba({
        "frequency_anomaly": 0.95, "noise_inconsistency": 0.95,
        "compression_anomaly": 0.95, "model_score": 0.95,
    })
    assert clearly_real < 0.3
    assert clearly_fake > 0.7


def test_fit_requires_both_classes():
    clf = HybridClassifier(IMAGE_FEATURE_NAMES)
    with pytest.raises(ValueError):
        clf.fit([{"model_score": 0.1}, {"model_score": 0.2}], [0, 0])


def test_save_and_load_roundtrip():
    feature_dicts, labels = _synthetic_dataset(n_per_class=15)
    clf = HybridClassifier(IMAGE_FEATURE_NAMES)
    clf.fit(feature_dicts, labels)

    sample_features = feature_dicts[0]
    original_score = clf.predict_proba(sample_features)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "hybrid.joblib"
        clf.save(path)
        assert path.exists()

        loaded = HybridClassifier.load(path)
        loaded_score = loaded.predict_proba(sample_features)

    assert original_score == pytest.approx(loaded_score)


def test_predict_proba_before_fit_raises():
    clf = HybridClassifier(IMAGE_FEATURE_NAMES)
    with pytest.raises(RuntimeError):
        clf.predict_proba({"model_score": 0.5})


def test_coefficients_reflect_feature_names():
    feature_dicts, labels = _synthetic_dataset()
    clf = HybridClassifier(IMAGE_FEATURE_NAMES)
    clf.fit(feature_dicts, labels)
    coefs = clf.coefficients()
    assert set(coefs.keys()) == set(IMAGE_FEATURE_NAMES)
    # every synthetic feature was constructed to push toward "fake", so all
    # learned weights should be positive
    assert all(w > 0 for w in coefs.values())
