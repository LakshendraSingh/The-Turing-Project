"""Trainable hybrid classifier that fuses heuristic signals with a pretrained
model's score into a single calibrated probability, via logistic-regression
stacking instead of a hand-picked fixed weight (e.g. the 70/30 blend used as
a fallback when no hybrid model has been trained yet).

Why this over the fixed blend: the "right" weighting between heuristics and
a pretrained model depends on your actual data — how good the heuristics are
on your specific media, how well the pretrained model generalizes to it, and
how they correlate. A fixed 0.7/0.3 split is a guess. Fitting logistic
regression on labeled examples lets the data decide, and gives you inspectable
coefficients showing how much each signal actually contributes.

Deliberately simple (a handful of scalar features into logistic regression) —
the goal is calibration and interpretability on modest labeled datasets, not
squeezing out maximum accuracy. If you have thousands of labeled examples,
consider swapping LogisticRegression for a gradient-boosted tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Feature order doesn't matter across calls since HybridClassifier vectorizes
# by name — these constants just document what each domain expects.
IMAGE_FEATURE_NAMES = [
    "frequency_anomaly",
    "noise_inconsistency",
    "compression_anomaly",
    "model_score",
]
AUDIO_FEATURE_NAMES = [
    "spectral_flatness_anomaly",
    "pitch_variance_anomaly",
    "model_score",
]


@dataclass
class HybridTrainResult:
    feature_names: list[str]
    n_train: int
    n_test: int
    test_report: dict
    test_auc: float | None


class HybridClassifier:
    """Wraps a scikit-learn LogisticRegression with feature-name bookkeeping,
    so features stay correctly aligned between training and inference
    regardless of dict key ordering."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self._fitted = False

    def _vectorize(self, features: dict) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def fit(
        self,
        feature_dicts: list[dict],
        labels: list[int],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> HybridTrainResult:
        X = np.array([self._vectorize(f) for f in feature_dicts])
        y = np.array(labels)

        if len(set(y.tolist())) < 2:
            raise ValueError(
                "Need both real (label 0) and fake (label 1) examples to train a hybrid classifier"
            )

        if len(y) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # Too little data for a meaningful holdout — fit on everything and
            # self-report. Treat these metrics as a sanity check, not real evaluation.
            X_train, y_train = X, y
            X_test, y_test = X, y

        self.model.fit(X_train, y_train)
        self._fitted = True

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        auc = None
        if len(set(y_test.tolist())) > 1:
            y_proba = self.model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))

        return HybridTrainResult(
            feature_names=self.feature_names,
            n_train=len(y_train),
            n_test=len(y_test),
            test_report=report,
            test_auc=auc,
        )

    def predict_proba(self, features: dict) -> float:
        if not self._fitted:
            raise RuntimeError("HybridClassifier not fitted/loaded")
        X = np.array([self._vectorize(features)])
        return float(self.model.predict_proba(X)[0, 1])

    def coefficients(self) -> dict:
        """Human-readable feature -> learned weight, for explainability."""
        if not self._fitted:
            raise RuntimeError("HybridClassifier not fitted/loaded")
        return dict(zip(self.feature_names, self.model.coef_[0].tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"feature_names": self.feature_names, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "HybridClassifier":
        data = joblib.load(path)
        obj = cls(data["feature_names"])
        obj.model = data["model"]
        obj._fitted = True
        return obj
