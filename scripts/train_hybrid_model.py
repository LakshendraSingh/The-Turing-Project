#!/usr/bin/env python
"""Train the hybrid (heuristics + pretrained-model-score) fusion classifier
on your own labeled real/fake examples.

Expected directory layout — one folder with real/ and fake/ subfolders:

    data/train/image/real/*.jpg|png|...
    data/train/image/fake/*.jpg|png|...

    data/train/audio/real/*.wav|mp3|...
    data/train/audio/fake/*.wav|mp3|...

Usage:
    python scripts/train_hybrid_model.py --domain image --data-dir data/train/image
    python scripts/train_hybrid_model.py --domain audio --data-dir data/train/audio

    # Skip the pretrained model (faster, no download, heuristics-only features):
    python scripts/train_hybrid_model.py --domain image --data-dir data/train/image --no-pretrained

You need at least a handful of examples of each class to fit anything
meaningful, and at least 10 total before the script holds out a test split
to report honest metrics. More labeled data (and more diverse fake sources)
makes a real difference here — this fuses a small number of signals, it
doesn't learn visual features from scratch, so it can't compensate for a
tiny or unrepresentative training set.
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.media_forensics.hybrid_model import HybridClassifier, IMAGE_FEATURE_NAMES, AUDIO_FEATURE_NAMES, HybridTrainResult
from src.media_forensics.heuristics import analyze_image
from src.media_forensics.frame_extractor import load_image
from src.media_forensics.audio_analysis import analyze_audio
from src.media_forensics.deepfake_classifier import _maybe_run_model as _maybe_run_image_model
from src.media_forensics.audio_analysis import _maybe_run_model as _maybe_run_audio_model
from src.utils.config import IMAGE_HYBRID_MODEL_PATH, VOICE_HYBRID_MODEL_PATH

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _collect_files(data_dir: Path, exts: set[str]) -> list[tuple[Path, int]]:
    """Returns (path, label) pairs — label 0 for real/, 1 for fake/."""
    pairs = []
    for label_name, label in [("real", 0), ("fake", 1)]:
        subdir = data_dir / label_name
        if not subdir.exists():
            print(f"Warning: expected subfolder not found: {subdir}")
            continue
        for p in sorted(subdir.iterdir()):
            if p.suffix.lower() in exts:
                pairs.append((p, label))
    return pairs


def train_image_hybrid(data_dir: Path, use_pretrained: bool) -> None:
    pairs = _collect_files(data_dir, IMAGE_EXTS)
    if len(pairs) < 4:
        raise SystemExit(
            f"Found only {len(pairs)} labeled images under {data_dir}. Need real/ and "
            f"fake/ subfolders with at least a few examples each — more is much better."
        )

    feature_dicts, labels = [], []
    for i, (path, label) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {path}")
        img = load_image(str(path))
        report = analyze_image(img)
        model_score = _maybe_run_image_model(img) if use_pretrained else None
        features = asdict(report)
        features["model_score"] = model_score if model_score is not None else 0.0
        feature_dicts.append(features)
        labels.append(label)

    clf = HybridClassifier(IMAGE_FEATURE_NAMES)
    result = clf.fit(feature_dicts, labels)
    clf.save(IMAGE_HYBRID_MODEL_PATH)
    _print_result("image", result, clf, IMAGE_HYBRID_MODEL_PATH)


def train_audio_hybrid(data_dir: Path, use_pretrained: bool) -> None:
    pairs = _collect_files(data_dir, AUDIO_EXTS)
    if len(pairs) < 4:
        raise SystemExit(
            f"Found only {len(pairs)} labeled audio files under {data_dir}. Need real/ "
            f"and fake/ subfolders with at least a few examples each — more is much better."
        )

    feature_dicts, labels = [], []
    for i, (path, label) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {path}")
        report = analyze_audio(str(path))
        model_score = _maybe_run_audio_model(str(path)) if use_pretrained else None
        features = asdict(report)
        features["model_score"] = model_score if model_score is not None else 0.0
        feature_dicts.append(features)
        labels.append(label)

    clf = HybridClassifier(AUDIO_FEATURE_NAMES)
    result = clf.fit(feature_dicts, labels)
    clf.save(VOICE_HYBRID_MODEL_PATH)
    _print_result("audio", result, clf, VOICE_HYBRID_MODEL_PATH)


def _print_result(domain: str, result: HybridTrainResult, clf: HybridClassifier, path: Path) -> None:
    print()
    print(f"Trained {domain} hybrid classifier on {result.n_train} examples "
          f"(held out {result.n_test} for eval).")
    if result.n_test < 5:
        print("NOTE: very small holdout — treat these metrics as a sanity check, not a real evaluation.")
    if result.test_auc is not None:
        print(f"Held-out AUC: {result.test_auc:.3f}")
    print("Held-out classification report:")
    for label, metrics in result.test_report.items():
        if isinstance(metrics, dict):
            print(f"  {label}: precision={metrics.get('precision', 0):.3f} "
                  f"recall={metrics.get('recall', 0):.3f} f1={metrics.get('f1-score', 0):.3f}")
    print()
    print("Learned feature weights (higher magnitude = more influence on the fake score):")
    for name, weight in clf.coefficients().items():
        print(f"  {name}: {weight:+.3f}")
    print()
    print(f"Saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["image", "audio"], required=True)
    parser.add_argument("--data-dir", required=True, help="Directory with real/ and fake/ subfolders")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train on heuristics only, skip running the pretrained model (faster, no download)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    if args.domain == "image":
        train_image_hybrid(data_dir, use_pretrained=not args.no_pretrained)
    else:
        train_audio_hybrid(data_dir, use_pretrained=not args.no_pretrained)
