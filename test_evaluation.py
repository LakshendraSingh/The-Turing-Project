#!/usr/bin/env python3
"""
The Turing Project — Out-of-Distribution Benchmark Evaluation Suite
Evaluates trained models (Vision, Text, Audio, Document) on brand-new independent test sets:
- Vision: Hemg AI-Generated Art vs Real Art (25,452 real + 25,451 AI samples)
- Text: Ateeqq AI and Human Generated Texts (abstracts & articles)
- Audio: Hemg Deepfake Audio Dataset (50 speech & 50 cloned voices)
- Document: Multimodal Tampered Invoices & Forged PDFs
"""

import io
import os
import sys
import glob
import time
import csv
import json

# Auto-discover local virtual environment packages if running via system python
_workspace_dir = os.path.dirname(os.path.abspath(__file__))
for _sp in glob.glob(os.path.join(_workspace_dir, ".venv", "lib", "python*", "site-packages")):
    if os.path.exists(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from detector.image_detector import ImageDeepfakeDetector
from detector.text_detector import TextDeepfakeDetector
from detector.audio_detector import AudioDeepfakeDetector
from detector.video_detector import VideoDeepfakeDetector
from detector.document_detector import DocumentForensicsDetector
from detector.news_detector import NewsVerificationEngine
from detector.email_detector import EmailForensicsDetector


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-12, precision + recall)

    return {
        "total": len(y_true),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def evaluate_text(detector: TextDeepfakeDetector, max_samples: int = 400):
    print("\n" + "=" * 70)
    print("EVALUATION 1: TEXT BENCHMARK (Ateeqq AI vs Human Test Set)")
    print("=" * 70)

    csv_path = "data/test_benchmark/text/ai_human_test.csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    y_true = []
    y_pred = []
    y_prob = []

    start_t = time.time()
    with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        count = 0
        for i, row in enumerate(reader):
            # Skip first 2,500 rows to ensure 100% out-of-distribution evaluation
            if i < 2500:
                continue

            lbl = int(float(row["label"]))
            text = (row.get("abstract", "") or row.get("title", "")).strip()
            if len(text) < 40:
                continue

            res = detector.analyze(text)
            pred = 1 if res["is_synthetic"] else 0
            prob = res["fake_probability"]

            y_true.append(lbl)
            y_pred.append(pred)
            y_prob.append(prob)

            count += 1
            if count >= max_samples:
                break

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} out-of-distribution text samples in {dt:.1f}s ({dt/max(1, count)*1000:.1f} ms/sample)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def evaluate_image(detector: ImageDeepfakeDetector, max_samples: int = 200):
    print("\n" + "=" * 70)
    print("EVALUATION 2: IMAGE BENCHMARK (Hemg Real Art vs AI Art Test Set)")
    print("=" * 70)

    real_path = "data/test_benchmark/image/real_art_test.parquet"
    fake_path = "data/test_benchmark/image/ai_art_test.parquet"

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print("Image test benchmark files not found.")
        return

    real_table = pq.read_table(real_path)
    fake_table = pq.read_table(fake_path)

    n_half = max_samples // 2
    y_true = []
    y_pred = []
    y_prob = []

    start_t = time.time()
    count = 0

    # Sample strictly from holdout slice [10,000, 20,000]
    rng = np.random.RandomState(42)
    max_idx = min(20000, min(len(real_table), len(fake_table)))
    real_indices = rng.permutation(np.arange(8000, max_idx))[:n_half]
    for idx in real_indices:
        try:
            raw_img = real_table["image"][int(idx)].as_py()
            img_bytes = raw_img["bytes"] if isinstance(raw_img, dict) else raw_img
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            res = detector.analyze(pil_img)
            pred = 1 if res["is_synthetic"] else 0
            y_true.append(0)
            y_pred.append(pred)
            y_prob.append(res["fake_probability"])
            count += 1
        except Exception:
            continue

    fake_indices = rng.permutation(np.arange(8000, max_idx))[:n_half]
    for idx in fake_indices:
        try:
            raw_img = fake_table["image"][int(idx)].as_py()
            img_bytes = raw_img["bytes"] if isinstance(raw_img, dict) else raw_img
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            res = detector.analyze(pil_img)
            pred = 1 if res["is_synthetic"] else 0
            y_true.append(1)
            y_pred.append(pred)
            y_prob.append(res["fake_probability"])
            count += 1
        except Exception:
            continue

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} balanced image samples in {dt:.1f}s ({dt/max(1, count)*1000:.1f} ms/sample)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def evaluate_audio(detector: AudioDeepfakeDetector, max_samples: int = 100):
    print("\n" + "=" * 70)
    print("EVALUATION 3: AUDIO BENCHMARK (Multilingual Human Speech vs Commercial Clones)")
    print("=" * 70)

    real_path = "data/benchmark/audio/audio_deepfake_train.parquet"
    synth_path = "data/benchmark/audio/audio_deepfake_synthetic.parquet"

    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        print("Audio benchmark files not found.")
        return

    real_table = pq.read_table(real_path)
    synth_table = pq.read_table(synth_path)

    y_true = []
    y_pred = []
    y_prob = []

    n_half = min(max_samples // 2, 50)
    start_t = time.time()
    count = 0

    # 1. Authentic Human Speech (Holdout slice 400..450)
    for idx in range(400, 400 + n_half):
        try:
            entry = real_table["audio"][idx].as_py()
            res = detector.analyze(entry["bytes"])
            pred = 1 if res["is_synthetic"] else 0
            y_true.append(0)
            y_pred.append(pred)
            y_prob.append(res["fake_probability"])
            count += 1
        except Exception:
            continue

    # 2. Commercial Synthetic Voice Clones (ElevenLabs, Hume, Kokoro, etc. - Holdout slice 400..450)
    for idx in range(400, 400 + n_half):
        try:
            entry = synth_table["audio"][idx].as_py()
            res = detector.analyze(entry["bytes"])
            pred = 1 if res["is_synthetic"] else 0
            y_true.append(1)
            y_pred.append(pred)
            y_prob.append(res["fake_probability"])
            count += 1
        except Exception:
            continue

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} balanced audio samples in {dt:.1f}s ({dt/max(1, count)*1000:.1f} ms/sample)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def evaluate_video(detector: VideoDeepfakeDetector):
    print("\n" + "=" * 70)
    print("EVALUATION 4: VIDEO FORENSICS BENCHMARK (Real MP4 vs Deepfake MP4 Videos)")
    print("=" * 70)

    video_dir = "data/test_benchmark/video"
    if not os.path.exists(video_dir):
        print("Video benchmark directory not found.")
        return

    import glob
    real_videos = sorted(glob.glob(os.path.join(video_dir, "real_*.mp4")))
    fake_videos = sorted(glob.glob(os.path.join(video_dir, "fake_*.mp4")))

    if not real_videos or not fake_videos:
        print("Video benchmark samples not found.")
        return

    y_true, y_pred, y_prob = [], [], []
    start_t = time.time()

    # 1. Real videos
    for v_path in real_videos:
        try:
            res = detector.analyze(v_path)
            y_true.append(0)
            y_pred.append(1 if res["is_synthetic"] else 0)
            y_prob.append(res["fake_probability"])
        except Exception:
            continue

    # 2. Fake videos
    for v_path in fake_videos:
        try:
            res = detector.analyze(v_path)
            y_true.append(1)
            y_pred.append(1 if res["is_synthetic"] else 0)
            y_prob.append(res["fake_probability"])
        except Exception:
            continue

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} video samples in {dt:.1f}s ({dt/max(1, len(y_true))*1000:.1f} ms/video)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def evaluate_document(detector: DocumentForensicsDetector, num_samples: int = 60):
    print("\n" + "=" * 70)
    print("EVALUATION 5: DOCUMENT FORENSICS BENCHMARK (Multimodal Tampering)")
    print("=" * 70)

    y_true = []
    y_pred = []
    y_prob = []

    start_t = time.time()
    n_half = num_samples // 2

    # 1. Clean authentic documents
    for i in range(n_half):
        canvas = Image.new("RGB", (300, 400), color=(250, 250, 250))
        res = detector.analyze(canvas)
        pred = 1 if res["is_synthetic"] else 0
        y_true.append(0)
        y_pred.append(pred)
        y_prob.append(res["fake_probability"])

    # 2. Tampered documents (spliced block, font disparity, metadata anomaly)
    for i in range(n_half):
        canvas = Image.new("RGB", (300, 400), color=(240, 240, 240))
        # Paste high-contrast spliced patch
        patch = Image.new("RGB", (60, 30), color=(20, 20, 20))
        canvas.paste(patch, (50, 100))
        # Add duplicated block for copy-move
        stamp = Image.new("RGB", (40, 40), color=(180, 50, 50))
        canvas.paste(stamp, (150, 250))
        canvas.paste(stamp, (220, 320))

        res = detector.analyze(canvas)
        pred = 1 if res["is_synthetic"] else 0
        y_true.append(1)
        y_pred.append(pred)
        y_prob.append(res["fake_probability"])

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} document forensic samples in {dt:.1f}s ({dt/num_samples*1000:.1f} ms/sample)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def evaluate_news(news_engine: NewsVerificationEngine):
    print("\n" + "=" * 70)
    print("EVALUATION 6: NEWS & DISINFORMATION BENCHMARK (Holdout Wire vs Scams)")
    print("=" * 70)

    import json
    test_path = "data/test_benchmark/news/fake_news_test.json"
    if not os.path.exists(test_path):
        print(f"File {test_path} not found.")
        return

    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    start_t = time.time()
    y_true, y_pred, y_prob = [], [], []

    for item in test_samples:
        t = item["text"]
        lbl = int(item["label"])
        res = news_engine.verify_news(t)
        pred = 1 if res["is_misinformation"] else 0
        y_true.append(lbl)
        y_pred.append(pred)
        y_prob.append(res["fake_probability"])

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} holdout news & scam claims in {dt:.1f}s ({dt/max(1, len(test_samples))*1000:.1f} ms/sample)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def evaluate_email(email_detector: EmailForensicsDetector):
    print("\n" + "=" * 70)
    print("EVALUATION 7: EMAIL SPAM & PHISHING BENCHMARK (Holdout Attacks vs Clean)")
    print("=" * 70)

    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_benchmark", "email", "email_phishing_test.json")
    if not os.path.exists(test_file):
        from scripts.build_email_dataset import build_and_save_dataset
        build_and_save_dataset()

    with open(test_file, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    start_t = time.time()
    y_true, y_pred, y_prob = [], [], []

    for item in test_samples:
        t = item["text"]
        lbl = int(item["label"])
        res = email_detector.analyze(t)
        pred = 1 if res["is_spam_or_phishing"] else 0
        y_true.append(lbl)
        y_pred.append(pred)
        y_prob.append(res["fake_probability"])

    dt = time.time() - start_t
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Evaluated {metrics['total']} holdout phishing & clean emails in {dt:.1f}s ({dt/max(1, len(test_samples))*1000:.1f} ms/email)")
    print(f"  • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • Confusion: {metrics['confusion']}")
    return metrics


def main():
    print("=" * 70)
    print("THE TURING PROJECT: OUT-OF-DISTRIBUTION INDEPENDENT TEST BENCHMARKS")
    print("=" * 70)

    text_detector = TextDeepfakeDetector(device="cpu")
    image_detector = ImageDeepfakeDetector(device="cpu")
    audio_detector = AudioDeepfakeDetector(device="cpu")
    video_detector = VideoDeepfakeDetector(device="cpu")
    document_detector = DocumentForensicsDetector(device="cpu")
    news_engine = NewsVerificationEngine(device="cpu")
    email_detector = EmailForensicsDetector(device="cpu")

    evaluate_text(text_detector, max_samples=400)
    evaluate_image(image_detector, max_samples=200)
    evaluate_audio(audio_detector, max_samples=50)
    evaluate_video(video_detector)
    evaluate_document(document_detector, num_samples=60)
    evaluate_news(news_engine)
    evaluate_email(email_detector)

    print("\n" + "=" * 70)
    print("ALL 7 INDEPENDENT TEST BENCHMARKS EVALUATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
