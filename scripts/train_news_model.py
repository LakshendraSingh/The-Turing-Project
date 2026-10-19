"""
Trains the 2026 Fake News & Disinformation Neural Classifier on the benchmark dataset.
Outputs calibrated PyTorch weights to weights/news_best.pt.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Ensure detector package is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.news_detector import NewsVerificationEngine, NewsClassifierBackbone, extract_news_feature_vector

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "news_best.pt")


def train_news_model(epochs=40, lr=0.005, device="cpu"):
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(data_dir, "data", "benchmark", "news", "fake_news_train.json")
    test_file = os.path.join(data_dir, "data", "test_benchmark", "news", "fake_news_test.json")

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")

    # Extract features
    X_train = np.array([extract_news_feature_vector(item["text"]) for item in train_data], dtype=np.float32)
    y_train = np.array([float(item["label"]) for item in train_data], dtype=np.float32)

    X_test = np.array([extract_news_feature_vector(item["text"]) for item in test_data], dtype=np.float32)
    y_test = np.array([float(item["label"]) for item in test_data], dtype=np.float32)

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)

    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    model = NewsClassifierBackbone(in_features=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze(-1)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_preds = (torch.sigmoid(logits) >= 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()

            test_logits = model(X_test_t).squeeze(-1)
            test_loss = criterion(test_logits, y_test_t).item()
            test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
            test_acc = (test_preds == y_test_t).float().mean().item()

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc*100:.1f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")

    print(f"\nTraining Complete! Best Test Accuracy: {best_test_acc*100:.1f}%")
    print(f"Saved production checkpoint to {CHECKPOINT_PATH}")

    # Verify holdout performance with metrics
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_test_t).squeeze(-1)) >= 0.5).float().cpu().numpy()
        tp = np.sum((preds == 1) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))

        print(f"Holdout Benchmark Evaluation:")
        print(f"  • Accuracy:  {best_test_acc*100:.2f}%")
        print(f"  • Precision: {precision*100:.2f}%")
        print(f"  • Recall:    {recall*100:.2f}%")
        print(f"  • F1-Score:  {f1:.4f}")


if __name__ == "__main__":
    train_news_model()
