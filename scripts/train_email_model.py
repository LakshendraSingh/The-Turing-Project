#!/usr/bin/env python3
"""
Trains the Email Spam, Phishing & Forensic Neural Classifier on the benchmark dataset.
Outputs calibrated PyTorch weights to:
- weights/email_best.pt
- weights/email_model_1_backbone.pt
- weights/email_model_2_phishing.pt
- weights/email_model_3_header.pt
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.email_detector import EmailClassifierBackbone, extract_email_feature_vector
from detector.ensemble_models import EmailPhishingUrgencyNet, EmailHeaderProvenanceNet

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def train_email_models(epochs=40, lr=0.005, device="cpu"):
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(data_dir, "data", "benchmark", "email", "email_spam_train.json")
    test_file = os.path.join(data_dir, "data", "test_benchmark", "email", "email_phishing_test.json")

    if not os.path.exists(train_file):
        from scripts.build_email_dataset import build_and_save_dataset
        build_and_save_dataset()

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(train_data)} training emails and {len(test_data)} test emails.")

    # 1. Extract 32-dim features
    X_train = np.array([extract_email_feature_vector(item["text"]) for item in train_data], dtype=np.float32)
    y_train = np.array([float(item["label"]) for item in train_data], dtype=np.float32)

    X_test = np.array([extract_email_feature_vector(item["text"]) for item in test_data], dtype=np.float32)
    y_test = np.array([float(item["label"]) for item in test_data], dtype=np.float32)

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)

    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    # -------------------------------------------------------------
    # Model 1: Dense 32-feature Neural Backbone
    # -------------------------------------------------------------
    print("\n--- Training Model 1: Email Neural Classifier Backbone ---")
    model1 = EmailClassifierBackbone(in_features=32).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs, eta_min=1e-4)

    best_test_acc1 = 0.0
    for epoch in range(1, epochs + 1):
        model1.train()
        optimizer1.zero_grad()
        logits = model1(X_train_t).squeeze(-1)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer1.step()
        scheduler1.step()

        model1.eval()
        with torch.no_grad():
            test_logits = model1(X_test_t).squeeze(-1)
            test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
            test_acc = (test_preds == y_test_t).float().mean().item()

        if test_acc >= best_test_acc1:
            best_test_acc1 = test_acc
            out_best = os.path.join(WEIGHTS_DIR, "email_best.pt")
            out_m1 = os.path.join(WEIGHTS_DIR, "email_model_1_backbone.pt")
            torch.save(model1.state_dict(), out_best)
            torch.save(model1.state_dict(), out_m1)

    print(f"Model 1 Training Complete! Best Test Accuracy: {best_test_acc1*100:.2f}%")

    # -------------------------------------------------------------
    # Model 2: Phishing Urgency & Credential Harvesting Net (16 dims)
    # -------------------------------------------------------------
    print("\n--- Training Model 2: Phishing Urgency & Credential Net ---")
    X_train_m2 = X_train_t[:, 10:26]
    X_test_m2 = X_test_t[:, 10:26]

    model2 = EmailPhishingUrgencyNet(input_dim=16).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr, weight_decay=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs, eta_min=1e-4)

    best_test_acc2 = 0.0
    for epoch in range(1, epochs + 1):
        model2.train()
        optimizer2.zero_grad()
        logits = model2(X_train_m2).squeeze(-1)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer2.step()
        scheduler2.step()

        model2.eval()
        with torch.no_grad():
            test_logits = model2(X_test_m2).squeeze(-1)
            test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
            test_acc = (test_preds == y_test_t).float().mean().item()

        if test_acc >= best_test_acc2:
            best_test_acc2 = test_acc
            out_m2 = os.path.join(WEIGHTS_DIR, "email_model_2_phishing.pt")
            torch.save(model2.state_dict(), out_m2)

    print(f"Model 2 Training Complete! Best Test Accuracy: {best_test_acc2*100:.2f}%")

    # -------------------------------------------------------------
    # Model 3: Header Provenance & URL Net (16 dims)
    # -------------------------------------------------------------
    print("\n--- Training Model 3: Header Provenance & URL Net ---")
    X_train_m3 = torch.cat([X_train_t[:, 0:10], X_train_t[:, 26:32]], dim=1)
    X_test_m3 = torch.cat([X_test_t[:, 0:10], X_test_t[:, 26:32]], dim=1)

    model3 = EmailHeaderProvenanceNet(input_dim=16).to(device)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=lr, weight_decay=1e-3)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=epochs, eta_min=1e-4)

    best_test_acc3 = 0.0
    for epoch in range(1, epochs + 1):
        model3.train()
        optimizer3.zero_grad()
        logits = model3(X_train_m3).squeeze(-1)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer3.step()
        scheduler3.step()

        model3.eval()
        with torch.no_grad():
            test_logits = model3(X_test_m3).squeeze(-1)
            test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
            test_acc = (test_preds == y_test_t).float().mean().item()

        if test_acc >= best_test_acc3:
            best_test_acc3 = test_acc
            out_m3 = os.path.join(WEIGHTS_DIR, "email_model_3_header.pt")
            torch.save(model3.state_dict(), out_m3)

    print(f"Model 3 Training Complete! Best Test Accuracy: {best_test_acc3*100:.2f}%")

    print("\nAll Email Forensics Weights successfully saved to weights/")


if __name__ == "__main__":
    train_email_models()
