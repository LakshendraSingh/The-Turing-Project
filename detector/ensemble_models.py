"""
The Turing Project (2026) — Specialized Neural Ensemble Architectures
Defines 3 specialized, odd-numbered deep neural backbones for each of the 6 modalities.
Each model focuses on distinct physical, mathematical, acoustic, or linguistic forensic signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. IMAGE FORENSICS SPECIALIZED MODELS
# =====================================================================

class SpectralPowerLawBackbone(nn.Module):
    """
    Model 2 for Images: Azimuthal Radial Power Spectral Density (PSD) Classifier.
    Detects power-law decay anomalies (-1.35 vs natural -2.00) and periodic
    high-frequency deconvolution grid peaks characteristic of GANs and Latent Diffusion.
    """
    def __init__(self, input_dim: int = 68):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MesoNet4Backbone(nn.Module):
    """
    Model 3 for Images: MesoNet-4 Architecture (Afchar et al.)
    Classic deepfake and micro-compression artifact CNN.
    Analyzes microscopic compression inconsistency across intermediate feature maps.
    """
    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv block 3
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Conv block 4
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)

        # Classifier
        self.fc1 = nn.Linear(16 * 2 * 2, 16)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, 3, 64, 64)
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x)), 0.1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.leaky_relu(self.fc1(x), 0.1))
        return self.fc2(x)


# =====================================================================
# 2. AUDIO FORENSICS SPECIALIZED MODELS
# =====================================================================

class VocoderPhaseNet(nn.Module):
    """
    Model 2 for Audio: Neural Vocoder Aliasing & Phase Discontinuity Net.
    Detects high-frequency phase inconsistencies and harmonic vocoder boundary
    artifacts produced by ElevenLabs, VALL-E, XTTS, and Diff-TTS.
    """
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PitchJitterBiomechanicNet(nn.Module):
    """
    Model 3 for Audio: Vocal Fold Biomechanics & Jitter Classifier.
    Analyzes human vocal micro-tremor, period perturbation (jitter, shimmer),
    and formant stability against synthetic robotic flatness.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# 3. VIDEO FORENSICS SPECIALIZED MODELS
# =====================================================================

class OpticalFlowDisparityNet(nn.Module):
    """
    Model 2 for Video: Inter-Frame Optical Flow Disparity Net.
    Analyzes dense optical flow gradient inconsistency, temporal jerkiness,
    and velocity vector divergence across consecutive video frames.
    """
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoundaryFlickerNet(nn.Module):
    """
    Model 3 for Video: Facial Landmark Margin & Boundary Shimmer Net.
    Detects high-frequency boundary flickering, blending seams, and
    spatiotemporal warping characteristic of Sora, Kling, and DeepFaceLab.
    """
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# 4. TEXT FORENSICS SPECIALIZED MODELS
# =====================================================================

class SentenceBurstinessNet(nn.Module):
    """
    Model 2 for Text: Sentence Burstiness Variance & Bigram Entropy Net.
    Measures sentence-length variance and cognitive rhythm irregularity,
    contrasting natural human writing against LLM token uniformity.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscourseClicheNet(nn.Module):
    """
    Model 3 for Text: Formulaic LLM Discourse Cliché & Transition Net.
    Detects synthetic transition hallmarks ('delve into', 'beacon of',
    'testament to', 'tapestry of') and formulaic sentence structure.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# 5. DOCUMENT FORENSICS SPECIALIZED MODELS
# =====================================================================

class ErrorLevelDisparityNet(nn.Module):
    """
    Model 2 for Documents: Error Level Analysis (ELA) Splicing Net.
    Detects JPEG recompression quantization discrepancies, block DCT variance,
    and localized pixel inpainting in receipts, IDs, and invoices.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FontSharpnessDisparityNet(nn.Module):
    """
    Model 3 for Documents: OCR Font Edge Step-Gradient Net.
    Identifies digitally inserted crisp typography superimposed on scanned paper backgrounds.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# 6. NEWS & CLAIMS VERIFICATION SPECIALIZED MODELS
# =====================================================================

class SensationalPanicNet(nn.Module):
    """
    Model 2 for News: Sensational Panic & Rural Scam Heuristic Net.
    Isolates urgent countdown timers, payment fee demands, mob violence triggers,
    and village panic rumors.
    """
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceCredibilityNet(nn.Module):
    """
    Model 3 for News: Source Attribution & Wire Credibility Net.
    Verifies institutional wire agency attribution (Reuters, AP, AFP),
    checks official government gazettes, and flags known satire publishers.
    """
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
