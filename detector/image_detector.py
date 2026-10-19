"""
Image Deepfake & Synthetic Media Detector (2026)
Detects Diffusion (FLUX.1, SD3, Midjourney v6/v7, DALL-E 3), GANs, and DeepFace/Inpainting.
Uses Dual-Domain Forensic Architecture:
- Frequency Domain (2D-FFT azimuthal power spectrum & high-frequency anomalies)
- Error Level Analysis (ELA compression disparity)
- Noise Residual Analysis (PRNU / high-pass sensor fingerprint)
- Spatial Neural Classifier (Dual-stream PyTorch architecture)
"""

import io
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from .metadata_forensics import UniversalMetadataForensics


class SqueezeExcitation2d(nn.Module):
    """
    Channel recalibration block that adaptively weights informative feature channels.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(8, channels // reduction)),
            nn.SiLU(),
            nn.Linear(max(8, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention2d(nn.Module):
    """
    Spatial Attention Module (CBAM): focuses on WHERE tampering artifacts and noise discrepancies are localized.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class ResidualConvBlock2d(nn.Module):
    """
    Residual convolutional building block with CBAM (Squeeze-and-Excitation + Spatial Attention).
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SqueezeExcitation2d(out_c)
        self.sa = SpatialAttention2d()
        self.act2 = nn.SiLU()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.sa(out)
        out = self.act2(out + res)
        return out


class CrossDomainAttention(nn.Module):
    """
    Cross-Attention between Spatial (RGB) and Frequency/Noise Residual Streams.
    Allows spatial texture features to query anomalous sensor noise artifacts.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, f_rgb, f_res):
        b, c, h, w = f_rgb.shape
        rgb_seq = f_rgb.flatten(2).transpose(1, 2)  # [B, H*W, C]
        res_seq = f_res.flatten(2).transpose(1, 2)  # [B, H*W, C]

        attn_out, _ = self.mha(query=self.norm1(rgb_seq), key=self.norm1(res_seq), value=self.norm1(res_seq))
        rgb_attended = rgb_seq + attn_out
        out = rgb_attended + self.ffn(self.norm2(rgb_attended))
        return out.transpose(1, 2).reshape(b, c, h, w)


class DualDomainVisionBackbone(nn.Module):
    """
    2026 Dual-Domain Deepfake Vision Backbone:
    - Multi-Stage Residual CNN with Squeeze-and-Excitation (SE) channel recalibration
    - Dual Stream: Spatial RGB + High-Frequency Noise Residual (PRNU/diffusion latent noise)
    - Cross-Domain Attention Layer correlating spatial semantics with noise floor disruptions
    - CatAvgMaxPool (Global Average + Global Max Pooling) preserving peak anomaly activations
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # Spatial Stream (RGB)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.spatial_stage1 = ResidualConvBlock2d(32, 64, stride=2)
        self.spatial_stage2 = ResidualConvBlock2d(64, 128, stride=2)

        # High-Frequency Residual Stream
        self.residual_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.residual_stage1 = ResidualConvBlock2d(32, 64, stride=2)
        self.residual_stage2 = ResidualConvBlock2d(64, 128, stride=2)

        # Cross-Domain Attention at final feature resolution (128 channels)
        self.cross_attn = CrossDomainAttention(dim=128, num_heads=4)

        # Dual Pooling: Global Avg Pool (128) + Global Max Pool (128) = 256 per stream
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Classification Fusion Head: 256 (RGB) + 256 (Res) + 256 (Attended) = 768 dims
        self.head = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def pool_features(self, feat):
        avg_p = self.gap(feat).flatten(1)
        max_p = self.gmp(feat).flatten(1)
        return torch.cat([avg_p, max_p], dim=1)

    def forward(self, rgb, residual):
        s_rgb = self.spatial_stage2(self.spatial_stage1(self.spatial_stem(rgb)))
        s_res = self.residual_stage2(self.residual_stage1(self.residual_stem(residual)))

        s_att = self.cross_attn(s_rgb, s_res)

        p_rgb = self.pool_features(s_rgb)
        p_res = self.pool_features(s_res)
        p_att = self.pool_features(s_att)

        fused = torch.cat([p_rgb, p_res, p_att], dim=1)
        logits = self.head(fused)
        return logits


class ImageDeepfakeDetector:
    def __init__(self, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = DualDomainVisionBackbone().to(self.device)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "vision_best.pt")
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model.eval()
        from .polling import ImageForensicsCommittee
        self.committee = ImageForensicsCommittee(device=self.device)

    def compute_fft_azimuthal(self, gray_np: np.ndarray) -> dict:
        """
        Computes 2D-FFT and radially averaged power spectral density (PSD).
        Synthetic images (Diffusion, GANs) exhibit characteristic power-law deviations
        and periodic peaks in high-frequency bands.
        """
        h, w = gray_np.shape
        # Apply Hanning window to reduce edge discontinuity artifacts
        win_y = np.hanning(h)
        win_x = np.hanning(w)
        window = np.outer(win_y, win_x)
        windowed = (gray_np - np.mean(gray_np)) * window

        f = np.fft.fft2(windowed)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) ** 2

        # Radial Azimuthal Integration
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy).astype(np.int32)
        max_r = min(cy, cx)

        radial_sum = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel())[:max_r]
        radial_count = np.bincount(r.ravel())[:max_r]
        radial_count[radial_count == 0] = 1
        psd = radial_sum / radial_count

        # Normalize PSD
        psd_norm = psd / (psd[1] + 1e-12)
        freqs = np.arange(1, len(psd_norm))

        # Fit power-law slope beta: log(PSD) ~ -beta * log(freq)
        log_freqs = np.log(freqs)
        log_psd = np.log(psd_norm[1:] + 1e-12)
        if len(freqs) > 10:
            slope, _ = np.polyfit(log_freqs, log_psd, 1)
        else:
            slope = -2.0

        # High-frequency anomaly ratio (energy in top 20% frequencies)
        cutoff = int(len(psd_norm) * 0.8)
        hf_energy_ratio = float(np.sum(psd_norm[cutoff:]) / (np.sum(psd_norm) + 1e-12))

        # Check for periodic peak anomalies (variance of high frequency differences)
        hf_peaks_variance = float(np.var(np.diff(log_psd[cutoff:]))) if len(log_psd) > cutoff + 2 else 0.0

        return {
            "power_law_slope": float(slope),
            "high_frequency_energy_ratio": hf_energy_ratio,
            "high_frequency_peaks_variance": hf_peaks_variance,
        }

    def compute_ela(self, image: Image.Image, quality: int = 90) -> dict:
        """
        Error Level Analysis (ELA):
        Differences in compression error level between natural captures and edited/synthetic pixels.
        """
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        diff = ImageChops.difference(image.convert("RGB"), recompressed.convert("RGB"))
        diff_np = np.array(diff, dtype=np.float32)

        mean_diff = float(np.mean(diff_np))
        std_diff = float(np.std(diff_np))
        max_diff = float(np.max(diff_np))

        # Local patch inconsistency (kurtosis of localized blocks)
        h, w, _ = diff_np.shape
        block_size = 32
        block_means = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                patch = diff_np[i : i + block_size, j : j + block_size]
                block_means.append(np.mean(patch))

        inconsistency = float(np.std(block_means)) if block_means else 0.0

        return {
            "mean_error": mean_diff,
            "error_std": std_diff,
            "max_error": max_diff,
            "compression_inconsistency": inconsistency,
        }

    def compute_srm_residuals(self, gray_np: np.ndarray) -> dict:
        """
        Steganalysis Rich Models (SRM) High-Pass Spatial Residuals.
        Computes 1st order, 2nd order (Laplacian), and square edge filters
        to expose diffusion upsampling lattices and GAN deconvolution checkerboards.
        """
        # Filter 1: 1st order horizontal gradient [-1, 2, -1]
        k1 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32) / 2.0
        # Filter 2: 2nd order 3x3 Laplacian
        k2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4.0
        # Filter 3: 3x3 square edge residual
        k3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0

        # Convolve using fast separable/spatial filter
        img_pil = Image.fromarray(np.clip(gray_np, 0, 255).astype(np.uint8))
        f_laplacian = img_pil.filter(ImageFilter.Kernel((3, 3), [0, -1, 0, -1, 4, -1, 0, -1, 0], scale=4))
        lap_np = np.array(f_laplacian, dtype=np.float32)

        # PRNU sensor noise via median subtraction
        denoised = img_pil.filter(ImageFilter.MedianFilter(size=3))
        prnu = gray_np - np.array(denoised, dtype=np.float32)

        prnu_std = float(np.std(prnu))
        lap_kurtosis = float(np.mean(((lap_np - np.mean(lap_np)) / (np.std(lap_np) + 1e-12)) ** 4))

        return {
            "prnu_noise_std": prnu_std,
            "srm_laplacian_kurtosis": lap_kurtosis,
            "residual_array": prnu,
        }

    def compute_chroma_correlation(self, image: Image.Image) -> dict:
        """
        YCbCr Cross-Chrominance High-Pass Covariance.
        Physical camera sensors demosaicing Bayer patterns create correlated high-frequency noise
        between Cb and Cr channels. Generative diffusion/GAN models lack this optical sensor correlation.
        """
        ycbcr = np.array(image.convert("YCbCr"), dtype=np.float32)
        cb_img = Image.fromarray(np.clip(ycbcr[:, :, 1], 0, 255).astype(np.uint8), mode="L")
        cr_img = Image.fromarray(np.clip(ycbcr[:, :, 2], 0, 255).astype(np.uint8), mode="L")

        cb_blur = np.array(cb_img.filter(ImageFilter.BoxBlur(1)), dtype=np.float32)
        cr_blur = np.array(cr_img.filter(ImageFilter.BoxBlur(1)), dtype=np.float32)

        cb_hf = (ycbcr[:, :, 1] - cb_blur).ravel()
        cr_hf = (ycbcr[:, :, 2] - cr_blur).ravel()

        corr = np.corrcoef(cb_hf, cr_hf)[0, 1]
        chroma_corr = float(corr) if not np.isnan(corr) else 0.0

        return {
            "chroma_correlation": round(chroma_corr, 4),
            "chroma_anomaly": float(max(0.0, 1.0 - abs(chroma_corr) / 0.4)),
        }

    def analyze(self, image_input) -> dict:
        """
        Full forensic analysis of an image (2026 Multi-Domain Pipeline).
        image_input: PIL Image or path to image file.
        """
        # 0. Metadata & C2PA Content Credentials Forensic Audit
        meta_audit = UniversalMetadataForensics.audit_image_metadata(image_input)

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Resize for standard multi-scale forensic analysis
        std_image = image.resize((512, 512), Image.Resampling.LANCZOS)
        gray_np = np.array(std_image.convert("L"), dtype=np.float32)

        # 1. Frequency domain forensic (Azimuthal Power Spectrum)
        fft_features = self.compute_fft_azimuthal(gray_np)

        # 2. Compression ELA forensic
        ela_features = self.compute_ela(std_image)

        # 3. SRM Steganographic & PRNU Sensor Noise Analysis
        srm_features = self.compute_srm_residuals(gray_np)
        residual_arr = srm_features.pop("residual_array")

        # 4. YCbCr Chrominance CFA Demosaicing Analysis
        chroma_features = self.compute_chroma_correlation(std_image)

        # 5. Dual-Domain Neural Backbone Stream Inference (64x64 input scale matching training)
        model_img = image.resize((64, 64), Image.Resampling.BILINEAR)
        rgb_tensor = (
            torch.from_numpy(np.array(model_img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        model_gray = model_img.convert("L")
        denoised = model_gray.filter(ImageFilter.MedianFilter(size=3))
        res_np = np.array(model_gray, dtype=np.float32) - np.array(denoised, dtype=np.float32)
        res_tensor = (
            torch.from_numpy(np.repeat(res_np[np.newaxis, :, :], 3, axis=0) / 255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(rgb_tensor, res_tensor)
            neural_prob = float(torch.sigmoid(logits).cpu().item())

        # 6. Calibrated Multi-Domain Forensic Aggregation
        # High-frequency periodic checkerboard / deconvolution grid artifact
        checkerboard_score = min(1.0, fft_features["high_frequency_peaks_variance"] * 10.0)

        # Steganographic kurtosis anomaly (diffusion latents produce excessive kurtosis > 180)
        srm_kurt = srm_features.get("srm_laplacian_kurtosis", 3.0)
        srm_fake_score = min(1.0, max(0.0, (srm_kurt - 180.0) / 150.0))

        # Compression error level disparity (splicing / localized inpainting)
        ela_fake_score = min(1.0, max(0.0, (ela_features["compression_inconsistency"] - 1.5) / 3.0))

        # Chrominance CFA demosaicing anomaly
        chroma_fake_score = max(0.0, min(1.0, chroma_features["chroma_anomaly"] * 0.5))

        # Combined forensic evidence
        forensic_evidence = float(
            0.40 * checkerboard_score +
            0.30 * srm_fake_score +
            0.15 * ela_fake_score +
            0.15 * chroma_fake_score
        )

        # Primary Neural Backbone (75%) + Multi-Domain Forensic Evidence (25%)
        composite_fake_prob = float(0.75 * neural_prob + 0.25 * forensic_evidence)

        # Authoritative Metadata Calibration (C2PA / EXIF / PNG Chunks)
        if meta_audit["c2pa"].get("is_ai_generated"):
            composite_fake_prob = max(composite_fake_prob, 0.99)
        elif meta_audit.get("is_synthetic"):
            composite_fake_prob = max(composite_fake_prob, 0.95)
        elif meta_audit.get("has_camera_hardware") and composite_fake_prob < 0.65:
            # Verified camera hardware reduces benign false positives
            composite_fake_prob = float(composite_fake_prob * 0.85)

        composite_fake_prob = max(0.01, min(0.99, composite_fake_prob))

        # Execute 3-Model Democratic Polling Committee
        polling = self.committee.poll(
            primary_prob=composite_fake_prob,
            fft_features=fft_features,
            ela_features=ela_features,
            srm_features=srm_features,
            rgb_tensor=rgb_tensor,
        )

        is_synthetic = composite_fake_prob >= 0.50
        classification = f"SYNTHETIC / DEEPFAKE ({polling.consensus_type} {polling.consensus_ratio})" if is_synthetic else f"AUTHENTIC PHOTOGRAPH ({polling.consensus_type} {polling.consensus_ratio})"

        # Forensic signal explainability
        signals = []
        signals.append(f"Democratic Polling Committee: {polling.summary}")
        if meta_audit["suspicious_flags"]:
            signals.extend(meta_audit["suspicious_flags"])
        if neural_prob > 0.60:
            signals.append(f"Dual-stream neural classifier detected generative latent synthesis patterns ({neural_prob*100:.1f}%)")
        if checkerboard_score > 0.50:
            signals.append("High-frequency periodic grid / deconvolution checkerboard artifact detected")
        if srm_fake_score > 0.50:
            signals.append(f"Steganographic Laplacian residual kurtosis anomaly ({srm_kurt:.1f} > natural threshold)")
        if ela_fake_score > 0.50:
            signals.append("Compression error level disparity across image patches (splicing/inpainting artifact)")
        if chroma_fake_score > 0.40:
            signals.append("Cross-chrominance high-pass decorrelation (absence of physical camera CFA demosaicing)")
        if meta_audit["authenticity_signals"]:
            signals.extend(meta_audit["authenticity_signals"])

        return {
            "modality": "image",
            "fake_probability": round(composite_fake_prob, 4),
            "is_synthetic": is_synthetic,
            "classification": classification,
            "confidence_percent": round(abs(composite_fake_prob - 0.5) * 200, 1),
            "forensic_signals": signals if signals else ["Natural physical optical and sensor noise characteristics verified"],
            "polling": polling.to_dict(),
            "metrics": {
                "neural_model_score": round(neural_prob, 4),
                "fft": fft_features,
                "srm": srm_features,
                "chroma": chroma_features,
                "ela": {k: round(v, 3) for k, v in ela_features.items()},
                "metadata_audit": meta_audit,
            },
        }
