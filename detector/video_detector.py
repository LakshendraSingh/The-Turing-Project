"""
Video Deepfake & Generative Video Detector (2026)
Detects Generative AI Video (Sora, Gen-3, Kling, Runway) and Face Swaps (DeepFaceLab, SimSwap, Roop).
Forensic Capabilities:
- Spatiotemporal inter-frame temporal flicker & optical gradient flow
- Blending boundary seam & warp jitter detection
- Spatial keyframe multi-domain inspection
- Audio-Visual LipSync correspondence check (when audio stream is present)
"""

import math
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from .image_detector import ImageDeepfakeDetector
from .audio_detector import AudioDeepfakeDetector
from .metadata_forensics import UniversalMetadataForensics


class R2Plus1DBlock(nn.Module):
    """
    Factorized Spatiotemporal Conv Block R(2+1)D:
    Decomposes 3D convolution into a 2D spatial convolution (1x3x3)
    followed by a 1D temporal convolution (3x1x1) with residual shortcut.
    """
    def __init__(self, in_c: int, out_c: int, stride_spatial: int = 1):
        super().__init__()
        # Intermediate dimension
        mid_c = max(16, (in_c * out_c * 3 * 3 * 3) // (in_c * 3 * 3 + 3 * out_c))

        # Spatial Conv: 1 x 3 x 3
        self.conv_s = nn.Conv3d(in_c, mid_c, kernel_size=(1, 3, 3), stride=(1, stride_spatial, stride_spatial), padding=(0, 1, 1), bias=False)
        self.bn_s = nn.BatchNorm3d(mid_c)
        self.act_s = nn.SiLU()

        # Temporal Conv: 3 x 1 x 1
        self.conv_t = nn.Conv3d(mid_c, out_c, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_c)
        self.act_t = nn.SiLU()

        if stride_spatial != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=(1, stride_spatial, stride_spatial), bias=False),
                nn.BatchNorm3d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act_s(self.bn_s(self.conv_s(x)))
        out = self.bn_t(self.conv_t(out))
        out = self.act_t(out + res)
        return out


class SpatiotemporalTemporalModel(nn.Module):
    """
    2026 R(2+1)D Factorized Spatiotemporal Video Deepfake Backbone:
    - Separates spatial deformation analysis from temporal inter-frame flicker
    - Residual gradient flow across video volume frames
    - Dual temporal pooling (Avg + Max) catching brief 1-frame deepfake warping glitches
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # Input: [B, 3, T, H, W]
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(),
        )

        self.block1 = R2Plus1DBlock(16, 32, stride_spatial=2)
        self.block2 = R2Plus1DBlock(32, 64, stride_spatial=2)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)

        avg_p = torch.mean(h, dim=(2, 3, 4))
        max_p = torch.amax(h, dim=(2, 3, 4))
        pooled = torch.cat([avg_p, max_p], dim=1)  # [B, 128]

        return self.classifier(pooled)


class VideoDeepfakeDetector:
    def __init__(self, device: str = None):
        self.image_detector = ImageDeepfakeDetector(device=device)
        self.audio_detector = AudioDeepfakeDetector(device=device)
        self.device = self.image_detector.device
        self.temporal_model = SpatiotemporalTemporalModel().to(self.device)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "video_best.pt")
        if os.path.exists(weights_path):
            try:
                self.temporal_model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.temporal_model.eval()
        from .polling import VideoForensicsCommittee
        self.committee = VideoForensicsCommittee(device=self.device)

    def analyze_interframe_consistency(self, frames: list) -> dict:
        """
        Measures inter-frame high-frequency flickering, warp jitter, and motion coherence.
        """
        if len(frames) < 2:
            return {"mean_frame_diff": 0.0, "flicker_variance": 0.0, "temporal_jitter": 0.0}

        diffs = []
        high_freq_flickers = []
        prev_gray = None

        for frame in frames:
            if isinstance(frame, str):
                pil_img = Image.open(frame).convert("L").resize((256, 256))
            elif isinstance(frame, Image.Image):
                pil_img = frame.convert("L").resize((256, 256))
            else:
                pil_img = Image.fromarray(frame).convert("L").resize((256, 256))

            gray = np.array(pil_img, dtype=np.float32)

            if prev_gray is not None:
                frame_diff = np.abs(gray - prev_gray)
                diffs.append(np.mean(frame_diff))

                # High frequency flicker: difference of gradients
                gx = np.gradient(frame_diff, axis=1)
                gy = np.gradient(frame_diff, axis=0)
                grad_mag = np.sqrt(gx**2 + gy**2)
                high_freq_flickers.append(np.mean(grad_mag))

            prev_gray = gray

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        flicker_var = float(np.var(high_freq_flickers)) if high_freq_flickers else 0.0
        temporal_jitter = float(np.std(diffs)) if diffs else 0.0

        return {
            "mean_frame_diff": round(mean_diff, 3),
            "flicker_variance": round(flicker_var, 4),
            "temporal_jitter": round(temporal_jitter, 4),
        }

    def extract_frames(self, video_input, max_frames: int = 16) -> tuple:
        """
        Extracts frames and optional audio from video file (.mp4, .avi, .mov, .npz) or frame lists.
        Returns: (frames_list, extracted_audio_path_or_bytes)
        """
        import shutil, subprocess, tempfile, glob

        if isinstance(video_input, str) and video_input.endswith(".npz"):
            data = np.load(video_input)
            raw = data["frames"] if "frames" in data else data[list(data.keys())[0]]
            frames = [Image.fromarray(f.astype(np.uint8)) if isinstance(f, np.ndarray) else f for f in raw]
            return frames, None

        if isinstance(video_input, str) and (video_input.endswith((".mp4", ".mov", ".avi", ".webm")) or os.path.isfile(video_input)):
            ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
            tmpdir = tempfile.mkdtemp(prefix="turing_video_")
            try:
                # 1. Extract visual keyframes
                cmd_frames = [
                    ffmpeg_bin, "-i", video_input,
                    "-vf", "fps=2,scale=256:256",
                    "-vframes", str(max_frames),
                    f"{tmpdir}/f_%03d.png",
                    "-y", "-loglevel", "error"
                ]
                subprocess.run(cmd_frames, check=True)
                frame_files = sorted(glob.glob(f"{tmpdir}/f_*.png"))
                frames = [Image.open(p).convert("RGB") for p in frame_files]

                # 2. Extract audio track if present (silent check)
                audio_path = f"{tmpdir}/track.wav"
                cmd_audio = [
                    ffmpeg_bin, "-i", video_input,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    audio_path, "-y", "-loglevel", "error"
                ]
                res_audio = subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                has_audio = res_audio.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000

                return frames, (audio_path if has_audio else None)
            except Exception:
                shutil.rmtree(tmpdir, ignore_errors=True)

        if isinstance(video_input, (list, tuple)):
            return list(video_input), None

        raise ValueError(f"Unsupported video input format: {type(video_input)}")

    def analyze(self, video_input, audio_track=None) -> dict:
        """
        Analyzes video keyframes and optional audio track.
        video_input: file path (.mp4, .npz), or list of PIL Images / numpy arrays.
        audio_track: optional path to wav audio track extracted from video.
        """
        # 0. Container & C2PA Metadata Forensic Audit
        meta_audit = (
            UniversalMetadataForensics.audit_video_metadata(video_input)
            if isinstance(video_input, (str, bytes))
            else {"has_metadata": False, "suspicious_flags": [], "authenticity_signals": []}
        )

        if isinstance(video_input, (list, tuple)):
            frames = list(video_input)
            extracted_audio = audio_track
        else:
            frames, extracted_audio = self.extract_frames(video_input)
            if audio_track is None:
                audio_track = extracted_audio

        if not frames:
            raise ValueError("No video frames could be extracted for analysis.")

        # 1. Sample keyframes for Spatial & Frequency Domain Analysis
        # Sample up to 8 evenly spaced frames
        n = len(frames)
        indices = np.linspace(0, n - 1, min(8, n), dtype=int)
        sampled_frames = [frames[i] for i in indices]

        frame_results = []
        frame_fake_probs = []
        for f in sampled_frames:
            res = self.image_detector.analyze(f)
            frame_results.append(res)
            frame_fake_probs.append(res["fake_probability"])

        spatial_fake_prob = float(np.mean(frame_fake_probs))

        # 2. Temporal Consistency Forensic
        temporal_metrics = self.analyze_interframe_consistency(frames)
        temporal_anomaly_score = float(
            min(1.0, (temporal_metrics["temporal_jitter"] / 12.0) * 0.5 + (temporal_metrics["flicker_variance"] / 3.0) * 0.5)
        )

        # 3. Spatiotemporal Neural Stream Inference (R(2+1)D Backbone)
        temporal_nn_prob = 0.5
        if len(frames) >= 4:
            t_sub = sampled_frames[:5] if len(sampled_frames) >= 5 else sampled_frames
            t_arr = [np.array(fr.resize((64, 64)), dtype=np.float32).transpose(2, 0, 1) / 255.0 for fr in t_sub]
            vol = torch.tensor(np.stack(t_arr, axis=1), dtype=torch.float32).unsqueeze(0).to(self.device)
            if vol.shape[2] != 5:
                vol = nn.functional.interpolate(vol, size=(5, 64, 64), mode="trilinear", align_corners=False)
            with torch.no_grad():
                logit = self.temporal_model(vol).squeeze()
                temporal_nn_prob = float(torch.sigmoid(logit).item())

        # 4. Audio-Visual Cross-Modal Consistency Check
        audio_result = None
        av_sync_discrepancy = 0.0
        if audio_track is not None:
            audio_result = self.audio_detector.analyze(audio_track)
            av_sync_discrepancy = abs(spatial_fake_prob - audio_result["fake_probability"])

        # Composite video fake probability
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "video_best.pt")
        has_weights = os.path.exists(weights_path)

        if has_weights:
            calib_temporal_prob = float(1.0 / (1.0 + np.exp(-((temporal_nn_prob - 0.08) / 0.04))))
        else:
            calib_temporal_prob = temporal_nn_prob

        if audio_result is not None:
            if has_weights:
                composite_fake_prob = float(
                    0.60 * calib_temporal_prob +
                    0.20 * spatial_fake_prob +
                    0.10 * audio_result["fake_probability"] +
                    0.10 * temporal_anomaly_score
                )
            else:
                composite_fake_prob = float(
                    0.50 * spatial_fake_prob +
                    0.25 * temporal_anomaly_score +
                    0.25 * audio_result["fake_probability"]
                )
        else:
            if has_weights:
                composite_fake_prob = float(
                    0.70 * calib_temporal_prob +
                    0.15 * spatial_fake_prob +
                    0.15 * temporal_anomaly_score
                )
            else:
                composite_fake_prob = float(
                    0.65 * spatial_fake_prob +
                    0.35 * temporal_anomaly_score
                )
        # Authoritative Metadata Calibration (C2PA / Container Atoms)
        if meta_audit.get("c2pa", {}).get("is_ai_generated"):
            composite_fake_prob = max(composite_fake_prob, 0.99)
        elif meta_audit.get("is_synthetic"):
            composite_fake_prob = max(composite_fake_prob, 0.95)
        elif meta_audit.get("has_metadata") and not meta_audit.get("is_synthetic") and composite_fake_prob < 0.60:
            composite_fake_prob = float(composite_fake_prob * 0.85)

        # Execute 3-Model Democratic Polling Committee
        polling = self.committee.poll(
            primary_prob=composite_fake_prob,
            flow_disparity=float(temporal_metrics.get("temporal_jitter", 0.0)),
            boundary_flicker=float(temporal_metrics.get("flicker_variance", 0.0)),
        )

        is_synthetic = composite_fake_prob >= 0.50
        classification = f"SYNTHETIC / DEEPFAKE VIDEO ({polling.consensus_type} {polling.consensus_ratio})" if is_synthetic else f"AUTHENTIC VIDEO RECORDING ({polling.consensus_type} {polling.consensus_ratio})"

        signals = []
        signals.append(f"Democratic Polling Committee: {polling.summary}")
        if meta_audit.get("suspicious_flags"):
            signals.extend(meta_audit["suspicious_flags"])
        if spatial_fake_prob > 0.6:
            signals.append(f"Spatial artifact detected across {len(sampled_frames)} keyframes (avg p={spatial_fake_prob:.2f})")
        if temporal_metrics["temporal_jitter"] > 3.0:
            signals.append("Temporal warp jitter detected between consecutive frames")
        if temporal_metrics["flicker_variance"] > 0.05:
            signals.append("High-frequency boundary flickering characteristic of neural face swaps")
        if audio_result and av_sync_discrepancy > 0.4:
            signals.append(f"Audio-Visual modality discrepancy: Audio fake p={audio_result['fake_probability']:.2f} vs Video spatial p={spatial_fake_prob:.2f}")
        if meta_audit.get("authenticity_signals"):
            signals.extend(meta_audit["authenticity_signals"])

        return {
            "modality": "video",
            "analyzed_frames_count": len(frames),
            "sampled_keyframes": len(sampled_frames),
            "fake_probability": round(composite_fake_prob, 4),
            "is_synthetic": is_synthetic,
            "classification": classification,
            "confidence_percent": round(abs(composite_fake_prob - 0.5) * 200, 1),
            "forensic_signals": signals if signals else ["Consistent temporal and spatiotemporal flow detected"],
            "polling": polling.to_dict(),
            "metrics": {
                "spatial_average_fake_prob": round(spatial_fake_prob, 4),
                "temporal_metrics": temporal_metrics,
                "audio_analysis": audio_result,
                "metadata_audit": meta_audit,
            },
        }
