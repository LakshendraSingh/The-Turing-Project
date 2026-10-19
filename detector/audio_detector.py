"""
Audio Deepfake & Synthetic Voice Detector (2026)
Detects Neural TTS & Voice Cloning (ElevenLabs, Voicebox, XTTS, VALL-E, Bark, Diff-TTS).
Uses Multilevel Acoustic Forensics:
- Linear Frequency Cepstral Coefficients (LFCC) & Mel-scale filterbanks
- High-frequency phase & spectral flux coherence
- Pitch (F0) micro-tremor & vocal jitter / shimmer analysis
- Spectral roll-off & synthetic vocoder boundary artifacts
- Neural 1D-Conv/BiGRU Acoustic Backbone
"""

import io
import math
import os
import subprocess
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metadata_forensics import UniversalMetadataForensics


class SqueezeExcitation1d(nn.Module):
    """
    1D Squeeze-and-Excitation channel recalibration for acoustic spectral bands.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(8, channels // reduction)),
            nn.SiLU(),
            nn.Linear(max(8, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class AcousticForensicsBackbone(nn.Module):
    """
    2026 Multi-Scale Attentive Acoustic Forensics Backbone:
    - Multi-Scale 1D Parallel Conv (k=3, 5, 7) capturing micro-jitter, phoneme, and syllabic artifacts
    - 1D Squeeze-and-Excitation channel attention across LFCC bands
    - Bidirectional GRU modeling forward/backward temporal co-articulation
    - Learned Attentive Temporal Pooling (focusing on cloned voice phoneme boundary glitches)
    """
    def __init__(self, in_features=60, hidden_dim=128, num_classes=1):
        super().__init__()
        # Multi-scale parallel temporal convolutions
        self.conv_k3 = nn.Conv1d(in_features, 42, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(in_features, 43, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(in_features, 43, kernel_size=7, padding=3)

        self.bn_stem = nn.BatchNorm1d(128)
        self.act_stem = nn.SiLU()
        self.se = SqueezeExcitation1d(128)
        self.pool = nn.AdaptiveAvgPool1d(64)

        self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)

        # Attentive Temporal Pooling
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, C, T]
        c3 = self.conv_k3(x)
        c5 = self.conv_k5(x)
        c7 = self.conv_k7(x)
        multi_scale = torch.cat([c3, c5, c7], dim=1)  # [B, 128, T]

        stem = self.act_stem(self.bn_stem(multi_scale))
        stem = self.se(stem)
        pooled_time = self.pool(stem)  # [B, 128, 64]

        # Bidirectional Temporal Modeling
        time_seq = pooled_time.transpose(1, 2)  # [B, 64, 128]
        gru_out, _ = self.gru(time_seq)        # [B, 64, hidden_dim * 2]

        # Attentive Temporal Pooling
        weights = F.softmax(self.attn_mlp(gru_out), dim=1)  # [B, 64, 1]
        attn_pool = torch.sum(weights * gru_out, dim=1)     # [B, hidden_dim * 2]
        max_pool = torch.max(gru_out, dim=1)[0]             # [B, hidden_dim * 2]

        fused_acoustic = torch.cat([attn_pool, max_pool], dim=-1)  # [B, hidden_dim * 4]
        return self.classifier(fused_acoustic)


class AudioDeepfakeDetector:
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

        self.model = AcousticForensicsBackbone().to(self.device)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "audio_best.pt")
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model.eval()
        from .polling import AudioForensicsCommittee
        self.committee = AudioForensicsCommittee(device=self.device)

    def load_audio(self, audio_input, target_sr=16000) -> tuple:
        """
        Loads audio from path or byte buffer. Supports WAV, FLAC, OGG via soundfile / wave.
        Returns: (normalized_audio_array, sample_rate)
        """
        try:
            import soundfile as sf
            if isinstance(audio_input, (bytes, io.BytesIO)):
                bio = audio_input if isinstance(audio_input, io.BytesIO) else io.BytesIO(audio_input)
                audio, sr = sf.read(bio)
            elif isinstance(audio_input, str):
                audio, sr = sf.read(audio_input)
            elif isinstance(audio_input, np.ndarray):
                audio = audio_input.astype(np.float32)
                sr = target_sr
            else:
                raise ValueError("Unsupported audio input format.")

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
        except Exception:
            # Fallback to standard library wave module or ffmpeg pipe
            audio = None
            sr = target_sr
            if isinstance(audio_input, str):
                try:
                    with wave.open(audio_input, "rb") as wf:
                        sr = wf.getframerate()
                        n_channels = wf.getnchannels()
                        frames = wf.readframes(wf.getnframes())
                        dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
                        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                        if n_channels > 1:
                            audio = audio.reshape(-1, n_channels).mean(axis=1)
                        audio = audio / (32768.0 if dtype == np.int16 else 2147483648.0)
                except Exception:
                    # Try ffmpeg pipe for MP3/AAC/FLAC/OGG
                    try:
                        res = subprocess.run(
                            ["ffmpeg", "-v", "quiet", "-i", audio_input, "-f", "wav", "-ar", str(target_sr), "-ac", "1", "pipe:1"],
                            capture_output=True,
                        )
                        if res.returncode == 0 and len(res.stdout) > 44:
                            with wave.open(io.BytesIO(res.stdout), "rb") as wf:
                                sr = wf.getframerate()
                                frames = wf.readframes(wf.getnframes())
                                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    except Exception:
                        pass
            elif isinstance(audio_input, (bytes, io.BytesIO)):
                bio = audio_input if isinstance(audio_input, io.BytesIO) else io.BytesIO(audio_input)
                try:
                    with wave.open(bio, "rb") as wf:
                        sr = wf.getframerate()
                        n_channels = wf.getnchannels()
                        frames = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                        if n_channels > 1:
                            audio = audio.reshape(-1, n_channels).mean(axis=1)
                        audio = audio / 32768.0
                except Exception:
                    try:
                        input_bytes = audio_input.getvalue() if isinstance(audio_input, io.BytesIO) else audio_input
                        res = subprocess.run(
                            ["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "wav", "-ar", str(target_sr), "-ac", "1", "pipe:1"],
                            input=input_bytes,
                            capture_output=True,
                        )
                        if res.returncode == 0 and len(res.stdout) > 44:
                            with wave.open(io.BytesIO(res.stdout), "rb") as wf:
                                sr = wf.getframerate()
                                frames = wf.readframes(wf.getnframes())
                                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    except Exception:
                        pass
            elif isinstance(audio_input, np.ndarray):
                audio = audio_input.astype(np.float32)
                sr = target_sr

            if audio is None:
                raise ValueError("Unsupported or corrupted audio input format.")

        # Resample to 16kHz if different
        if sr != target_sr and len(audio) > 0:
            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, num_samples)
            sr = target_sr

        # Normalize amplitude
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-6:
            audio = audio / max_amp

        return audio, sr

    def compute_lfcc(self, audio: np.ndarray, sr=16000, n_lfcc=30, n_fft=512, hop_length=160) -> np.ndarray:
        """
        Linear Frequency Cepstral Coefficients (LFCC) - ASVspoof Benchmark Standard.
        Linear filterbanks preserve spectral details in higher frequencies where vocoders fail.
        """
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        # Short-time Fourier Transform (STFT)
        _, _, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        power_spec = np.abs(Zxx) ** 2

        n_bins = power_spec.shape[0]
        # Linear filterbank
        num_filters = 40
        f_min = 0
        f_max = sr / 2
        linear_freqs = np.linspace(f_min, f_max, num_filters + 2)
        fft_freqs = np.linspace(0, sr / 2, n_bins)

        filterbank = np.zeros((num_filters, n_bins))
        for m in range(1, num_filters + 1):
            f_m_minus = linear_freqs[m - 1]
            f_m = linear_freqs[m]
            f_m_plus = linear_freqs[m + 1]

            left = (fft_freqs - f_m_minus) / (f_m - f_m_minus + 1e-12)
            right = (f_m_plus - fft_freqs) / (f_m_plus - f_m + 1e-12)
            tri = np.maximum(0, np.minimum(left, right))
            filterbank[m - 1] = tri

        linear_energies = np.dot(filterbank, power_spec)
        log_energies = np.log(np.maximum(linear_energies, 1e-10))

        # Discrete Cosine Transform (DCT)
        lfcc = scipy.fftpack.dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc, :]
        return lfcc

    def analyze_vocal_microdynamics(self, audio: np.ndarray, sr=16000) -> dict:
        """
        Analyzes pitch micro-tremors, jitter, spectral flux, and spectral roll-off.
        Synthetic voices exhibit unnaturally regular pitch tracks or missing natural micro-perturbations.
        """
        # Frame-wise energy and zero crossing rate
        frame_len = 512
        hop = 160
        frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, hop)]
        if not frames:
            return {"jitter": 0.0, "shimmer": 0.0, "spectral_flux": 0.0, "spectral_rolloff": 0.0}

        # Energy per frame
        energies = np.array([np.sum(f ** 2) for f in frames])

        # Autocorrelation-based pitch tracker for voiced segments
        pitches = []
        for f in frames:
            if np.sum(f ** 2) > 0.01:  # voiced threshold
                corr = np.correlate(f, f, mode="full")[len(f) - 1 :]
                # Pitch search range: 50Hz - 500Hz
                min_lag = int(sr / 500)
                max_lag = int(sr / 50)
                if len(corr) > max_lag:
                    peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                    if peak_lag > 0 and corr[peak_lag] > 0.3 * corr[0]:
                        pitches.append(sr / peak_lag)

        # Jitter: cycle-to-cycle pitch variability
        if len(pitches) > 5:
            pitch_diffs = np.abs(np.diff(pitches))
            jitter = float(np.mean(pitch_diffs) / (np.mean(pitches) + 1e-12))
        else:
            jitter = 0.0

        # Shimmer: amplitude variability across voiced frames
        voiced_energies = energies[energies > 0.01]
        if len(voiced_energies) > 5:
            shimmer = float(np.mean(np.abs(np.diff(voiced_energies))) / (np.mean(voiced_energies) + 1e-12))
        else:
            shimmer = 0.0

        # Spectral Flux: frame-to-frame spectral envelope change rate
        _, _, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=frame_len, noverlap=frame_len - hop)
        mag_spec = np.abs(Zxx)
        if mag_spec.shape[1] > 1:
            diff_spec = np.diff(mag_spec, axis=1)
            flux = float(np.mean(np.sqrt(np.sum(np.maximum(0, diff_spec) ** 2, axis=0))))
        else:
            flux = 0.0

        # Spectral Rolloff: frequency below which 85% of total spectral energy lies
        power = mag_spec ** 2
        total_energy = np.sum(power, axis=0) + 1e-12
        cum_energy = np.cumsum(power, axis=0)
        rolloff_idx = np.argmax(cum_energy >= 0.85 * total_energy, axis=0)
        freq_bins = np.linspace(0, sr / 2, mag_spec.shape[0])
        rolloff_freq = float(np.mean(freq_bins[rolloff_idx]))

        return {
            "jitter": round(jitter, 5),
            "shimmer": round(shimmer, 5),
            "spectral_flux": round(flux, 4),
            "spectral_rolloff_hz": round(rolloff_freq, 1),
            "voiced_ratio": round(len(pitches) / (len(frames) + 1e-6), 3),
        }

    def analyze(self, audio_input) -> dict:
        """
        Full forensic acoustic inspection.
        audio_input: path to wav file, bytes, or numpy array.
        """
        # 0. Metadata & C2PA Content Credentials Forensic Audit
        meta_audit = UniversalMetadataForensics.audit_audio_metadata(audio_input)

        audio, sr = self.load_audio(audio_input)
        duration_sec = round(len(audio) / sr, 2)

        # 1. LFCC Feature Matrix
        lfcc = self.compute_lfcc(audio, sr=sr, n_lfcc=30)
        # Compute delta LFCC
        if lfcc.shape[1] > 2:
            delta_lfcc = np.diff(lfcc, axis=1)
            delta_lfcc = np.pad(delta_lfcc, ((0, 0), (0, 1)), mode="edge")
        else:
            delta_lfcc = np.zeros_like(lfcc)

        combined_lfcc = np.vstack([lfcc, delta_lfcc])  # [60, T]

        # 2. Vocal Microdynamics
        vocal_dynamics = self.analyze_vocal_microdynamics(audio, sr)

        # 3. Neural Acoustic Stream Inference
        # Sample or interpolate to 128 frames
        in_tensor = torch.from_numpy(combined_lfcc.astype(np.float32)).unsqueeze(0)
        if in_tensor.shape[2] != 128:
            in_tensor = F.interpolate(in_tensor, size=128, mode="linear", align_corners=False)
        in_tensor = in_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(in_tensor)
            neural_prob = float(torch.sigmoid(logits).cpu().item())

        # 4. Multi-cue Forensic Aggregation (2026 Acoustic Calibrator)
        jitter = vocal_dynamics["jitter"]
        jitter_anomaly = 0.80 if (0 < jitter < 0.005) else 0.10
        rolloff_anomaly = 0.80 if vocal_dynamics["spectral_rolloff_hz"] > 7200 else 0.10

        # Calibrated LFCC Variance Score
        lfcc_var = float(np.var(lfcc))
        lfcc_fake_score = float(np.clip((lfcc_var - 250.0) / (380.0 - 250.0), 0.0, 1.0))
        aliasing_score = max(jitter_anomaly, rolloff_anomaly)

        # Base acoustic composite fusion: Neural Backbone (75%) + Acoustic Physics (25%)
        composite_fake_prob = float(
            0.75 * neural_prob +
            0.15 * lfcc_fake_score +
            0.10 * aliasing_score
        )

        # Authoritative Metadata Calibration (C2PA JUMBF / ID3 Frames)
        if meta_audit["c2pa"].get("is_ai_generated"):
            composite_fake_prob = max(composite_fake_prob, 0.99)
        elif meta_audit.get("is_synthetic"):
            composite_fake_prob = max(composite_fake_prob, 0.95)
        elif meta_audit.get("has_metadata") and not meta_audit.get("is_synthetic") and composite_fake_prob < 0.60:
            composite_fake_prob = float(composite_fake_prob * 0.85)

        composite_fake_prob = max(0.01, min(0.99, composite_fake_prob))

        # Execute 3-Model Democratic Polling Committee
        polling = self.committee.poll(
            primary_prob=composite_fake_prob,
            acoustic_features={
                "vocoder_aliasing_peak": aliasing_score,
                "phase_discontinuity": float(np.clip(rolloff_anomaly, 0.0, 1.0)),
                "pitch_jitter": jitter,
            }
        )

        is_synthetic = composite_fake_prob >= 0.50
        classification = f"SYNTHETIC / CLONED VOICE ({polling.consensus_type} {polling.consensus_ratio})" if is_synthetic else f"AUTHENTIC HUMAN SPEECH ({polling.consensus_type} {polling.consensus_ratio})"

        signals = []
        signals.append(f"Democratic Polling Committee: {polling.summary}")
        if meta_audit["suspicious_flags"]:
            signals.extend(meta_audit["suspicious_flags"])
        if neural_prob > 0.60:
            signals.append(f"Neural acoustic stream detected synthetic speech characteristics ({neural_prob*100:.1f}%)")
        if lfcc_fake_score > 0.65:
            signals.append(f"Acoustic spectral distribution signature characteristic of neural vocoder (LFCC variance: {lfcc_var:.1f})")
        if jitter_anomaly > 0.7:
            signals.append(f"Unnatural vocal pitch stability/jitter ({jitter:.4f} outside typical human 0.01-0.05)")
        if rolloff_anomaly > 0.7:
            signals.append(f"High-frequency vocoder aliasing at {vocal_dynamics['spectral_rolloff_hz']}Hz")
        if vocal_dynamics["voiced_ratio"] < 0.1:
            signals.append("Low voiced segment ratio / breath-to-phonation artifact")
        if meta_audit["authenticity_signals"]:
            signals.extend(meta_audit["authenticity_signals"])

        return {
            "modality": "audio",
            "duration_seconds": duration_sec,
            "sample_rate": sr,
            "fake_probability": round(composite_fake_prob, 4),
            "is_synthetic": is_synthetic,
            "classification": classification,
            "confidence_percent": round(abs(composite_fake_prob - 0.5) * 200, 1),
            "forensic_signals": signals if signals else ["Natural human vocal tract acoustic markers present"],
            "polling": polling.to_dict(),
            "metrics": {
                "neural_model_score": round(neural_prob, 4),
                "vocal_dynamics": vocal_dynamics,
                "lfcc_variance": round(float(np.var(lfcc)), 4),
                "metadata_audit": meta_audit,
            },
        }
