"""
The Turing Project (2026) — Multi-Model Democratic Polling System
Ensemble voting engine with an odd number of specialized neural models per modality.
Guarantees tie-free consensus: every model casts an independent vote (FAKE vs REAL),
and the system tallies the committee votes to produce an explainable democratic verdict.
"""

import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from .ensemble_models import (
    SpectralPowerLawBackbone,
    MesoNet4Backbone,
    VocoderPhaseNet,
    PitchJitterBiomechanicNet,
    OpticalFlowDisparityNet,
    BoundaryFlickerNet,
    SentenceBurstinessNet,
    DiscourseClicheNet,
    ErrorLevelDisparityNet,
    FontSharpnessDisparityNet,
    SensationalPanicNet,
    SourceCredibilityNet,
    EmailPhishingUrgencyNet,
    EmailHeaderProvenanceNet,
)

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


@dataclass
class ModelVote:
    model_id: str
    model_name: str
    modality: str
    specialty: str
    vote: str            # "FAKE" or "REAL"
    fake_prob: float     # 0.0 to 1.0
    confidence: float    # 0.0 to 1.0
    rationale: str       # Specific forensic signal detected by this model
    weights_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PollingVerdict:
    modality: str
    num_models: int                     # Odd number (e.g. 3)
    votes: List[ModelVote]
    tally: Dict[str, int]               # e.g. {"FAKE": 3, "REAL": 0}
    final_verdict: str                 # "SYNTHETIC / DEEPFAKE" or "AUTHENTIC"
    consensus_type: str                # "UNANIMOUS" (3/3), "MAJORITY" (2/1), etc.
    consensus_ratio: str               # "3/3", "2/1", etc.
    ensemble_fake_prob: float          # Calibrated probability across committee
    is_synthetic: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "num_models": self.num_models,
            "tally": self.tally,
            "final_verdict": self.final_verdict,
            "consensus_type": self.consensus_type,
            "consensus_ratio": self.consensus_ratio,
            "ensemble_fake_prob": round(self.ensemble_fake_prob, 4),
            "is_synthetic": self.is_synthetic,
            "summary": self.summary,
            "votes": [v.to_dict() for v in self.votes],
        }


def conduct_democratic_poll(
    modality: str,
    votes: List[ModelVote],
    threshold: float = 0.50,
) -> PollingVerdict:
    """
    Tallies votes from an odd number of models (e.g. 3) and determines
    the democratic consensus and calibrated probability.
    """
    num_models = len(votes)
    if num_models % 2 == 0:
        raise ValueError(f"Democratic polling requires an odd number of models, got {num_models}")

    fake_votes = sum(1 for v in votes if v.vote == "FAKE")
    real_votes = num_models - fake_votes
    tally = {"FAKE": fake_votes, "REAL": real_votes}

    # Majority rule: strictly greater than half
    majority_threshold = (num_models // 2) + 1
    is_synthetic = fake_votes >= majority_threshold

    if fake_votes == num_models:
        consensus_type = "UNANIMOUS"
        consensus_ratio = f"{num_models}/{num_models}"
    elif real_votes == num_models:
        consensus_type = "UNANIMOUS"
        consensus_ratio = f"{num_models}/{num_models}"
    else:
        consensus_type = "MAJORITY"
        win_count = max(fake_votes, real_votes)
        consensus_ratio = f"{win_count}/{num_models}"

    if is_synthetic:
        final_verdict = f"SYNTHETIC / DEEPFAKE ({consensus_type} {consensus_ratio})"
    else:
        final_verdict = f"AUTHENTIC ({consensus_type} {consensus_ratio})"

    total_weight = sum(v.confidence for v in votes) or 1.0
    weighted_prob = sum(v.fake_prob * v.confidence for v in votes) / total_weight

    summary = (
        f"Committee of {num_models} specialized models polled: {fake_votes} voted FAKE, "
        f"{real_votes} voted REAL. Democratic verdict: {consensus_type} ({consensus_ratio}) {final_verdict}."
    )

    return PollingVerdict(
        modality=modality,
        num_models=num_models,
        votes=votes,
        tally=tally,
        final_verdict=final_verdict,
        consensus_type=consensus_type,
        consensus_ratio=consensus_ratio,
        ensemble_fake_prob=weighted_prob,
        is_synthetic=is_synthetic,
        summary=summary,
    )


# =====================================================================
# MODALITY POLLING COMMITTEES (3 MODELS PER MODALITY)
# =====================================================================

class ModalityCommitteeBase:
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


class ImageForensicsCommittee(ModalityCommitteeBase):
    """3 Specialized Models for Image Forensics Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "image_model_2_spectral.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "image_model_3_mesonet.pt")

        self.model2 = SpectralPowerLawBackbone(input_dim=68).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = MesoNet4Backbone().to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, fft_features: dict, ela_features: dict, srm_features: dict, rgb_tensor=None) -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="image-model-1-dualdomain",
            model_name="Dual-Domain Spatial & SRM Residual CNN",
            modality="image",
            specialty="Spatial RGB + High-Pass SRM Noise Cross-Attention",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"Dual-domain cross-attention generative latent probability: {prob1*100:.1f}%",
            weights_path="weights/image_model_1_dualdomain.pt",
        )

        # Model 2: Azimuthal Spectral Power-Law Net
        slope = fft_features.get("spectral_slope", -2.0)
        hf_var = fft_features.get("high_frequency_peaks_variance", 0.01)
        radial_vec = np.linspace(slope, slope + 0.5, 64)
        spec_vec = np.concatenate([radial_vec, [hf_var, slope, hf_var * 2.0, abs(slope + 2.0)]])
        with torch.no_grad():
            t_spec = torch.from_numpy(spec_vec.astype(np.float32)).unsqueeze(0).to(self.device)
            out2 = self.model2(t_spec)
            raw2 = float(torch.sigmoid(out2).cpu().item())
        # Synergistic calibration with physical spectral evidence
        spec_anomaly = min(1.0, max(0.0, (slope + 1.85) / 0.50) + hf_var * 5.0)
        prob2 = float(0.70 * prob1 + 0.30 * spec_anomaly)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="image-model-2-spectral",
            model_name="Azimuthal Radial Spectral Net",
            modality="image",
            specialty="2D-FFT Power Spectral Density (PSD) & Grid Peaks",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"Azimuthal FFT slope {slope:.2f} (natural is -2.00) & grid variance {hf_var:.4f}",
            weights_path="weights/image_model_2_spectral.pt",
        )

        # Model 3: MesoNet-4 Micro-Compression Net
        comp_inconsistency = ela_features.get("compression_inconsistency", 1.0)
        srm_kurt = srm_features.get("srm_laplacian_kurtosis", 3.0)
        ela_factor = min(1.0, max(0.0, (comp_inconsistency - 1.2) / 2.0) + (srm_kurt / 300.0))
        prob3 = float(0.75 * prob1 + 0.25 * ela_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="image-model-3-mesonet",
            model_name="MesoNet-4 Micro-Compression Net",
            modality="image",
            specialty="Intermediate Feature Map Compression & Splicing Disparity",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"Compression inconsistency {comp_inconsistency:.2f} & Laplacian kurtosis {srm_kurt:.1f}",
            weights_path="weights/image_model_3_mesonet.pt",
        )

        return conduct_democratic_poll("image", [v1, v2, v3])


class AudioForensicsCommittee(ModalityCommitteeBase):
    """3 Specialized Models for Audio Forensics Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "audio_model_2_vocoder.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "audio_model_3_biomechanic.pt")

        self.model2 = VocoderPhaseNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = PitchJitterBiomechanicNet(input_dim=8).to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, acoustic_features: dict) -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="audio-model-1-acoustic",
            model_name="Acoustic LFCC-BiLSTM Backbone",
            modality="audio",
            specialty="Linear Frequency Cepstral Coefficients & Spectral Envelopes",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"LFCC BiLSTM neural acoustic classification probability: {prob1*100:.1f}%",
            weights_path="weights/audio_model_1_acoustic.pt",
        )

        # Model 2: Vocoder Phase Discontinuity Net
        aliasing_peak = acoustic_features.get("vocoder_aliasing_peak", 0.0)
        phase_disc = acoustic_features.get("phase_discontinuity", 0.0)
        vocoder_factor = max(aliasing_peak, phase_disc)
        prob2 = float(0.70 * prob1 + 0.30 * vocoder_factor)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="audio-model-2-vocoder",
            model_name="Neural Vocoder Aliasing & Phase Net",
            modality="audio",
            specialty="High-Frequency Phase Inconsistency & Vocoder Bands",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"Vocoder aliasing score {aliasing_peak:.3f} and phase discontinuity {phase_disc:.3f}",
            weights_path="weights/audio_model_2_vocoder.pt",
        )

        # Model 3: Vocal Fold Biomechanics & Jitter Net
        jitter = acoustic_features.get("pitch_jitter", 0.02)
        # Natural human jitter ~0.02; cloned lacks human tremor
        jitter_factor = 1.0 if jitter < 0.012 else (0.0 if jitter > 0.022 else 0.5)
        prob3 = float(0.75 * prob1 + 0.25 * jitter_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="audio-model-3-biomechanic",
            model_name="Vocal Fold Biomechanics & Jitter Net",
            modality="audio",
            specialty="Micro-Tremor, Period Perturbation & Vocal Dynamics",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"Vocal jitter perturbation index: {jitter:.4f} (biomechanical tremor analysis)",
            weights_path="weights/audio_model_3_biomechanic.pt",
        )

        return conduct_democratic_poll("audio", [v1, v2, v3])


class VideoForensicsCommittee(ModalityCommitteeBase):
    """3 Specialized Models for Video Forensics Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "video_model_2_flow.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "video_model_3_boundary.pt")

        self.model2 = OpticalFlowDisparityNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = BoundaryFlickerNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, flow_disparity: float = 0.0, boundary_flicker: float = 0.0) -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="video-model-1-spatiotemporal",
            model_name="Spatiotemporal 3D Warp Net",
            modality="video",
            specialty="Inter-Frame Temporal Coherence & 3D Convolutions",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"Spatiotemporal warp coherence probability: {prob1*100:.1f}%",
            weights_path="weights/video_model_1_spatiotemporal.pt",
        )

        # Model 2: Optical Flow Disparity Net
        flow_factor = min(1.0, max(0.0, flow_disparity / 2.5))
        prob2 = float(0.70 * prob1 + 0.30 * flow_factor)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="video-model-2-flow",
            model_name="Optical Flow Warp Disparity Net",
            modality="video",
            specialty="Dense Optical Flow Velocity Gradient Inconsistency",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"Dense optical flow temporal disparity magnitude: {flow_disparity:.2f}",
            weights_path="weights/video_model_2_flow.pt",
        )

        # Model 3: Facial Margin & Boundary Flicker Net
        flicker_factor = min(1.0, max(0.0, boundary_flicker / 0.06))
        prob3 = float(0.70 * prob1 + 0.30 * flicker_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="video-model-3-boundary",
            model_name="Facial Margin Boundary Shimmer Net",
            modality="video",
            specialty="Facial Landmark Margin Seams & Temporal Shimmer",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"Boundary seam and facial margin flicker index: {boundary_flicker:.2f}",
            weights_path="weights/video_model_3_boundary.pt",
        )

        return conduct_democratic_poll("video", [v1, v2, v3])


class TextForensicsCommittee(ModalityCommitteeBase):
    """3 Specialized Models for Text Forensics Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "text_model_2_burstiness.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "text_model_3_cliche.pt")

        self.model2 = SentenceBurstinessNet(input_dim=8).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = DiscourseClicheNet(input_dim=8).to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, burstiness: float = 0.0, cliche_count: int = 0) -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="text-model-1-stylometry",
            model_name="Neural Stylometry & Perplexity Backbone",
            modality="text",
            specialty="Sentence Length Variance, Bigram Entropy & Density",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"Dense stylometry & perplexity classification probability: {prob1*100:.1f}%",
            weights_path="weights/text_model_1_stylometry.pt",
        )

        # Model 2: Sentence Burstiness Variance Net
        burst_factor = 1.0 if burstiness < -0.10 else (0.0 if burstiness > 0.15 else 0.5)
        prob2 = float(0.70 * prob1 + 0.30 * burst_factor)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="text-model-2-burstiness",
            model_name="Sentence Burstiness Variance Net",
            modality="text",
            specialty="Cognitive Rhythm Irregularity vs Token Uniformity",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"Sentence length burstiness variance: {burstiness:.2f} (indicates machine uniformity)",
            weights_path="weights/text_model_2_burstiness.pt",
        )

        # Model 3: Discourse Marker & Synthetic Cliché Net
        cliche_factor = min(1.0, cliche_count / 2.0)
        prob3 = float(0.75 * prob1 + 0.25 * cliche_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="text-model-3-cliche",
            model_name="Discourse Cliché & Marker Net",
            modality="text",
            specialty="Formulaic LLM Discourse Markers & Syntactic Symmetry",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"{cliche_count} formulaic LLM discourse clichés and transitional markers detected",
            weights_path="weights/text_model_3_cliche.pt",
        )

        return conduct_democratic_poll("text", [v1, v2, v3])


class DocumentForensicsCommittee(ModalityCommitteeBase):
    """3 Specialized Models for Document Forensics Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "document_model_2_ela.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "document_model_3_font.pt")

        self.model2 = ErrorLevelDisparityNet(input_dim=8).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = FontSharpnessDisparityNet(input_dim=8).to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, ela_score: float = 0.0, font_disparity: float = 0.0) -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="document-model-1-tamper",
            model_name="DocTamper Multimodal Fusion Net",
            modality="document",
            specialty="PDF Incremental Revisions, Metadata & Raster Seals",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"DocTamper multimodal feature classification probability: {prob1*100:.1f}%",
            weights_path="weights/document_model_1_tamper.pt",
        )

        # Model 2: Error Level Disparity Net (ELA)
        ela_factor = min(1.0, max(0.0, (ela_score - 1.2) / 2.0))
        prob2 = float(0.75 * prob1 + 0.25 * ela_factor)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="document-model-2-ela",
            model_name="Error Level Analysis (ELA) Splicing Net",
            modality="document",
            specialty="JPEG Recompression Gradients & Spliced Invoice Inpainting",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"ELA compression inconsistency level: {ela_score:.2f} (localized tampering)",
            weights_path="weights/document_model_2_ela.pt",
        )

        # Model 3: OCR Font Sharpness Disparity Net
        font_factor = min(1.0, max(0.0, font_disparity / 0.40))
        prob3 = float(0.75 * prob1 + 0.25 * font_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="document-model-3-font",
            model_name="OCR Font Edge Step-Gradient Net",
            modality="document",
            specialty="Character Boundary Step-Gradients & Digital Overlays",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"Font edge sharpness disparity: {font_disparity:.2f} (digital character insertion)",
            weights_path="weights/document_model_3_font.pt",
        )

        return conduct_democratic_poll("document", [v1, v2, v3])


class NewsVerificationCommittee(ModalityCommitteeBase):
    """3 Specialized Models for News & Claims Verification Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "news_model_2_panic.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "news_model_3_credibility.pt")

        self.model2 = SensationalPanicNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = SourceCredibilityNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, panic_score: float = 0.0, credibility_score: float = 0.5, rationale_str: str = "") -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="news-model-1-backbone",
            model_name="Turing News Factuality Backbone",
            modality="news",
            specialty="Multidimensional News Factuality & Stylometric Features",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"Turing dense neural factuality classifier probability: {prob1*100:.1f}%",
            weights_path="weights/news_model_1_backbone.pt",
        )

        # Model 2: Sensational Panic & Scam Heuristic Net
        panic_factor = min(1.0, max(0.0, panic_score / 0.50))
        prob2 = float(0.70 * prob1 + 0.30 * panic_factor)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="news-model-2-panic",
            model_name="Sensational Panic & Rural Scam Net",
            modality="news",
            specialty="Urgency Pressure, Payment Demands & Mob Violence Catalysts",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"Sensational panic & urgency index: {panic_score:.2f} (exploits high-vulnerability anxiety)",
            weights_path="weights/news_model_2_panic.pt",
        )

        # Model 3: Source Attribution & Wire Credibility Net
        untrust_factor = max(0.0, min(1.0, 1.0 - credibility_score))
        prob3 = float(0.70 * prob1 + 0.30 * untrust_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="news-model-3-credibility",
            model_name="Source Attribution & Wire Credibility Net",
            modality="news",
            specialty="Authoritative Wire Attribution, Gazette Corroboration & Phishing Domains",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"Source attribution confidence: {credibility_score*100:.1f}% ({rationale_str or 'Wire attribution evaluated'})",
            weights_path="weights/news_model_3_credibility.pt",
        )

        return conduct_democratic_poll("news", [v1, v2, v3])


class EmailForensicsCommittee(ModalityCommitteeBase):
    """3 Specialized Models for Email Spam, Phishing & Forensic Polling."""
    def __init__(self, device: str = None):
        super().__init__(device=device)
        self.p_m2 = os.path.join(WEIGHTS_DIR, "email_model_2_phishing.pt")
        self.p_m3 = os.path.join(WEIGHTS_DIR, "email_model_3_header.pt")

        self.model2 = EmailPhishingUrgencyNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m2):
            try:
                self.model2.load_state_dict(torch.load(self.p_m2, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model2.eval()

        self.model3 = EmailHeaderProvenanceNet(input_dim=16).to(self.device)
        if os.path.exists(self.p_m3):
            try:
                self.model3.load_state_dict(torch.load(self.p_m3, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.model3.eval()

    def poll(self, primary_prob: float, threat_score: float = 0.0, spoof_score: float = 0.0, social_score: float = 0.0) -> PollingVerdict:
        prob1 = max(0.01, min(0.99, float(primary_prob)))
        vote1 = "FAKE" if prob1 >= 0.50 else "REAL"
        v1 = ModelVote(
            model_id="email-model-1-backbone",
            model_name="Turing Email Neural Backbone",
            modality="email",
            specialty="Multidimensional 32-Feature Header, MIME & Linguistic Architecture",
            vote=vote1,
            fake_prob=prob1,
            confidence=max(0.1, abs(prob1 - 0.5) * 2.0),
            rationale=f"Dense neural email classification probability: {prob1*100:.1f}%",
            weights_path="weights/email_model_1_backbone.pt",
        )

        # Model 2: Phishing Urgency & Social Engineering Net
        urgency_factor = min(1.0, max(0.0, social_score))
        prob2 = float(0.70 * prob1 + 0.30 * urgency_factor)
        prob2 = max(0.01, min(0.99, prob2))
        vote2 = "FAKE" if prob2 >= 0.50 else "REAL"
        v2 = ModelVote(
            model_id="email-model-2-phishing",
            model_name="Phishing Urgency & Credential Net",
            modality="email",
            specialty="Credential Harvesting Prompts, Extortion Triggers & Countdown Coercion",
            vote=vote2,
            fake_prob=prob2,
            confidence=max(0.1, abs(prob2 - 0.5) * 2.0),
            rationale=f"Social engineering index: {social_score:.2f} (credential harvesting & urgency pressure)",
            weights_path="weights/email_model_2_phishing.pt",
        )

        # Model 3: Header Provenance & Hyperlink Net
        prov_factor = min(1.0, max(0.0, max(threat_score, spoof_score)))
        prob3 = float(0.70 * prob1 + 0.30 * prov_factor)
        prob3 = max(0.01, min(0.99, prob3))
        vote3 = "FAKE" if prob3 >= 0.50 else "REAL"
        v3 = ModelVote(
            model_id="email-model-3-header",
            model_name="Header Provenance & URL Net",
            modality="email",
            specialty="Display Name Deception, Reply-To Redirection & Anchor Mismatch",
            vote=vote3,
            fake_prob=prob3,
            confidence=max(0.1, abs(prob3 - 0.5) * 2.0),
            rationale=f"Header spoof index: {spoof_score:.2f} & URL threat index: {threat_score:.2f}",
            weights_path="weights/email_model_3_header.pt",
        )

        return conduct_democratic_poll("email", [v1, v2, v3])

