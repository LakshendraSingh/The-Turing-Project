"""
The Turing Project — Multimodal Deepfake & Synthetic Media Detector (2026)
Supports Audio, Video, Text, Images, and Documents with explainable forensic signals.
"""

from .image_detector import ImageDeepfakeDetector
from .audio_detector import AudioDeepfakeDetector
from .video_detector import VideoDeepfakeDetector
from .text_detector import TextDeepfakeDetector
from .document_detector import DocumentForensicsDetector
from .metadata_forensics import UniversalMetadataForensics
from .news_detector import NewsVerificationEngine
from .polling import ModelVote, PollingVerdict, conduct_democratic_poll
from .fusion import MultimodalFusionDetector, ForensicReport, scan, identify_input_modality

__version__ = "2026.1.0"
__all__ = [
    "scan",
    "identify_input_modality",
    "ImageDeepfakeDetector",
    "AudioDeepfakeDetector",
    "VideoDeepfakeDetector",
    "TextDeepfakeDetector",
    "DocumentForensicsDetector",
    "UniversalMetadataForensics",
    "NewsVerificationEngine",
    "MultimodalFusionDetector",
    "ForensicReport",
    "ModelVote",
    "PollingVerdict",
    "conduct_democratic_poll",
]
