"""
Multimodal Fusion Engine & Unified Forensic Reporter (2026)
Aggregates forensic signals across Audio, Video, Text, Images, and Documents.
Performs cross-modal discrepancy auditing and outputs structured risk reports.
"""

import io
import os
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image

from .image_detector import ImageDeepfakeDetector
from .audio_detector import AudioDeepfakeDetector
from .video_detector import VideoDeepfakeDetector
from .text_detector import TextDeepfakeDetector
from .document_detector import DocumentForensicsDetector
from .news_detector import NewsVerificationEngine
from .email_detector import EmailForensicsDetector

# Known file extensions by modality
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".avif", ".heic"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus", ".aiff", ".alac"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
DOC_EXTS = {".pdf", ".docx", ".doc"}
TEXT_EXTS = {".txt", ".md", ".rtf", ".csv", ".log", ".json"}
EMAIL_EXTS = {".eml", ".msg"}
MEDIA_EXTS = IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS | DOC_EXTS | TEXT_EXTS | EMAIL_EXTS

NEWS_PREFIXES = ("news:", "claim:", "factcheck:", "fact-check:", "breaking:", "rumor:", "forwarded:", "verify:")
EMAIL_PREFIXES = ("email:", "mail:", "phish:", "spam:")


def identify_input_modality(target: Any) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Automagically identifies the modality of an input and returns:
    (modality_name, processed_target, extra_kwargs)
    where modality_name is in: 'image', 'audio', 'video', 'document', 'text'
    """
    # 1. String Inputs (File path or raw text)
    if isinstance(target, str):
        target_stripped = target.strip()

        # If it is an existing file path on disk
        if os.path.exists(target_stripped):
            ext = os.path.splitext(target_stripped)[1].lower()
            if ext in IMAGE_EXTS:
                return ("image", target_stripped, {})
            if ext in AUDIO_EXTS:
                return ("audio", target_stripped, {})
            if ext in VIDEO_EXTS:
                return ("video", target_stripped, {})
            if ext in DOC_EXTS:
                return ("document", target_stripped, {"is_pdf": ext == ".pdf"})
            if ext in EMAIL_EXTS:
                return ("email", target_stripped, {})
            if ext in TEXT_EXTS:
                try:
                    with open(target_stripped, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "From:" in content and ("Subject:" in content or "To:" in content):
                            return ("email", content, {})
                        return ("text", content, {})
                except Exception:
                    pass

            # Inspect magic bytes if extension is unlisted or missing
            try:
                with open(target_stripped, "rb") as f:
                    head = f.read(512)
                if head.startswith(b"%PDF-"):
                    return ("document", target_stripped, {"is_pdf": True})
                if head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff") or (head.startswith(b"RIFF") and b"WEBP" in head[:16]):
                    return ("image", target_stripped, {})
                if head.startswith(b"ID3") or head.startswith(b"fLaC") or head.startswith(b"OggS") or (head.startswith(b"RIFF") and b"WAVE" in head[:16]) or head.startswith(b"\xff\xfb") or head.startswith(b"\xff\xf3") or head.startswith(b"\xff\xf2"):
                    return ("audio", target_stripped, {})
                if len(head) >= 8 and (head[4:8] == b"ftyp" or head.startswith(b"\x1aE\xdf\xa3")):
                    return ("video", target_stripped, {})
                # Check if it is clean UTF-8 text file
                text_content = head.decode("utf-8")
                if "\x00" not in text_content:
                    with open(target_stripped, "r", encoding="utf-8", errors="ignore") as f:
                        return ("text", f.read(), {})
            except Exception:
                pass

        # If it is NOT an existing file, check if it was intended as a path (typo or missing file)
        has_media_ext = any(target_stripped.lower().endswith(e) for e in MEDIA_EXTS)
        has_path_sep = "/" in target_stripped or "\\" in target_stripped
        if has_media_ext or (has_path_sep and len(target_stripped.splitlines()) == 1):
            raise FileNotFoundError(f"Specified media file does not exist: '{target_stripped}'")

        # Check if text is prefixed as a news or email verification request
        lower_target = target_stripped.lower()
        for prefix in EMAIL_PREFIXES:
            if lower_target.startswith(prefix):
                clean_text = target_stripped[len(prefix):].strip()
                return ("email", clean_text, {})

        for prefix in NEWS_PREFIXES:
            if lower_target.startswith(prefix):
                clean_text = target_stripped[len(prefix):].strip()
                return ("news", clean_text, {})

        # If text looks like a standard MIME email with headers
        if "from:" in lower_target and ("subject:" in lower_target or "to:" in lower_target):
            return ("email", target_stripped, {})

        # Otherwise, treat string as raw text content (articles, prompts, claims)
        return ("text", target_stripped, {})

    # 2. PIL Image
    if isinstance(target, Image.Image):
        return ("image", target, {})

    # 3. Numpy Arrays
    if isinstance(target, np.ndarray):
        if target.ndim == 1:
            return ("audio", target, {})
        elif target.ndim in (2, 3):
            return ("image", target, {})
        elif target.ndim >= 4:
            return ("video", target, {})

    # 4. Sequences (List / Tuple of frames)
    if isinstance(target, (list, tuple)):
        if all(isinstance(x, (Image.Image, np.ndarray)) for x in target):
            return ("video", target, {})
        elif all(isinstance(x, str) for x in target):
            return ("text", " ".join(target), {})

    # 5. Raw Bytes or BytesIO Buffer
    if isinstance(target, (bytes, bytearray, io.BytesIO)):
        data = target.getvalue() if isinstance(target, io.BytesIO) else bytes(target)
        head = data[:512]
        if head.startswith(b"%PDF-"):
            return ("document", target, {"is_pdf": True})
        if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or (head.startswith(b"RIFF") and b"WEBP" in head[:16]):
            return ("image", target, {})
        if head.startswith(b"ID3") or head.startswith(b"fLaC") or head.startswith(b"OggS") or (head.startswith(b"RIFF") and b"WAVE" in head[:16]) or head.startswith(b"\xff\xfb") or head.startswith(b"\xff\xf3"):
            return ("audio", target, {})
        if len(head) >= 8 and (head[4:8] == b"ftyp" or head.startswith(b"\x1aE\xdf\xa3")):
            return ("video", target, {})
        try:
            text_str = data.decode("utf-8")
            if "\x00" not in text_str:
                return ("text", text_str, {})
        except Exception:
            pass

    raise ValueError(f"Unable to automagically determine modality for input of type: {type(target)}")


@dataclass
class ForensicReport:
    overall_verdict: str
    overall_fake_probability: float
    risk_level: str  # CRITICAL, HIGH, MODERATE, LOW, VERIFIED_CLEAN
    modality_results: Dict[str, Any]
    detected_manipulations: list
    cross_modal_discrepancies: list

    def to_dict(self):
        return asdict(self)


class MultimodalFusionDetector:
    def __init__(self, device: str = None):
        self.image_detector = ImageDeepfakeDetector(device=device)
        self.audio_detector = AudioDeepfakeDetector(device=device)
        self.video_detector = VideoDeepfakeDetector(device=device)
        self.text_detector = TextDeepfakeDetector(device=device)
        self.document_detector = DocumentForensicsDetector(device=device)
        self.news_engine = NewsVerificationEngine()
        self.email_detector = EmailForensicsDetector(device=device)

    def scan(self, target: Any = None, **kwargs) -> ForensicReport:
        """
        Automagically-dispatched single point of entry:
        Automatically inspects target (Image, Audio, Video, Document, Text, News Claim) and executes
        the corresponding forensic model.
        """
        # Explicit news verification override
        if kwargs.get("verify_news"):
            kwargs.pop("verify_news", None)
            if target is not None and isinstance(target, str):
                return self.scan_multimodal(news_input=target, **kwargs)

        if target is None:
            return self.scan_multimodal(**kwargs)

        if isinstance(target, dict):
            return self.scan_multimodal(**{**target, **kwargs})

        modality, processed_target, extra_kw = identify_input_modality(target)
        merged_args = {**extra_kw, **kwargs}

        if modality == "image":
            merged_args.pop("image_input", None)
            return self.scan_multimodal(image_input=processed_target, **merged_args)
        elif modality == "audio":
            merged_args.pop("audio_input", None)
            return self.scan_multimodal(audio_input=processed_target, **merged_args)
        elif modality == "video":
            merged_args.pop("video_frames", None)
            return self.scan_multimodal(video_frames=processed_target, **merged_args)
        elif modality == "document":
            merged_args.pop("document_input", None)
            return self.scan_multimodal(document_input=processed_target, **merged_args)
        elif modality == "news":
            merged_args.pop("news_input", None)
            return self.scan_multimodal(news_input=processed_target, **merged_args)
        elif modality == "email":
            merged_args.pop("email_input", None)
            return self.scan_multimodal(email_input=processed_target, **merged_args)
        elif modality == "text":
            merged_args.pop("text_input", None)
            return self.scan_multimodal(text_input=processed_target, **merged_args)
        else:
            raise ValueError(f"Unrecognized input modality: {modality}")

    def scan_multimodal(
        self,
        image_input=None,
        audio_input=None,
        video_frames=None,
        video_audio=None,
        text_input: Optional[str] = None,
        document_input=None,
        news_input: Optional[str] = None,
        email_input: Optional[str] = None,
        is_pdf: bool = False,
    ) -> ForensicReport:
        """
        Runs comprehensive forensic detection across all submitted modalities.
        """
        modality_results = {}
        all_signals = []
        fake_probabilities = []
        weights = []

        # 1. Image Modality
        if image_input is not None:
            res = self.image_detector.analyze(image_input)
            modality_results["image"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(1.0)
            if res["is_synthetic"]:
                all_signals.extend([f"[Image] {s}" for s in res["forensic_signals"]])

        # 2. Audio Modality
        if audio_input is not None:
            res = self.audio_detector.analyze(audio_input)
            modality_results["audio"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(1.0)
            if res["is_synthetic"]:
                all_signals.extend([f"[Audio] {s}" for s in res["forensic_signals"]])

        # 3. Video Modality
        if video_frames is not None:
            res = self.video_detector.analyze(video_frames, audio_track=video_audio)
            modality_results["video"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(1.5)  # Video has high forensic gravity
            if res["is_synthetic"]:
                all_signals.extend([f"[Video] {s}" for s in res["forensic_signals"]])

        # 4. Text Modality
        if text_input is not None and len(text_input.strip()) >= 30:
            res = self.text_detector.analyze(text_input)
            modality_results["text"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(0.8)
            if res["is_synthetic"]:
                all_signals.extend([f"[Text] {s}" for s in res["forensic_signals"]])

        # 5. Document Modality
        if document_input is not None:
            res = self.document_detector.analyze(document_input, is_pdf=is_pdf)
            modality_results["document"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(1.2)
            if res["is_synthetic"]:
                all_signals.extend([f"[Document] {s}" for s in res["forensic_signals"]])

        # 6. News & Claim Veracity Modality
        if news_input is not None:
            res = self.news_engine.verify_news(
                news_input,
                attached_image=image_input,
                attached_video=video_frames,
            )
            modality_results["news"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(1.3)  # Factual veracity carries high gravity
            if res["is_misinformation"]:
                all_signals.extend([f"[News] {s}" for s in res["forensic_signals"]])
            elif res["forensic_signals"]:
                for s in res["forensic_signals"]:
                    if any(k in s.lower() for k in ["authoritative", "fact-checker", "credible", "authentic"]):
                        all_signals.append(f"[News] {s}")

        # 7. Email Spam & Phishing Modality
        if email_input is not None:
            res = self.email_detector.analyze(email_input)
            modality_results["email"] = res
            fake_probabilities.append(res["fake_probability"])
            weights.append(1.2)  # High security priority
            if res["is_spam_or_phishing"]:
                all_signals.extend([f"[Email] {s}" for s in res["forensic_signals"]])
            elif res["forensic_signals"]:
                for s in res["forensic_signals"]:
                    if any(k in s.lower() for k in ["verified", "authentic", "clean"]):
                        all_signals.append(f"[Email] {s}")

        # If no input provided
        if not fake_probabilities:
            return ForensicReport(
                overall_verdict="NO_DATA",
                overall_fake_probability=0.0,
                risk_level="UNKNOWN",
                modality_results={},
                detected_manipulations=[],
                cross_modal_discrepancies=[],
            )

        # Cross-modal discrepancy check
        discrepancies = []
        if "video" in modality_results and "audio" in modality_results:
            vid_fake = modality_results["video"]["fake_probability"]
            aud_fake = modality_results["audio"]["fake_probability"]
            if abs(vid_fake - aud_fake) > 0.45:
                discrepancies.append(
                    f"Video-Audio Asymmetry: Video synthetic confidence={vid_fake:.2f} vs Audio confidence={aud_fake:.2f}. "
                    "Likely audio dubbing or face-reenactment."
                )

        if "image" in modality_results and "text" in modality_results:
            img_fake = modality_results["image"]["fake_probability"]
            txt_fake = modality_results["text"]["fake_probability"]
            if abs(img_fake - txt_fake) > 0.50:
                discrepancies.append(
                    f"Meme / News Integrity Mismatch: Image fake p={img_fake:.2f} vs Text fake p={txt_fake:.2f}. "
                    "Authentic media repurposed with synthetic narrative, or synthetic media with human commentary."
                )

        if "news" in modality_results and "image" in modality_results:
            img_fake = modality_results["image"]["fake_probability"]
            news_fake = modality_results["news"]["fake_probability"]
            if img_fake >= 0.70 and news_fake < 0.40:
                discrepancies.append(
                    f"Cross-Modal Cheapfake Detected: Authentic news claim paired with an AI-generated synthetic deepfake image (Image fake p={img_fake:.2f})."
                )
            elif img_fake < 0.30 and news_fake >= 0.75:
                discrepancies.append(
                    f"Context Hijacking Detected: Real, authentic camera photograph repurposed with a fabricated disinformation claim (News fake p={news_fake:.2f})."
                )

        if "email" in modality_results and "document" in modality_results:
            email_fake = modality_results["email"]["fake_probability"]
            doc_fake = modality_results["document"]["fake_probability"]
            if doc_fake >= 0.70 and email_fake < 0.40:
                discrepancies.append(
                    f"Trojan Attachment Detected: Authentic-looking email carrier paired with a forged/tampered document attachment (Document fake p={doc_fake:.2f})."
                )

        # Calculate weighted average fake probability
        w_sum = sum(weights)
        weighted_prob = sum(p * w for p, w in zip(fake_probabilities, weights)) / w_sum

        # Penalize overall score if maximum single modality is severe
        max_prob = max(fake_probabilities)
        composite_score = float(max(weighted_prob, max_prob * 0.85))

        # Determine Risk Level
        if composite_score >= 0.80:
            risk_level = "CRITICAL"
            verdict = "DEEPFAKE / SYNTHETIC MANIPULATION CONFIRMED"
        elif composite_score >= 0.55:
            risk_level = "HIGH"
            verdict = "LIKELY SYNTHETIC / SUSPICIOUS ARTIFACTS FOUND"
        elif composite_score >= 0.35:
            risk_level = "MODERATE"
            verdict = "INCONCLUSIVE / LOW MANIPULATION SIGNALS"
        else:
            risk_level = "VERIFIED_CLEAN"
            verdict = "AUTHENTIC / NO SYNTHETIC TRACES DETECTED"

        return ForensicReport(
            overall_verdict=verdict,
            overall_fake_probability=round(composite_score, 4),
            risk_level=risk_level,
            modality_results=modality_results,
            detected_manipulations=all_signals if all_signals else ["All forensic indicators within natural human/camera bounds"],
            cross_modal_discrepancies=discrepancies,
        )


# Global default detector instance for zero-boilerplate automagic calls
_DEFAULT_DETECTOR = None


def scan(target: Any = None, **kwargs) -> ForensicReport:
    """
    Automagically-dispatched single point of entry for The Turing Project.
    Automatically inspects target (Image, Audio, Video, Document, Text) and executes
    the corresponding forensic model.

    Examples:
        import detector
        report = detector.scan("photo.png")
        report = detector.scan("song.mp3")
        report = detector.scan("clip.mp4")
        report = detector.scan("invoice.pdf")
        report = detector.scan("Text snippet to verify...")
        report = detector.scan("BREAKING: Free Solar Scheme announced...", verify_news=True)
    """
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = MultimodalFusionDetector()
    return _DEFAULT_DETECTOR.scan(target, **kwargs)
