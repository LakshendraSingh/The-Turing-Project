"""
Comprehensive Unit & Integration Test Suite for The Turing Project (2026 Detector).
Tests all 5 modalities: Image, Audio, Video, Text, Document, and Multimodal Fusion.
"""

import unittest
import numpy as np
from PIL import Image
from detector.image_detector import ImageDeepfakeDetector
from detector.audio_detector import AudioDeepfakeDetector
from detector.video_detector import VideoDeepfakeDetector
from detector.text_detector import TextDeepfakeDetector
from detector.document_detector import DocumentForensicsDetector
from detector.fusion import MultimodalFusionDetector


class TestTuringProjectForensics(unittest.TestCase):
    def test_image_detector(self):
        detector = ImageDeepfakeDetector()
        # Create a test synthetic-like canvas with high-frequency grid
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[::4, ::4] = 255  # grid artifact
        test_img = Image.fromarray(arr)

        res = detector.analyze(test_img)
        self.assertEqual(res["modality"], "image")
        self.assertIn("fake_probability", res)
        self.assertIn("forensic_signals", res)
        self.assertIsInstance(res["metrics"], dict)

    def test_audio_detector(self):
        detector = AudioDeepfakeDetector()
        # Generate 1-second synthetic sine wave (abnormally flat pitch, no jitter)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz pure tone

        res = detector.analyze(audio)
        self.assertEqual(res["modality"], "audio")
        self.assertIn("fake_probability", res)
        self.assertIn("vocal_dynamics", res["metrics"])

    def test_video_detector(self):
        detector = VideoDeepfakeDetector()
        # Create a sequence of 4 test frames
        frames = [Image.new("RGB", (128, 128), color=(i * 40, i * 20, 100)) for i in range(4)]
        res = detector.analyze(frames)
        self.assertEqual(res["modality"], "video")
        self.assertEqual(res["analyzed_frames_count"], 4)
        self.assertIn("fake_probability", res)

    def test_text_detector(self):
        detector = TextDeepfakeDetector()
        # Test sample with typical LLM markers and uniform phrasing
        llm_sample = (
            "Furthermore, it is crucial to delve into the tapestry of modern technological progress. "
            "A testament to human ingenuity, navigating the digital ecosystem plays a pivotal role in our society. "
            "In conclusion, embracing a holistic approach will ensure a sustainable future."
        )
        res = detector.analyze(llm_sample)
        self.assertEqual(res["modality"], "text")
        self.assertIn("fake_probability", res)
        self.assertTrue(res["metrics"]["discourse_markers"]["marker_count"] >= 3)

    def test_document_detector(self):
        detector = DocumentForensicsDetector()
        doc_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        res = detector.analyze(doc_img)
        self.assertEqual(res["modality"], "document")
        self.assertIn("fake_probability", res)

    def test_multimodal_fusion(self):
        fusion = MultimodalFusionDetector()
        test_text = (
            "Delve into the multifaceted tapestry of modern synthetic generation. "
            "Furthermore, it is important to remember that artificial intelligence plays a pivotal role."
        )
        report = fusion.scan_multimodal(text_input=test_text)
        self.assertIn(report.risk_level, ["CRITICAL", "HIGH", "MODERATE", "LOW", "VERIFIED_CLEAN"])
        self.assertIn("text", report.modality_results)


if __name__ == "__main__":
    unittest.main()
