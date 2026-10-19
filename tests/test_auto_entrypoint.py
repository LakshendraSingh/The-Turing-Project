"""
Unit Tests for Automagic Single Point of Entry (scan) in The Turing Project
"""

import io
import unittest
import numpy as np
from PIL import Image
from detector import scan, identify_input_modality


class TestAutoEntrypoint(unittest.TestCase):
    def test_identify_text(self):
        mod, processed, kw = identify_input_modality("A simple sentence to test the text auto detector.")
        self.assertEqual(mod, "text")
        self.assertIsInstance(processed, str)

    def test_identify_image_object(self):
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        mod, processed, kw = identify_input_modality(img)
        self.assertEqual(mod, "image")

    def test_identify_audio_numpy(self):
        audio_arr = np.zeros(16000, dtype=np.float32)
        mod, processed, kw = identify_input_modality(audio_arr)
        self.assertEqual(mod, "audio")

    def test_identify_video_sequence(self):
        frames = [Image.new("RGB", (32, 32)) for _ in range(3)]
        mod, processed, kw = identify_input_modality(frames)
        self.assertEqual(mod, "video")

    def test_identify_pdf_bytes(self):
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<\n>>\nendobj\n%%EOF"
        mod, processed, kw = identify_input_modality(pdf_bytes)
        self.assertEqual(mod, "document")
        self.assertTrue(kw.get("is_pdf"))

    def test_scan_text(self):
        report = scan("In conclusion, delving into this multifaceted topic is crucial for human progress.")
        self.assertIn("text", report.modality_results)
        self.assertIn(report.risk_level, ["CRITICAL", "HIGH", "MODERATE", "LOW", "VERIFIED_CLEAN"])

    def test_scan_image(self):
        img = Image.new("RGB", (64, 64), color=(120, 150, 180))
        report = scan(img)
        self.assertIn("image", report.modality_results)


if __name__ == "__main__":
    unittest.main()
