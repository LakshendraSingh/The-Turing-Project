"""
Unit & Integration Test Suite for Universal Metadata & C2PA Forensics (The Turing Project)
"""

import io
import unittest
from PIL import Image, PngImagePlugin
from detector.metadata_forensics import UniversalMetadataForensics
from detector.image_detector import ImageDeepfakeDetector
from detector.audio_detector import AudioDeepfakeDetector
from detector.video_detector import VideoDeepfakeDetector


class TestMetadataForensics(unittest.TestCase):
    def test_c2pa_manifest_audit(self):
        # Simulate binary stream with Google C2PA manifest assertions
        simulated_c2pa_bytes = (
            b"\x00\x00\x00\x1capplication/c2pa\x00c2pa\x00c2pa manifest store\x00"
            b"jumbc2pa Google C2PA Media Services 1P ICA G3 "
            b"claim_generator_info Google C2PA Core Generator Library "
            b"action c2pa.created description: Created by Google Generative AI. "
            b"digitalSourceType: http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia "
            b"action c2pa.edited description: Applied imperceptible SynthID watermark. "
        )

        res = UniversalMetadataForensics.audit_c2pa(simulated_c2pa_bytes)
        self.assertTrue(res["has_c2pa"])
        self.assertTrue(res["is_ai_generated"])
        self.assertTrue(res["synthid_watermark_present"])
        self.assertIn("trainedAlgorithmicMedia", res["digital_source_type"])
        self.assertEqual(res["confidence"], 1.0)

    def test_image_png_prompt_metadata(self):
        # Create an image with embedded Stable Diffusion prompt parameters in PNG text chunks
        img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", "photorealistic portrait of astronaut on mars, 8k, highly detailed\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 7, Seed: 1234567")
        pnginfo.add_text("Software", "Automatic1111 WebUI")

        bio = io.BytesIO()
        img.save(bio, format="PNG", pnginfo=pnginfo)
        img_bytes = bio.getvalue()

        # Run audit on byte buffer
        res = UniversalMetadataForensics.audit_image_metadata(img_bytes)
        self.assertTrue(res["has_metadata"])
        self.assertTrue(res["is_synthetic"])
        self.assertGreaterEqual(res["metadata_fake_score"], 0.95)
        self.assertTrue(any("Diffusion" in f or "Generative" in f for f in res["suspicious_flags"]))

        # Also test via ImageDeepfakeDetector
        detector = ImageDeepfakeDetector()
        img_loaded = Image.open(io.BytesIO(img_bytes))
        det_res = detector.analyze(img_loaded)
        self.assertTrue(det_res["is_synthetic"])
        self.assertGreaterEqual(det_res["fake_probability"], 0.95)
        self.assertIn("metadata_audit", det_res["metrics"])

    def test_audio_c2pa_metadata(self):
        # Simulate MP3 stream with ID3 GEOB frame containing C2PA assertions
        simulated_mp3_header = (
            b"ID3\x03\x00\x00\x00\x00\x10\x00"
            b"GEOB\x00\x00\x05\x00\x00\x00\x00application/c2pa\x00c2pa\x00c2pa manifest store\x00"
            b"Google C2PA Media Services 1P ICA G3 "
            b"action c2pa.created Created by Google Generative AI. "
            b"digitalSourceType http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia "
            b"Applied imperceptible SynthID watermark."
        )

        res = UniversalMetadataForensics.audit_audio_metadata(simulated_mp3_header)
        self.assertTrue(res["has_metadata"])
        self.assertTrue(res["is_synthetic"])
        self.assertEqual(res["metadata_fake_score"], 0.99)
        self.assertTrue(any("Google Generative AI" in f or "C2PA" in f for f in res["suspicious_flags"]))

    def test_video_container_metadata(self):
        # Simulate MP4 atoms containing AI video generator tool signature
        simulated_mp4_bytes = (
            b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42"
            b"\x00\x00\x00\x40moov"
            b"\x00\x00\x00\x28udtameta"
            b"encoder: Runway Gen-3 Alpha Cinematic Model"
        )

        res = UniversalMetadataForensics.audit_video_metadata(simulated_mp4_bytes)
        self.assertTrue(res["has_metadata"])
        self.assertTrue(res["is_synthetic"])
        self.assertGreaterEqual(res["metadata_fake_score"], 0.95)
        self.assertTrue(any("Runway" in f for f in res["suspicious_flags"]))


if __name__ == "__main__":
    unittest.main()
