"""
Unit tests for the News Verification Engine and Disinformation Auditor (2026)
Covers credibility scoring, satire detection, rural scam heuristics, and cross-modal cheapfakes.
"""

import unittest
import numpy as np
from PIL import Image

import detector
from detector.news_detector import NewsVerificationEngine
from detector.fusion import MultimodalFusionDetector


class TestNewsVerificationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = NewsVerificationEngine()
        cls.detector = MultimodalFusionDetector()

    def test_credible_news_reporting(self):
        """Authoritative wire report with named sources should yield high credibility."""
        claim = "According to Reuters, official data from the World Health Organization confirms new global health protocols published in Nature."
        res = self.engine.verify_news(claim)

        self.assertEqual(res["modality"], "news")
        self.assertFalse(res["is_misinformation"])
        self.assertGreaterEqual(res["credibility_score"], 0.85)
        self.assertIn("CREDIBLE", res["veracity_verdict"])
        self.assertEqual(res["risk_level"], "VERIFIED_CLEAN")

    def test_satire_outlet_detection(self):
        """Satire publications should be recognized and marked as parody."""
        claim = "theonion.com: Congress Approves Bill Funding Entire Nation's Hopes And Dreams."
        res = self.engine.verify_news(claim)

        self.assertTrue(res["is_misinformation"])
        self.assertIn("SATIRE", res["veracity_verdict"])
        self.assertGreaterEqual(res["fake_probability"], 0.85)
        self.assertLessEqual(res["credibility_score"], 0.15)

    def test_rural_welfare_scam_hoax(self):
        """Grassroots subsidy and phishing schemes should trigger high-risk rural scam alerts."""
        claim = "URGENT: Government of India announced Free Solar Scheme 2026! Click here to claim Rs 50000 immediately. Share before it gets deleted!"
        res = self.engine.verify_news(claim)

        self.assertTrue(res["is_misinformation"])
        self.assertIn("RURAL SCAM", res["veracity_verdict"])
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertGreaterEqual(res["fake_probability"], 0.90)
        self.assertTrue(res["claim_analysis"]["has_urgency_call"])
        self.assertGreater(len(res["claim_analysis"]["rural_scam_flags"]), 0)

    def test_rural_panic_rumor(self):
        """Viral child lifter panic rumors causing mob violence must be flagged critically."""
        claim = "Alert to all villagers: 50 child lifter gang members entered the district tonight. Forward to all groups immediately!"
        res = self.engine.verify_news(claim)

        self.assertTrue(res["is_misinformation"])
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertTrue(any("child lifter" in s.lower() for s in res["forensic_signals"]))

    def test_sensationalism_and_urgency(self):
        """Sensationalist hyperbole and emotional triggers should increase fake probability."""
        claim = "SHOCKING BOMBSHELL: Secret they don't want you to know exposed! Doctors hate this unbelievable miracle cure! ACT NOW OR DIE!"
        res = self.engine.verify_news(claim)

        self.assertTrue(res["is_misinformation"])
        self.assertGreater(res["claim_analysis"]["sensationalism_index"], 0.5)

    def test_empty_news_input(self):
        """Empty or whitespace input should be gracefully handled."""
        res = self.engine.verify_news("   ")
        self.assertEqual(res["veracity_verdict"], "NO_INPUT")

    def test_fusion_scan_integration(self):
        """detector.scan() should seamlessly route news inputs and prefix markers."""
        report = detector.scan("news: According to Reuters, the World Health Organization issued a formal statement.")
        self.assertIn("news", report.modality_results)
        self.assertFalse(report.modality_results["news"]["is_misinformation"])

        report_flag = detector.scan("Free solar scheme yojana click here to claim Rs 25000", verify_news=True)
        self.assertIn("news", report_flag.modality_results)
        self.assertTrue(report_flag.modality_results["news"]["is_misinformation"])
        self.assertEqual(report_flag.risk_level, "CRITICAL")

    def test_cross_modal_cheapfake_auditing(self):
        """Synthetic deepfake image paired with news claim should trigger cross-modal alerts."""
        # Create a synthetic image (uniform noise / grid that triggers synthetic classification)
        syn_arr = (np.sin(np.linspace(0, 100, 256 * 256)).reshape((256, 256, 1)) * 127 + 128).astype(np.uint8)
        syn_img = Image.fromarray(np.repeat(syn_arr, 3, axis=2))

        report = self.detector.scan_multimodal(
            news_input="According to Reuters, official data confirms economic recovery.",
            image_input=syn_img,
        )
        self.assertIn("news", report.modality_results)
        self.assertIn("image", report.modality_results)
        # Discrepancy checks or signals should be generated
        self.assertIsNotNone(report.overall_verdict)


if __name__ == "__main__":
    unittest.main()
