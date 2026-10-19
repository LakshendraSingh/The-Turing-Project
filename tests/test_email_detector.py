"""
Unit tests for the Email Spam, Phishing & Forensic Security Engine (2026)
Covers sender spoofing, credential harvesting, dangerous attachment detection,
and automagic multimodal entrypoints.
"""

import unittest
import detector
from detector.email_detector import EmailForensicsDetector
from detector.fusion import MultimodalFusionDetector


class TestEmailForensicsDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = EmailForensicsDetector()
        cls.fusion = MultimodalFusionDetector()

    def test_phishing_credential_harvesting(self):
        """Phishing email mimicking PayPal with urgent suspension threats must be flagged."""
        phish_email = """From: "PayPal Security Department" <security-alerts@paypal-service-verify.xyz>
To: target@victim.com
Subject: URGENT: Your PayPal account has been limited! Immediate action required
Date: Mon, 19 Oct 2026 10:00:00 +0000
Authentication-Results: spf=fail; dkim=fail

Dear Valued Customer,
We detected unauthorized login attempts to your account.
You must verify your password and confirm your account within 24 hours at http://paypal-service-verify.xyz/signin.
Failure to comply will result in complete account closure.
PayPal Security Team"""

        res = self.detector.analyze(phish_email)
        self.assertEqual(res["modality"], "email")
        self.assertTrue(res["is_spam_or_phishing"])
        self.assertGreaterEqual(res["fake_probability"], 0.80)
        self.assertIn("PHISHING", res["classification"])
        self.assertGreaterEqual(len(res["forensic_signals"]), 1)
        self.assertEqual(res["polling"]["tally"]["FAKE"], 3)

    def test_clean_corporate_correspondence(self):
        """Authentic engineering email should be verified clean with low threat score."""
        clean_email = """From: "Priya Sharma" <priya.sharma@techcorp.internal>
To: eng-team@techcorp.internal
Subject: Engineering Standup Notes — Monday Oct 19
Date: Mon, 19 Oct 2026 09:30:00 +0000
Authentication-Results: spf=pass; dkim=pass

Hi Team,
Quick summary from today's sprint sync:
1. Backend: Payment webhook refactor merged into staging (PR #412).
2. QA: Regression testing ongoing, zero P0 blockers found.
Next sprint planning is scheduled for Wednesday at 2:00 PM.

Best,
Priya Sharma"""

        res = self.detector.analyze(clean_email)
        self.assertEqual(res["modality"], "email")
        self.assertFalse(res["is_spam_or_phishing"])
        self.assertLessEqual(res["fake_probability"], 0.25)
        self.assertIn("AUTHENTIC", res["classification"])
        self.assertEqual(res["polling"]["tally"]["REAL"], 3)

    def test_dangerous_attachment_flagged(self):
        """Executable or macro-enabled double-extension attachments must be flagged critically."""
        malware_email = """From: "Accounting" <billing@invoice-sender.top>
Subject: OVERDUE INVOICE #98124
Content-Disposition: attachment; filename="Overdue_Invoice_98124.pdf.exe"

Please find the overdue invoice attached.
Attachment: Overdue_Invoice_98124.pdf.exe"""

        res = self.detector.analyze(malware_email)
        self.assertTrue(res["is_spam_or_phishing"])
        self.assertGreaterEqual(res["fake_probability"], 0.90)
        self.assertTrue(any("attachment" in s.lower() for s in res["forensic_signals"]))

    def test_automagic_fusion_entrypoint(self):
        """Testing top-level detector.scan automagic dispatch on email text."""
        email_text = """From: "Apple Support" <support@notifications-apple-verify.club>
To: user@example.com
Subject: Final Notice: iCloud Account Storage Exceeded

Your mailbox quota exceeded. Reset your password immediately to avoid deletion.
Click here: http://192.168.1.10/login"""

        report = detector.scan(email_text)
        self.assertIn("email", report.modality_results)
        self.assertTrue(report.overall_fake_probability > 0.60)
        self.assertIn(report.risk_level, ["HIGH", "CRITICAL"])


if __name__ == "__main__":
    unittest.main()
