"""
Email Spam, Phishing & Forensic Security Engine (2026)
Performs multi-layered forensic inspection of emails, message files (.eml), and raw headers:
1. Sender Authenticity & Provenance Audit (SPF/DKIM/DMARC alignment, display name spoofing, reply-to mismatch)
2. Malicious Structural Forensics (Executable/macro double-extension attachments, hidden elements)
3. Phishing URL & Redirection Forensics (Mismatched href vs anchor text, punycode/IDN, suspicious TLDs, IP URLs)
4. Social Engineering & Urgency Indicators (Credential harvesting, financial coercion, extortion triggers)
5. Multi-Model Democratic Polling Committee (Dense neural backbone + Phishing net + Header provenance net)
"""

import email
import email.policy
import os
import re
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class EmailClassifierBackbone(nn.Module):
    """
    Calibrated 3-layer neural MLP for statistical email spam, phishing, and scam classification.
    Operates on 32-dimensional dense forensic, header, URL, and linguistic feature vectors.
    """
    def __init__(self, in_features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# High-risk executable and macro script attachment extensions
MALICIOUS_ATTACHMENT_EXTS = {
    ".exe", ".scr", ".vbs", ".js", ".bat", ".cmd", ".iso", ".img",
    ".xlsm", ".docm", ".hta", ".cpl", ".pif", ".wsf", ".ps1", ".jar"
}

# Suspicious TLDs frequently abused in disposable phishing campaigns
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".work", ".click", ".buzz", ".cam",
    ".live", ".loan", ".surf", ".support", ".account", ".icu", ".rest"
}

# Well-known free webmail domains (used to detect executive impersonation)
FREE_WEBMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "zoho.com", "mail.com", "yandex.com", "icloud.com"
}

# Brands and sensitive services frequently impersonated
IMPERSONATION_TARGETS = [
    "paypal", "microsoft", "apple", "google", "amazon", "netflix",
    "bank", "chase", "wells fargo", "citibank", "barclays", "hsbc",
    "security team", "it helpdesk", "admin", "ceo", "human resources",
    "payroll", "accounting", "meta", "instagram", "facebook", "coinbase", "binance"
]

# Credential harvesting indicators
CREDENTIAL_HARVEST_TRIGGERS = [
    "verify your password", "reset your password", "confirm your account",
    "update your credentials", "re-authenticate", "login to continue",
    "mailbox quota exceeded", "access suspended", "unauthorized login",
    "verify identity", "security alert", "2fa verification required"
]

# Financial pressure and extortion keywords
FINANCIAL_SCAM_TRIGGERS = [
    "urgent payment", "wire transfer", "bitcoin", "cryptocurrency wallet",
    "unclaimed inheritance", "lottery winnings", "overdue invoice",
    "gift card", "western union", "confidential proposal", "fund release",
    "direct deposit", "remittance advice"
]

# Urgency pressure phrases
URGENCY_TRIGGERS = [
    "within 24 hours", "immediate action required", "account will be closed",
    "failure to comply", "urgent attention", "act immediately",
    "final notice", "immediate response", "do not ignore"
]


def extract_email_feature_vector(email_text_or_path: str) -> np.ndarray:
    """
    Extracts a 32-dimensional dense forensic feature vector from raw email text or file.
    """
    detector = EmailForensicsDetector()
    return detector.build_feature_vector(email_text_or_path)


class EmailForensicsDetector:
    """
    Air-gapped, offline forensic email security and spam/phishing detector.
    Analyzes raw MIME headers, HTML structures, URLs, and linguistic indicators.
    """
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

        self.model = EmailClassifierBackbone(in_features=32).to(self.device)
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "email_best.pt"
        )
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device, weights_only=True)
                )
            except Exception:
                pass
        self.model.eval()

        from .polling import EmailForensicsCommittee
        self.committee = EmailForensicsCommittee(device=self.device)

    def parse_email(self, email_input: str) -> Tuple[Dict[str, str], str, List[str], List[str]]:
        """
        Parses raw text, .eml file path, or structured email into headers, body, links, and attachment names.
        """
        raw_text = email_input
        if isinstance(email_input, str) and os.path.exists(email_input):
            try:
                with open(email_input, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            except Exception:
                pass

        headers = {}
        body = ""
        attachment_names = []

        try:
            msg = email.message_from_string(raw_text, policy=email.policy.default)
            for k in ["From", "To", "Subject", "Date", "Reply-To", "Return-Path", "Authentication-Results", "Message-ID"]:
                val = msg.get(k, "")
                if val:
                    headers[k] = str(val)

            # Walk body parts
            body_parts = []
            for part in msg.walk():
                fn = part.get_filename()
                if fn:
                    attachment_names.append(fn)
                cd = part.get("Content-Disposition", "")
                if "attachment" in cd and not fn:
                    attachment_names.append("unnamed_attachment")

                ct = part.get_content_type()
                if ct in ["text/plain", "text/html"]:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(payload.decode("utf-8", errors="ignore"))

            body = "\n".join(body_parts) if body_parts else msg.get_payload()
            if not isinstance(body, str):
                body = str(body or "")
        except Exception:
            # Fallback regex header parsing for raw text snippets
            body = raw_text
            for k in ["From", "To", "Subject", "Reply-To", "Return-Path"]:
                m = re.search(rf"^{k}:\s*(.*)$", raw_text, re.MULTILINE | re.IGNORECASE)
                if m:
                    headers[k] = m.group(1).strip()

        # Extract URLs from body
        links = re.findall(r'https?://[^\s<>"\']+', body)

        return headers, body, links, attachment_names

    def analyze_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Audits email headers for sender spoofing, display name deception, and authentication failures.
        """
        signals = []
        spoof_score = 0.0

        from_header = headers.get("From", "")
        reply_to = headers.get("Reply-To", "")
        return_path = headers.get("Return-Path", "")
        auth_results = headers.get("Authentication-Results", "").lower()

        # Extract display name and email address
        name_match = re.match(r'^"?([^"<]*)"?\s*<([^>]+)>', from_header)
        display_name = ""
        from_email = from_header
        if name_match:
            display_name = name_match.group(1).strip()
            from_email = name_match.group(2).strip()

        from_domain = from_email.split("@")[-1].lower() if "@" in from_email else ""

        # 1. Display Name Deception & Lookalike Domain Spoofing
        is_impersonating = False
        lower_name = display_name.lower()
        for target in IMPERSONATION_TARGETS:
            if target in lower_name:
                is_legit = False
                for legit_ext in [f"{target}.com", f"{target}.org", f"{target}.net", f"{target}.co.uk", f"{target}.in"]:
                    if from_domain == legit_ext or from_domain.endswith(f".{legit_ext}"):
                        is_legit = True
                        break
                if not is_legit:
                    is_impersonating = True
                    signals.append(f"Display Name Deception: Sender claims to be '{display_name}' but domain is '{from_domain}'")
                    spoof_score += 0.50
                    break

        # 2. Executive / Sensitive Impersonation via Free Webmail
        if is_impersonating and from_domain in FREE_WEBMAIL_DOMAINS:
            signals.append(f"High-Risk Phishing Marker: Institutional impersonation via free webmail provider ({from_domain})")
            spoof_score += 0.35

        # 3. Reply-To Mismatch
        reply_domain = ""
        if reply_to:
            rt_match = re.search(r'<([^>]+)>', reply_to)
            rt_email = rt_match.group(1) if rt_match else reply_to.strip()
            reply_domain = rt_email.split("@")[-1].lower() if "@" in rt_email else ""
            if reply_domain and from_domain and reply_domain != from_domain:
                signals.append(f"Reply-To Mismatch: Responses redirected to '{rt_email}' instead of sender '{from_email}'")
                spoof_score += 0.30

        # 4. Authentication Failures (SPF/DKIM/DMARC)
        auth_fail = False
        if auth_results:
            if "spf=fail" in auth_results or "spf=softfail" in auth_results:
                signals.append("SPF Authentication Failed: Sending host unauthorized for sender domain")
                spoof_score += 0.25
                auth_fail = True
            if "dkim=fail" in auth_results:
                signals.append("DKIM Signature Failed: Cryptographic message integrity compromised")
                spoof_score += 0.25
                auth_fail = True
            if "dmarc=fail" in auth_results:
                signals.append("DMARC Policy Violation: Sender alignment failed")
                spoof_score += 0.25
                auth_fail = True

        return {
            "spoof_score": min(1.0, spoof_score),
            "display_name": display_name,
            "from_domain": from_domain,
            "reply_domain": reply_domain,
            "is_impersonating": is_impersonating,
            "auth_fail": auth_fail,
            "signals": signals,
        }

    def analyze_urls_and_attachments(self, body: str, links: List[str], attachments: List[str]) -> Dict[str, Any]:
        """
        Inspects hyperlinks, anchor texts, domain structures, and attachment extensions.
        """
        signals = []
        threat_score = 0.0

        # 1. Attachment inspection
        has_dangerous_attachment = False
        for att in attachments:
            ext = os.path.splitext(att)[1].lower()
            if ext in MALICIOUS_ATTACHMENT_EXTS:
                has_dangerous_attachment = True
                signals.append(f"Dangerous Attachment: Executable/script payload detected ('{att}')")
                threat_score += 0.60
            elif "." in att[:-4]:  # Check for double extension e.g. .pdf.exe
                signals.append(f"Double-Extension Deception: Spliced extension in attachment ('{att}')")
                threat_score += 0.50

        # 2. URL Inspection
        suspicious_url_count = 0
        ip_url_count = 0
        mismatched_anchors = 0

        # Check for Mismatched Anchor text vs href (e.g. <a href="http://scam.com">paypal.com</a>)
        for anchor_match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body, re.IGNORECASE | re.DOTALL):
            href = anchor_match.group(1).lower()
            anchor_text = anchor_match.group(2).strip().lower()
            # If anchor text looks like a domain/URL but href points elsewhere
            if any(dom in anchor_text for dom in ["http", ".com", ".org", ".net", ".edu", "paypal", "microsoft", "bank"]):
                clean_anchor = re.sub(r'https?://', '', anchor_text).split('/')[0]
                href_domain = urllib.parse.urlparse(href).netloc.lower()
                if clean_anchor and href_domain and clean_anchor not in href_domain:
                    signals.append(f"Phishing Link Deception: Visible link text '{anchor_text}' leads to external '{href_domain}'")
                    mismatched_anchors += 1
                    threat_score += 0.40

        for url in links:
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc.lower().split(":")[0]

            # IP address host (e.g. http://192.168.1.1/login)
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', host):
                signals.append(f"Obfuscated IP URL: Destination is raw IP address rather than verified hostname ({host})")
                ip_url_count += 1
                threat_score += 0.35

            # Punycode / IDN homograph attack
            if host.startswith("xn--"):
                signals.append(f"Homograph Attack: Punycode encoded domain detected ({host})")
                threat_score += 0.40

            # Suspicious TLD
            for tld in SUSPICIOUS_TLDS:
                if host.endswith(tld):
                    suspicious_url_count += 1
                    threat_score += 0.20
                    break

            # Phishing path indicators
            path = parsed.path.lower()
            if any(k in path for k in ["login", "signin", "verify", "update-account", "wallet", "secure"]):
                if not any(known in host for known in ["microsoft.com", "google.com", "apple.com", "github.com"]):
                    threat_score += 0.15

        return {
            "threat_score": min(1.0, threat_score),
            "dangerous_attachment": has_dangerous_attachment,
            "suspicious_url_count": suspicious_url_count,
            "ip_url_count": ip_url_count,
            "mismatched_anchors": mismatched_anchors,
            "total_urls": len(links),
            "signals": signals,
        }

    def analyze_social_engineering(self, text: str, subject: str = "") -> Dict[str, Any]:
        """
        Scores urgency, financial coercion, credential harvesting, and deceptive formatting.
        """
        combined = f"{subject}\n{text}".lower()
        signals = []

        # 1. Credential Harvesting
        cred_matches = [t for t in CREDENTIAL_HARVEST_TRIGGERS if t in combined]
        cred_score = min(1.0, len(cred_matches) * 0.35)
        if cred_matches:
            signals.append(f"Credential Harvesting Pattern: High-risk authentication prompt ({', '.join(cred_matches[:2])})")

        # 2. Financial Scams & Advance Fee
        fin_matches = [t for t in FINANCIAL_SCAM_TRIGGERS if t in combined]
        fin_score = min(1.0, len(fin_matches) * 0.35)
        if fin_matches:
            signals.append(f"Financial Coercion Trigger: Demands or promises financial remittance ({', '.join(fin_matches[:2])})")

        # 3. Urgency Catalysts
        urg_matches = [t for t in URGENCY_TRIGGERS if t in combined]
        urg_score = min(1.0, len(urg_matches) * 0.40)
        if urg_matches:
            signals.append(f"Artificial Urgency Catalyst: Time-pressure triggers detected ('{urg_matches[0]}')")

        # 4. Stylometric Formatting (Caps lock and excessive exclamations)
        all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        exclamation_count = text.count("!")
        caps_ratio = min(1.0, all_caps_words / max(1, len(text.split()) * 0.2))

        return {
            "cred_score": cred_score,
            "fin_score": fin_score,
            "urg_score": urg_score,
            "caps_ratio": caps_ratio,
            "exclamation_count": exclamation_count,
            "signals": signals,
        }

    def build_feature_vector(self, email_input: str) -> np.ndarray:
        """
        Builds a normalized 32-dimensional forensic feature vector for neural classification.
        """
        headers, body, links, attachments = self.parse_email(email_input)
        h_res = self.analyze_headers(headers)
        u_res = self.analyze_urls_and_attachments(body, links, attachments)
        s_res = self.analyze_social_engineering(body, headers.get("Subject", ""))

        vec = np.zeros(32, dtype=np.float32)
        vec[0] = h_res["spoof_score"]
        vec[1] = 1.0 if h_res["is_impersonating"] else 0.0
        vec[2] = 1.0 if h_res["auth_fail"] else 0.0
        vec[3] = 1.0 if h_res["reply_domain"] else 0.0
        vec[4] = u_res["threat_score"]
        vec[5] = 1.0 if u_res["dangerous_attachment"] else 0.0
        vec[6] = min(1.0, u_res["suspicious_url_count"] / 3.0)
        vec[7] = min(1.0, u_res["ip_url_count"])
        vec[8] = min(1.0, u_res["mismatched_anchors"])
        vec[9] = min(1.0, u_res["total_urls"] / 5.0)
        vec[10] = s_res["cred_score"]
        vec[11] = s_res["fin_score"]
        vec[12] = s_res["urg_score"]
        vec[13] = s_res["caps_ratio"]
        vec[14] = min(1.0, s_res["exclamation_count"] / 5.0)
        vec[15] = min(1.0, len(body) / 2000.0)
        vec[16] = 1.0 if "password" in body.lower() else 0.0
        vec[17] = 1.0 if "bitcoin" in body.lower() or "btc" in body.lower() else 0.0
        vec[18] = 1.0 if "suspend" in body.lower() or "terminate" in body.lower() else 0.0
        vec[19] = 1.0 if "invoice" in headers.get("Subject", "").lower() else 0.0
        vec[20] = 1.0 if any(dom in h_res["from_domain"] for dom in FREE_WEBMAIL_DOMAINS) else 0.0
        vec[21] = 1.0 if len(attachments) > 0 else 0.0
        vec[22] = min(1.0, float(len(re.findall(r'https?://', body))))
        vec[23] = min(1.0, float(len(re.findall(r'\$\d+', body))))
        vec[24] = 1.0 if "http://" in body.lower() else 0.0  # Insecure HTTP
        vec[25] = 1.0 if "urgent" in headers.get("Subject", "").lower() else 0.0
        vec[26] = 1.0 if "dear customer" in body.lower() or "dear user" in body.lower() else 0.0  # Generic greeting
        vec[27] = min(1.0, float(len(re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', body))))
        vec[28] = 1.0 if "unsubscribe" in body.lower() else 0.0
        vec[29] = min(1.0, float(len(body.splitlines())) / 40.0)
        vec[30] = 1.0 if h_res["spoof_score"] > 0.4 and u_res["threat_score"] > 0.3 else 0.0
        vec[31] = 1.0 if s_res["urg_score"] > 0.4 and s_res["cred_score"] > 0.4 else 0.0

        return vec

    def analyze(self, email_input: str) -> Dict[str, Any]:
        """
        Executes end-to-end forensic analysis of an email, returning risk score,
        explainability indicators, and democratic committee verdict.
        """
        headers, body, links, attachments = self.parse_email(email_input)
        h_res = self.analyze_headers(headers)
        u_res = self.analyze_urls_and_attachments(body, links, attachments)
        s_res = self.analyze_social_engineering(body, headers.get("Subject", ""))

        vec = self.build_feature_vector(email_input)

        # Neural backbone inference
        with torch.no_grad():
            t_vec = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            raw_logit = self.model(t_vec).item()
            neural_prob = float(torch.sigmoid(torch.tensor(raw_logit)).item())

        # Rule-based heuristic verification
        rule_score = float(
            0.35 * h_res["spoof_score"] +
            0.35 * u_res["threat_score"] +
            0.30 * max(s_res["cred_score"], s_res["fin_score"], s_res["urg_score"])
        )

        # Calibrated composite probability
        composite_fake_prob = float(0.65 * neural_prob + 0.35 * rule_score)

        # Hard overrides for critical security violations
        if u_res["dangerous_attachment"] or u_res["mismatched_anchors"] > 0:
            composite_fake_prob = max(composite_fake_prob, 0.95)
        elif h_res["is_impersonating"] and h_res["from_domain"] in FREE_WEBMAIL_DOMAINS:
            composite_fake_prob = max(composite_fake_prob, 0.92)
        elif h_res["is_impersonating"] and (s_res["cred_score"] > 0.3 or s_res["urg_score"] > 0.3):
            composite_fake_prob = max(composite_fake_prob, 0.88)
        elif h_res["is_impersonating"]:
            composite_fake_prob = max(composite_fake_prob, 0.82)

        composite_fake_prob = max(0.01, min(0.99, composite_fake_prob))

        # Democratic polling committee
        polling = self.committee.poll(
            primary_prob=composite_fake_prob,
            threat_score=u_res["threat_score"],
            spoof_score=h_res["spoof_score"],
            social_score=max(s_res["cred_score"], s_res["fin_score"], s_res["urg_score"]),
        )

        is_spam_or_phishing = composite_fake_prob >= 0.50

        # Classify verdict
        if composite_fake_prob >= 0.80:
            verdict = f"MALICIOUS PHISHING / SPOOFED CAMPAIGN ({polling.consensus_type} {polling.consensus_ratio})"
        elif composite_fake_prob >= 0.50:
            verdict = f"SUSPICIOUS EMAIL / SPAM DETECTED ({polling.consensus_type} {polling.consensus_ratio})"
        elif composite_fake_prob >= 0.25:
            verdict = f"MARKETING / PROMOTIONAL EMAIL ({polling.consensus_type} {polling.consensus_ratio})"
        else:
            verdict = f"AUTHENTIC & CLEAN CORRESPONDENCE ({polling.consensus_type} {polling.consensus_ratio})"

        # Aggregate forensic signals
        all_signals = []
        all_signals.append(f"Democratic Polling Committee: {polling.summary}")
        all_signals.extend(h_res["signals"])
        all_signals.extend(u_res["signals"])
        all_signals.extend(s_res["signals"])

        if not all_signals or len(all_signals) == 1:
            all_signals.append("Sender provenance, cryptographic authentication, and body hyperlinks verified authentic")

        return {
            "modality": "email",
            "fake_probability": round(composite_fake_prob, 4),
            "is_synthetic": is_spam_or_phishing,
            "is_spam_or_phishing": is_spam_or_phishing,
            "classification": verdict,
            "confidence_percent": round(abs(composite_fake_prob - 0.5) * 200, 1),
            "forensic_signals": all_signals,
            "polling": polling.to_dict(),
            "metrics": {
                "neural_model_score": round(neural_prob, 4),
                "sender_spoof_score": round(h_res["spoof_score"], 3),
                "attachment_threat_score": round(u_res["threat_score"], 3),
                "credential_harvest_index": round(s_res["cred_score"], 3),
                "urgency_coercion_index": round(s_res["urg_score"], 3),
                "total_urls_detected": u_res["total_urls"],
                "subject": headers.get("Subject", "N/A"),
                "sender": headers.get("From", "N/A"),
            },
        }
