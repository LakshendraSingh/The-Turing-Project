#!/usr/bin/env python3
"""
Builds the Email Spam, Phishing & Forensic Benchmark Dataset for The Turing Project.
Generates comprehensive real-world scenarios:
- Authentic corporate correspondence, pull requests, engineering discussions, calendar invites, receipts
- Malicious phishing, CEO fraud / BEC, credential harvesting, double extensions, Nigerian 419, urgent suspensions
"""

import json
import os

EMAIL_TRAIN_DATA = [
    # --- PHISHING & MALICIOUS SAMPLES (Label: 1.0) ---
    {
        "id": "phish_001",
        "label": 1.0,
        "type": "credential_harvesting",
        "text": """From: "PayPal Security Center" <support@service-update-security-alert.xyz>
To: victim@example.com
Subject: URGENT: Your PayPal account has been limited! Immediate action required.
Date: Mon, 19 Oct 2026 10:14:00 +0000
Authentication-Results: spf=fail; dkim=fail; dmarc=fail

Dear Valued Customer,

We detected unauthorized login attempts to your PayPal wallet from an unrecognized device in Vladivostok, Russia. For your safety, your account access has been temporarily restricted.

You must verify your password and confirm your account within 24 hours, or your account will be closed permanently and remaining funds confiscated.

Click here to verify: https://paypal.com.secure-login-verification.xyz/signin

Sincerely,
PayPal Security Department"""
    },
    {
        "id": "phish_002",
        "label": 1.0,
        "type": "ceo_fraud_bec",
        "text": """From: "Satya Nadella (CEO)" <ceo.direct.urgent@gmail.com>
To: finance@enterprise-corp.com
Reply-To: executive-wire-transfers@consultant.ru
Subject: Quick confidential request - Wire transfer today
Date: Mon, 19 Oct 2026 11:22:15 +0000

Are you at your desk? I am currently in an all-day executive board meeting and cannot take calls.

We are closing a confidential overseas acquisition today. I need you to initiate an urgent payment of $48,500 immediately to our outside legal counsel via wire transfer.

Please confirm when you are ready so I can forward the bank details. Keep this strictly between us until the press release.

Sent from my iPad"""
    },
    {
        "id": "phish_003",
        "label": 1.0,
        "type": "malware_double_extension",
        "text": """From: "Billing Dept" <invoices@quick-accounting-service.top>
To: target@company.org
Subject: OVERDUE: Unpaid Invoice #INV-984210 - Final Notice
Content-Disposition: attachment; filename="Overdue_Invoice_984210.pdf.exe"

Attached is the final notice for invoice #INV-984210 which is now 45 days past due.
Please review the attached invoice immediately to avoid legal collection action and credit damage.

Attachment: Overdue_Invoice_984210.pdf.exe

Regards,
Accounts Receivable"""
    },
    {
        "id": "phish_004",
        "label": 1.0,
        "type": "anchor_text_deception",
        "text": """From: "Microsoft 365 Admin" <no-reply@auth-notifications.club>
To: user@firm.com
Subject: Warning: Your Microsoft 365 Mailbox quota exceeded (99% full)

Dear User,

Your email mailbox has exceeded its storage limit. You will not be able to send or receive new emails until you re-authenticate your mailbox storage.

<a href="http://192.168.10.45/harvest/login.php">https://login.microsoftonline.com/common/oauth2/authorize</a>

Immediate response required within 12 hours to prevent permanent email deletion."""
    },
    {
        "id": "phish_005",
        "label": 1.0,
        "type": "419_advance_fee",
        "text": """From: "Barrister Ahmed Bello" <barrister.bello.chambers@yahoo.com>
Subject: CONFIDENTIAL: Unclaimed inheritance fund ($18.5 Million USD)

Dear Friend,

I am the personal attorney to a late foreign national who died in a tragic plane crash along with his immediate family. He left behind a deposit of Eighteen Million Five Hundred Thousand United States Dollars ($18,500,000.00) in a top security bank here.

Because you share the same surname, I seek your partnership to present you as the legal heir. We will share the funds 50/50. Please reply with your full legal name, private telephone number, and bank account details immediately."""
    },
    {
        "id": "phish_006",
        "label": 1.0,
        "type": "crypto_extortion",
        "text": """From: "DarkWeb Hacker Group" <hacker@ransom-syndicate.live>
Subject: Security Alert: Your device has been compromised! Payment required.

I have placed a remote trojan on your computer and recorded your webcam while you visited adult websites. I have copied your entire contact list from WhatsApp and email.

If you do not send $1,200 in Bitcoin (BTC) to wallet bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh within 24 hours, I will send the video to all your friends, family, and coworkers.

Do not try to contact the police or the timer will expire immediately."""
    },
    {
        "id": "phish_007",
        "label": 1.0,
        "type": "homograph_attack",
        "text": """From: "Apple Support" <notifications@xn--appl-43a.com>
Subject: Critical: Your Apple ID has been suspended

Your Apple ID was used to sign in from an unknown device in Nigeria.
To safeguard your iCloud photos and contacts, verify your identity immediately:
http://xn--appl-43a.com/verify-identity/session_id=89234"""
    },
    {
        "id": "phish_008",
        "label": 1.0,
        "type": "lottery_scam",
        "text": """From: "UK National Lottery" <prizes@lottery-winner-notification.loan>
Subject: CONGRATULATIONS! You have won £2,500,000 in the EuroMillions draw!

Your email was randomly selected from internet users across the globe.
To claim your lottery winnings, send processing fee of £350 via Western Union or Google Play gift cards to our claim agent."""
    },

    # --- AUTHENTIC SAMPLES (Label: 0.0) ---
    {
        "id": "real_001",
        "label": 0.0,
        "type": "corporate_standup",
        "text": """From: "Priya Sharma" <priya.sharma@techcorp.internal>
To: eng-team@techcorp.internal
Subject: Engineering Standup Notes — Monday Oct 19
Date: Mon, 19 Oct 2026 09:30:00 +0000

Hi Team,

Quick summary from today's sprint sync:
1. Backend: Payment webhook refactor merged into staging (PR #412).
2. Mobile: iOS release candidate 2.4.1 submitted to TestFlight.
3. QA: Regression testing across auth flows ongoing, zero P0 blockers found.

Next sprint planning is scheduled for Wednesday at 2:00 PM in Conference Room B.
Have a great week!

Best,
Priya Sharma
Engineering Lead | TechCorp"""
    },
    {
        "id": "real_002",
        "label": 0.0,
        "type": "github_notification",
        "text": """From: "GitHub" <notifications@github.com>
To: developer@github.com
Subject: [LakshendraSingh/The-Turing-Project] Pull Request #12 opened: Add email spam detection engine
Date: Mon, 19 Oct 2026 14:00:22 +0000
Authentication-Results: dkim=pass (good signature); spf=pass

@contributor opened a pull request in LakshendraSingh/The-Turing-Project:

Feature: Modular Email Security & Phishing Forensic Engine
- Extends multimodal fusion pipeline to support RFC 822 / MIME emails
- Evaluated against standard phishing and corporate communication benchmarks

View on GitHub: https://github.com/LakshendraSingh/The-Turing-Project/pull/12
You are receiving this because you authored the repository."""
    },
    {
        "id": "real_003",
        "label": 0.0,
        "type": "calendar_invitation",
        "text": """From: "Arun Verma" <arun.verma@university.edu>
To: lakshendra@university.edu
Subject: Invitation: B.Tech Major Project Review @ Thu Oct 22, 2026 3pm - 4pm (IST)
Date: Mon, 19 Oct 2026 12:00:00 +0000

You have been invited to the following event:
Event: B.Tech Major Project Mid-Term Evaluation
When: Thursday Oct 22, 2026 · 3pm – 4pm (IST)
Where: Department Seminar Hall 3

Agenda:
- 15 min project architecture walkthrough
- 10 min live demonstration
- 15 min evaluation panel Q&A

Please bring 3 printed copies of your project report synopsis."""
    },
    {
        "id": "real_004",
        "label": 0.0,
        "type": "transactional_receipt",
        "text": """From: "Amazon.in Orders" <auto-confirm@amazon.in>
To: customer@gmail.com
Subject: Your Amazon.in order #408-9812401-1928471 has been dispatched
Authentication-Results: spf=pass; dkim=pass

Hello,

Your package containing 'Clean Code: A Handbook of Agile Software Craftsmanship' is on its way.
Estimated delivery date: Wednesday, 21 October 2026.

Courier: BlueDart Express
Tracking ID: AWB982140129

Track your package directly inside the Amazon app or visit:
https://www.amazon.in/gp/your-account/order-history

Thank you for shopping with us."""
    },
    {
        "id": "real_005",
        "label": 0.0,
        "type": "airline_boarding",
        "text": """From: "Air India Bookings" <noreply@airindia.com>
To: passenger@corp.com
Subject: Web Check-In Confirmation: Flight AI-804 (DEL to BOM)
Date: Mon, 19 Oct 2026 16:45:00 +0000

Dear Passenger,

Your web check-in is complete for flight AI-804.
Seat: 12F (Window)
Departure: Indira Gandhi International Airport (Terminal 3)
Boarding Time: 07:15 AM

Please ensure you carry a valid government-issued photo ID at airport security.
Have a pleasant flight."""
    },
    {
        "id": "real_006",
        "label": 0.0,
        "type": "academic_newsletter",
        "text": """From: "ACM TechNews" <technews@acm.org>
To: acm-members@acm.org
Subject: ACM TechNews: Recent Advances in Digital Media Provenance and Cryptographic Watermarking
Authentication-Results: spf=pass; dkim=pass

ACM TechNews — October 2026 Edition
In this issue:
1. C2PA consortium releases upgraded manifest specifications for edge devices.
2. University researchers unveil benchmark dataset for acoustic voice cloning forensics.
3. Call for papers: IEEE International Workshop on Information Forensics and Security.

Read full articles at https://technews.acm.org
To update your subscription preferences, visit your ACM member dashboard."""
    }
]

EMAIL_TEST_DATA = [
    # Test holdout phishing
    {
        "id": "test_phish_001",
        "label": 1.0,
        "type": "bank_phish",
        "text": """From: "HDFC Bank NetBanking" <alert@hdfc-security-update-portal.top>
To: customer@gmail.com
Subject: URGENT: Your Debit Card has been blocked due to suspicious activity
Authentication-Results: spf=fail; dkim=fail

Dear Customer,

We noticed unusual withdrawal attempts on your HDFC account. Your card is temporarily suspended.
Please update your credentials and verify your password within 24 hours at http://hdfc-security-update-portal.top/login.php

Failure to act will result in complete account closure.
HDFC Security Team"""
    },
    {
        "id": "test_phish_002",
        "label": 1.0,
        "type": "macro_malware",
        "text": """From: "Accounts Payable" <invoices@vendor-remittance.xyz>
Subject: Remittance Advice - Urgent Payment confirmation
Content-Disposition: attachment; filename="Payment_Remittance_Advice.xlsm"

Please find attached the signed remittance confirmation. Enable macros upon opening to view encrypted financial table.

Attachment: Payment_Remittance_Advice.xlsm"""
    },
    {
        "id": "test_phish_003",
        "label": 1.0,
        "type": "free_webmail_ceo",
        "text": """From: "Sundar Pichai (CEO)" <sundar.pichai.personal@gmail.com>
Reply-To: confidential.fund.handler@mail.ru
Subject: Quick favor - Are you available right now?

I am currently tied up in client meetings. Need you to purchase 10 Apple gift cards ($100 each) for our customer appreciation event.
Send photos of the scratch codes on the back to this email directly. I will reimburse you by tomorrow morning."""
    },
    {
        "id": "test_phish_004",
        "label": 1.0,
        "type": "cloud_storage_harvest",
        "text": """From: "Google Drive Notification" <share-docs@drive-shared-file-access.buzz>
Subject: Document shared with you: 'Q3 Financial Bonus Allocation.pdf'

A confidential document has been shared with you on Google Drive.
<a href="http://185.220.101.5/google_phish/index.html">https://drive.google.com/file/d/10842918471/view</a>

Sign in with your Google account credentials to view the document."""
    },
    # Test holdout clean
    {
        "id": "test_clean_001",
        "label": 0.0,
        "type": "code_review",
        "text": """From: "Kavita Rao" <kavita.rao@techcorp.internal>
To: lakshendra@techcorp.internal
Subject: Code review feedback on PR #89 (Forensic Fusion Pipeline)
Date: Mon, 19 Oct 2026 15:10:00 +0000

Hey Lakshendra,

Left a few comments on the PR:
- Loved the modular separation in detector/fusion.py!
- Let's make sure the unit tests run under 20s in CI.
- Added one nit on error handling in the MIME parser.

LGTM once those minor comments are addressed.

Cheers,
Kavita"""
    },
    {
        "id": "test_clean_002",
        "label": 0.0,
        "type": "payroll_stub",
        "text": """From: "TechCorp Payroll" <payroll-noreply@techcorp.internal>
To: employee@techcorp.internal
Subject: Your TechCorp Payslip for October 2026 is now available
Authentication-Results: spf=pass; dkim=pass

Dear Employee,

Your monthly compensation statement for the period ending 31 October 2026 has been generated.
You can view and download your payslip by logging into the internal Workday portal:
https://workday.techcorp.internal/payroll

For any tax deduction queries, please raise a ticket with HR Helpdesk.

Warm regards,
Global Payroll Operations"""
    },
    {
        "id": "test_clean_003",
        "label": 0.0,
        "type": "zoom_meeting",
        "text": """From: "Zoom Video Communications" <no-reply@zoom.us>
To: attendee@example.com
Subject: Meeting Scheduled: Architecture Review with Deepmind AI Team
Date: Mon, 19 Oct 2026 17:00:00 +0000
Authentication-Results: spf=pass; dkim=pass

Hi there,

Deepmind AI Team is inviting you to a scheduled Zoom meeting.
Topic: Architecture Review: Edge Forensics Pipeline
Time: Oct 20, 2026 04:00 PM India

Join Zoom Meeting: https://zoom.us/j/98124018241
Meeting ID: 981 2401 8241

Passcode: 481920"""
    },
    {
        "id": "test_clean_004",
        "label": 0.0,
        "type": "github_sponsor",
        "text": """From: "GitHub Sponsors" <sponsors-noreply@github.com>
To: lakshendra@users.noreply.github.com
Subject: Monthly Sponsorship Receipt for Open Source Projects
Authentication-Results: spf=pass; dkim=pass

Thank you for supporting open source software!
Your monthly sponsorship of $10 to PyTorch Foundation has been processed successfully.

Transaction ID: gh_sp_9812401
Payment Method: Mastercard ending in 4102

Manage your sponsorships: https://github.com/sponsors"""
    }
]


def build_and_save_dataset():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(data_dir, "data", "benchmark", "email")
    test_dir = os.path.join(data_dir, "data", "test_benchmark", "email")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_file = os.path.join(train_dir, "email_spam_train.json")
    test_file = os.path.join(test_dir, "email_phishing_test.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(EMAIL_TRAIN_DATA, f, indent=2)

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(EMAIL_TEST_DATA, f, indent=2)

    print(f"Created Email Train Dataset with {len(EMAIL_TRAIN_DATA)} samples at {train_file}")
    print(f"Created Email Test Dataset with {len(EMAIL_TEST_DATA)} samples at {test_file}")


if __name__ == "__main__":
    build_and_save_dataset()
