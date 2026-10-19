#!/usr/bin/env python3
"""
The Turing Project (2026 CLI) — Multimodal Multimodal Deepfake & Synthetic Media Detector
Usage:
  python cli.py datasets
  python cli.py scan --image sample.jpg
  python cli.py scan --audio sample.wav
  python cli.py scan --text "Sample text to check..."
  python cli.py scan --doc invoice.png
  python cli.py scan --image sample.jpg --text "Breaking news headline..." --json
"""

import argparse
import glob
import json
import os
import sys

# Auto-discover local virtual environment packages if running via system python
_workspace_dir = os.path.dirname(os.path.abspath(__file__))
for _sp in glob.glob(os.path.join(_workspace_dir, ".venv", "lib", "python*", "site-packages")):
    if os.path.exists(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

from detector.fusion import MultimodalFusionDetector
from detector.datasets_catalog import print_catalog_summary, DATASETS_CATALOG


# Terminal ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_risk_badge(risk_level: str) -> str:
    if risk_level == "CRITICAL":
        return f"{RED}{BOLD}[CRITICAL RISK - SYNTHETIC DETECTED]{RESET}"
    elif risk_level == "HIGH":
        return f"{RED}{BOLD}[HIGH RISK - LIKELY FAKE]{RESET}"
    elif risk_level == "MODERATE":
        return f"{YELLOW}{BOLD}[MODERATE RISK - SUSPICIOUS]{RESET}"
    else:
        return f"{GREEN}{BOLD}[VERIFIED CLEAN / AUTHENTIC]{RESET}"


def print_report(report_dict: dict):
    print("\n" + "=" * 70)
    print(f"{CYAN}{BOLD}THE TURING PROJECT — 2026 FORENSIC ANALYSIS REPORT{RESET}")
    print("=" * 70)
    print(f"Overall Verdict:     {BOLD}{report_dict['overall_verdict']}{RESET}")
    print(f"Risk Assessment:     {format_risk_badge(report_dict['risk_level'])}")
    print(f"Fake Probability:    {BOLD}{report_dict['overall_fake_probability'] * 100:.1f}%{RESET}")
    print("-" * 70)

    print(f"{BOLD}Modality Breakdown:{RESET}")
    for mod, data in report_dict["modality_results"].items():
        p = data.get("fake_probability", 0.0)
        status_color = RED if data.get("is_synthetic", False) or data.get("is_misinformation", False) else GREEN

        if mod == "news":
            verdict = data.get("veracity_verdict", "N/A")
            cred = data.get("credibility_score", 0.0)
            print(f"  • {BOLD}NEWS / CLAIM{RESET:8}: {status_color}{verdict}{RESET} (Credibility: {cred * 100:.1f}%)")
            ca = data.get("claim_analysis", {})
            if "ml_model_prediction" in ca:
                mlp = ca["ml_model_prediction"]
                ml_prob = mlp.get("fake_probability", 0.0)
                ml_color = RED if ml_prob >= 0.50 else GREEN
                print(f"    {CYAN}↳{RESET} ML Model Prediction: {ml_color}{mlp.get('verdict', 'N/A')}{RESET} (Disinformation Prob: {ml_prob * 100:.1f}%)")
            if "sensationalism_index" in ca:
                print(f"    {CYAN}↳{RESET} Sensationalism Index: {ca['sensationalism_index'] * 100:.0f}%")
            if ca.get("rural_scam_flags"):
                for flag in ca["rural_scam_flags"]:
                    print(f"    {RED}↳ Rural Scam Alert:{RESET} {flag}")
            if ca.get("detected_domains"):
                print(f"    {CYAN}↳{RESET} Recognized Domains: {', '.join(ca['detected_domains'])}")
            if data.get("recommendation"):
                print(f"    {YELLOW}↳ Guidance:{RESET} {data['recommendation']}")
        else:
            print(f"  • {BOLD}{mod.upper()}{RESET:10}: {status_color}{data.get('classification', 'N/A')}{RESET} (Confidence: {p * 100:.1f}%)")
            if mod == "document" and "extraction_telemetry" in data:
                telem = data["extraction_telemetry"]
                print(f"    {CYAN}↳{RESET} Extracted Text: {telem.get('text_word_count', 0)} words analyzed")
                print(f"    {CYAN}↳{RESET} Extracted Images: {telem.get('extracted_images_count', 0)} embedded images/stamps analyzed")

            if "metrics" in data and "metadata_audit" in data["metrics"]:
                ma = data["metrics"]["metadata_audit"]
                if ma.get("c2pa", {}).get("has_c2pa"):
                    c2pa_status = f"{RED}AI-Generated Claim{RESET}" if ma["c2pa"].get("is_ai_generated") else f"{GREEN}Camera Verified{RESET}"
                    auth = ma["c2pa"].get("signing_authority") or "Verified Issuer"
                    print(f"    {CYAN}↳{RESET} C2PA Credentials: [{c2pa_status}] (Issuer: {auth})")
                    if ma["c2pa"].get("synthid_watermark_present"):
                        print(f"    {CYAN}↳{RESET} Watermark: {YELLOW}Google SynthID Imperceptible Watermark Confirmed{RESET}")
                if ma.get("has_camera_hardware"):
                    make_model = f"{ma.get('camera_make', '')} {ma.get('camera_model', '')}".strip()
                    print(f"    {CYAN}↳{RESET} Camera Hardware EXIF: {GREEN}{make_model}{RESET}")
                elif ma.get("software"):
                    print(f"    {CYAN}↳{RESET} Software / Tool Tag: {YELLOW}{ma.get('software')}{RESET}")

        if "polling" in data:
            pol = data["polling"]
            tally_str = f"{RED if pol['tally']['FAKE'] > 0 else GREEN}{pol['tally']['FAKE']} FAKE{RESET} vs {GREEN}{pol['tally']['REAL']} REAL{RESET}"
            print(f"\n    {BOLD}Democratic Polling Committee ({pol['num_models']} Specialized Models):{RESET}")
            print(f"    Democratic Consensus: {CYAN}{pol['consensus_type']} ({pol['consensus_ratio']}){RESET} | Tally: [{tally_str}]")
            for idx, v in enumerate(pol["votes"], 1):
                v_color = RED if v["vote"] == "FAKE" else GREEN
                print(f"      Model {idx}: {BOLD}{v['model_name']}{RESET}")
                print(f"        • Vote: {v_color}[{v['vote']}]{RESET} (p={v['fake_prob']*100:.1f}%) | Specialty: {v['specialty']}")
                print(f"        • Forensic Signal: {v['rationale']}")

    if report_dict["detected_manipulations"]:
        print(f"\n{BOLD}Detected Forensic Signals & Artifacts:{RESET}")
        for sig in report_dict["detected_manipulations"]:
            print(f"  {YELLOW}[!]{RESET} {sig}")

    if report_dict["cross_modal_discrepancies"]:
        print(f"\n{BOLD}{RED}Cross-Modal Asymmetries / Inconsistencies:{RESET}")
        for disc in report_dict["cross_modal_discrepancies"]:
            print(f"  {RED}[!]{RESET} {disc}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="The Turing Project — 2026 Multimodal Deepfake Detector")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Catalog command
    subparsers.add_parser("datasets", help="Display 2026 Deepfake Benchmark Datasets Catalog")

    # Verify command (Specialized news & rumor fact-checker)
    verify_p = subparsers.add_parser("verify", help="Verify news headline, viral claim, or WhatsApp rumor")
    verify_p.add_argument("claim", type=str, help="Headline, statement, article URL or rumor text to verify")
    verify_p.add_argument("--image", type=str, help="Optional attached image to audit for cheapfake/out-of-context synthesis")
    verify_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Scan command (Automagically routes or accepts explicit overrides)
    scan_p = subparsers.add_parser("scan", help="Run forensic scan (automagically detects input modality)")
    scan_p.add_argument("target", nargs="?", type=str, default=None, help="File path (image/audio/video/doc) or raw text string to auto-analyze")
    scan_p.add_argument("--image", type=str, help="Explicit path to image file (jpg/png/webp)")
    scan_p.add_argument("--audio", type=str, help="Explicit path to audio file (mp3/wav/flac)")
    scan_p.add_argument("--video", type=str, help="Explicit path to video file (mp4/avi/mov/npz)")
    scan_p.add_argument("--text", type=str, help="Explicit text string or path to text file")
    scan_p.add_argument("--doc", type=str, help="Explicit path to document page image or PDF")
    scan_p.add_argument("--news", type=str, help="Explicit news headline, viral claim, or rumor to verify")
    scan_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.command == "datasets":
        print_catalog_summary()
        return

    detector = MultimodalFusionDetector()

    if args.command == "verify":
        report = detector.scan_multimodal(
            news_input=args.claim,
            image_input=args.image,
        )
        res_dict = report.to_dict()
        if args.json:
            print(json.dumps(res_dict, indent=2))
        else:
            print_report(res_dict)
        return

    if args.command == "scan":
        # Check if text is a file path
        text_content = args.text
        if text_content and text_content.endswith(".txt"):
            try:
                with open(text_content, "r", encoding="utf-8") as tf:
                    text_content = tf.read()
            except Exception:
                pass

        # If a single positional target is provided, automagically route to correct model
        if args.target:
            report = detector.scan(
                args.target,
                image_input=args.image,
                audio_input=args.audio,
                video_frames=args.video,
                text_input=text_content,
                document_input=args.doc,
                news_input=args.news,
            )
        else:
            report = detector.scan_multimodal(
                image_input=args.image,
                audio_input=args.audio,
                video_frames=args.video,
                text_input=text_content,
                document_input=args.doc,
                news_input=args.news,
            )

        res_dict = report.to_dict()
        if args.json:
            print(json.dumps(res_dict, indent=2))
        else:
            print_report(res_dict)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
