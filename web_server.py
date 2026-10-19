#!/usr/bin/env python3
"""
The Turing Project (2026) — Standalone Forensic Web Server & REST API
Zero external dependencies (uses standard library http.server).
Serves futuristic UI and provides /api/scan and /api/datasets endpoints.
"""

import http.server
import json
import os
import socketserver
import sys
import urllib.parse
from detector.fusion import MultimodalFusionDetector
from detector.datasets_catalog import DATASETS_CATALOG


PORT = 8080
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class TuringProjectHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/datasets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(DATASETS_CATALOG, indent=2).encode("utf-8"))
            return
        elif parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "version": "2026.1.0"}).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/scan":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            text_input = data.get("text")
            news_input = data.get("news")
            verify_news = data.get("verify_news", False)

            detector = MultimodalFusionDetector()
            if verify_news or news_input:
                report = detector.scan_multimodal(news_input=news_input or text_input)
            elif text_input:
                report = detector.scan(text_input)
            else:
                report = detector.scan_multimodal()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(report.to_dict(), indent=2).encode("utf-8"))
            return
        elif parsed.path == "/api/verify":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            claim = data.get("claim") or data.get("text", "")
            detector = MultimodalFusionDetector()
            report = detector.scan_multimodal(news_input=claim)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(report.to_dict(), indent=2).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_server(port=PORT):
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), TuringProjectHandler) as httpd:
        print(f"The Turing Project (2026) Web Server running at http://127.0.0.1:{port}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(p)
