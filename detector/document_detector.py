"""
Document Forgery, Tampering & Deepfake Detector (2026)
Detects Forged PDFs, Altered Invoices, Fake IDs, Tampered Contracts, and Spliced Receipts.
Comprehensive Multimodal Document Forensics:
- Automated Text Extraction & LLM Synthetic Content Analysis (Fast-DetectGPT / Burstiness)
- Automated Embedded Image / Stamp Extraction & Diffusion/GAN Forensic Inspection
- Metadata & Revision History Integrity Audit (Producer signatures, incremental %%EOF updates)
- Error Level Analysis (ELA) on Document Canvas (spliced numbers, pasted stamps)
- Font Anti-Aliasing Edge Gradient Disparity (digitally inserted text vs scanned background)
- Copy-Move Forgery Detection (CMFD) for duplicated stamps, signatures, or numbers
"""

import io
import math
import os
import re
import sys
import glob
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import torch
import torch.nn as nn

# Auto-discover local virtual environment packages (pypdf, etc.)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sp in glob.glob(os.path.join(_repo_root, ".venv", "lib", "python*", "site-packages")):
    if os.path.exists(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

from .image_detector import ImageDeepfakeDetector
from .text_detector import TextDeepfakeDetector


class DocumentTamperClassifier(nn.Module):
    """
    2026 Self-Attentive Multimodal Document Classifier:
    - Tokenizes each forensic modality indicator (ELA, gradient disparity, CMFD, metadata, text LLM, image deepfake)
    - Multi-Head Self-Attention across forensic channels to capture non-linear correlation attacks
    - Residual transformer encoder block with LayerNorm and feedforward expansion
    """
    def __init__(self, in_features=8, embed_dim=16, num_heads=2, num_classes=1, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        # Token projection: map each scalar feature to embed_dim token
        self.token_proj = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: [B, in_features]
        b = x.shape[0]
        tokens = self.token_proj(x.unsqueeze(-1))  # [B, in_features, embed_dim]
        cls = self.cls_token.expand(b, -1, -1)     # [B, 1, embed_dim]
        seq = torch.cat([cls, tokens], dim=1)      # [B, in_features + 1, embed_dim]

        # Multi-Head Self-Attention
        norm_seq = self.norm1(seq)
        attn_out, _ = self.self_attn(norm_seq, norm_seq, norm_seq)
        seq = seq + attn_out
        seq = seq + self.ffn(self.norm2(seq))

        # Combine CLS token with Mean-pooled tokens
        cls_out = seq[:, 0, :]
        mean_out = torch.mean(seq[:, 1:, :], dim=1)
        pooled = torch.cat([cls_out, mean_out], dim=-1)

        return self.head(pooled)


class DocumentForensicsDetector:
    # Known graphic editor signatures that should not be the producer of authentic official records
    SUSPICIOUS_PRODUCERS = [
        "photoshop", "gimp", "canva", "illustrator", "paint.net",
        "inkscape", "pdf-xchange editor", "ilovepdf", "sejda", "smallpdf"
    ]

    def __init__(self, device: str = None):
        self.image_detector = ImageDeepfakeDetector(device=device)
        self.text_detector = TextDeepfakeDetector(device=device)
        self.device = self.image_detector.device
        self.tamper_classifier = DocumentTamperClassifier().to(self.device)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "document_best.pt")
        if os.path.exists(weights_path):
            try:
                self.tamper_classifier.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.tamper_classifier.eval()
        from .polling import DocumentForensicsCommittee
        self.committee = DocumentForensicsCommittee(device=self.device)

    @staticmethod
    def build_feature_vector(
        ela_score: float,
        edge_disparity: float,
        cmfd_score: float,
        meta_score: float,
        text_fake_score: float,
        img_fake_score: float,
        text_word_count: int,
        img_count: int,
    ) -> np.ndarray:
        return np.array([
            float(ela_score),
            float(edge_disparity),
            float(cmfd_score),
            float(meta_score),
            float(text_fake_score),
            float(img_fake_score),
            min(1.0, float(text_word_count) / 500.0),
            min(1.0, float(img_count) / 5.0)
        ], dtype=np.float32)

    def extract_pdf_content(self, pdf_input) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Extracts all textual content and embedded images/figures/stamps from a PDF document.
        Returns: (extracted_text, extracted_images_list, raw_pdf_stream)
        """
        extracted_text = ""
        extracted_images = []
        raw_stream = ""

        # Read raw stream for metadata audit
        if isinstance(pdf_input, str):
            with open(pdf_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(pdf_input, bytes):
            raw_bytes = pdf_input
        elif isinstance(pdf_input, io.BytesIO):
            raw_bytes = pdf_input.getvalue()
        else:
            raw_bytes = b""

        raw_stream = raw_bytes.decode("latin-1", errors="ignore")

        # Use pypdf for deep extraction
        try:
            import pypdf
            bio = io.BytesIO(raw_bytes)
            reader = pypdf.PdfReader(bio)

            text_chunks = []
            for page_idx, page in enumerate(reader.pages):
                # 1. Extract Text
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text.strip())

                # 2. Extract Embedded Images / Stamps / Photos
                try:
                    for img_idx, img_obj in enumerate(page.images):
                        try:
                            pil_img = img_obj.image.convert("RGB")
                            # Only analyze meaningful images (larger than 32x32)
                            if pil_img.width >= 32 and pil_img.height >= 32:
                                extracted_images.append({
                                    "name": f"page_{page_idx+1}_{img_obj.name}",
                                    "page": page_idx + 1,
                                    "image": pil_img,
                                    "size": pil_img.size,
                                })
                        except Exception:
                            continue
                except Exception:
                    pass

            extracted_text = "\n\n".join(text_chunks)
            if not extracted_text:
                text_matches = re.findall(r"\((.*?)\)\s*Tj", raw_stream)
                if text_matches:
                    extracted_text = " ".join(text_matches)
        except Exception:
            # Fallback regex text stream extraction
            text_matches = re.findall(r"\((.*?)\)\s*Tj", raw_stream)
            if text_matches:
                extracted_text = " ".join(text_matches)

        return extracted_text, extracted_images, raw_stream

    def extract_rois_from_scanned_image(self, page_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Heuristic ROI segmenter for scanned document images (e.g. driver licenses, ID cards, receipts).
        Isolates embedded stamps, seals, or headshot photos for deepfake image analysis.
        """
        rois = []
        w, h = page_image.size
        # Look for photo-like regions (e.g. ID card headshot typically occupying top-right or top-left)
        # Check standard passport/ID headshot window: 20%-45% width, 25%-50% height
        if w > 200 and h > 200:
            # Candidate 1: Top Right Box (standard for international IDs)
            tr_box = (int(w * 0.65), int(h * 0.15), int(w * 0.95), int(h * 0.60))
            crop_tr = page_image.crop(tr_box)
            # Candidate 2: Bottom Right Stamp/Seal window
            br_box = (int(w * 0.60), int(h * 0.65), int(w * 0.95), int(h * 0.95))
            crop_br = page_image.crop(br_box)

            # Keep candidate if not solid white/black
            for name, crop in [("id_headshot_candidate", crop_tr), ("stamp_seal_candidate", crop_br)]:
                gray_crop = np.array(crop.convert("L"))
                if 15 < np.std(gray_crop) < 95:
                    rois.append({
                        "name": name,
                        "page": 1,
                        "image": crop,
                        "size": crop.size,
                    })

        return rois

    def audit_pdf_metadata(self, pdf_bytes_or_text: str) -> dict:
        """
        Scans raw PDF bytes or stream for metadata anomalies, producer tools,
        and modification timestamp discrepancies.
        """
        metadata = {}
        flags = []

        producer_match = re.search(r"/Producer\s*\((.*?)\)", pdf_bytes_or_text, re.IGNORECASE)
        creator_match = re.search(r"/Creator\s*\((.*?)\)", pdf_bytes_or_text, re.IGNORECASE)
        mod_date_match = re.search(r"/ModDate\s*\((.*?)\)", pdf_bytes_or_text, re.IGNORECASE)
        creation_date_match = re.search(r"/CreationDate\s*\((.*?)\)", pdf_bytes_or_text, re.IGNORECASE)

        producer = producer_match.group(1) if producer_match else "Unknown"
        creator = creator_match.group(1) if creator_match else "Unknown"
        mod_date = mod_date_match.group(1) if mod_date_match else None
        creation_date = creation_date_match.group(1) if creation_date_match else None

        metadata["Producer"] = producer
        metadata["Creator"] = creator
        metadata["ModDate"] = mod_date
        metadata["CreationDate"] = creation_date

        for sus in self.SUSPICIOUS_PRODUCERS:
            if sus in producer.lower() or sus in creator.lower():
                flags.append(f"Suspicious editing software signature in metadata: '{sus.title()}'")

        if mod_date and creation_date and mod_date != creation_date:
            flags.append(f"Post-creation alteration detected: Created {creation_date} != Modified {mod_date}")

        eof_count = pdf_bytes_or_text.count("%%EOF")
        if eof_count > 1:
            flags.append(f"PDF contains {eof_count} incremental revisions (potential revision-masking)")

        return {
            "metadata": metadata,
            "suspicious_flags": flags,
            "has_tampering_metadata": len(flags) > 0,
        }

    def detect_edge_gradient_disparity(self, page_image: Image.Image) -> dict:
        """
        Examines edge sharpness and anti-aliasing gradient variance across text blocks.
        Scanned paper documents have optical blur, while pasted digital text has sharp step edges.
        """
        gray = np.array(page_image.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        edge_threshold = 40.0
        edge_pixels = grad_mag[grad_mag > edge_threshold]

        if len(edge_pixels) > 50:
            mean_edge_sharpness = float(np.mean(edge_pixels))
            edge_sharpness_variance = float(np.var(edge_pixels))
            sharpness_kurtosis = float(np.mean(((edge_pixels - mean_edge_sharpness) / (np.std(edge_pixels) + 1e-12))**4))
        else:
            mean_edge_sharpness = 0.0
            edge_sharpness_variance = 0.0
            sharpness_kurtosis = 0.0

        disparity_score = min(1.0, (edge_sharpness_variance / 500.0) * 0.5 + min(1.0, sharpness_kurtosis / 10.0) * 0.5)

        return {
            "mean_edge_sharpness": round(mean_edge_sharpness, 2),
            "edge_sharpness_variance": round(edge_sharpness_variance, 2),
            "sharpness_kurtosis": round(sharpness_kurtosis, 2),
            "edge_disparity_score": round(disparity_score, 4),
        }

    def detect_copy_move_forgery(self, page_image: Image.Image, patch_size=16) -> dict:
        """
        Fast spatial block-matching to detect duplicated stamps, signatures, or digits.
        """
        small_img = page_image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        arr = np.array(small_img, dtype=np.float32)

        h, w = arr.shape
        stride = 8
        patches = []
        positions = []

        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                patch = arr[y : y + patch_size, x : x + patch_size]
                if np.std(patch) > 10.0:
                    patches.append(patch.ravel() - np.mean(patch))
                    positions.append((y, x))

        duplicate_pairs = 0
        if len(patches) > 1:
            patches = np.array(patches)
            norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-12
            norm_patches = patches / norms

            sample_idx = np.random.choice(len(patches), min(len(patches), 100), replace=False)
            sim_matrix = np.dot(norm_patches[sample_idx], norm_patches.T)

            for i_local, i_orig in enumerate(sample_idx):
                sims = sim_matrix[i_local]
                pos_i = positions[i_orig]
                for j in range(len(positions)):
                    if j != i_orig and sims[j] > 0.985:
                        pos_j = positions[j]
                        dist = math.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
                        if dist > 32:
                            duplicate_pairs += 1

        copy_move_detected = duplicate_pairs > 3
        return {
            "duplicated_block_pairs": duplicate_pairs,
            "copy_move_detected": copy_move_detected,
        }

    def analyze(self, document_input, is_pdf: bool = False) -> dict:
        """
        Performs full end-to-end document forensics:
        1. Extracts text & analyzes for LLM disinformation / synthetic writing
        2. Extracts embedded images/stamps & analyzes for deepfake/diffusion artifacts
        3. Audits PDF metadata and incremental updates
        4. Calculates ELA compression inconsistencies and font anti-aliasing gradient disparities
        """
        metadata_audit = {"has_tampering_metadata": False, "suspicious_flags": []}
        extracted_text = ""
        extracted_images = []

        is_pdf_file = is_pdf or (isinstance(document_input, str) and document_input.lower().endswith(".pdf"))

        if is_pdf_file:
            extracted_text, extracted_images, raw_stream = self.extract_pdf_content(document_input)
            metadata_audit = self.audit_pdf_metadata(raw_stream)
            # Create a base rendering canvas if pypdf has images
            if extracted_images:
                page_image = extracted_images[0]["image"]
            else:
                page_image = Image.new("RGB", (512, 512), color=(255, 255, 255))
        else:
            # Image-based document (scanned PNG/JPG)
            if isinstance(document_input, str):
                page_image = Image.open(document_input).convert("RGB")
            elif isinstance(document_input, Image.Image):
                page_image = document_input.convert("RGB")
            else:
                page_image = Image.new("RGB", (512, 512), color=(255, 255, 255))

            # Extract sub-regions (headshots / stamps) from scanned image
            extracted_images = self.extract_rois_from_scanned_image(page_image)

        # 1. Structural & ELA Forensics on Document Canvas
        ela_res = self.image_detector.compute_ela(page_image)
        ela_fake_score = min(1.0, ela_res["compression_inconsistency"] / 4.0)
        edge_metrics = self.detect_edge_gradient_disparity(page_image)
        cmfd_metrics = self.detect_copy_move_forgery(page_image)
        cmfd_fake_score = 0.85 if cmfd_metrics["copy_move_detected"] else 0.1
        meta_score = 0.9 if metadata_audit["has_tampering_metadata"] else 0.1

        # 2. Forensics on Extracted Text (LLM / Disinformation Analysis)
        text_analysis = None
        text_fake_score = 0.1
        if extracted_text and len(extracted_text.strip()) >= 30:
            text_analysis = self.text_detector.analyze(extracted_text)
            text_fake_score = text_analysis["fake_probability"]

        # 3. Forensics on Extracted Embedded Images (Diffusion / GAN Analysis)
        image_analyses = []
        max_img_fake_score = 0.1
        for item in extracted_images:
            img_res = self.image_detector.analyze(item["image"])
            image_analyses.append({
                "name": item["name"],
                "page": item["page"],
                "size": list(item["size"]),
                "fake_probability": img_res["fake_probability"],
                "is_synthetic": img_res["is_synthetic"],
                "forensic_signals": img_res["forensic_signals"],
            })
            if img_res["fake_probability"] > max_img_fake_score:
                max_img_fake_score = img_res["fake_probability"]

        # 4. Neural Document Tamper Classifier
        text_words = len(extracted_text.split()) if extracted_text else 0
        img_cnt = len(image_analyses)
        feat_vec = self.build_feature_vector(
            ela_fake_score,
            edge_metrics["edge_disparity_score"],
            cmfd_fake_score,
            meta_score,
            text_fake_score,
            max_img_fake_score,
            text_words,
            img_cnt
        )

        with torch.no_grad():
            feat_tensor = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            nn_logit = self.tamper_classifier(feat_tensor).squeeze()
            nn_prob = float(torch.sigmoid(nn_logit).item())

        # 5. Composite Forensic Aggregation
        # Weights: Layout/ELA (25%), Edge Disparity (20%), CMFD (15%), Metadata (15%), Extracted Text (15%), Extracted Images (10%)
        weights = [0.25, 0.20, 0.15, 0.15]
        scores = [ela_fake_score, edge_metrics["edge_disparity_score"], cmfd_fake_score, meta_score]

        if text_analysis is not None:
            weights.append(0.15)
            scores.append(text_fake_score)

        if image_analyses:
            weights.append(0.15)
            scores.append(max_img_fake_score)

        w_sum = sum(weights)
        rule_score = sum(s * w for s, w in zip(scores, weights)) / w_sum

        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "document_best.pt")
        if os.path.exists(weights_path):
            composite_fake_prob = 0.40 * rule_score + 0.60 * nn_prob
        else:
            composite_fake_prob = rule_score

        # If any embedded photo or text is aggressively fake, raise composite
        if max_img_fake_score > 0.80 or text_fake_score > 0.85:
            composite_fake_prob = max(composite_fake_prob, 0.78)
        if metadata_audit["has_tampering_metadata"]:
            composite_fake_prob = max(composite_fake_prob, 0.75)
        if cmfd_metrics["copy_move_detected"]:
            composite_fake_prob = max(composite_fake_prob, 0.85)
        if edge_metrics["edge_disparity_score"] > 0.35 and ela_res["max_error"] > 30.0:
            composite_fake_prob = max(composite_fake_prob, 0.80)

        # Execute 3-Model Democratic Polling Committee
        polling = self.committee.poll(
            primary_prob=composite_fake_prob,
            ela_score=float(ela_res.get("compression_inconsistency", 1.0)),
            font_disparity=float(edge_metrics.get("edge_disparity_score", 0.0)),
        )

        is_tampered = composite_fake_prob >= 0.50

        # Classification label
        if max_img_fake_score > 0.70 and not is_tampered:
            classification = f"DOCUMENT CONTAINS SYNTHETIC / DEEPFAKE EMBEDDED PHOTO ({polling.consensus_type} {polling.consensus_ratio})"
            is_tampered = True
        elif text_fake_score > 0.75 and not is_tampered:
            classification = f"DOCUMENT CONTAINS AI-GENERATED DISINFORMATION TEXT ({polling.consensus_type} {polling.consensus_ratio})"
            is_tampered = True
        elif is_tampered:
            classification = f"TAMPERED / FORGED DOCUMENT ({polling.consensus_type} {polling.consensus_ratio})"
        else:
            classification = f"AUTHENTIC DOCUMENT ({polling.consensus_type} {polling.consensus_ratio})"

        # Forensic Signals
        signals = []
        signals.append(f"Democratic Polling Committee: {polling.summary}")
        if metadata_audit["suspicious_flags"]:
            signals.extend(metadata_audit["suspicious_flags"])
        if ela_res["compression_inconsistency"] > 3.0:
            signals.append("Localized ELA compression inconsistency (pasted numbers/blocks)")
        if edge_metrics["edge_disparity_score"] > 0.5:
            signals.append("Font anti-aliasing sharpness disparity (digitally inserted text on scanned document)")
        if cmfd_metrics["copy_move_detected"]:
            signals.append(f"Copy-Move duplication detected: {cmfd_metrics['duplicated_block_pairs']} identical visual clusters found")
        if os.path.exists(weights_path) and nn_prob > 0.65:
            signals.append(f"Multimodal Document Tamper Classifier anomaly flagged (p={nn_prob:.2f})")

        if text_analysis and text_analysis["is_synthetic"]:
            signals.append(f"Extracted Text is AI-generated (p={text_fake_score:.2f}): {', '.join(text_analysis['forensic_signals'][:2])}")

        for ia in image_analyses:
            if ia["is_synthetic"]:
                signals.append(f"Extracted Image '{ia['name']}' is Synthetic (p={ia['fake_probability']:.2f}): {ia['forensic_signals'][0]}")

        return {
            "modality": "document",
            "fake_probability": round(composite_fake_prob, 4),
            "is_synthetic": is_tampered,
            "classification": classification,
            "confidence_percent": round(abs(composite_fake_prob - 0.5) * 200, 1),
            "forensic_signals": signals if signals else ["Document structure, metadata, extracted text, and images are authentic"],
            "polling": polling.to_dict(),
            "extraction_telemetry": {
                "extracted_text_snippet": (extracted_text[:200] + "...") if len(extracted_text) > 200 else extracted_text,
                "text_word_count": len(extracted_text.split()) if extracted_text else 0,
                "text_analysis": text_analysis,
                "extracted_images_count": len(image_analyses),
                "image_analyses": image_analyses,
            },
            "metrics": {
                "metadata_audit": metadata_audit,
                "ela": {k: round(v, 3) for k, v in ela_res.items()},
                "edge_gradient": edge_metrics,
                "copy_move": cmfd_metrics,
                "neural_tamper_prob": round(nn_prob, 4),
            },
        }
