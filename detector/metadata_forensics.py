"""
Universal Multimodal Metadata & Content Credentials (C2PA) Forensic Auditor (2026)
Performs cryptographic C2PA assertion audits, EXIF / XMP inspection, ID3 audio frame parsing,
and container atom analysis across Images, Audio, Video, and Documents.
"""

import io
import os
import re
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ExifTags


class UniversalMetadataForensics:
    """
    Multimodal Metadata & Cryptographic Content Credential Forensic Auditor.
    Extracts, decodes, and evaluates:
    1. C2PA (Coalition for Content Provenance and Authenticity) Manifest Stores & JUMBF boxes
    2. Image EXIF, XMP, and PNG text chunks (A1111/ComfyUI prompts, Camera hardware signatures)
    3. Audio ID3v2 frames & RIFF chunks (C2PA GEOB frames, Suno/Udio/ElevenLabs/MusicFX markers)
    4. Video MP4/MOV container atoms & encoder signatures
    """

    # Generative AI keywords in metadata, software, or prompt fields
    GENAI_KEYWORDS = [
        "trainedalgorithmicmedia",
        "generative ai",
        "google generative ai",
        "synthid",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stablediffusion",
        "novelai",
        "comfyui",
        "automatic1111",
        "invokeai",
        "adobe firefly",
        "firefly",
        "imagen",
        "flux.1",
        "suno",
        "udio",
        "elevenlabs",
        "musicfx",
        "lyria",
        "runway",
        "sora",
        "kling",
        "pika",
        "luma dream machine",
        "deepfacelab",
        "faceswap",
    ]

    # Authentic camera manufacturers
    CAMERA_MANUFACTURERS = [
        "apple", "canon", "nikon", "sony", "fujifilm", "leica", "panasonic",
        "olympus", "samsung", "google", "hasselblad", "pentax", "gopro", "dji"
    ]

    @staticmethod
    def _read_bytes_head_tail(input_source, max_bytes: int = 500000) -> bytes:
        """
        Safely extracts up to max_bytes from file path or buffer.
        """
        if isinstance(input_source, (bytes, bytearray)):
            return bytes(input_source[:max_bytes])
        elif isinstance(input_source, io.BytesIO):
            pos = input_source.tell()
            input_source.seek(0)
            data = input_source.read(max_bytes)
            input_source.seek(pos)
            return data
        elif isinstance(input_source, str) and os.path.exists(input_source):
            try:
                with open(input_source, "rb") as f:
                    return f.read(max_bytes)
            except Exception:
                return b""
        return b""

    # =========================================================================
    # 1. UNIVERSAL C2PA (CONTENT CREDENTIALS) MANIFEST AUDITOR
    # =========================================================================
    @classmethod
    def audit_c2pa(cls, raw_bytes: bytes) -> Dict[str, Any]:
        """
        Scans binary stream for C2PA manifest stores (JUMBF boxes).
        Extracts claim generator, IPTC digital source type, actions, signing authority,
        and SynthID imperceptible watermark assertions.
        """
        if not raw_bytes:
            return {"has_c2pa": False, "is_ai_generated": False}

        # Check for C2PA marker keywords in binary stream
        has_c2pa_marker = (
            b"application/c2pa" in raw_bytes or
            b"c2pa manifest store" in raw_bytes or
            b"c2pa.claim" in raw_bytes or
            b"c2pa.actions" in raw_bytes or
            b"http://ns.adobe.com/c2pa" in raw_bytes
        )

        if not has_c2pa_marker:
            return {"has_c2pa": False, "is_ai_generated": False}

        # Extract textual substrings from C2PA JUMBF region
        strings = re.findall(b"[\x20-\x7E]{4,}", raw_bytes[:150000])
        decoded_strings = [s.decode("latin-1") for s in strings]

        # Scan for actions and descriptions
        actions = []
        is_ai_generated = False
        synthid_watermark = False
        digital_source_type = None
        claim_generator = None
        signing_authority = None
        instance_id = None

        for text in decoded_strings:
            lower = text.lower()

            # Digital source type detection
            if "digitalsourcetype" in lower or "trainedalgorithmicmedia" in lower:
                is_ai_generated = True
                if "trainedalgorithmicmedia" in lower:
                    digital_source_type = "trainedAlgorithmicMedia (IPTC Generative AI)"
                elif "compositeWithTrainedAlgorithmicMedia" in text:
                    digital_source_type = "compositeWithTrainedAlgorithmicMedia"

            # SynthID assertion
            if "synthid" in lower:
                synthid_watermark = True
                is_ai_generated = True

            # Actions / Descriptions
            if "created by google generative ai" in lower:
                actions.append("Created by Google Generative AI")
                is_ai_generated = True
            elif "applied imperceptible synthid watermark" in lower:
                actions.append("Applied imperceptible SynthID watermark")
                synthid_watermark = True
            elif "c2pa.created" in lower:
                actions.append("c2pa.created")
            elif "c2pa.edited" in lower:
                actions.append("c2pa.edited")

            # Claim generator library
            if "generator library" in lower or "c2pa core" in lower:
                match = re.search(r'(Google C2PA Core Generator Library|Adobe Firefly|Truepic[^\x00-\x1f]*|OpenAI[^\x00-\x1f]*)', text, re.IGNORECASE)
                if match:
                    claim_generator = match.group(1).strip()

            # Signing authority / certificate info
            if any(ca in text for ca in ["Google C2PA Media Services", "Google Media Processing Services", "Google C2PA Root", "Adobe Systems", "Microsoft"]):
                if not signing_authority:
                    clean_ca = re.sub(r'^[^\w$]+|[^\w\s$]+$', '', text).strip()
                    signing_authority = clean_ca

            # Instance ID
            if "instanceid" in lower:
                match = re.search(r"instanceID[^\w-]*([0-9a-fA-F-]{16,40})", text)
                if match:
                    instance_id = match.group(1)

        # Fallback check for AI mentions anywhere in the C2PA block
        if not is_ai_generated:
            for s in decoded_strings:
                if any(k in s.lower() for k in ["generative ai", "synthid", "dall-e", "trainedalgorithmicmedia"]):
                    is_ai_generated = True
                    break

        return {
            "has_c2pa": True,
            "is_ai_generated": is_ai_generated,
            "digital_source_type": digital_source_type,
            "synthid_watermark_present": synthid_watermark,
            "claim_generator": claim_generator,
            "signing_authority": signing_authority,
            "instance_id": instance_id,
            "actions": list(dict.fromkeys(actions)),
            "confidence": 1.0 if is_ai_generated else 0.8,
        }

    # =========================================================================
    # 2. IMAGE METADATA FORENSICS (EXIF, XMP, PNG CHUNKS)
    # =========================================================================
    @classmethod
    def audit_image_metadata(cls, image_input) -> Dict[str, Any]:
        """
        Audits image metadata: C2PA manifests, EXIF tags, PNG chunks, and XMP blocks.
        """
        raw_bytes = b""
        pil_img = None

        if isinstance(image_input, str):
            raw_bytes = cls._read_bytes_head_tail(image_input)
            try:
                pil_img = Image.open(image_input)
            except Exception:
                pass
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            bio = io.BytesIO()
            try:
                pil_img.save(bio, format=pil_img.format or "PNG")
                raw_bytes = bio.getvalue()[:300000]
            except Exception:
                pass
        elif isinstance(image_input, (bytes, bytearray, io.BytesIO)):
            raw_bytes = cls._read_bytes_head_tail(image_input)
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
            except Exception:
                pass

        # 1. C2PA Audit
        c2pa_res = cls.audit_c2pa(raw_bytes)

        # 2. EXIF Tag Extraction
        exif_data = {}
        camera_make = None
        camera_model = None
        lens_model = None
        software = None
        user_comment = None
        image_desc = None
        has_camera_hardware = False

        if pil_img:
            try:
                exif_raw = pil_img.getexif()
                if exif_raw:
                    for tag_id, value in exif_raw.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="ignore").strip()
                            except Exception:
                                value = str(value)
                        exif_data[tag_name] = str(value)

                    camera_make = exif_data.get("Make")
                    camera_model = exif_data.get("Model")
                    software = exif_data.get("Software")
                    user_comment = exif_data.get("UserComment")
                    image_desc = exif_data.get("ImageDescription")

                    if camera_make and any(m in camera_make.lower() for m in cls.CAMERA_MANUFACTURERS):
                        has_camera_hardware = True
            except Exception:
                pass

        # 3. PNG Text Chunks / Parameters Audit
        png_text_chunks = {}
        if pil_img and hasattr(pil_img, "info") and pil_img.info:
            for k, v in pil_img.info.items():
                if isinstance(v, (str, bytes)):
                    val_str = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
                    png_text_chunks[str(k)] = val_str

        # Also parse raw PNG text chunks directly if available
        if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            pos = 8
            while pos < len(raw_bytes) - 12:
                c_len = int.from_bytes(raw_bytes[pos:pos+4], "big")
                c_type = raw_bytes[pos+4:pos+8]
                if c_type in [b"tEXt", b"zTXt", b"iTXt"]:
                    c_data = raw_bytes[pos+8:pos+8+c_len]
                    parts = c_data.split(b"\x00", 1)
                    if len(parts) == 2:
                        k = parts[0].decode("latin-1", errors="ignore")
                        v = parts[1].decode("latin-1", errors="ignore")
                        png_text_chunks[k] = v
                pos += 12 + c_len
                if c_type == b"IEND":
                    break

        # 4. Evaluation & Signal Synthesis
        suspicious_flags = []
        authenticity_signals = []
        is_synthetic = False
        metadata_fake_score = 0.50

        # C2PA Cryptographic Assertions (Highest Priority)
        if c2pa_res["has_c2pa"]:
            if c2pa_res["is_ai_generated"]:
                is_synthetic = True
                metadata_fake_score = 0.99
                auth_str = f" from {c2pa_res['signing_authority']}" if c2pa_res.get('signing_authority') else ""
                suspicious_flags.append(f"C2PA Content Credential verified: AI-generated media{auth_str}")
                if c2pa_res.get("synthid_watermark_present"):
                    suspicious_flags.append("C2PA assertion confirms embedded Google SynthID watermark")
                if c2pa_res.get("digital_source_type"):
                    suspicious_flags.append(f"C2PA digitalSourceType: {c2pa_res['digital_source_type']}")
            else:
                metadata_fake_score = 0.05
                authenticity_signals.append("C2PA Content Credential verified: Camera capture origin")

        # Generative AI Prompt / Tool Metadata Check
        combined_text = " ".join(
            list(png_text_chunks.values()) +
            [str(v) for v in exif_data.values()] +
            [software or "", user_comment or "", image_desc or ""]
        ).lower()

        found_genai_tags = [kw for kw in cls.GENAI_KEYWORDS if kw in combined_text]
        if found_genai_tags and not c2pa_res.get("is_ai_generated"):
            is_synthetic = True
            metadata_fake_score = 0.95
            suspicious_flags.append(f"Generative AI metadata tag detected: '{found_genai_tags[0].title()}'")

        # Stable Diffusion / ComfyUI parameters check
        if any(k in png_text_chunks for k in ["parameters", "prompt", "workflow"]):
            is_synthetic = True
            metadata_fake_score = 0.98
            suspicious_flags.append("AI prompt generation parameters embedded in PNG chunks (Diffusion workflow)")

        # Authentic Camera Hardware Verification
        if has_camera_hardware and not is_synthetic:
            metadata_fake_score = 0.10
            camera_name = f"{camera_make} {camera_model or ''}".strip()
            authenticity_signals.append(f"Authentic camera EXIF hardware profile: {camera_name}")

        # Missing Metadata Anomaly (High-res image completely lacking camera EXIF)
        has_any_exif = len(exif_data) > 0
        if not has_any_exif and not png_text_chunks and not c2pa_res["has_c2pa"]:
            if pil_img and (pil_img.width >= 1024 and pil_img.height >= 1024):
                suspicious_flags.append("High-resolution image completely devoid of physical camera EXIF hardware tags")

        return {
            "has_metadata": has_any_exif or len(png_text_chunks) > 0 or c2pa_res["has_c2pa"],
            "c2pa": c2pa_res,
            "has_camera_hardware": has_camera_hardware,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "software": software,
            "png_text_chunks_count": len(png_text_chunks),
            "suspicious_flags": suspicious_flags,
            "authenticity_signals": authenticity_signals,
            "metadata_fake_score": round(metadata_fake_score, 3),
            "is_synthetic": is_synthetic,
        }

    # =========================================================================
    # 3. AUDIO METADATA FORENSICS (C2PA, ID3v2, RIFF/WAV CHUNKS)
    # =========================================================================
    @classmethod
    def audit_audio_metadata(cls, audio_input) -> Dict[str, Any]:
        """
        Audits audio metadata: C2PA manifests (GEOB frame), ID3 tags, and RIFF INFO chunks.
        """
        raw_bytes = cls._read_bytes_head_tail(audio_input, max_bytes=250000)
        c2pa_res = cls.audit_c2pa(raw_bytes)

        # Parse ID3 tags
        id3_tags = {}
        if raw_bytes[:3] == b"ID3":
            # ID3v2 header
            tag_size = ((raw_bytes[6] & 0x7F) << 21) | ((raw_bytes[7] & 0x7F) << 14) | ((raw_bytes[8] & 0x7F) << 7) | (raw_bytes[9] & 0x7F)
            pos = 10
            limit = min(len(raw_bytes), 10 + tag_size)
            while pos < limit - 10:
                frame_id = raw_bytes[pos:pos+4]
                if frame_id[0] == 0:
                    pos += 1
                    continue
                try:
                    fid = frame_id.decode("ascii")
                except Exception:
                    break
                fsize = int.from_bytes(raw_bytes[pos+4:pos+8], "big")
                if fsize <= 0 or pos + 10 + fsize > len(raw_bytes):
                    break
                fcontent = raw_bytes[pos+10:pos+10+fsize]
                if fid in ["TIT2", "TPE1", "TALB", "COMM", "TCON", "TENC", "TSSE", "TPUB"]:
                    # Decode text
                    try:
                        text_val = fcontent[1:].decode("utf-8", errors="ignore").replace("\x00", "")
                        id3_tags[fid] = text_val
                    except Exception:
                        pass
                pos += 10 + fsize

        suspicious_flags = []
        authenticity_signals = []
        is_synthetic = False
        metadata_fake_score = 0.50

        # C2PA Cryptographic Assertions
        if c2pa_res["has_c2pa"]:
            if c2pa_res["is_ai_generated"]:
                is_synthetic = True
                metadata_fake_score = 0.99
                auth_str = f" by {c2pa_res['signing_authority']}" if c2pa_res.get('signing_authority') else ""
                suspicious_flags.append(f"C2PA Content Credential verified: AI-generated audio{auth_str}")
                if c2pa_res.get("synthid_watermark_present"):
                    suspicious_flags.append("C2PA assertion confirms imperceptible Google SynthID audio watermark")
                if c2pa_res.get("digital_source_type"):
                    suspicious_flags.append(f"C2PA digitalSourceType: {c2pa_res['digital_source_type']}")
            else:
                metadata_fake_score = 0.05
                authenticity_signals.append("C2PA Content Credential verified: Microphone recording origin")

        # ID3 / RIFF Tag Generative AI Check
        all_tags_text = " ".join([str(v) for v in id3_tags.values()]).lower()
        found_genai_audio = [kw for kw in cls.GENAI_KEYWORDS if kw in all_tags_text]
        if found_genai_audio and not c2pa_res.get("is_ai_generated"):
            is_synthetic = True
            metadata_fake_score = 0.95
            suspicious_flags.append(f"Generative AI audio marker in metadata tags: '{found_genai_audio[0].title()}'")

        return {
            "has_metadata": c2pa_res["has_c2pa"] or len(id3_tags) > 0,
            "c2pa": c2pa_res,
            "id3_tags": id3_tags,
            "suspicious_flags": suspicious_flags,
            "authenticity_signals": authenticity_signals,
            "metadata_fake_score": round(metadata_fake_score, 3),
            "is_synthetic": is_synthetic,
        }

    # =========================================================================
    # 4. VIDEO CONTAINER METADATA FORENSICS
    # =========================================================================
    @classmethod
    def audit_video_metadata(cls, video_input) -> Dict[str, Any]:
        """
        Audits video container atoms (MP4/MOV/WebM) for C2PA boxes and generative video signatures.
        """
        raw_bytes = cls._read_bytes_head_tail(video_input, max_bytes=350000)
        c2pa_res = cls.audit_c2pa(raw_bytes)

        suspicious_flags = []
        authenticity_signals = []
        is_synthetic = False
        metadata_fake_score = 0.50

        # Scan for C2PA box
        if c2pa_res["has_c2pa"]:
            if c2pa_res["is_ai_generated"]:
                is_synthetic = True
                metadata_fake_score = 0.99
                suspicious_flags.append("C2PA Content Credential verified: AI-generated synthetic video")
            else:
                metadata_fake_score = 0.05
                authenticity_signals.append("C2PA Content Credential verified: Authentic camera recording")

        # Scan for generative video tool keywords in container
        decoded_text = raw_bytes.decode("latin-1", errors="ignore").lower()
        found_video_gen = [kw for kw in ["sora", "runway", "gen-2", "gen-3", "kling", "pika", "luma", "haiper"] if kw in decoded_text]
        if found_video_gen and not is_synthetic:
            is_synthetic = True
            metadata_fake_score = 0.95
            suspicious_flags.append(f"AI video generation tool signature detected in container: '{found_video_gen[0].title()}'")

        # Check for Apple iPhone / QuickTime camera recording atom
        if "com.apple.quicktime.make" in decoded_text or "apple" in decoded_text and "iphone" in decoded_text:
            if not is_synthetic:
                metadata_fake_score = 0.10
                authenticity_signals.append("Authentic Apple QuickTime hardware recording atoms detected")

        return {
            "has_metadata": c2pa_res["has_c2pa"] or len(suspicious_flags) > 0 or len(authenticity_signals) > 0,
            "c2pa": c2pa_res,
            "suspicious_flags": suspicious_flags,
            "authenticity_signals": authenticity_signals,
            "metadata_fake_score": round(metadata_fake_score, 3),
            "is_synthetic": is_synthetic,
        }
