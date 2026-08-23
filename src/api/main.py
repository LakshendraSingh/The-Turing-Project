from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.claim_verification.verifier import verify_claim
from src.media_forensics.deepfake_classifier import assess_image, assess_video_frames
from src.media_forensics.frame_extractor import extract_frames, load_image
from src.media_forensics.audio_analysis import assess_voice

app = FastAPI(
    title="The Turing Project API",
    description=(
        "Detection support tooling for human fact-checkers. All scores are "
        "signals for review, not automated verdicts."
    ),
    version="0.1.0",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class ClaimRequest(BaseModel):
    text: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify-claim")
def verify_claim_endpoint(req: ClaimRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    result = verify_claim(req.text, top_k=req.top_k)
    return {
        "query": result.query,
        "verdict": result.top_verdict(),
        "confident_match": result.confident_match,
        "matches": [m.__dict__ for m in result.matches],
    }


@app.post("/detect-deepfake")
async def detect_deepfake_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if suffix in IMAGE_EXTS:
            img = load_image(tmp_path)
            assessment = assess_image(img)
        else:
            frames = extract_frames(tmp_path, max_frames=30)
            if not frames:
                raise HTTPException(status_code=400, detail="Could not extract frames from video")
            assessment = assess_video_frames(frames)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return assessment.__dict__


@app.post("/detect-voice")
async def detect_voice_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        report = assess_voice(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return report
