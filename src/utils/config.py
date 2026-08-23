import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT_DIR / "data" / "corpus"
CORPUS_FILE = CORPUS_DIR / "factchecks.jsonl"
INDEX_DIR = ROOT_DIR / "data" / "index"
INDEX_FILE = INDEX_DIR / "factcheck.index"
INDEX_META_FILE = INDEX_DIR / "factcheck_meta.jsonl"

MODELS_DIR = ROOT_DIR / "data" / "models"
IMAGE_HYBRID_MODEL_PATH = MODELS_DIR / "image_hybrid.joblib"
VOICE_HYBRID_MODEL_PATH = MODELS_DIR / "voice_hybrid.joblib"

TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "misinfo_detector_session")
PIB_TELEGRAM_CHANNEL = "PIB_FactCheck"

GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
GOOGLE_FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Pretrained image/video deepfake classifier (HuggingFace image-classification model).
# ViT fine-tuned specifically for real-vs-deepfake; outputs "Realism"/"Deepfake" labels.
# Swap for any other image-classification checkpoint by changing this one value —
# see src/media_forensics/deepfake_classifier.py for the label-matching logic.
DEEPFAKE_MODEL_NAME = os.getenv("DEEPFAKE_MODEL_NAME", "prithivMLmods/Deep-Fake-Detector-v2-Model")

# Optional: set to "" to disable the model and fall back to heuristics-only
# (e.g. if you're offline or don't want the ~350MB download).

# Pretrained audio anti-spoofing / synthetic-voice classifier (HuggingFace
# audio-classification model). Defaults to MelodyMachine/Deepfake-audio-detection-V2.
# Set to "" to disable and use spectral heuristics only.
VOICE_MODEL_NAME = os.getenv("VOICE_MODEL_NAME", "MelodyMachine/Deepfake-audio-detection-V2")

# Retrieval / decision thresholds — tune against your own validation set
CLAIM_MATCH_THRESHOLD = 0.55       # cosine similarity below this => "no confident match"
MEDIA_FLAG_THRESHOLD = 0.6         # heuristic manipulation score above this => "flag for review"

CORPUS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
