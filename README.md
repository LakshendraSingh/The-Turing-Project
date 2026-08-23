# The Turing Project

# IN DEVELOPEMENT
A starter prototype for detecting (a) synthetically manipulated media ("deepfakes")
and (b) text claims that misrepresent government proceedings/policy, grounded against
PIB (Press Information Bureau, Government of India) Fact Check outputs and other
open ClaimReview-format fact-check corpora.

This is **defensive infrastructure**: it does not generate or alter media, it flags
content for human review. Treat every model score as a _signal_, not a verdict —
route high-confidence flags to a human fact-checker before any public action.

## Architecture

```
                 ┌─────────────────────┐
   video/image → │  Media Forensics    │ → manipulation_score, artifact_report
   /audio input  │  (frames, spectral,  │
                 │   model ensemble)    │
                 └─────────────────────┘

                 ┌─────────────────────┐
   text claim  → │  Claim Verification │ → matched_factcheck, verdict, confidence
                 │  (embed + retrieve   │
                 │   vs fact-check DB)  │
                 └─────────────────────┘

                 ┌─────────────────────┐
   PIB Telegram, │  Data Ingestion      │ → data/corpus/*.jsonl
   Data Commons  │  (ToS-respecting)    │
   ClaimReview   │                      │
                 └─────────────────────┘
```

Everything is exposed via a FastAPI service in `src/api/main.py`.

## Why not scrape Instagram

Instagram's ToS prohibits automated scraping and its API doesn't expose a public
fact-check feed, so an Instagram-based pipeline breaks constantly and creates legal
exposure. PIB Fact Check publishes the **same content** to more scrape-friendly,
API-accessible channels:

- **Telegram** (`t.me/PIB_FactCheck`) — public channel, readable via the official
  Telegram API (Telethon), which is within Telegram's ToS for public channel data.
- **Google Fact Check Tools API / Data Commons ClaimReview feed** — if PIB's
  fact-checks carry ClaimReview markup, this gives you structured
  `claim / verdict / date / url` records for free, no scraping at all.
- **factcheck.pib.gov.in** — the official portal; check `robots.txt` before
  crawling and prefer any official bulk-download or RTI/NDSAP data request over
  crawling it directly.

`src/data/pib_telegram_ingest.py` and `src/data/claimreview_fetch.py` implement
the first two. Wire in additional fact-checkers (BOOM, AltNews, Factly, etc. — all
IFCN-verified) the same way for broader coverage.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in Telegram API id/hash, Google API key
```

## Ingest fact-check corpus

```bash
python scripts/ingest_pib_data.py --source telegram --limit 500
python scripts/ingest_pib_data.py --source claimreview --query "India government"
```

This populates `data/corpus/factchecks.jsonl`, one record per fact-check:

```json
{
  "claim": "...",
  "verdict": "False",
  "explanation": "...",
  "source_url": "...",
  "date": "2026-01-15"
}
```

## Build the retrieval index

```bash
python scripts/build_index.py
```

## Hybrid fusion classifier (optional, recommended once you have labeled data)

Both media-forensics pipelines default to a fixed blend (70% pretrained model
score, 30% heuristics). That weighting is a guess — the right balance depends
on how well each signal actually performs on _your_ media. Once you've
collected some labeled real/fake examples (even a few dozen of each helps),
train a small logistic-regression fusion model that learns the weighting from
data instead:

```bash
# expects data/train/image/real/*.jpg and data/train/image/fake/*.jpg
python scripts/train_hybrid_model.py --domain image --data-dir data/train/image

# expects data/train/audio/real/*.wav and data/train/audio/fake/*.wav
python scripts/train_hybrid_model.py --domain audio --data-dir data/train/audio
```

This prints a held-out classification report, AUC, and the learned weight for
each feature (heuristics + pretrained model score), then saves the trained
classifier to `data/models/image_hybrid.joblib` / `data/models/voice_hybrid.joblib`.
Once saved, `/detect-deepfake` and `/detect-voice` pick it up automatically and
use it in place of the fixed blend — no code changes or restarts needed beyond
reloading the process. Delete the `.joblib` file to fall back to the fixed
blend again.

Where to get labeled data: public benchmark sets (FaceForensics++, DFDC,
Celeb-DF for video/image; ASVspoof, SpoofCeleb for audio) get you started;
for something specifically tuned to fakes of government proceedings, the most
valuable data will be the fakes PIB and other fact-checkers have actually
flagged, paired with genuine footage from official broadcasts.

## Run the API

```bash
uvicorn src.api.main:app --reload
```

- `POST /verify-claim` — `{"text": "..."}` → best-matching fact-check(s) + verdict
- `POST /detect-deepfake` — upload image/video → manipulation score + artifact breakdown
- `POST /detect-voice` — upload audio → synthetic-voice likelihood
- `GET /health`

## What's a stub vs what actually runs

- **Claim verification**: fully functional once you run the ingestion + indexing
  scripts — uses `sentence-transformers` embeddings + FAISS, no training required.
- **Media forensics (image/video)**: combines the heuristic baseline with a
  pretrained deepfake classifier, on by default —
  [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model),
  a ViT fine-tuned specifically for real-vs-deepfake image classification
  (~92% macro F1 on its reported test set). First call downloads the model
  (~350MB) from HuggingFace and caches it locally. `/detect-deepfake` returns
  `heuristics`, `model_score`, and a `combined_score` (70% model / 30%
  heuristics) so you can see both signals separately.
- **Audio**: same pattern — spectral heuristics blended with
  [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2),
  a wav2vec2-based synthetic-voice classifier, on by default.
- **Swapping models**: set `DEEPFAKE_MODEL_NAME` / `VOICE_MODEL_NAME` in `.env`
  to any other HuggingFace image-classification / audio-classification
  checkpoint, or set to `none` to fall back to heuristics-only (e.g. if you're
  offline or don't want the download). Label matching in
  `deepfake_classifier.py` / `audio_analysis.py` looks for "fake"/"deepfake"/
  "spoof"/"synthetic" in the model's output labels — if you pick a model with
  numeric labels (`LABEL_0`/`LABEL_1`), check its `id2label` config and map
  them explicitly.
- **Important**: even a good pretrained classifier is a signal, not a verdict.
  Both shipped models were trained on specific generator families and
  datasets — validate against your own labeled samples before trusting scores
  in production, and re-check periodically as generation techniques evolve
  (this is an active arms race, not a solved problem).

## Honest limitations (read this before deploying anything)

- Heuristic media forensics scores are **not** reliable deepfake detectors on their
  own — modern generators can defeat naive frequency/compression checks. Validate
  against a labeled dataset (e.g. FaceForensics++, DFDC) before trusting scores.
- Claim verification only ever tells you "this resembles/doesn't resemble a claim
  we already have fact-checked" — it can't verify genuinely novel claims. Route
  unmatched claims to human reviewers, don't auto-label them "true."
- This system should support human fact-checkers, not replace them or auto-flag
  accounts/content for takedown. Building automated content-removal on top of these
  scores risks false positives against legitimate speech.
