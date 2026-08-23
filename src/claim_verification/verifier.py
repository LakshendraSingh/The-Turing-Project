"""Given an arbitrary text claim, retrieve the closest matching fact-check(s)
from the indexed corpus and surface their verdict.

This is retrieval, not classification: it tells you "this resembles a claim
we've already seen fact-checked," with a similarity score and full traceability
back to the original source. It will NOT reliably judge a genuinely novel claim
it has no match for — those should be routed to human review, not auto-labeled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import INDEX_FILE, INDEX_META_FILE, EMBEDDING_MODEL, CLAIM_MATCH_THRESHOLD


@dataclass
class Match:
    claim: str
    verdict: str
    explanation: str
    source: str
    source_url: str
    date: str | None
    similarity: float


@dataclass
class VerificationResult:
    query: str
    matches: list[Match] = field(default_factory=list)
    confident_match: bool = False

    def top_verdict(self) -> str:
        if self.confident_match and self.matches:
            return self.matches[0].verdict
        return "Unverified — no confident match in corpus, route to human review"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_index():
    if not INDEX_FILE.exists():
        raise RuntimeError(
            "No index found. Run `python scripts/build_index.py` after ingesting fact-check data."
        )
    index = faiss.read_index(str(INDEX_FILE))
    meta = []
    with INDEX_META_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return index, meta


def verify_claim(text: str, top_k: int = 3) -> VerificationResult:
    model = _load_model()
    index, meta = _load_index()

    query_vec = model.encode([text], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, ids = index.search(query_vec, top_k)
    matches = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        record = meta[idx]
        matches.append(
            Match(
                claim=record["claim"],
                verdict=record["verdict"],
                explanation=record.get("explanation", ""),
                source=record.get("source", ""),
                source_url=record.get("source_url", ""),
                date=record.get("date"),
                similarity=float(score),
            )
        )

    result = VerificationResult(query=text, matches=matches)
    result.confident_match = bool(matches) and matches[0].similarity >= CLAIM_MATCH_THRESHOLD
    return result
