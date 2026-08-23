"""Build a semantic-search index over the fact-check corpus.

Uses sentence-transformers for embeddings and FAISS for nearest-neighbor
search. Re-run this any time data/corpus/factchecks.jsonl is updated.
"""
from __future__ import annotations

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import CORPUS_FILE, INDEX_FILE, INDEX_META_FILE, EMBEDDING_MODEL
from src.data.schema import load_records


def build_index() -> int:
    records = load_records(CORPUS_FILE)
    if not records:
        raise RuntimeError(
            f"No records found in {CORPUS_FILE}. Run an ingestion script first, "
            f"e.g. scripts/ingest_pib_data.py"
        )

    model = SentenceTransformer(EMBEDDING_MODEL)
    claims = [r["claim"] for r in records]
    embeddings = model.encode(claims, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    with INDEX_META_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Indexed {len(records)} fact-checks -> {INDEX_FILE}")
    return len(records)


if __name__ == "__main__":
    build_index()
