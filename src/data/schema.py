"""Shared record schema for the fact-check corpus.

Every ingestion source (Telegram, ClaimReview API, manual CSV, ...) should
normalize its output into this shape before appending to data/corpus/factchecks.jsonl.
Keeping one schema means the retrieval index doesn't care where a record came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class FactCheckRecord:
    claim: str                 # the (para)phrased claim being checked
    verdict: str                # e.g. "False", "Misleading", "True", "Unverified"
    explanation: str = ""       # short summary of why
    source: str = ""            # "PIB Fact Check", "BOOM", etc.
    source_url: str = ""        # link back to the original fact-check
    date: Optional[str] = None  # ISO date string
    language: str = "en"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append_records(records: list[FactCheckRecord], path: Path) -> int:
    """Append records to a jsonl corpus file, skipping empty claims. Returns count written."""
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            if not r.claim or not r.claim.strip():
                continue
            f.write(r.to_json() + "\n")
            written += 1
    return written


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
