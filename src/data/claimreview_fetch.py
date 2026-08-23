"""Pull structured ClaimReview-format fact-checks from Google's Fact Check Tools API.

This is the highest-quality free source available: entries are already
structured as claim/verdict/publisher/url, contributed by IFCN-affiliated
fact-checkers (which may include PIB and Indian outlets like BOOM, AltNews,
Factly). Requires a free Google API key with "Fact Check Tools API" enabled.

Docs: https://developers.google.com/fact-check/tools/api
"""
from __future__ import annotations

import argparse
import requests

from src.utils.config import GOOGLE_FACTCHECK_API_KEY, GOOGLE_FACTCHECK_ENDPOINT, CORPUS_FILE
from src.data.schema import FactCheckRecord, append_records


def search_claims(query: str, language_code: str = "en", page_size: int = 50) -> list[FactCheckRecord]:
    if not GOOGLE_FACTCHECK_API_KEY:
        raise RuntimeError("GOOGLE_FACTCHECK_API_KEY not set — see .env.example")

    params = {
        "query": query,
        "languageCode": language_code,
        "pageSize": page_size,
        "key": GOOGLE_FACTCHECK_API_KEY,
    }
    resp = requests.get(GOOGLE_FACTCHECK_ENDPOINT, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    records: list[FactCheckRecord] = []
    for claim in data.get("claims", []):
        claim_text = claim.get("text", "")
        for review in claim.get("claimReview", []):
            records.append(
                FactCheckRecord(
                    claim=claim_text,
                    verdict=review.get("textualRating", "Unverified"),
                    explanation=review.get("title", ""),
                    source=review.get("publisher", {}).get("name", "ClaimReview"),
                    source_url=review.get("url", ""),
                    date=review.get("reviewDate"),
                    language=language_code,
                )
            )
    return records


def run(query: str, language_code: str = "en") -> int:
    records = search_claims(query, language_code=language_code)
    count = append_records(records, CORPUS_FILE)
    print(f"Wrote {count} records from ClaimReview search '{query}' to {CORPUS_FILE}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="e.g. 'India government', 'PIB', 'Parliament India'")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    run(args.query, language_code=args.lang)
