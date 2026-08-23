#!/usr/bin/env python
"""Orchestration CLI for populating data/corpus/factchecks.jsonl.

Examples:
    python scripts/ingest_pib_data.py --source telegram --limit 500
    python scripts/ingest_pib_data.py --source claimreview --query "India government"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import pib_telegram_ingest, claimreview_fetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["telegram", "claimreview"], required=True)
    parser.add_argument("--limit", type=int, default=500, help="telegram: max messages to fetch")
    parser.add_argument("--query", type=str, default="India government PIB", help="claimreview: search query")
    parser.add_argument("--lang", type=str, default="en", help="claimreview: language code")
    args = parser.parse_args()

    if args.source == "telegram":
        pib_telegram_ingest.run(limit=args.limit)
    elif args.source == "claimreview":
        claimreview_fetch.run(args.query, language_code=args.lang)


if __name__ == "__main__":
    main()
