"""Ingest fact-checks from the PUBLIC t.me/PIB_FactCheck Telegram channel.

Reading messages from a public Telegram channel via the official API is within
Telegram's terms — this is not a workaround. Requires a Telegram API id/hash
from https://my.telegram.org/apps (free, personal account).

PIB's Telegram posts are short-form summaries (not structured ClaimReview), so
`parse_verdict` uses simple keyword heuristics to pull a verdict out of the post
text. Review and correct these labels before trusting them for training — this
is a starting point, not a ground-truth labeler.
"""
from __future__ import annotations

import argparse
import asyncio
import re

from telethon import TelegramClient

from src.utils.config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_SESSION_NAME,
    PIB_TELEGRAM_CHANNEL,
    CORPUS_FILE,
)
from src.data.schema import FactCheckRecord, append_records

VERDICT_PATTERNS = [
    (re.compile(r"\bfake\b|\bfalse\b|\bmisleading\b|\bmorphed\b|\bimpersonat", re.I), "False"),
    (re.compile(r"\bpartly true\b|\bmisleading context\b", re.I), "Partly False"),
    (re.compile(r"\bconfirms?\b|\btrue\b|\bcorrect\b", re.I), "True"),
]


def parse_verdict(text: str) -> str:
    for pattern, label in VERDICT_PATTERNS:
        if pattern.search(text):
            return label
    return "Unverified"


async def fetch_channel(limit: int = 500) -> list[FactCheckRecord]:
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        raise RuntimeError(
            "TELEGRAM_API_ID / TELEGRAM_API_HASH not set — see .env.example. "
            "Get free credentials at https://my.telegram.org/apps"
        )

    records: list[FactCheckRecord] = []
    async with TelegramClient(TELEGRAM_SESSION_NAME, int(TELEGRAM_API_ID), TELEGRAM_API_HASH) as client:
        async for message in client.iter_messages(PIB_TELEGRAM_CHANNEL, limit=limit):
            if not message.text:
                continue
            records.append(
                FactCheckRecord(
                    claim=message.text.strip(),
                    verdict=parse_verdict(message.text),
                    explanation="",
                    source="PIB Fact Check (Telegram)",
                    source_url=f"https://t.me/{PIB_TELEGRAM_CHANNEL}/{message.id}",
                    date=message.date.isoformat() if message.date else None,
                )
            )
    return records


def run(limit: int = 500) -> int:
    records = asyncio.run(fetch_channel(limit=limit))
    count = append_records(records, CORPUS_FILE)
    print(f"Wrote {count} records from Telegram to {CORPUS_FILE}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    run(limit=args.limit)
