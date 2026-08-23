#!/usr/bin/env python
"""Copy the bundled sample fact-checks into the working corpus, so you can
run build_index.py and try /verify-claim immediately without live ingestion."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import CORPUS_FILE, CORPUS_DIR

SAMPLE_FILE = CORPUS_DIR / "factchecks.sample.jsonl"

if __name__ == "__main__":
    if not SAMPLE_FILE.exists():
        raise SystemExit(f"Sample file not found: {SAMPLE_FILE}")
    shutil.copy(SAMPLE_FILE, CORPUS_FILE)
    print(f"Seeded {CORPUS_FILE} from sample data. Now run: python scripts/build_index.py")
