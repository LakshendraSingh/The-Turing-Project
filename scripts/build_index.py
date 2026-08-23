#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.claim_verification.index_builder import build_index

if __name__ == "__main__":
    build_index()
