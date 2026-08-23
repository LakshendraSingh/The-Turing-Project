import tempfile
from pathlib import Path

from src.data.schema import FactCheckRecord, append_records, load_records


def test_append_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "factchecks.jsonl"
        records = [
            FactCheckRecord(claim="Test claim one", verdict="False", source="unit-test"),
            FactCheckRecord(claim="Test claim two", verdict="True", source="unit-test"),
        ]
        count = append_records(records, path)
        assert count == 2

        loaded = load_records(path)
        assert len(loaded) == 2
        assert loaded[0]["claim"] == "Test claim one"
        assert loaded[1]["verdict"] == "True"


def test_append_skips_empty_claims():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "factchecks.jsonl"
        records = [FactCheckRecord(claim="  ", verdict="False")]
        count = append_records(records, path)
        assert count == 0
        assert load_records(path) == []
