import json

import pytest

from scanner.edge.retrieval import EdgeRecord, load_edge_index, save_edge_index
from scanner.learning import outcome_store
from scanner.utils.atomic_io import atomic_write_json, atomic_write_text


def test_atomic_write_replaces_content(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    # No stray temp files left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "state.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def _decision(ticker="AAA", day="2026-06-01"):
    return {
        "ticker": ticker,
        "mode": "research_scan",
        "direction": "bullish",
        "entry_price": 10.0,
        "stage_failed": "validation",
        "decision_ts": f"{day}T15:00:00-04:00",
        "outcome_status": "pending",
    }


def test_load_decisions_recovers_torn_final_line(tmp_path, monkeypatch):
    journal = tmp_path / "scan_decisions.jsonl"
    quarantine = tmp_path / "scan_decisions.quarantine.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", journal)
    monkeypatch.setattr(outcome_store, "QUARANTINE_PATH", quarantine)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    good = json.dumps(_decision())
    journal.write_text(good + "\n" + good[: len(good) // 2], encoding="utf-8")

    rows = outcome_store.load_decisions()

    assert len(rows) == 1
    quarantined = quarantine.read_text(encoding="utf-8").strip().splitlines()
    assert len(quarantined) == 1
    assert json.loads(quarantined[0])["source_line"] == 2


def test_load_decisions_fails_closed_on_mid_file_corruption(tmp_path, monkeypatch):
    journal = tmp_path / "scan_decisions.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", journal)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    good = json.dumps(_decision())
    journal.write_text("NOT-JSON\n" + good + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt at line 1"):
        outcome_store.load_decisions()


def test_save_decisions_roundtrip_is_atomic_write(tmp_path, monkeypatch):
    journal = tmp_path / "scan_decisions.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", journal)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    outcome_store.save_decisions([_decision("AAA"), _decision("BBB")])

    rows = outcome_store.load_decisions()
    assert [row["ticker"] for row in rows] == ["AAA", "BBB"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_edge_index_tolerates_unknown_fields(tmp_path):
    path = tmp_path / "edge_index.json"
    record = EdgeRecord(
        ticker="TEST",
        timestamp="2026-01-01T00:00:00-05:00",
        direction="bullish",
        features={"volume_expansion": 1.2},
        outcome_return_pct=1.0,
        outcome_label="win",
        r_multiple=0.5,
        mae_pct=-0.4,
        mfe_pct=1.5,
    )
    save_edge_index([record], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[0]["some_future_field"] = "added by a newer schema"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_edge_index(path)

    assert len(records) == 1
    assert records[0].ticker == "TEST"


def test_load_edge_index_raises_clear_error_on_corrupt_json(tmp_path):
    path = tmp_path / "edge_index.json"
    path.write_text('[{"ticker": "TEST", ', encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        load_edge_index(path)
