import json

from scanner.learning import outcome_store


def _record(**overrides):
    payload = {
        "ticker": "TEST",
        "mode": "research_scan",
        "decision_ts": "2026-06-06T10:00:00-04:00",
        "direction": "bullish",
        "entry_price": 10.1234,
        "stage_failed": "potter_box_research",
        "outcome_status": "pending",
    }
    payload.update(overrides)
    return payload


def test_append_decision_skips_duplicate_setup_on_same_day(monkeypatch, tmp_path):
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", path)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    first = outcome_store.append_decision(_record())
    second = outcome_store.append_decision(_record(decision_ts="2026-06-06T15:00:00-04:00", entry_price=10.5678))

    assert first is True
    assert second is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_decision_skips_duplicate_source_session(monkeypatch, tmp_path):
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", path)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    first = outcome_store.append_decision(
        _record(
            decision_ts="2026-07-10T14:00:00-04:00",
            source_session_date="2026-07-10",
        )
    )
    second = outcome_store.append_decision(
        _record(
            decision_ts="2026-07-11T15:00:00-04:00",
            source_session_date="2026-07-10",
            entry_price=10.5678,
        )
    )

    assert first is True
    assert second is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_decision_keeps_later_candidate_after_same_session_non_sample(monkeypatch, tmp_path):
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", path)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    below_threshold = _record(
        outcome_status="not_applicable",
        research_score=55,
        research_diagnostics={"passed": False},
        source_session_date="2026-07-15",
    )
    candidate = _record(
        decision_ts="2026-07-15T15:30:00-04:00",
        entry_price=10.5678,
        research_score=73,
        research_diagnostics={"passed": True},
        source_session_date="2026-07-15",
    )

    first = outcome_store.append_decision(below_threshold)
    second = outcome_store.append_decision(candidate)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert first is True
    assert second is True
    assert len(rows) == 2
    assert [row["outcome_status"] for row in rows] == ["not_applicable", "pending"]


def test_append_decision_enriches_duplicate_without_adding_sample(monkeypatch, tmp_path):
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", path)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    first = outcome_store.append_decision(_record(research_score=68))
    second = outcome_store.append_decision(
        _record(
            decision_ts="2026-06-06T15:00:00-04:00",
            entry_price=10.5678,
            research_score=68,
            doctrine_v2_score=74,
            doctrine_v2_punchback_state="reclaim",
            doctrine_v2_diagnostics={"score": 74, "punchback_state": "reclaim"},
        )
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert first is True
    assert second is False
    assert len(rows) == 1
    assert rows[0]["doctrine_v2_score"] == 74
    assert rows[0]["doctrine_v2_diagnostics"]["punchback_state"] == "reclaim"


def test_deduplicate_decisions_keeps_resolved_version():
    records = [
        _record(outcome_status="pending"),
        _record(
            decision_ts="2026-06-06T15:00:00-04:00",
            entry_price=10.5678,
            outcome_status="resolved",
            outcome_label="win",
        ),
    ]

    clean, report = outcome_store.deduplicate_decisions(records)

    assert len(clean) == 1
    assert clean[0]["outcome_status"] == "resolved"
    assert clean[0]["outcome_label"] == "win"
    assert report["duplicates_removed"] == 1


def test_deduplicate_decisions_collapses_legacy_weekend_rerun():
    records = [
        _record(decision_ts="2026-07-10T14:00:00-04:00"),
        _record(decision_ts="2026-07-11T15:00:00-04:00", entry_price=10.5678),
    ]

    clean, report = outcome_store.deduplicate_decisions(records)

    assert len(clean) == 1
    assert report["duplicates_removed"] == 1


def test_deduplicate_decisions_keeps_non_sample_and_later_candidate():
    records = [
        _record(
            outcome_status="not_applicable",
            research_score=55,
            research_diagnostics={"passed": False},
            source_session_date="2026-07-15",
        ),
        _record(
            decision_ts="2026-07-15T15:30:00-04:00",
            entry_price=10.5678,
            research_score=73,
            research_diagnostics={"passed": True},
            source_session_date="2026-07-15",
        ),
    ]

    clean, report = outcome_store.deduplicate_decisions(records)

    assert len(clean) == 2
    assert [row["outcome_status"] for row in clean] == ["not_applicable", "pending"]
    assert report["duplicates_removed"] == 0


def test_deduplicate_decisions_handles_nested_diagnostics():
    records = [
        _record(
            research_diagnostics={"scorecard": {"touches": 2}},
            doctrine_v2_diagnostics={"score": 74, "punchback_state": "fresh_breakout"},
        ),
        _record(
            decision_ts="2026-06-06T15:00:00-04:00",
            entry_price=10.5678,
            outcome_status="resolved",
            outcome_label="win",
            research_diagnostics={"scorecard": {"touches": 3}},
        ),
    ]

    clean, report = outcome_store.deduplicate_decisions(records)

    assert len(clean) == 1
    assert clean[0]["outcome_status"] == "resolved"
    assert clean[0]["doctrine_v2_diagnostics"]["punchback_state"] == "fresh_breakout"
    assert report["duplicates_removed"] == 1
