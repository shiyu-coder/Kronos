import json
import logging

import pandas as pd

from scanner.edge.retrieval import EdgeRecord, save_edge_index
from scanner.evidence.store import start_evidence_run
from scanner.main import (
    run_audit_edge,
    run_build_retrieval_index,
    run_diagnose_edge,
    run_edge_lab,
    run_validate_edge,
)


def _daily_bars():
    idx = pd.date_range("2026-01-05", periods=3, freq="D", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.0],
            "High": [10.6, 11.1, 11.6],
            "Low": [9.8, 10.3, 10.8],
            "Close": [10.5, 11.0, 11.5],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def _edge_record(ticker="TEST", timestamp="2026-01-01T00:00:00-05:00"):
    features = {
        "ticker": ticker,
        "timestamp": timestamp,
        "direction": "bullish",
        "research_score": 72.0,
        "potter_passed": 1.0,
        "empty_space_score": 3.0,
        "rr_ratio": 2.0,
        "kronos_directional_agreement": 0.7,
        "kronos_median_forecast_return_pct": 1.0,
        "data_quality_score": 1.0,
        "feed_confidence": 0.8,
        "options_spread_pct": 0.04,
    }
    return EdgeRecord(
        ticker=ticker,
        timestamp=timestamp,
        direction="bullish",
        features=features,
        outcome_return_pct=2.0,
        outcome_label="win",
        r_multiple=1.0,
        mae_pct=-0.5,
        mfe_pct=3.0,
    )


def test_build_retrieval_index_records_evidence(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    evidence = start_evidence_run("build_retrieval_index", tmp_path)
    monkeypatch.setattr("scanner.main.EDGE_INDEX_PATH", tmp_path / "edge_index.json")
    monkeypatch.setattr("scanner.main.REPORT_DIR", tmp_path)
    monkeypatch.setattr("scanner.main.EDGE_INDEX_EXTRA_UNIVERSE", [])
    monkeypatch.setattr("scanner.main.fetch_daily_bars", lambda ticker, research=False, adjustment="raw": _daily_bars())
    monkeypatch.setattr("scanner.main.build_edge_records_from_bars", lambda ticker, bars, horizon: [_edge_record(ticker)])

    payload = run_build_retrieval_index(["AAA", "BBB"], logger, evidence_run=evidence)
    evidence.flush()

    rows = (tmp_path / evidence.run_id / "edge_index_records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = (tmp_path / evidence.run_id / "metrics.jsonl").read_text(encoding="utf-8")

    assert payload["records"] == 2
    assert len(rows) == 2
    assert json.loads(rows[0])["ticker"] == "AAA"
    assert "index_records" in metrics


def test_build_retrieval_index_fails_closed_on_bar_contract_violation(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    monkeypatch.setattr("scanner.main.EDGE_INDEX_PATH", tmp_path / "edge_index.json")
    monkeypatch.setattr("scanner.main.REPORT_DIR", tmp_path)
    monkeypatch.setattr("scanner.main.EDGE_INDEX_EXTRA_UNIVERSE", [])

    def fake_fetch(ticker, research=False, adjustment="raw"):
        if ticker == "BAD":
            bars = _daily_bars()
            bars.loc[bars.index[0], "High"] = 0.0  # High below body: hard violation
            return bars
        return _daily_bars()

    monkeypatch.setattr("scanner.main.fetch_daily_bars", fake_fetch)
    monkeypatch.setattr("scanner.main.build_edge_records_from_bars", lambda ticker, bars, horizon: [_edge_record(ticker)])

    payload = run_build_retrieval_index(["GOOD", "BAD"], logger)

    assert payload["records"] == 1
    assert "BAD" in payload["errors"]
    assert "bar contract violation" in payload["errors"]["BAD"]
    assert payload["bars_adjustment"]


def test_build_retrieval_index_extends_universe_without_duplicates(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    seen = []
    monkeypatch.setattr("scanner.main.EDGE_INDEX_PATH", tmp_path / "edge_index.json")
    monkeypatch.setattr("scanner.main.REPORT_DIR", tmp_path)
    monkeypatch.setattr("scanner.main.EDGE_INDEX_EXTRA_UNIVERSE", ["XTRA", "AAA"])
    monkeypatch.setattr(
        "scanner.main.fetch_daily_bars",
        lambda ticker, research=False, adjustment="raw": seen.append(ticker) or _daily_bars(),
    )
    monkeypatch.setattr("scanner.main.build_edge_records_from_bars", lambda ticker, bars, horizon: [_edge_record(ticker)])

    payload = run_build_retrieval_index(["AAA", "BBB"], logger)

    assert seen == ["AAA", "BBB", "XTRA"]
    assert payload["tickers"] == 3
    assert payload["watchlist_tickers"] == 2
    assert payload["extra_universe_tickers"] == 1


def test_validate_and_diagnose_record_evidence(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    index_path = tmp_path / "edge_index.json"
    save_edge_index(
        [
            _edge_record("AAA", "2026-01-01T00:00:00-05:00"),
            _edge_record("BBB", "2026-02-01T00:00:00-05:00"),
            _edge_record("CCC", "2026-03-01T00:00:00-05:00"),
        ],
        index_path,
    )
    evidence = start_evidence_run("validate_edge", tmp_path)
    monkeypatch.setattr("scanner.main.EDGE_INDEX_PATH", index_path)
    monkeypatch.setattr("scanner.main.EDGE_VALIDATION_REPORT_PATH", tmp_path / "edge_validation_report.json")
    monkeypatch.setattr("scanner.main.EDGE_DIAGNOSTIC_REPORT_PATH", tmp_path / "edge_diagnostic_report.json")
    monkeypatch.setattr("scanner.main.EDGE_SCAN_REPORT_PATH", tmp_path / "missing_scan_report.json")

    validation = run_validate_edge(logger, evidence_run=evidence)
    diagnostic = run_diagnose_edge(logger, evidence_run=evidence)
    evidence.flush()

    validation_rows = (tmp_path / evidence.run_id / "validation_candidates.jsonl").read_text(encoding="utf-8")
    diagnostic_rows = (tmp_path / evidence.run_id / "diagnostics.jsonl").read_text(encoding="utf-8")

    assert validation["candidate_count"] == 3
    assert diagnostic["index_records"] == 3
    assert "edge_score" in validation_rows
    assert "diagnose_edge" in diagnostic_rows
    # Calibration suite block is always present; every objective fails
    # CLOSED on tiny history and nothing ships.
    meta_block = validation["meta_model"]
    assert meta_block["shipped_objective"] is None
    assert set(meta_block["objectives"]) == {"expected_r", "tail_prob", "p_win"}
    for objective_result in meta_block["objectives"].values():
        assert objective_result["acceptance"]["passed"] is False
        assert objective_result["metrics"]["insufficient"] is True
        assert "predictions" not in objective_result
        assert "final_model" not in objective_result
    # No two-touch pass -> no live advisory model on disk.
    from scanner import main as scanner_main

    assert not scanner_main.META_MODEL_PATH.exists()


def test_validate_edge_reuses_analog_index(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    index_path = tmp_path / "edge_index.json"
    save_edge_index(
        [
            _edge_record("AAA", "2026-01-01T00:00:00-05:00"),
            _edge_record("BBB", "2026-02-01T00:00:00-05:00"),
            _edge_record("CCC", "2026-03-01T00:00:00-05:00"),
        ],
        index_path,
    )
    seen_record_types = []
    seen_allow_future = []

    def fake_find_analogs(features, records, k=7, embargo_days=5, allow_future=True, **kwargs):
        seen_record_types.append(type(records).__name__)
        seen_allow_future.append(allow_future)
        return []

    monkeypatch.setattr("scanner.main.EDGE_INDEX_PATH", index_path)
    monkeypatch.setattr("scanner.main.EDGE_VALIDATION_REPORT_PATH", tmp_path / "edge_validation_report.json")
    monkeypatch.setattr("scanner.main.find_analogs", fake_find_analogs)

    validation = run_validate_edge(logger)

    assert seen_record_types == ["EdgeAnalogIndex", "EdgeAnalogIndex", "EdgeAnalogIndex"]
    assert seen_allow_future == [False, False, False]
    assert validation["validation_method"] == "purged_walk_forward"


def test_run_edge_lab_orchestrates_single_evidence_run(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    monkeypatch.setattr("scanner.main.EVIDENCE_DIR", tmp_path)
    calls = []

    def fake_build(watchlist, logger, evidence_run=None):
        calls.append(("build", evidence_run.run_id))
        return {"mode": "build_retrieval_index", "records": 1}

    def fake_validate(logger, evidence_run=None):
        calls.append(("validate", evidence_run.run_id))
        return {"mode": "validate_edge", "candidate_count": 1, "thresholds": {}}

    def fake_scan(watchlist, logger, evidence_run=None):
        calls.append(("scan", evidence_run.run_id))
        return {"mode": "edge_scan", "total": 1, "candidates": []}

    def fake_diagnose(logger, evidence_run=None):
        calls.append(("diagnose", evidence_run.run_id))
        return {"mode": "diagnose_edge", "index_records": 1}

    monkeypatch.setattr("scanner.main.run_build_retrieval_index", fake_build)
    monkeypatch.setattr("scanner.main.run_validate_edge", fake_validate)
    monkeypatch.setattr("scanner.main.run_edge_scan", fake_scan)
    monkeypatch.setattr("scanner.main.run_diagnose_edge", fake_diagnose)

    payload = run_edge_lab(["AAA"], logger)

    assert [name for name, _run_id in calls] == ["build", "validate", "scan", "diagnose"]
    assert len({_run_id for _name, _run_id in calls}) == 1
    assert payload["mode"] == "run_edge_lab"
    assert (tmp_path / calls[0][1] / "manifest.json").exists()


def test_audit_edge_writes_readiness_report(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    validation_path = tmp_path / "edge_validation_report.json"
    scan_path = tmp_path / "edge_scan_report.json"
    audit_path = tmp_path / "edge_audit_report.json"
    validation_path.write_text(
        json.dumps(
            {
                "completed_at": "2026-08-01T18:00:00+00:00",
                "evidence_run_id": "lab-run",
                "runtime_fingerprint": "sha256:test-runtime",
                "validation_method": "purged_walk_forward",
                "future_analogs_allowed": False,
                "thresholds": {"55": {"signal_count": 0, "precision": 0.0, "average_r_multiple": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    scan_path.write_text(
        json.dumps(
            {
                "completed_at": "2026-08-01T18:01:00+00:00",
                "evidence_run_id": "lab-run",
                "runtime_fingerprint": "sha256:test-runtime",
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scanner.main.EDGE_VALIDATION_REPORT_PATH", validation_path)
    monkeypatch.setattr("scanner.main.EDGE_SCAN_REPORT_PATH", scan_path)
    monkeypatch.setattr("scanner.main.EDGE_AUDIT_REPORT_PATH", audit_path)
    monkeypatch.setattr("scanner.main._edge_runtime_fingerprint", lambda: "sha256:test-runtime")

    report = run_audit_edge(logger)

    assert report["readiness"] == "blocked"
    assert "validation_threshold_55_unsupported" in report["blockers"]
    assert report["evidence_provenance"] == {
        "scan_completed_at": "2026-08-01T18:01:00+00:00",
        "validation_completed_at": "2026-08-01T18:00:00+00:00",
        "scan_run_id": "lab-run",
        "validation_run_id": "lab-run",
        "scan_runtime_fingerprint": "sha256:test-runtime",
        "validation_runtime_fingerprint": "sha256:test-runtime",
        "runtime_fingerprint": "sha256:test-runtime",
    }
    assert report["checks"]["runtime_source"]["passed"] is True
    assert report["generated_at"]
    assert json.loads(audit_path.read_text(encoding="utf-8"))["mode"] == "audit_edge"


def test_audit_edge_blocks_mixed_runtime_source(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    validation_path = tmp_path / "edge_validation_report.json"
    scan_path = tmp_path / "edge_scan_report.json"
    audit_path = tmp_path / "edge_audit_report.json"
    validation_path.write_text(
        json.dumps(
            {
                "runtime_fingerprint": "sha256:old-runtime",
                "validation_method": "purged_walk_forward",
                "future_analogs_allowed": False,
                "thresholds": {},
            }
        ),
        encoding="utf-8",
    )
    scan_path.write_text(
        json.dumps({"runtime_fingerprint": "sha256:current-runtime", "candidates": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr("scanner.main.EDGE_VALIDATION_REPORT_PATH", validation_path)
    monkeypatch.setattr("scanner.main.EDGE_SCAN_REPORT_PATH", scan_path)
    monkeypatch.setattr("scanner.main.EDGE_AUDIT_REPORT_PATH", audit_path)
    monkeypatch.setattr("scanner.main._edge_runtime_fingerprint", lambda: "sha256:current-runtime")

    report = run_audit_edge(logger)

    assert report["checks"]["runtime_source"]["passed"] is False
    assert "evidence_runtime_source_mismatch" in report["blockers"]
    assert report["readiness"] == "blocked"
