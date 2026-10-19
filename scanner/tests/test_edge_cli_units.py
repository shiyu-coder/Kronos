import logging

import pandas as pd

from scanner.edge.retrieval import EdgeRecord
from scanner.data.synthetic_sessions import build_synthetic_sessions
from scanner.utils.validation import OptionsContractResult
from scanner.utils.validation import TickerValidationResult
from scanner.main import (
    _build_edge_diagnostic_payload,
    _data_provenance,
    _edge_data_quality,
    _run_single_ticker,
    _score_edge_for_bars,
    run_watchlist_scan,
)


def _bars():
    rows = []
    for i in range(45):
        if i < 29:
            rows.append([100, 104, 96, 100 + (0.2 if i % 2 == 0 else -0.2), 1000])
        elif i < 44:
            rows.append([100, 101, 99, 100 + (0.05 if i % 2 == 0 else -0.05), 1200])
        else:
            rows.append([101, 104, 100.5, 103, 2600])
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def _analog_records():
    base_features = {
        "ticker": "HIST",
        "timestamp": "2025-01-01T00:00:00-05:00",
        "direction": "bullish",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.8,
        "rr_ratio": 2.0,
        "data_quality_score": 1.0,
        "feed_confidence": 0.9,
    }
    return [
        EdgeRecord("H1", "2025-01-01T00:00:00-05:00", "bullish", base_features, 3.0, "win", 1.2, -0.5, 4.0),
        EdgeRecord("H2", "2025-01-10T00:00:00-05:00", "bullish", base_features, 2.0, "win", 0.9, -0.7, 3.0),
        EdgeRecord("H3", "2025-01-20T00:00:00-05:00", "bullish", base_features, -0.5, "loss", -0.2, -1.2, 1.0),
    ]


def _valid_options_contract(*_args, **_kwargs):
    return OptionsContractResult(
        passed=True,
        expiration="2026-02-20",
        dte=35,
        contract_type="call",
        strike=105.0,
        bid=1.0,
        ask=1.1,
        midpoint=1.05,
        spread_pct=0.09,
        open_interest=420,
        volume=75,
        implied_volatility=0.5,
    )


def test_score_edge_for_bars_returns_scorecard_without_network():
    result = _score_edge_for_bars(
        "TEST",
        _bars(),
        _analog_records(),
        logging.getLogger("test"),
        options_selector=_valid_options_contract,
    )

    assert result["status"] == "candidate"
    assert result["ticker"] == "TEST"
    assert "edge_score" in result
    assert "scorecard" in result
    assert result["analog_summary"]["count"] > 0
    assert result["gross_analog_summary"]["count"] > 0
    assert result["analog_cost_model"]["analog_rows"] > 0
    assert result["analog_cost_model"]["basis"] == "net_of_costs"


def test_score_edge_for_bars_includes_option_liquidity(monkeypatch):
    monkeypatch.setattr("scanner.main.select_options_contract", _valid_options_contract)

    result = _score_edge_for_bars("TEST", _bars(), _analog_records(), logging.getLogger("test"))

    assert result["status"] == "candidate"
    assert result["features"]["options_passed"] == 1.0
    assert result["features"]["options_open_interest"] == 420.0
    assert result["features"]["options_volume"] == 75.0
    assert result["features"]["options_spread_pct"] == 0.09


def test_score_edge_for_bars_includes_doctrine_v2_payload(monkeypatch):
    monkeypatch.setattr("scanner.main.select_options_contract", _valid_options_contract)

    result = _score_edge_for_bars("TEST", _bars(), _analog_records(), logging.getLogger("test"))

    assert result["status"] == "candidate"
    assert "doctrine_v2_score" in result["features"]
    assert result["doctrine_v2"]["score"] >= 0
    assert "doctrine_v2" in result["scorecard"]


def test_edge_data_quality_uses_provider_and_staleness_metadata():
    bars = _bars()
    bars.index = pd.date_range("2026-01-02 10:00", periods=len(bars), freq="1min", tz="America/New_York")
    now = bars.index[-1] + pd.Timedelta(minutes=90)

    sip_quality = _edge_data_quality(
        bars,
        provider="alpaca",
        alpaca_feed="sip",
        alpaca_credentials_available=True,
        now=now,
    )
    iex_quality = _edge_data_quality(
        bars,
        provider="alpaca",
        alpaca_feed="iex",
        alpaca_credentials_available=True,
        now=now,
    )
    yfinance_quality = _edge_data_quality(
        bars,
        provider="yfinance",
        alpaca_feed="sip",
        alpaca_credentials_available=False,
        now=now,
    )

    assert sip_quality["feed_confidence"] == 0.9
    assert iex_quality["feed_confidence"] == 0.7
    assert yfinance_quality["feed_confidence"] == 0.5
    assert sip_quality["stale_minutes"] == 90
    assert sip_quality["missing_bars"] == 0
    assert sip_quality["quality_score"] < 1.0


def test_edge_data_quality_keeps_fresh_complete_bars_pristine():
    bars = _bars()
    now = bars.index[-1]

    quality = _edge_data_quality(
        bars,
        provider="alpaca",
        alpaca_feed="sip",
        alpaca_credentials_available=True,
        now=now,
    )

    assert quality["stale_minutes"] == 0
    assert quality["missing_bars"] == 0
    assert quality["quality_score"] == 1.0


def test_edge_data_quality_prefers_synthetic_source_timestamp():
    idx = pd.date_range("2026-01-02 09:30", periods=4, freq="30min", tz="America/New_York")
    intraday = pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.5, 1000],
            [100.5, 101.5, 100.0, 101.0, 1100],
            [101.0, 102.0, 100.5, 101.5, 1200],
            [101.5, 102.5, 101.0, 102.0, 1300],
        ],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )

    synthetic, _ = build_synthetic_sessions(
        intraday,
        session_anchor_hour=20,
        session_anchor_minute=0,
        source_interval="30m",
        prepost_enabled=True,
    )
    quality = _edge_data_quality(
        synthetic,
        provider="alpaca",
        alpaca_feed="sip",
        alpaca_credentials_available=True,
        now=idx[-1] + pd.Timedelta(minutes=5),
    )

    assert synthetic.index[-1] < idx[-1]
    assert quality["stale_minutes"] == 5
    assert quality["quality_score"] == 1.0


def test_edge_data_quality_does_not_penalize_market_closed_weekend_gap():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-26 15:30", tz="America/New_York")])
    bars = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1000]],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )

    quality = _edge_data_quality(
        bars,
        provider="alpaca",
        alpaca_feed="iex",
        alpaca_credentials_available=True,
        now=pd.Timestamp("2026-06-28 18:00", tz="America/New_York"),
    )

    assert quality["stale_minutes"] == 30
    assert quality["quality_score"] == 1.0

    reopened_quality = _edge_data_quality(
        bars,
        provider="alpaca",
        alpaca_feed="iex",
        alpaca_credentials_available=True,
        now=pd.Timestamp("2026-06-29 10:30", tz="America/New_York"),
    )

    assert reopened_quality["stale_minutes"] == 90
    assert reopened_quality["quality_score"] < 1.0


def test_data_provenance_reads_bar_metadata():
    bars = _bars()
    bars.attrs.update({"data_provider": "alpaca", "data_feed": "sip", "data_delay_minutes": 16})

    provenance = _data_provenance(bars)

    assert provenance == {
        "data_provider": "alpaca",
        "data_feed": "sip",
        "data_delay_minutes": 16,
        "latest_source_timestamp": "2026-02-14T00:00:00-05:00",
        "source_session_date": "2026-02-14",
    }


def test_build_edge_diagnostic_payload_summarizes_state():
    payload = _build_edge_diagnostic_payload(
        index_records=_analog_records(),
        validation_report={"samples": 10, "thresholds": {"65": {"precision": 0.5}}},
        scan_report={"candidates": [{"recommendation": "research"}, {"recommendation": "promote"}]},
    )

    assert payload["index_records"] == 3
    assert payload["validation_samples"] == 10
    assert payload["recommendation_counts"]["promote"] == 1


def test_build_edge_diagnostic_payload_counts_rejection_reasons():
    payload = _build_edge_diagnostic_payload(
        index_records=_analog_records(),
        validation_report={"samples": 10, "thresholds": {"55": {"precision": 0.0}}},
        scan_report={
            "candidates": [
                {
                    "ticker": "AAA",
                    "recommendation": "reject",
                    "rejection_reasons": ["setup_gate_failed", "options_data_not_execution_grade"],
                    "blocking_reasons": ["setup_gate_failed", "options_data_not_execution_grade"],
                },
                {
                    "ticker": "BBB",
                    "recommendation": "reject",
                    "rejection_reasons": ["setup_gate_failed"],
                    "blocking_reasons": ["setup_gate_failed"],
                },
            ]
        },
    )

    assert payload["rejection_reason_counts"] == {
        "setup_gate_failed": 2,
        "options_data_not_execution_grade": 1,
    }
    assert payload["blocking_reason_counts"]["setup_gate_failed"] == 2


def test_research_scan_decision_records_doctrine_v2(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "scanner.main.validate_ticker",
        lambda ticker, logger: TickerValidationResult(ticker, True, 100.0, True, True),
    )
    monkeypatch.setattr("scanner.main._resolve_calibrated_anchor", lambda ticker: (10, 0))
    monkeypatch.setattr("scanner.main.fetch_intraday_bars", lambda ticker, research=False: _bars())
    monkeypatch.setattr(
        "scanner.main.build_synthetic_sessions",
        lambda intraday_df, **kwargs: (intraday_df, {"source_interval": "test"}),
    )
    monkeypatch.setattr("scanner.main.append_decision", lambda record: captured.append(record))

    result = _run_single_ticker("TEST", "research_scan", {}, None, None, logging.getLogger("test"))

    assert result["ticker"] == "TEST"
    assert captured
    assert "doctrine_v2_score" in captured[0]
    assert "doctrine_v2_diagnostics" in captured[0]
    assert captured[0]["source_session_date"] == "2026-02-14"
    assert "latest_source_timestamp" in captured[0]


def test_watchlist_scan_reports_runtime_metadata(monkeypatch):
    clock = iter([100.0, 101.0, 103.5, 107.0, 108.0])
    timestamps = iter(["start", "done"])
    monkeypatch.setattr("scanner.main._monotonic_seconds", lambda: next(clock))
    monkeypatch.setattr("scanner.main._utc_now_iso", lambda: next(timestamps))
    monkeypatch.setattr(
        "scanner.main._run_single_ticker",
        lambda ticker, mode, env, kronos, minimax, logger: {"ticker": ticker, "status": "skip", "reason": "test"},
    )

    summary = run_watchlist_scan(["AAA", "BBB"], "research_scan", {}, logging.getLogger("test"))

    assert summary["started_at"] == "start"
    assert summary["completed_at"] == "done"
    assert summary["duration_seconds"] == 8.0
    assert summary["ticker_timings"] == [
        {"ticker": "AAA", "status": "skip", "duration_seconds": 2.5},
        {"ticker": "BBB", "status": "skip", "duration_seconds": 1.0},
    ]
