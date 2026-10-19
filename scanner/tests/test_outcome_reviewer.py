import logging

import pandas as pd

from scanner.learning import outcome_reviewer


def test_review_pending_outcomes_anchors_after_hours_decision_to_signal_session(monkeypatch, tmp_path):
    synthetic = pd.DataFrame(
        {"Close": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-06-24 00:00", tz="America/New_York"),
                pd.Timestamp("2026-06-25 00:00", tz="America/New_York"),
                pd.Timestamp("2026-06-26 00:00", tz="America/New_York"),
                pd.Timestamp("2026-06-29 00:00", tz="America/New_York"),
                pd.Timestamp("2026-06-30 00:00", tz="America/New_York"),
                pd.Timestamp("2026-07-01 00:00", tz="America/New_York"),
            ]
        ),
    )
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-06-24T19:03:00-04:00",
            "direction": "bullish",
            "entry_price": 10.0,
            "outcome_status": "pending",
            "counterfactual": True,
        }
    ]

    seen = {}

    def fake_fetch(ticker, research=False):
        seen["research"] = research
        return pd.DataFrame()

    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", fake_fetch)
    monkeypatch.setattr(
        outcome_reviewer,
        "build_synthetic_sessions",
        lambda intraday, anchor_hour, anchor_minute, source_interval, prepost_enabled: (synthetic, {}),
    )

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    # Outcome resolution must use the delayed consolidated (SIP) feed - IEX
    # highs/lows systematically miss real stop/target touches.
    assert seen["research"] is True
    assert summary["resolved_now"] == 1
    assert summary["resolved_counterfactual"] == 1
    assert reviewed[0]["outcome_status"] == "resolved"
    assert reviewed[0]["outcome_label"] == "win"
    assert reviewed[0]["outcome_ret_5bar_pct"] == 25.0


def test_source_timestamp_maps_to_shifted_synthetic_session_label():
    sessions = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-12 00:00", tz="America/New_York"),
            pd.Timestamp("2026-07-13 00:00", tz="America/New_York"),
        ]
    )
    record = {
        "latest_source_timestamp": "2026-07-13T14:00:00-04:00",
        "anchor_hour": 20,
        "anchor_minute": 0,
    }
    created = pd.Timestamp("2026-07-13T14:32:00-04:00")

    assert outcome_reviewer._record_session_position(sessions, record, created) == 0
    # The legacy decision-time lookup demonstrates the old off-by-one: it
    # selects the next session label, whose bars come from July 14.
    assert outcome_reviewer._decision_session_position(sessions, created) == 1


def test_review_waits_for_complete_target_synthetic_session(monkeypatch, tmp_path):
    idx = pd.DatetimeIndex(
        [pd.Timestamp(day, tz="America/New_York") for day in (
            "2026-07-12",
            "2026-07-13",
            "2026-07-14",
            "2026-07-15",
            "2026-07-16",
            "2026-07-19",
        )]
    )
    synthetic = pd.DataFrame(
        {
            "Open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "High": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "Low": [9.9, 10.0, 10.1, 10.2, 10.3, 10.4],
            "Close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
        },
        index=idx,
    )
    synthetic.attrs["data_delay_minutes"] = 16
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-07-13T14:32:00-04:00",
            "latest_source_timestamp": "2026-07-13T14:00:00-04:00",
            "anchor_hour": 20,
            "anchor_minute": 0,
            "direction": "bullish",
            "entry_price": 10.0,
            "outcome_status": "pending",
            "counterfactual": True,
        }
    ]

    now = {"value": pd.Timestamp("2026-07-20T14:30:00-04:00")}
    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "_now_timestamp", lambda: now["value"])
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", lambda ticker, research=False: pd.DataFrame())
    monkeypatch.setattr(
        outcome_reviewer,
        "build_synthetic_sessions",
        lambda intraday, anchor_hour, anchor_minute, source_interval, prepost_enabled: (synthetic, {}),
    )

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))
    assert summary["resolved_now"] == 0
    assert reviewed[0]["outcome_status"] == "pending"

    now["value"] = pd.Timestamp("2026-07-20T20:16:00-04:00")
    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))
    assert summary["resolved_now"] == 1
    assert reviewed[0]["outcome_status"] == "resolved"
    assert reviewed[0]["outcome_ret_5bar_pct"] == 5.0
    assert (
        reviewed[0]["outcome_session_alignment_version"]
        == outcome_reviewer.OUTCOME_SESSION_ALIGNMENT_VERSION
    )


def test_resolved_source_timestamp_outcome_is_repaired(monkeypatch, tmp_path):
    idx = pd.date_range("2026-07-01", periods=7, freq="D", tz="America/New_York")
    synthetic = pd.DataFrame(
        {
            "Open": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
            "High": [10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3],
            "Low": [9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1],
            "Close": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
        },
        index=idx,
    )
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-07-02T14:32:00-04:00",
            "latest_source_timestamp": "2026-07-02T14:00:00-04:00",
            "anchor_hour": 0,
            "anchor_minute": 0,
            "direction": "bullish",
            "entry_price": 10.2,
            "outcome_status": "resolved",
            "outcome_label": "loss",
            "outcome_ret_5bar_pct": -99.0,
            "counterfactual": True,
        }
    ]

    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "_now_timestamp", lambda: pd.Timestamp("2026-07-20", tz="America/New_York"))
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", lambda ticker, research=False: pd.DataFrame())
    monkeypatch.setattr(
        outcome_reviewer,
        "build_synthetic_sessions",
        lambda intraday, anchor_hour, anchor_minute, source_interval, prepost_enabled: (synthetic, {}),
    )

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["alignment_repairs_reviewed"] == 1
    assert summary["alignment_repairs_resolved"] == 1
    assert summary["alignment_repairs_pending"] == 0
    assert reviewed[0]["outcome_status"] == "resolved"
    assert reviewed[0]["outcome_label"] == "win"
    assert reviewed[0]["outcome_ret_5bar_pct"] != -99.0


def _ohlc_sessions():
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"2026-06-{day:02d} 00:00", tz="America/New_York") for day in (22, 23, 24, 25, 26, 29, 30)]
    )
    # Entry session close 10.0; path then runs to 13.0 without touching a
    # 2R-style target exit because the shipped plan has NO target.
    return pd.DataFrame(
        {
            "Open": [10.0, 10.2, 10.6, 11.2, 11.8, 12.4, 12.9],
            "High": [10.1, 10.7, 11.4, 12.0, 12.6, 13.1, 13.4],
            "Low": [9.9, 10.1, 10.5, 11.1, 11.7, 12.3, 12.8],
            "Close": [10.0, 10.6, 11.3, 11.9, 12.5, 13.0, 13.3],
        },
        index=idx,
    )


def test_journal_outcomes_follow_shipped_no_target_geometry(monkeypatch, tmp_path):
    # The reviewer used to hardcode a 2R target, so the adaptive policy
    # learned from an exit geometry the lab no longer trades. Under the
    # shipped no-target plan, this runaway winner must exit at the horizon,
    # not at a phantom target.
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-06-22T15:00:00-04:00",
            "direction": "bullish",
            "entry_price": 10.0,
            "outcome_status": "pending",
            "counterfactual": True,
        }
    ]

    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", lambda ticker, research=False: pd.DataFrame())
    monkeypatch.setattr(
        outcome_reviewer,
        "build_synthetic_sessions",
        lambda intraday, anchor_hour, anchor_minute, source_interval, prepost_enabled: (_ohlc_sessions(), {}),
    )

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["resolved_now"] == 1
    assert reviewed[0]["outcome_method"] == "triple_barrier"
    assert reviewed[0]["outcome_target_mode"] == "none"
    assert reviewed[0]["outcome_exit_reason"] == "horizon"
    assert reviewed[0]["outcome_return_pct"] == 30.0


def test_split_rescaled_history_quarantines_record(monkeypatch, tmp_path):
    # A 1:10 reverse split between decision and review re-scales the fetched
    # (split-adjusted) history; walking barriers against the old-scale entry
    # would fabricate a catastrophic outcome. The record must be quarantined,
    # not resolved.
    sessions = _ohlc_sessions() * 10.0  # decision-session close now 100, entry recorded at 10
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-06-22T15:00:00-04:00",
            "direction": "bullish",
            "entry_price": 10.0,
            "outcome_status": "pending",
            "counterfactual": True,
        }
    ]

    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", lambda ticker, research=False: pd.DataFrame())
    monkeypatch.setattr(
        outcome_reviewer,
        "build_synthetic_sessions",
        lambda intraday, anchor_hour, anchor_minute, source_interval, prepost_enabled: (sessions, {}),
    )

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["resolved_now"] == 0
    assert summary["quarantined_scale_mismatch"] == 1
    assert reviewed[0]["outcome_status"] == "not_applicable"
    assert "scale mismatch" in reviewed[0]["outcome_error"]


def test_corrupt_intraday_bars_block_resolution(monkeypatch, tmp_path):
    # NaN OHLC in the fetched bars must leave the record pending (retry on a
    # cleaner fetch), never resolve into a label.
    idx = pd.date_range("2026-06-22", periods=3, freq="30min", tz="America/New_York")
    corrupt = pd.DataFrame(
        {
            "Open": [10.0, float("nan"), 10.2],
            "High": [10.1, 10.4, 10.5],
            "Low": [9.9, 10.0, 10.1],
            "Close": [10.0, float("nan"), 10.3],
            "Volume": [100, 100, 100],
        },
        index=idx,
    )
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-06-22T15:00:00-04:00",
            "direction": "bullish",
            "entry_price": 10.0,
            "outcome_status": "pending",
            "counterfactual": True,
        }
    ]

    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", lambda ticker, research=False: corrupt)

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["resolved_now"] == 0
    assert summary["blocked_by_bar_contract"] == 1
    assert reviewed[0]["outcome_status"] == "pending"
    assert "bar contract violation" in reviewed[0]["outcome_error"]
