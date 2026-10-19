import logging

import pandas as pd
import pytest

from scanner import config as scanner_config
from scanner.edge.outcomes import resolve_plan_target_pct
from scanner.edge.retrieval import _future_outcome, resolve_trade_risk_pct
from scanner.learning import outcome_reviewer


@pytest.fixture(autouse=True)
def _pin_baseline_geometry(monkeypatch):
    # These tests exercise barrier mechanics; pin the plan geometry so a
    # change to the shipped exit-geometry default cannot silently reshape
    # their expected targets.
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_MODE", "nearest_empty_space")
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_R_FLOOR", 0.0)
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_ATR_MULT", 2.0)


def _bars(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def test_stop_hit_first_is_a_loss_even_if_price_recovers():
    # Entry 100, risk 2% -> stop 98. Bar 2 dips to 97, then price runs to 106.
    # The old close-at-horizon label called this a win.
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],  # idx 0 = entry bar
            [100, 102, 99.5, 101, 1000],
            [101, 101.5, 97.0, 99, 1000],  # stop touched
            [99, 104, 98.5, 103, 1000],
            [103, 106, 102, 105, 1000],
            [105, 107, 104, 106, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=8.0)

    assert outcome["exit_reason"] == "stop"
    assert outcome["label"] == "loss"
    assert outcome["return_pct"] == -2.0
    assert outcome["r_multiple"] == -1.0


def test_target_hit_first_is_a_win_with_target_r_multiple():
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 102, 99.5, 101, 1000],
            [101, 104.5, 100.5, 104, 1000],  # target 104 touched, stop never
            [104, 105, 103, 104.5, 1000],
            [104, 105, 103, 104.5, 1000],
            [104, 105, 103, 104.5, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)

    assert outcome["exit_reason"] == "target"
    assert outcome["label"] == "win"
    assert outcome["return_pct"] == 4.0
    assert outcome["r_multiple"] == 2.0


def test_same_bar_stop_and_target_resolves_conservatively_to_stop():
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 105, 97, 104, 1000],  # both barriers inside one bar
            [104, 105, 103, 104, 1000],
            [104, 105, 103, 104, 1000],
            [104, 105, 103, 104, 1000],
            [104, 105, 103, 104, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)

    assert outcome["exit_reason"] == "stop"
    assert outcome["label"] == "loss"


def test_horizon_exit_uses_final_close_sign():
    bars = _bars(
        [
            [100, 100.5, 99.5, 100, 1000],
            [100, 100.5, 99.5, 100.2, 1000],
            [100, 100.5, 99.5, 100.3, 1000],
            [100, 100.5, 99.5, 100.4, 1000],
            [100, 100.5, 99.5, 100.5, 1000],
            [100, 101.0, 99.5, 100.8, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)

    assert outcome["exit_reason"] == "horizon"
    assert outcome["label"] == "win"
    assert round(outcome["return_pct"], 2) == 0.8


def test_bearish_direction_mirrors_barriers():
    # Bearish entry 100, risk 2% -> stop 102 above, target 96 below.
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 101, 98, 99, 1000],
            [99, 100, 95.5, 96, 1000],  # target touched
            [96, 97, 95, 96.5, 1000],
            [96, 97, 95, 96.5, 1000],
            [96, 97, 95, 96.5, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bearish", 100.0, risk_pct=2.0, target_pct=4.0)

    assert outcome["exit_reason"] == "target"
    assert outcome["label"] == "win"
    assert outcome["return_pct"] == 4.0


def test_gap_through_stop_uses_open_fill_not_stop_price():
    # Entry 100, risk 2% -> stop 98, but the exit bar OPENS at 90. Flooring
    # the loss at -1R flattered the expectancy that gates promotion.
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [90, 92, 85, 88, 1000],  # gap-down open straight through the stop
            [88, 90, 86, 89, 1000],
            [89, 91, 87, 90, 1000],
            [90, 92, 88, 91, 1000],
            [91, 93, 89, 92, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)

    assert outcome["exit_reason"] == "stop"
    assert outcome["return_pct"] == -10.0
    assert outcome["r_multiple"] == -5.0


def test_gap_through_target_stays_capped_at_target():
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [106, 108, 105, 107, 1000],  # favorable gap beyond the 104 target
            [107, 108, 106, 107, 1000],
            [107, 108, 106, 107, 1000],
            [107, 108, 106, 107, 1000],
            [107, 108, 106, 107, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)

    assert outcome["exit_reason"] == "target"
    assert outcome["return_pct"] == 4.0  # conservative side of a favorable gap


def test_risk_fallback_uses_atr_then_default():
    assert resolve_trade_risk_pct(0.0, atr_value=1.5, entry=100.0) == 1.5
    assert resolve_trade_risk_pct(0.0, atr_value=0.0, entry=100.0) == 2.0
    assert resolve_trade_risk_pct(3.0, atr_value=1.5, entry=100.0) == 3.0
    # Clamped so a near-zero denominator can't manufacture huge R values.
    assert resolve_trade_risk_pct(0.06, atr_value=0.0, entry=100.0) == 0.25


def test_plan_target_baseline_passes_nearest_through():
    plan = resolve_plan_target_pct(3.0, 6.0, 1.0, 100.0, 2.0, mode="nearest_empty_space", r_floor=0.0)
    assert plan == {"target_pct": 3.0, "target_mode": "nearest_empty_space"}


def test_plan_target_baseline_degenerate_falls_back_to_2r():
    plan = resolve_plan_target_pct(0.0, 0.0, 0.0, 100.0, 2.0, mode="nearest_empty_space", r_floor=0.0)
    assert plan["target_pct"] == 4.0
    assert plan["target_mode"] == "nearest_empty_space:fallback_2r"


def test_plan_target_r_floor_lifts_too_close_targets():
    plan = resolve_plan_target_pct(1.0, 0.0, 0.0, 100.0, 2.0, mode="nearest_empty_space", r_floor=1.5)
    assert plan["target_pct"] == 3.0  # 1.5 x 2% resolved risk
    assert plan["target_mode"] == "nearest_empty_space+floor1.5"


def test_plan_target_r_floor_keeps_structural_targets_beyond_it():
    plan = resolve_plan_target_pct(5.0, 0.0, 0.0, 100.0, 2.0, mode="nearest_empty_space", r_floor=1.5)
    assert plan == {"target_pct": 5.0, "target_mode": "nearest_empty_space"}


def test_plan_target_next_level_prefers_next_then_falls_back():
    plan = resolve_plan_target_pct(2.0, 6.0, 0.0, 100.0, 2.0, mode="next_empty_space", r_floor=0.0)
    assert plan == {"target_pct": 6.0, "target_mode": "next_empty_space"}
    fallback = resolve_plan_target_pct(2.0, 0.0, 0.0, 100.0, 2.0, mode="next_empty_space", r_floor=0.0)
    assert fallback["target_pct"] == 2.0
    assert fallback["target_mode"] == "next_empty_space:fallback_nearest"
    degenerate = resolve_plan_target_pct(0.0, 0.0, 0.0, 100.0, 2.0, mode="next_empty_space", r_floor=0.0)
    assert degenerate["target_pct"] == 4.0
    assert degenerate["target_mode"] == "next_empty_space:fallback_2r"


def test_plan_target_atr_multiple_and_fallback():
    plan = resolve_plan_target_pct(1.0, 0.0, 3.0, 100.0, 2.0, mode="atr_multiple", r_floor=0.0, atr_mult=2.0)
    assert plan["target_pct"] == 6.0  # 2 x 3% ATR
    assert plan["target_mode"] == "atr_multiple"
    fallback = resolve_plan_target_pct(1.0, 0.0, 0.0, 100.0, 2.0, mode="atr_multiple", r_floor=0.0, atr_mult=2.0)
    assert fallback["target_pct"] == 4.0
    assert fallback["target_mode"] == "atr_multiple:fallback_2r"


def test_plan_target_none_mode_disables_the_target():
    plan = resolve_plan_target_pct(3.0, 6.0, 1.0, 100.0, 2.0, mode="none", r_floor=1.5)
    assert plan == {"target_pct": None, "target_mode": "none"}


def test_no_target_plan_rides_past_would_be_targets_to_horizon(monkeypatch):
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_MODE", "none")
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 102, 99.5, 101, 1000],
            [101, 104.5, 100.5, 104, 1000],  # a 4% target would exit here...
            [104, 108, 103, 107, 1000],
            [107, 110, 106, 109, 1000],
            [109, 112, 108, 111, 1000],  # ...but the plan rides to +11%
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)
    assert outcome["exit_reason"] == "horizon"
    assert outcome["return_pct"] == pytest.approx(11.0)
    assert outcome["r_multiple"] == pytest.approx(5.5)
    assert outcome["target_mode"] == "none"
    assert outcome["target_pct_used"] == 0.0


def test_no_target_plan_still_stops_out(monkeypatch):
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_MODE", "none")
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 101, 97.5, 98, 1000],  # stop 98 touched
            [98, 104, 97.9, 103, 1000],
            [103, 106, 102, 105, 1000],
            [105, 107, 104, 106, 1000],
            [106, 108, 105, 107, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)
    assert outcome["exit_reason"] == "stop"
    assert outcome["label"] == "loss"
    assert outcome["r_multiple"] == -1.0
    assert outcome["target_mode"] == "none"


def test_plan_target_reads_config_defaults(monkeypatch):
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_MODE", "nearest_empty_space")
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_R_FLOOR", 2.0)
    plan = resolve_plan_target_pct(1.0, 0.0, 0.0, 100.0, 2.0)
    assert plan["target_pct"] == 4.0
    assert plan["target_mode"] == "nearest_empty_space+floor2"


def test_future_outcome_stamps_target_geometry():
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 102, 99.5, 101, 1000],
            [101, 104.5, 100.5, 104, 1000],
            [104, 105, 103, 104.5, 1000],
            [104, 105, 103, 104.5, 1000],
            [104, 105, 103, 104.5, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=4.0)
    assert outcome["target_pct_used"] == 4.0
    assert outcome["target_mode"] == "nearest_empty_space"


def test_future_outcome_r_floor_changes_exit(monkeypatch):
    # Nearest level 1% away would exit at bar 1; a 2R floor (4%) holds the
    # trade until the 104 target prints at bar 2.
    monkeypatch.setattr(scanner_config, "EDGE_EXIT_TARGET_R_FLOOR", 2.0)
    bars = _bars(
        [
            [100, 101, 99, 100, 1000],
            [100, 102, 99.5, 101, 1000],
            [101, 104.5, 100.5, 104, 1000],
            [104, 105, 103, 104.5, 1000],
            [104, 105, 103, 104.5, 1000],
            [104, 105, 103, 104.5, 1000],
        ]
    )
    outcome = _future_outcome(bars, 0, 5, "bullish", 100.0, risk_pct=2.0, target_pct=1.0)
    assert outcome["target_pct_used"] == 4.0
    assert outcome["target_mode"] == "nearest_empty_space+floor2"
    assert outcome["exit_reason"] == "target"
    assert outcome["return_pct"] == 4.0
    assert outcome["r_multiple"] == 2.0


def _synthetic_sessions():
    closes = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
    return pd.DataFrame(
        {
            "Close": closes,
            "High": [c + 0.2 for c in closes],
            "Low": [c - 0.3 for c in closes],
        },
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


def _patch_reviewer(monkeypatch, tmp_path, synthetic):
    monkeypatch.setattr(outcome_reviewer, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(outcome_reviewer, "OUTCOME_MIN_AGE_DAYS", -1_000_000)
    monkeypatch.setattr(outcome_reviewer, "fetch_intraday_bars", lambda ticker, research=False: pd.DataFrame())
    monkeypatch.setattr(
        outcome_reviewer,
        "build_synthetic_sessions",
        lambda intraday, anchor_hour, anchor_minute, source_interval, prepost_enabled: (synthetic, {}),
    )


def test_review_applies_triple_barrier_to_journal_outcomes(monkeypatch, tmp_path):
    # Session ATR proxy: (High-Low)=0.5 on a 10.0 entry -> risk 5%, target 10%
    # (11.0). Session 2's high (11.2) hits the target first.
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
    _patch_reviewer(monkeypatch, tmp_path, _synthetic_sessions())

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["resolved_now"] == 1
    record = reviewed[0]
    assert record["outcome_method"] == "triple_barrier"
    assert record["outcome_label"] == "win"
    assert record["outcome_exit_reason"] == "target"
    assert record["outcome_risk_pct_used"] == pytest.approx(5.0)
    assert record["outcome_r_multiple"] == pytest.approx(2.0)
    assert record["outcome_return_pct"] == pytest.approx(10.0)  # barrier exit, matches the label
    assert record["outcome_ret_5bar_pct"] == pytest.approx(25.0)  # legacy metric preserved
    assert record["outcome_mae_pct"] == pytest.approx(2.0)  # path low 10.2 vs entry, up to exit
    assert record["outcome_mfe_pct"] == pytest.approx(12.0)  # path high 11.2 vs entry, up to exit


def test_review_falls_back_to_close_horizon_without_ohlc(monkeypatch, tmp_path):
    closes_only = _synthetic_sessions()[["Close"]]
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
    _patch_reviewer(monkeypatch, tmp_path, closes_only)

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["resolved_now"] == 1
    assert reviewed[0]["outcome_method"] == "close_horizon"
    assert reviewed[0]["outcome_label"] == "win"
    assert reviewed[0]["outcome_ret_5bar_pct"] == pytest.approx(25.0)


def test_review_expires_pending_older_than_resolution_window(monkeypatch, tmp_path):
    records = [
        {
            "ticker": "TEST",
            "decision_ts": "2026-01-02T10:00:00-05:00",
            "direction": "bullish",
            "entry_price": 10.0,
            "outcome_status": "pending",
            "counterfactual": True,
        }
    ]
    _patch_reviewer(monkeypatch, tmp_path, _synthetic_sessions())

    reviewed, summary = outcome_reviewer.review_pending_outcomes(records, logging.getLogger("test"))

    assert summary["expired_unresolvable"] == 1
    assert reviewed[0]["outcome_status"] == "not_applicable"
    assert reviewed[0]["outcome_error"] == "expired_before_resolution_window"
