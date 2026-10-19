import pandas as pd

from scanner.strategy.potter_doctrine import score_potter_doctrine_v2
from scanner.utils.validation import PotterBoxResult


def _bars(closes):
    rows = []
    for close in closes:
        rows.append([close, close + 0.8, close - 0.8, close, 1000])
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def _potter_box(close=104.5):
    return PotterBoxResult(
        ticker="TEST",
        passed=False,
        direction="bullish",
        box_top=102.0,
        box_bottom=98.0,
        cost_basis=100.0,
        prior_close=101.2,
        breakout_close=close,
        breakout_strength_pct=2.45,
        atr_value=1.0,
        range_compression_ratio=0.7,
        no_trend_score=0.001,
        skip_reason="research",
        diagnostics={
            "control_top": 102.0,
            "control_bottom": 98.0,
            "top_touches": 3,
            "bottom_touches": 3,
            "touch_tolerance": 0.4,
        },
    )


def test_doctrine_scores_punchback_reclaim_after_breakout():
    bars = _bars([100.0] * 24 + [104.0, 102.2, 104.5])

    doctrine = score_potter_doctrine_v2("TEST", bars, _potter_box(), None)

    assert doctrine["direction"] == "bullish"
    assert doctrine["punchback_state"] == "reclaim"
    assert doctrine["cost_basis_state"] == "held"
    assert doctrine["box_stack_score"] > 0
    assert doctrine["score"] >= 70
    assert doctrine["passed"] is True
    assert "punchback_reclaim" in doctrine["reasons"]


def test_doctrine_does_not_call_pre_breakout_control_touch_a_reclaim():
    bars = _bars([100.0] * 24 + [101.5, 102.0, 104.5])

    doctrine = score_potter_doctrine_v2("TEST", bars, _potter_box(), None)

    assert doctrine["direction"] == "bullish"
    assert doctrine["punchback_state"] == "fresh_breakout"
    assert "fresh_breakout" in doctrine["reasons"]
    assert "punchback_reclaim" not in doctrine["reasons"]


def test_doctrine_rejects_failed_punchback_back_inside_box():
    bars = _bars([100.0] * 24 + [104.0, 101.9, 99.4])

    doctrine = score_potter_doctrine_v2("TEST", bars, _potter_box(close=99.4), None)

    assert doctrine["direction"] == "bullish"
    assert doctrine["punchback_state"] == "failed_reentry"
    assert doctrine["cost_basis_state"] == "lost"
    assert doctrine["passed"] is False
    assert "failed_reentry" in doctrine["risk_flags"]
