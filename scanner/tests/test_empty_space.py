import pandas as pd
import pytest

from scanner.strategy.empty_space import score_empty_space


def _bars():
    idx = pd.date_range("2025-01-01", periods=80, freq="D", tz="America/New_York")
    base = [100.0 for _ in range(80)]
    data = {
        "Open": base,
        "High": [101.0 for _ in range(80)],
        "Low": [99.0 for _ in range(80)],
        "Close": base,
        "Volume": [1000] * 80,
    }
    return pd.DataFrame(data, index=idx)


def test_score_zero_when_blocked():
    bars = _bars()
    bars.iloc[-2, bars.columns.get_loc("High")] = 110.1
    res = score_empty_space(bars, "bullish", breakout_close=110.0, cost_basis=109.7)
    assert res.score == 0
    assert res.passed is False


def test_score_two_rr_ge_15():
    bars = _bars()
    bars.iloc[:-1, bars.columns.get_loc("High")] = 120.0
    res = score_empty_space(bars, "bullish", breakout_close=110.0, cost_basis=106.0)
    assert res.rr_ratio >= 1.5
    assert res.score >= 2


def test_score_three_rr_ge_25():
    bars = _bars()
    bars.iloc[:-1, bars.columns.get_loc("Low")] = 90.0
    res = score_empty_space(bars, "bearish", breakout_close=100.0, cost_basis=104.0)
    assert res.rr_ratio >= 2.5
    assert res.score == 3


def test_next_target_reported_bullish():
    bars = _bars()
    # Two resistance shelves above the 110 breakout: 112 (nearest), 118 (next).
    bars.iloc[:-1, bars.columns.get_loc("High")] = 112.0
    bars.iloc[10:20, bars.columns.get_loc("High")] = 118.0
    res = score_empty_space(bars, "bullish", breakout_close=110.0, cost_basis=106.0)
    assert res.nearest_target == 112.0
    assert res.diagnostics["next_target"] == 118.0
    assert res.diagnostics["distance_to_next_target_pct"] == pytest.approx((118.0 - 110.0) / 110.0 * 100)


def test_next_target_reported_bearish():
    bars = _bars()
    # Two support shelves below the 100 breakdown: 95 (nearest), 90 (next).
    bars.iloc[:-1, bars.columns.get_loc("Low")] = 95.0
    bars.iloc[10:20, bars.columns.get_loc("Low")] = 90.0
    res = score_empty_space(bars, "bearish", breakout_close=100.0, cost_basis=104.0)
    assert res.nearest_target == 95.0
    assert res.diagnostics["next_target"] == 90.0
    assert res.diagnostics["distance_to_next_target_pct"] == pytest.approx(10.0)


def test_next_target_none_when_only_one_level():
    bars = _bars()
    # Every historical high sits at 101: one level above the breakout, no next.
    res = score_empty_space(bars, "bullish", breakout_close=100.5, cost_basis=99.0)
    assert res.nearest_target == 101.0
    assert res.diagnostics["next_target"] is None
    assert res.diagnostics["distance_to_next_target_pct"] is None
