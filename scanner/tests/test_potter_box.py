import pandas as pd

from scanner import config
from scanner.strategy.potter_box import detect_potter_box, score_potter_research_candidate


def _make_synthetic_df():
    rows = []
    price = 100.0
    for i in range(40):
        if i < 24:
            high = price + 3.0
            low = price - 3.0
            close = price + (0.2 if i % 2 == 0 else -0.2)
        elif i < 39:
            high = 101.0
            low = 99.0
            close = 100.0 + (0.05 if i % 2 == 0 else -0.05)
        else:
            high = 103.0
            low = 100.5
            close = 102.5
        rows.append([price, high, low, close, 1000])
    idx = pd.date_range("2025-01-01", periods=40, freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def test_consolidation_excludes_breakout_candle():
    df = _make_synthetic_df()
    result = detect_potter_box("TEST", df)
    assert result.box_top == 101.0
    assert result.breakout_close > result.box_top


def test_bullish_requires_prior_close_above_cost_basis():
    df = _make_synthetic_df()
    df.iloc[-2, df.columns.get_loc("Close")] = 99.0
    result = detect_potter_box("TEST", df)
    assert result.passed is False


def test_compression_metrics_are_advisory_not_setup_gates(monkeypatch):
    df = _make_synthetic_df()
    monkeypatch.setattr(config, "ATR_COMPRESSION", 0.01)
    monkeypatch.setattr(config, "RANGE_COMPRESSION", 0.01)
    monkeypatch.setattr(config, "NO_TREND_SLOPE_ABS_MAX", 0.0)

    result = detect_potter_box("TEST", df)

    assert result.diagnostics["atr_compressed"] is False
    assert result.diagnostics["range_compressed"] is False
    assert result.diagnostics["no_trend"] is False
    assert result.diagnostics["top_touches_ok"] is True
    assert result.diagnostics["bottom_touches_ok"] is True
    assert result.diagnostics["bullish_breakout"] is True
    assert result.passed is True


def test_documented_setup_can_pass_with_less_than_atr_plus_box_history():
    df = _make_synthetic_df().iloc[-18:].copy()

    result = detect_potter_box("TEST", df)

    assert result.diagnostics["top_touches_ok"] is True
    assert result.diagnostics["bottom_touches_ok"] is True
    assert result.diagnostics["bullish_breakout"] is True
    assert result.passed is True


def test_research_candidate_scores_near_breakout():
    df = _make_synthetic_df()
    df.iloc[-1, df.columns.get_loc("Close")] = 100.7
    df.iloc[-1, df.columns.get_loc("High")] = 101.0
    df.iloc[-1, df.columns.get_loc("Volume")] = 2000
    result = detect_potter_box("TEST", df)
    candidate = score_potter_research_candidate(result, df)
    assert candidate["direction"] == "bullish"
    assert candidate["score"] > 0
    assert candidate["breakout_state"] in {"near_breakout", "confirmed_breakout"}


def test_research_candidate_uses_current_runtime_threshold(monkeypatch):
    df = _make_synthetic_df()
    df.iloc[-1, df.columns.get_loc("Close")] = 100.7
    df.iloc[-1, df.columns.get_loc("High")] = 101.0
    df.iloc[-1, df.columns.get_loc("Volume")] = 2000
    result = detect_potter_box("TEST", df)

    score = score_potter_research_candidate(result, df)["score"]
    monkeypatch.setattr(config, "RESEARCH_CANDIDATE_MIN_SCORE", score + 1)

    candidate = score_potter_research_candidate(result, df)

    assert candidate["passed"] is False
    assert candidate["reason"] == "score below research threshold"
    assert candidate["min_score"] == score + 1
