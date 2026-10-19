import logging

import pandas as pd

from scanner.data import cross_check


def _daily(closes, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(closes), tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=idx,
    )


def _reference_from(df, scale=1.0):
    return {
        pd.Timestamp(ts).date().isoformat(): float(close) * scale
        for ts, close in df["Close"].items()
    }


def test_skips_without_token(monkeypatch):
    monkeypatch.delenv("TRADIER_API_TOKEN", raising=False)
    result = cross_check.cross_check_daily_bars("TEST", _daily([10.0] * 30), logging.getLogger("t"))
    assert result["status"] == "skipped"
    assert result["reason"] == "no_tradier_token"


def test_agreeing_series_passes(monkeypatch):
    closes = [10.0 + 0.1 * i for i in range(30)]
    df = _daily(closes)
    monkeypatch.setenv("TRADIER_API_TOKEN", "token")
    # A constant scale factor (different adjustment basis) must NOT flag -
    # returns are identical.
    monkeypatch.setattr(
        cross_check, "_tradier_daily_closes", lambda *a, **k: _reference_from(df, scale=0.5)
    )

    result = cross_check.cross_check_daily_bars("TEST", df, logging.getLogger("t"))

    assert result["status"] == "ok"
    assert result["compared_sessions"] == 29
    assert result["worst_return_diff_pp"] < 0.001


def test_missed_split_is_disagreement(monkeypatch):
    closes = [100.0] * 15 + [50.0] * 15  # primary carries an unadjusted 2:1 gap
    df = _daily(closes)
    reference = _reference_from(df)
    # The reference vendor adjusted the split: its series is smooth.
    for day, value in list(reference.items())[:15]:
        reference[day] = value / 2.0

    monkeypatch.setenv("TRADIER_API_TOKEN", "token")
    monkeypatch.setattr(cross_check, "_tradier_daily_closes", lambda *a, **k: reference)

    result = cross_check.cross_check_daily_bars("TEST", df, logging.getLogger("t"))

    assert result["status"] == "disagreement"
    assert "split-scale" in result["reason"]


def test_transport_failure_skips(monkeypatch):
    monkeypatch.setenv("TRADIER_API_TOKEN", "token")
    monkeypatch.setattr(cross_check, "_tradier_daily_closes", lambda *a, **k: {})

    result = cross_check.cross_check_daily_bars("TEST", _daily([10.0] * 30), logging.getLogger("t"))

    assert result["status"] == "skipped"
    assert result["reason"] == "tradier_unavailable"


def test_too_few_overlapping_sessions_skips(monkeypatch):
    df = _daily([10.0] * 10)
    monkeypatch.setenv("TRADIER_API_TOKEN", "token")
    monkeypatch.setattr(cross_check, "_tradier_daily_closes", lambda *a, **k: _reference_from(df))

    result = cross_check.cross_check_daily_bars("TEST", df, logging.getLogger("t"))

    assert result["status"] == "skipped"
    assert "overlapping sessions" in result["reason"]
