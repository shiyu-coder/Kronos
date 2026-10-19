import logging

import pandas as pd

from scanner.data import events


class _FakeTicker:
    def __init__(self, calendar=None, info=None):
        self.calendar = calendar
        self.info = info or {}


def _patch_ticker(monkeypatch, calendar, info=None):
    monkeypatch.setattr(events.yf, "Ticker", lambda _t: _FakeTicker(calendar, info))


def _calendar_for(days_from_now: int):
    date = pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(days=days_from_now)
    return {"Earnings Date": [date.date().isoformat()]}


def test_earnings_inside_block_window_blocks(monkeypatch):
    _patch_ticker(monkeypatch, _calendar_for(5))
    result = events.assess_event_risk("TEST", logging.getLogger("t"))
    assert result.passed is False
    assert "earnings within" in result.skip_reason


def test_earnings_at_cushion_edge_blocks(monkeypatch):
    # Yahoo dates are estimates; day BLOCK+1 must still block via the cushion.
    _patch_ticker(monkeypatch, _calendar_for(events.EARNINGS_BLOCK_DAYS + events.EARNINGS_ESTIMATE_CUSHION_DAYS))
    result = events.assess_event_risk("TEST", logging.getLogger("t"))
    assert result.passed is False
    assert "estimate cushion" in result.skip_reason


def test_earnings_far_out_passes(monkeypatch):
    _patch_ticker(monkeypatch, _calendar_for(30))
    result = events.assess_event_risk("TEST", logging.getLogger("t"))
    assert result.passed is True


def test_stale_past_earnings_date_blocks(monkeypatch):
    # A past date means the provider has not published the next date yet.
    _patch_ticker(monkeypatch, _calendar_for(-20))
    result = events.assess_event_risk("TEST", logging.getLogger("t"))
    assert result.passed is False
    assert "stale past earnings date" in result.skip_reason


def test_unknown_earnings_fails_closed(monkeypatch):
    _patch_ticker(monkeypatch, None)
    result = events.assess_event_risk("TEST", logging.getLogger("t"))
    assert result.passed is False
    assert "unavailable" in result.skip_reason


def test_ex_dividend_epoch_is_utc(monkeypatch):
    # Yahoo sends midnight-UTC epochs; local-time parsing shifted the date
    # back a day on this Central-time machine.
    epoch = int(pd.Timestamp("2026-07-10", tz="UTC").timestamp())
    _patch_ticker(monkeypatch, _calendar_for(30), info={"exDividendDate": epoch})
    result = events.assess_event_risk("TEST", logging.getLogger("t"))
    assert result.ex_dividend_date is not None
    assert result.ex_dividend_date.date().isoformat() == "2026-07-10"
