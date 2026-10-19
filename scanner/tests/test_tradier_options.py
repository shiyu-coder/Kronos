import logging

import pandas as pd
import pytest

from scanner.data import options_data


def _exp_str(days=45):
    return (pd.Timestamp.now() + pd.Timedelta(days=days)).date().isoformat()


def _fresh_ms():
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def _chain_row(strike=10.0, bid=2.0, ask=2.1, oi=5000, volume=300, bid_date=None, option_type="call"):
    stamp = bid_date if bid_date is not None else _fresh_ms()
    return {
        "symbol": f"TEST{strike}",
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "bidsize": 12,
        "asksize": 9,
        "volume": volume,
        "open_interest": oi,
        "bid_date": stamp,
        "ask_date": stamp,
        "greeks": {"mid_iv": 0.62, "delta": 0.51},
    }


def _patch_tradier(monkeypatch, chain_rows, expirations=None):
    calls = []

    def fake_get(path, params, token, logger, retries=2):
        calls.append(path)
        if "expirations" in path:
            return {"expirations": {"date": expirations if expirations is not None else [_exp_str()]}}
        return {"options": {"option": chain_rows}}

    monkeypatch.setattr(options_data, "_tradier_token", lambda: "test-token")
    monkeypatch.setattr(options_data, "_tradier_get", fake_get)
    return calls


def _block_yfinance(monkeypatch, marker="yfinance must not be called"):
    def boom(ticker):
        raise AssertionError(marker)

    monkeypatch.setattr(options_data.yf, "Ticker", boom)


def test_tradier_selects_atm_contract_with_execution_grade_quality(monkeypatch):
    rows = [
        _chain_row(strike=9.0, oi=6000),
        _chain_row(strike=10.0, oi=5000),  # closest to breakout 10.1
        _chain_row(strike=12.0, oi=9000),
        _chain_row(strike=10.0, option_type="put"),
    ]
    _patch_tradier(monkeypatch, rows)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is True
    assert result.contract_type == "call"
    assert result.strike == 10.0
    assert result.data_provider == "tradier"
    assert result.data_feed == "opra-consolidated"
    assert result.open_interest_source == "tradier"
    assert result.bid_size == 12
    assert result.ask_size == 9
    assert result.greeks_available is True
    assert result.implied_volatility == pytest.approx(0.62)
    assert result.quote_age_minutes is not None and result.quote_age_minutes < 5
    assert result.options_data_quality >= 0.75


def test_tradier_gate_failure_is_authoritative_no_fallback(monkeypatch):
    # 50% spread fails every configured bound; the result must be a Tradier
    # fail, not a silent fallback to lower-grade data that might pass.
    rows = [_chain_row(bid=1.0, ask=2.0)]
    _patch_tradier(monkeypatch, rows)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is False
    assert result.data_provider == "tradier"
    assert "no call contract passed" in result.skip_reason


def test_tradier_infra_failure_falls_back_to_legacy_pipeline(monkeypatch):
    monkeypatch.setattr(options_data, "_tradier_token", lambda: "test-token")
    monkeypatch.setattr(options_data, "_tradier_get", lambda *args, **kwargs: None)

    def yf_marker(ticker):
        raise RuntimeError("legacy pipeline reached")

    monkeypatch.setattr(options_data.yf, "Ticker", yf_marker)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is False
    assert "legacy pipeline reached" in result.skip_reason


def test_without_token_tradier_is_never_called(monkeypatch):
    monkeypatch.setattr(options_data, "_tradier_token", lambda: "")

    def fail_get(*args, **kwargs):
        raise AssertionError("tradier called without token")

    monkeypatch.setattr(options_data, "_tradier_get", fail_get)

    def yf_marker(ticker):
        raise RuntimeError("legacy pipeline reached")

    monkeypatch.setattr(options_data.yf, "Ticker", yf_marker)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert "legacy pipeline reached" in result.skip_reason


def test_future_dated_quote_is_not_execution_grade(monkeypatch):
    # Clock skew / corrupt feed epoch must not read as a fresh quote.
    tomorrow = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    rows = [_chain_row(bid_date=tomorrow)]
    _patch_tradier(monkeypatch, rows)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is True
    assert result.quote_age_minutes is None
    assert result.options_data_quality < 0.75


def test_stale_tradier_quotes_are_not_execution_grade(monkeypatch):
    two_hours_ago = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)).timestamp() * 1000)
    rows = [_chain_row(bid_date=two_hours_ago)]
    _patch_tradier(monkeypatch, rows)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is True
    assert result.options_data_quality < 0.75  # honest after-hours degradation


def test_tradier_single_item_payload_collapse(monkeypatch):
    # Tradier returns bare objects (not lists) for single-item results.
    single_row = _chain_row()

    def fake_get(path, params, token, logger, retries=2):
        if "expirations" in path:
            return {"expirations": {"date": _exp_str()}}
        return {"options": {"option": single_row}}

    monkeypatch.setattr(options_data, "_tradier_token", lambda: "test-token")
    monkeypatch.setattr(options_data, "_tradier_get", fake_get)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is True
    assert result.strike == 10.0


def test_dead_nearest_strike_does_not_disqualify_expiration(monkeypatch):
    rows = [
        _chain_row(strike=10.0, bid=0.0, ask=0.0),  # ATM but unquotable
        _chain_row(strike=10.5, bid=1.8, ask=1.85, oi=4000),
    ]
    _patch_tradier(monkeypatch, rows)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bullish", 10.1, logging.getLogger("test"))

    assert result.passed is True
    assert result.strike == 10.5


def test_tradier_bearish_selects_puts(monkeypatch):
    rows = [
        _chain_row(strike=10.0, option_type="call"),
        _chain_row(strike=10.0, option_type="put"),
    ]
    _patch_tradier(monkeypatch, rows)
    _block_yfinance(monkeypatch)

    result = options_data.select_options_contract("TEST", "bearish", 10.1, logging.getLogger("test"))

    assert result.passed is True
    assert result.contract_type == "put"
