from types import SimpleNamespace

import pandas as pd

from scanner.data.options_data import select_options_contract


class DummyTicker:
    def __init__(self, options, chain_df):
        self._options = options
        self._chain_df = chain_df

    @property
    def options(self):
        return self._options

    def option_chain(self, _exp):
        return SimpleNamespace(calls=self._chain_df, puts=self._chain_df)


class DummyLogger:
    def error(self, *args, **kwargs):
        return None


def test_fails_empty_chain(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda _t: DummyTicker([], pd.DataFrame()))
    res = select_options_contract("TEST", "bullish", 100.0, DummyLogger())
    assert res.passed is False


def test_passes_valid_contract(monkeypatch):
    import yfinance as yf

    chain_df = pd.DataFrame(
        [
            {
                "strike": 100,
                "bid": 1.0,
                "ask": 1.1,
                "openInterest": 1000,
                "volume": 50,
                "impliedVolatility": 0.5,
            }
        ]
    )
    monkeypatch.setattr(yf, "Ticker", lambda _t: DummyTicker([(pd.Timestamp.now() + pd.Timedelta(days=35)).strftime("%Y-%m-%d")], chain_df))
    res = select_options_contract("TEST", "bullish", 100.0, DummyLogger())
    assert res.passed is True


def test_fails_spread_too_wide(monkeypatch):
    import yfinance as yf

    chain_df = pd.DataFrame(
        [
            {
                "strike": 100,
                "bid": 1.0,
                "ask": 2.0,
                "openInterest": 1000,
                "volume": 50,
                "impliedVolatility": 0.5,
            }
        ]
    )
    monkeypatch.setattr(yf, "Ticker", lambda _t: DummyTicker([(pd.Timestamp.now() + pd.Timedelta(days=35)).strftime("%Y-%m-%d")], chain_df))
    res = select_options_contract("TEST", "bullish", 100.0, DummyLogger())
    assert res.passed is False


def test_enriches_yfinance_open_interest_with_alpaca_indicative_quote(monkeypatch):
    import yfinance as yf

    expiration = (pd.Timestamp.now() + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    contract_symbol = f"TEST{pd.Timestamp(expiration).strftime('%y%m%d')}C00100000"
    chain_df = pd.DataFrame(
        [
            {
                "contractSymbol": contract_symbol,
                "strike": 100,
                "bid": 0.8,
                "ask": 1.4,
                "openInterest": 1000,
                "volume": 10,
                "impliedVolatility": 0.4,
            }
        ]
    )
    snapshot = {
        contract_symbol: {
            "latestQuote": {
                "bp": 1.0,
                "ap": 1.1,
                "t": pd.Timestamp.now(tz="UTC").isoformat(),
            },
            "dailyBar": {"v": 75},
            "greeks": {"delta": 0.55},
            "impliedVolatility": 0.5,
        }
    }
    monkeypatch.setattr(yf, "Ticker", lambda _t: DummyTicker([expiration], chain_df))
    monkeypatch.setattr("scanner.data.options_data._fetch_alpaca_option_snapshots", lambda _ticker, _logger: snapshot)

    result = select_options_contract("TEST", "bullish", 100.0, DummyLogger())

    assert result.passed is True
    assert result.bid == 1.0
    assert result.ask == 1.1
    assert result.open_interest == 1000
    assert result.volume == 75
    assert result.data_provider == "alpaca+yfinance"
    assert result.data_feed == "indicative"
    assert result.open_interest_source == "yfinance"
    assert result.quote_source == "alpaca"
    assert result.options_data_quality < 0.75
