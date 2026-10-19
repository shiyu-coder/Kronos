import pandas as pd

from scanner.models.kronos_adapter import KronosAdapter


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _bars(n=80):
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [100 + i * 0.1 for i in range(n)],
            "High": [101 + i * 0.1 for i in range(n)],
            "Low": [99 + i * 0.1 for i in range(n)],
            "Close": [100 + i * 0.1 for i in range(n)],
            "Volume": [1000] * n,
        },
        index=idx,
    )


def test_unknown_output_format_fails_safely(monkeypatch):
    adapter = KronosAdapter(DummyLogger())

    class DummyPredictor:
        def predict(self, **kwargs):
            return {"unexpected": True}

    monkeypatch.setattr(adapter, "_load_once", lambda: DummyPredictor())
    result = adapter.evaluate("TEST", _bars(), "bullish")
    assert result.passed is False
    assert result.output_mode == "unknown"


def test_single_path_alignment_not_probability(monkeypatch):
    adapter = KronosAdapter(DummyLogger())

    class DummyPredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            idx = kwargs["y_timestamp"]
            return pd.DataFrame(
                {
                    "open": [110] * len(idx),
                    "high": [111] * len(idx),
                    "low": [109] * len(idx),
                    "close": [112] * len(idx),
                    "volume": [1000] * len(idx),
                    "amount": [112000] * len(idx),
                },
                index=idx,
            )

    monkeypatch.setattr(adapter, "_load_once", lambda: DummyPredictor())
    result = adapter.evaluate("TEST", _bars(), "bullish")
    assert result.output_mode in ("multi_path_agreement", "forecast_alignment")
