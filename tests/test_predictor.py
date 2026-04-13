"""Unit tests for KronosPredictor input validation and batch consistency."""

import numpy as np
import pandas as pd
import pytest
import torch

from model.kronos import Kronos, KronosPredictor, KronosTokenizer


@pytest.fixture
def predictor(small_tokenizer_config, small_model_config):
    torch.manual_seed(42)
    tokenizer = KronosTokenizer(**small_tokenizer_config)
    model = Kronos(**small_model_config)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=32)


def _make_ohlcv_df(n_rows=20):
    """Create a simple OHLCV DataFrame."""
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    data = {
        "open": np.random.uniform(100, 200, n_rows),
        "high": np.random.uniform(100, 200, n_rows),
        "low": np.random.uniform(100, 200, n_rows),
        "close": np.random.uniform(100, 200, n_rows),
        "volume": np.random.uniform(1000, 5000, n_rows),
    }
    # Return as Series (not DatetimeIndex) so .dt accessor works in calc_time_stamps
    return pd.DataFrame(data), pd.Series(dates)


class TestPredictorInputValidation:

    def test_non_dataframe_raises(self, predictor):
        with pytest.raises(ValueError, match="pandas DataFrame"):
            predictor.predict(
                df="not a dataframe",
                x_timestamp=pd.date_range("2024-01-01", periods=5, freq="h"),
                y_timestamp=pd.date_range("2024-01-02", periods=3, freq="h"),
                pred_len=3,
            )

    def test_missing_price_columns_raises(self, predictor):
        df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        with pytest.raises(ValueError, match="Price columns"):
            predictor.predict(
                df=df,
                x_timestamp=pd.date_range("2024-01-01", periods=3, freq="h"),
                y_timestamp=pd.date_range("2024-01-02", periods=2, freq="h"),
                pred_len=2,
            )

    def test_nan_values_raise(self, predictor):
        df = pd.DataFrame({
            "open": [1.0, np.nan],
            "high": [2.0, 3.0],
            "low": [1.0, 2.0],
            "close": [1.5, 2.5],
            "volume": [100.0, 200.0],
        })
        with pytest.raises(ValueError, match="NaN"):
            predictor.predict(
                df=df,
                x_timestamp=pd.date_range("2024-01-01", periods=2, freq="h"),
                y_timestamp=pd.date_range("2024-01-02", periods=2, freq="h"),
                pred_len=2,
            )

    def test_missing_volume_filled(self, predictor):
        """DataFrame without volume should still work (filled with zeros)."""
        df = pd.DataFrame({
            "open": np.random.uniform(100, 200, 10),
            "high": np.random.uniform(100, 200, 10),
            "low": np.random.uniform(100, 200, 10),
            "close": np.random.uniform(100, 200, 10),
        })
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="h"))
        y_ts = pd.Series(pd.date_range("2024-01-02", periods=3, freq="h"))
        result = predictor.predict(df=df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=3, verbose=False, sample_count=1, top_k=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestPredictorTimestamp:

    def test_various_timestamp_formats(self, predictor):
        """Various timestamp frequencies should work."""
        df, _ = _make_ohlcv_df(20)
        for freq in ["h", "min", "D"]:
            x_ts = pd.Series(pd.date_range("2024-01-01", periods=20, freq=freq))
            y_ts = pd.Series(pd.date_range("2024-01-22", periods=3, freq=freq))
            result = predictor.predict(df=df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=3, verbose=False, sample_count=1, top_k=1)
            assert isinstance(result, pd.DataFrame)


class TestPredictorBatchConsistency:

    def test_batch_single_matches_predict(self, predictor):
        """predict_batch with a single item should give same result as predict."""
        torch.manual_seed(42)
        np.random.seed(42)

        df, dates = _make_ohlcv_df(20)
        x_ts = dates[:20].reset_index(drop=True)
        y_ts = pd.Series(pd.date_range(dates.iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="h"))

        torch.manual_seed(99)
        single_result = predictor.predict(
            df=df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=3, verbose=False, sample_count=1, top_k=1, top_p=1.0
        )

        torch.manual_seed(99)
        batch_result = predictor.predict_batch(
            df_list=[df], x_timestamp_list=[x_ts], y_timestamp_list=[y_ts],
            pred_len=3, verbose=False, sample_count=1, top_k=1, top_p=1.0
        )

        assert len(batch_result) == 1
        # Values should be identical (same seed, same input)
        np.testing.assert_allclose(
            single_result.values, batch_result[0].values, rtol=1e-4
        )


class TestPredictorNoGradient:

    def test_no_gradients_during_prediction(self, predictor):
        """Predict should run under no_grad context."""
        df, dates = _make_ohlcv_df(20)
        x_ts = dates[:20].reset_index(drop=True)
        y_ts = pd.Series(pd.date_range(dates.iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="h"))

        # Ensure model params don't accumulate grads
        result = predictor.predict(
            df=df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=3, verbose=False, sample_count=1, top_k=1
        )
        for p in predictor.model.parameters():
            assert p.grad is None
        for p in predictor.tokenizer.parameters():
            assert p.grad is None
