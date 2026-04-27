from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yfinance as yf
from errors.errors import (
    DataFormatError,
    DataNotFoundError,
    DataProviderError,
    DataProviderTimeoutError,
    InsufficientDataError,
)


class OHLCVDataService:
    """Load candle data from Binance, Yahoo Finance, or local files."""

    def get_data_from_binance(
        self,
        symbol: str,
        interval: str = "1h",
        lookback: int = 400,
        pred_len: int = 120,
        limit: int | None = None,
    ):
        """Fetch OHLCV candles from Binance and return prediction-ready data."""
        fetch_limit = limit or lookback
        params = urlencode(
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": fetch_limit,
            }
        )
        url = f"https://api.binance.com/api/v3/klines?{params}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            rows = response.json()
        except requests.Timeout as exc:
            raise DataProviderTimeoutError(
                "Binance took too long to respond. Try again or use a smaller limit.",
                details={
                    "data_source": "binance",
                    "symbol": symbol,
                    "interval": interval,
                },
            ) from exc
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            provider_message = self._extract_provider_message(exc.response)
            raise DataProviderError(
                "Binance rejected the data request. Check the symbol and interval.",
                details={
                    "data_source": "binance",
                    "symbol": symbol,
                    "interval": interval,
                    "provider_status_code": status_code,
                    "provider_message": provider_message,
                },
            ) from exc
        except requests.RequestException as exc:
            raise DataProviderError(
                "Could not connect to Binance. Please try again later.",
                details={
                    "data_source": "binance",
                    "symbol": symbol,
                    "interval": interval,
                },
            ) from exc
        except ValueError as exc:
            raise DataProviderError(
                "Binance returned an invalid response.",
                details={
                    "data_source": "binance",
                    "symbol": symbol,
                    "interval": interval,
                },
            ) from exc

        if not rows:
            raise DataNotFoundError(
                f"No Binance data found for symbol '{symbol}'.",
                details={
                    "data_source": "binance",
                    "symbol": symbol,
                    "interval": interval,
                },
            )

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
        df["amount"] = pd.to_numeric(df["quote_asset_volume"], errors="coerce")

        return self._prepare_prediction_data(df, lookback, pred_len)

    def get_data_from_yfinance(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "1h",
        lookback: int = 400,
        pred_len: int = 120,
    ):
        """Fetch OHLCV candles from Yahoo Finance and return prediction-ready data."""

        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        except requests.Timeout as exc:
            raise DataProviderTimeoutError(
                "Yahoo Finance took too long to respond. Try again later.",
                details={
                    "data_source": "yfinance",
                    "symbol": symbol,
                    "interval": interval,
                },
            ) from exc
        except Exception as exc:
            raise DataProviderError(
                "Could not fetch data from Yahoo Finance. Check the symbol, period, and interval.",
                details={
                    "data_source": "yfinance",
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                },
            ) from exc

        if df.empty:
            raise DataNotFoundError(
                f"No Yahoo Finance data found for symbol '{symbol}'.",
                details={
                    "data_source": "yfinance",
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                },
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        timestamp_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(
            columns={
                timestamp_col: "timestamps",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return self._prepare_prediction_data(df, lookback, pred_len)

    def get_data_from_local_storage(
        self,
        path: str,
        lookback: int = 400,
        pred_len: int = 120,
    ):
        """Load candle data from a CSV or Feather file and return prediction-ready data."""
        file_path = Path(path)
        if not file_path.exists():
            raise DataNotFoundError(
                f"Data file not found: {file_path}",
                details={"data_source": "local", "path": str(file_path)},
            )

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == ".feather":
                df = pd.read_feather(file_path)
            else:
                raise DataFormatError(
                    "Only CSV and Feather files are supported.",
                    details={"data_source": "local", "path": str(file_path)},
                )
        except DataFormatError:
            raise
        except Exception as exc:
            raise DataFormatError(
                "Could not read the local data file. Make sure it is a valid CSV or Feather file.",
                details={"data_source": "local", "path": str(file_path)},
            ) from exc

        return self._prepare_prediction_data(df, lookback, pred_len)

    def _prepare_prediction_data(self, df, lookback: int, pred_len: int):
        """Normalize candle columns and build the inputs expected by PredictCandles."""
        df = self._normalize_columns(df)

        if len(df) < lookback:
            raise InsufficientDataError(
                f"Need at least {lookback} valid rows, got {len(df)}.",
                details={"required_rows": lookback, "available_rows": len(df)},
            )

        x_df = df.iloc[-lookback:][
            ["timestamps", "open", "high", "low", "close", "volume", "amount"]
        ]
        x_timestamp = df.iloc[-lookback:]["timestamps"]
        y_timestamp = self._make_future_timestamps(df["timestamps"], pred_len)

        return {
            "df": x_df,
            "x_timestamp": x_timestamp,
            "y_timestamp": y_timestamp,
            "pred_len": pred_len,
            "source_df": df,
        }

    def _normalize_columns(self, df):
        """Convert common candle column names into Kronos predictor column names."""
        df = df.copy()
        rename_map = {
            "timestamp": "timestamps",
            "date": "timestamps",
            "datetime": "timestamps",
            "time": "timestamps",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
        }
        df = df.rename(columns=rename_map)

        required_cols = ["timestamps", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataFormatError(
                "Input data is missing required OHLC timestamp columns.",
                details={
                    "missing_columns": missing_cols,
                    "required_columns": required_cols,
                },
            )

        try:
            df["timestamps"] = pd.to_datetime(df["timestamps"])
        except Exception as exc:
            raise DataFormatError(
                "Could not parse the timestamps column.",
                details={"column": "timestamps"},
            ) from exc
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(
                axis=1
            )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        df = df.dropna(subset=required_cols)
        if df.empty:
            raise DataFormatError(
                "No valid candle rows remain after parsing timestamps and OHLC values.",
                details={"required_columns": required_cols},
            )
        return df.sort_values("timestamps").reset_index(drop=True)

    def _make_future_timestamps(self, timestamps, pred_len: int):
        """Create future timestamps using the last observed time interval."""
        if len(timestamps) < 2:
            freq = pd.Timedelta(hours=1)
        else:
            freq = timestamps.iloc[-1] - timestamps.iloc[-2]

        return pd.Series(
            pd.date_range(
                start=timestamps.iloc[-1] + freq,
                periods=pred_len,
                freq=freq,
            ),
            name="timestamps",
        )

    def _extract_provider_message(self, response):
        if response is None:
            return None
        try:
            body = response.json()
        except ValueError:
            return response.text[:300] if response.text else None
        if isinstance(body, dict):
            return body.get("msg") or body.get("message") or body
        return body
