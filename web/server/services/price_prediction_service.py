import pandas as pd
from errors.errors import (
    ModelLoadError,
    PredictionAPIError,
    PredictionRuntimeError,
    UnsupportedPredictionOptionError,
)

from constants.available_models import AVAILABLE_MODELS
from schemas.prediction import PricePredictionRequest
from services.ohlcv_data import OHLCVDataService
from services.prediction import KronosPredictionService


class PricePredictionService:
    """API-facing service that fetches candles and runs Kronos predictions."""

    def __init__(
        self,
        data_service: OHLCVDataService | None = None,
        prediction_service: KronosPredictionService | None = None,
    ):
        """Create the service with default or injected data/prediction services."""
        self.data_service = data_service or OHLCVDataService()
        self.prediction_service = prediction_service or KronosPredictionService()

    def predict_price(self, request: PricePredictionRequest):
        """Fetch selected data, run the selected model, and return JSON-safe output."""
        prepared_data = self._load_data(request)

        self.prediction_service.configure(
            model_name=request.model_name,
            device=request.device,
        )

        try:
            prediction_df = self.prediction_service.predict(
                df=prepared_data["df"],
                x_timestamp=prepared_data["x_timestamp"],
                y_timestamp=prepared_data["y_timestamp"],
                pred_len=prepared_data["pred_len"],
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                sample_count=request.sample_count,
                verbose=request.verbose,
            )
        except PredictionAPIError:
            raise
        except (OSError, FileNotFoundError) as exc:
            raise ModelLoadError(
                "Could not load the selected Kronos model. Check model files or network access.",
                details={"model_name": request.model_name, "device": request.device},
            ) from exc
        except RuntimeError as exc:
            raise PredictionRuntimeError(
                "Prediction failed while running the model. Check the selected device and inputs.",
                details={
                    "model_name": request.model_name,
                    "device": request.device,
                    "reason": str(exc),
                },
            ) from exc
        except Exception as exc:
            raise PredictionRuntimeError(
                "Prediction failed unexpectedly while generating results.",
                details={"model_name": request.model_name, "device": request.device},
            ) from exc

        # Move prediction timestamps from the DataFrame index into the API response.
        prediction_df = prediction_df.reset_index(names="timestamps")

        return {
            "request": request.model_dump(),
            "model": AVAILABLE_MODELS[request.model_name],
            "lookback_start_timestamp": self._timestamp_to_json(
                prepared_data["x_timestamp"].iloc[0]
            ),
            "lookback_end_timestamp": self._timestamp_to_json(
                prepared_data["x_timestamp"].iloc[-1]
            ),
            "prediction_start_timestamp": self._timestamp_to_json(
                prepared_data["y_timestamp"].iloc[0]
            ),
            "history": self._records_to_json(prepared_data["df"]),
            "future_timestamps": self._timestamps_to_json(prepared_data["y_timestamp"]),
            "prediction": self._records_to_json(prediction_df),
        }

    def get_options(self):
        """Return values a client can show as selectable prediction options."""
        return {
            "data_sources": ["binance", "yfinance", "local"],
            "models": AVAILABLE_MODELS,
            "devices": ["cpu", "cuda:0", "mps"],
            "common_intervals": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            "common_periods": ["5d", "1mo", "3mo", "6mo", "1y", "2y"],
            "defaults": PricePredictionRequest().model_dump(),
        }

    def _load_data(self, request: PricePredictionRequest):
        """Load OHLCV data from the selected source and prepare it for prediction."""
        if request.model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise UnsupportedPredictionOptionError(
                f"Unknown model '{request.model_name}'. Available models: {available}",
                details={
                    "field": "model_name",
                    "received": request.model_name,
                    "available": list(AVAILABLE_MODELS.keys()),
                },
            )

        if request.data_source == "binance":
            if not request.symbol:
                raise UnsupportedPredictionOptionError(
                    "symbol is required for Binance data.",
                    details={"field": "symbol", "data_source": "binance"},
                )
            return self.data_service.get_data_from_binance(
                symbol=request.symbol,
                interval=request.interval,
                lookback=request.lookback,
                pred_len=request.pred_len,
                limit=request.limit,
            )

        if request.data_source == "yfinance":
            if not request.symbol:
                raise UnsupportedPredictionOptionError(
                    "symbol is required for Yahoo Finance data.",
                    details={"field": "symbol", "data_source": "yfinance"},
                )
            return self.data_service.get_data_from_yfinance(
                symbol=request.symbol,
                period=request.period,
                interval=request.interval,
                lookback=request.lookback,
                pred_len=request.pred_len,
            )

        if not request.local_path:
            raise UnsupportedPredictionOptionError(
                "local_path is required for local data.",
                details={"field": "local_path", "data_source": "local"},
            )
        return self.data_service.get_data_from_local_storage(
            path=request.local_path,
            lookback=request.lookback,
            pred_len=request.pred_len,
        )

    def _records_to_json(self, df: pd.DataFrame):
        """Convert a DataFrame into JSON-safe row dictionaries."""
        result = df.copy()
        for column in result.columns:
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                result[column] = result[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return result.to_dict(orient="records")

    def _timestamps_to_json(self, timestamps):
        """Convert timestamp values into ISO strings for JSON responses."""
        return [self._timestamp_to_json(timestamp) for timestamp in timestamps]

    def _timestamp_to_json(self, timestamp):
        """Convert one timestamp value into an ISO string for JSON responses."""
        if hasattr(timestamp, "isoformat"):
            ts = timestamp
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            return ts.isoformat()
        return str(timestamp)
