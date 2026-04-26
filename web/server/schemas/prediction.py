from typing import Literal

from pydantic import BaseModel, Field

DataSource = Literal["binance", "yfinance", "local"]
ModelName = Literal["kronos-mini", "kronos-small", "kronos-base"]
Device = Literal["cpu", "cuda:0", "mps"]


class PricePredictionRequest(BaseModel):
    """User-selectable inputs for price prediction."""

    data_source: DataSource = Field(default="binance")
    symbol: str | None = Field(
        default="ETHUSDT", examples=["BTCUSDT", "BTC-USDT", "AAPL"]
    )
    local_path: str | None = None
    period: str = "30d"
    interval: str = "1h"
    lookback: int = Field(default=400, ge=2)
    pred_len: int = Field(default=120, ge=1)
    limit: int | None = Field(default=400, ge=1)
    model_name: ModelName = "kronos-mini"
    device: Device = "cpu"
    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    sample_count: int = Field(default=1, ge=1)
    verbose: bool = False


class AvailableModel(BaseModel):
    """Metadata for one selectable Kronos model."""

    name: str
    model_id: str
    tokenizer_id: str
    context_length: int
    params: str
    description: str


class CandleRecord(BaseModel):
    """One OHLCV row returned by the prediction API."""

    timestamps: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


class PredictionOptionsResponse(BaseModel):
    """Selectable values and defaults for prediction requests."""

    data_sources: list[DataSource]
    models: dict[ModelName, AvailableModel]
    devices: list[Device]
    common_intervals: list[str]
    common_periods: list[str]
    defaults: PricePredictionRequest


class PricePredictionResponse(BaseModel):
    """Prediction output with source history and future candles."""

    request: PricePredictionRequest
    model: AvailableModel
    lookback_start_timestamp: str
    lookback_end_timestamp: str
    prediction_start_timestamp: str
    history: list[CandleRecord]
    future_timestamps: list[str]
    prediction: list[CandleRecord]
