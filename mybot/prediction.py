"""
Binance OHLCV (CCXT) + indicators + ``KronosPredictor`` — shared by API and CLI.

Formerly ``predict_service.py``.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import ta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import Kronos, KronosPredictor, KronosTokenizer

_MODEL_LOCK = threading.Lock()
_CACHED: Dict[str, Any] = {}


def _parse_timeframe(tf: str) -> Tuple[pd.Timedelta, str]:
    """CCXT-style timeframe -> (step as Timedelta, pandas freq for date_range)."""
    m = re.match(r"^(\d+)([mhdw])$", tf.strip().lower())
    if not m:
        raise ValueError(f"Unsupported timeframe: {tf!r} (expected e.g. 5m, 1h, 1d)")
    n, u = int(m.group(1)), m.group(2)
    if u == "m":
        return pd.Timedelta(minutes=n), f"{n}min"
    if u == "h":
        return pd.Timedelta(hours=n), f"{n}h"
    if u == "d":
        return pd.Timedelta(days=n), f"{n}D"
    if u == "w":
        return pd.Timedelta(weeks=n), f"{n}W"
    raise ValueError(f"Unsupported timeframe: {tf}")


def fetch_ohlcv_limited(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
    since_ms: Optional[int] = None,
) -> List[list]:
    """Fetch up to `limit` candles, paging when exchange max per request < limit."""
    out: List[list] = []
    remaining = max(1, min(limit, 5000))
    since = since_ms
    max_batch = 1000
    while remaining > 0:
        batch_n = min(max_batch, remaining)
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_n)
        if not batch:
            break
        out.extend(batch)
        remaining = limit - len(out)
        since = batch[-1][0] + 1
        if len(batch) < batch_n:
            break
    return out[:limit]


def fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_candles: int = 2500,
) -> List[list]:
    """Fetch candles from start to end (inclusive), capped by max_candles."""
    start = pd.Timestamp(start).tz_localize(None)
    end = pd.Timestamp(end).tz_localize(None)
    if end <= start:
        raise ValueError("end must be after start")
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out: List[list] = []
    since = since_ms
    while len(out) < max_candles:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        for row in batch:
            if row[0] > end_ms:
                return out
            if row[0] >= since_ms:
                out.append(row)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    return out


@dataclass
class PredictParams:
    symbol: str = "BTC/USDT"
    timeframe: str = "1d"
    limit: int = 300
    since_iso: Optional[str] = None
    range_start_iso: Optional[str] = None
    range_end_iso: Optional[str] = None
    window_size: int = 200
    pred_len: int = 30
    rsi_period: int = 14
    ema_fast: int = 50
    ema_slow: int = 200
    model_id: str = "NeoQuasar/Kronos-small"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    max_context: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1
    verbose: bool = False


def get_predictor(model_id: str, tokenizer_id: str, max_context: int) -> KronosPredictor:
    cache_key = f"{model_id}|{tokenizer_id}|{max_context}"
    with _MODEL_LOCK:
        if cache_key not in _CACHED:
            model = Kronos.from_pretrained(model_id)
            tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
            _CACHED[cache_key] = KronosPredictor(model, tokenizer, max_context=max_context)
        return _CACHED[cache_key]


def run_prediction(params: PredictParams) -> Dict[str, Any]:
    step, pd_freq = _parse_timeframe(params.timeframe)

    if params.window_size > params.max_context:
        raise ValueError(f"window_size ({params.window_size}) must be <= max_context ({params.max_context})")
    if params.pred_len < 1 or params.pred_len > 512:
        raise ValueError("pred_len must be in 1..512")
    need_rows = max(params.ema_slow, params.ema_fast, params.rsi_period) + 5
    if params.window_size < 20:
        raise ValueError("window_size should be at least 20")

    exchange = ccxt.binance({"enableRateLimit": True})

    if params.range_start_iso and params.range_end_iso:
        bars = fetch_ohlcv_range(
            exchange,
            params.symbol,
            params.timeframe,
            pd.Timestamp(params.range_start_iso),
            pd.Timestamp(params.range_end_iso),
        )
    else:
        since_ms = None
        if params.since_iso:
            since_ts = pd.Timestamp(params.since_iso)
            utc_today = pd.Timestamp.utcnow().normalize()
            if since_ts.normalize() >= utc_today:
                raise ValueError(
                    "Parameter 'since' must be strictly before today (UTC). "
                    "Today or future dates often return 0–1 candles (especially on 1d). "
                    "Omit 'since' to fetch the latest N candles instead."
                )
            since_ms = int(since_ts.timestamp() * 1000)
        bars = fetch_ohlcv_limited(
            exchange,
            params.symbol,
            params.timeframe,
            params.limit,
            since_ms=since_ms,
        )

    min_bars = need_rows + params.window_size
    if len(bars) < min_bars:
        hint = ""
        if len(bars) == 0:
            hint = (
                " No candles returned: check symbol, network/VPN, or use a past 'since' date."
            )
        elif params.since_iso and len(bars) < min_bars:
            hint = " Try increasing 'limit' or setting an earlier 'since' for more history."
        raise ValueError(
            f"Not enough candles ({len(bars)}). Need at least {min_bars} "
            f"(slow EMA {params.ema_slow} + window {params.window_size} + margin). "
            f"Increase limit (e.g. ≥ {min_bars + 50}) or widen the date range.{hint}"
        )

    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=params.rsi_period).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=params.ema_fast).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=params.ema_slow).ema_indicator()
    df = df.dropna()

    if len(df) < params.window_size + 10:
        raise ValueError("After indicator warmup, not enough rows. Lower EMA periods or fetch more data.")

    df_input = df.tail(params.window_size).copy()
    x_timestamp = df_input["timestamps"]
    last_ts = df_input["timestamps"].iloc[-1]
    y_timestamp = pd.date_range(start=last_ts + step, periods=params.pred_len, freq=pd_freq)

    predictor = get_predictor(params.model_id, params.tokenizer_id, params.max_context)
    result = predictor.predict(
        df_input,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=params.pred_len,
        T=params.temperature,
        top_p=params.top_p,
        sample_count=params.sample_count,
        verbose=params.verbose,
    )

    pred_price = float(result["close"].iloc[-1])
    start_price = float(result["close"].iloc[0])
    last_price = float(df["close"].iloc[-1])
    trend = pred_price - start_price

    rsi = float(df["rsi"].iloc[-1])
    ema_f = float(df["ema_fast"].iloc[-1])
    ema_s = float(df["ema_slow"].iloc[-1])

    signal = "HOLD"
    if trend > 0 and last_price > ema_f > ema_s and rsi < 70:
        signal = "BUY"
    elif trend < 0 and last_price < ema_f < ema_s:
        signal = "SELL"

    def _ohlc_row(ts, row) -> Dict[str, Any]:
        return {
            "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    history = []
    for _, r in df.iterrows():
        history.append(_ohlc_row(r["timestamps"], r))

    forecast = []
    for ts, r in result.iterrows():
        forecast.append(_ohlc_row(ts, r))

    return {
        "symbol": params.symbol,
        "timeframe": params.timeframe,
        "signal": signal,
        "last_price": last_price,
        "pred_start_price": start_price,
        "pred_end_price": pred_price,
        "trend": trend,
        "rsi": rsi,
        "ema_fast": ema_f,
        "ema_slow": ema_s,
        "history": history,
        "forecast": forecast,
        "meta": {
            "window_size": params.window_size,
            "pred_len": params.pred_len,
            "rsi_period": params.rsi_period,
            "ema_fast": params.ema_fast,
            "ema_slow": params.ema_slow,
            "model_id": params.model_id,
            "rows_used": len(df),
        },
    }
