"""
FastAPI REST API for the web UI / clients — wraps ``mybot.prediction``.

From the repository root::

    uvicorn mybot.server:app --reload --host 127.0.0.1 --port 8765

The Vite app in ``mybot/web`` proxies ``/api`` to this server.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mybot.prediction import PredictParams, run_prediction

app = FastAPI(title="Kronos Mybot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symbol: str = Field(default="BTC/USDT", description="CCXT symbol, e.g. ETH/USDT")
    timeframe: str = Field(default="1d", description="e.g. 1m, 5m, 1h, 1d")
    limit: int = Field(default=500, ge=50, le=5000)
    since_iso: str | None = Field(default=None, description="Optional ISO start for fetch (with limit)")
    range_start_iso: str | None = None
    range_end_iso: str | None = None
    window_size: int = Field(default=200, ge=20, le=512)
    pred_len: int = Field(default=30, ge=1, le=200)
    rsi_period: int = Field(default=14, ge=2, le=100)
    ema_fast: int = Field(default=50, ge=2, le=500)
    ema_slow: int = Field(default=200, ge=2, le=500)
    model_id: str = Field(default="NeoQuasar/Kronos-small")
    tokenizer_id: str = Field(default="NeoQuasar/Kronos-Tokenizer-base")
    max_context: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    sample_count: int = Field(default=1, ge=1, le=8)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/options")
def options():
    return {
        "symbols": [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "DOGE/USDT",
        ],
        "timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"],
        "models": [
            {"id": "NeoQuasar/Kronos-small", "label": "Kronos small (faster)"},
            {"id": "NeoQuasar/Kronos-base", "label": "Kronos base (slower, larger)"},
        ],
    }


@app.post("/api/predict")
def predict(body: PredictRequest):
    if body.ema_fast >= body.ema_slow:
        raise HTTPException(status_code=400, detail="ema_fast must be smaller than ema_slow")
    if body.range_start_iso and not body.range_end_iso:
        raise HTTPException(status_code=400, detail="range_end_iso required when range_start_iso is set")
    if body.range_end_iso and not body.range_start_iso:
        raise HTTPException(status_code=400, detail="range_start_iso required when range_end_iso is set")

    params = PredictParams(
        symbol=body.symbol,
        timeframe=body.timeframe,
        limit=body.limit,
        since_iso=body.since_iso,
        range_start_iso=body.range_start_iso,
        range_end_iso=body.range_end_iso,
        window_size=body.window_size,
        pred_len=body.pred_len,
        rsi_period=body.rsi_period,
        ema_fast=body.ema_fast,
        ema_slow=body.ema_slow,
        model_id=body.model_id,
        tokenizer_id=body.tokenizer_id,
        max_context=body.max_context,
        temperature=body.temperature,
        top_p=body.top_p,
        sample_count=body.sample_count,
        verbose=False,
    )
    try:
        return run_prediction(params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream or model error: {e!s}") from e
