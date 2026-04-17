"""
Kronos K-line Prediction API Service
Provides HTTP endpoints for OpenClaw Gateway to call Kronos model predictions.
"""

import os
import sys
import json
import traceback
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Kronos K-line Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_models = {}


def get_predictor(model_size: str = "base"):
    """Load and cache a KronosPredictor for the given model size."""
    if model_size in _models:
        return _models[model_size]

    tok_name = "Kronos-Tokenizer-2k" if model_size == "mini" else "Kronos-Tokenizer-base"
    model_name = f"Kronos-{model_size}"

    tok_path = os.path.join(MODEL_DIR, tok_name)
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.isdir(tok_path) or not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model files not found: {model_path} / {tok_path}")

    tokenizer = KronosTokenizer.from_pretrained(tok_path)
    model = Kronos.from_pretrained(model_path)

    ctx = 2048 if model_size == "mini" else 512
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=ctx)
    _models[model_size] = predictor
    return predictor


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Prediction request body."""
    stock_code: Optional[str] = Field(None, description="Stock code, e.g. '600036'. If provided, data is fetched via akshare.")
    kline_data: Optional[list[dict]] = Field(None, description="Raw OHLCV rows: [{open,high,low,close,volume,amount,timestamps}, ...]")
    pred_days: int = Field(30, ge=1, le=120, description="Number of trading days to predict")
    lookback: int = Field(0, ge=0, le=400, description="Historical bars to feed the model. 0 = auto-calculate")
    model_size: str = Field("base", description="Model variant: mini / small / base")
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)


class PredictResponse(BaseModel):
    stock_code: Optional[str]
    model_used: str
    device: str
    lookback: int
    pred_days: int
    current_price: float
    predicted_end_price: float
    change_pct: float
    predictions: list[dict]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_stock_data(stock_code: str) -> pd.DataFrame:
    """Fetch stock data via akshare."""
    try:
        import akshare as ak
    except ImportError:
        raise HTTPException(400, "akshare not installed. pip install akshare")

    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
    if df is None or df.empty:
        raise HTTPException(404, f"No data found for {stock_code}")

    col_map = {
        '日期': 'timestamps', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    return df


def generate_future_dates(last_date, n: int):
    dates, cur = [], last_date + timedelta(days=1)
    while len(dates) < n:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "models_loaded": list(_models.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Resolve data
    if req.kline_data:
        df = pd.DataFrame(req.kline_data)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)
    elif req.stock_code:
        df = fetch_stock_data(req.stock_code)
    else:
        raise HTTPException(400, "Provide either stock_code or kline_data")

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']

    # 2. Compute parameters
    pred_len = min(req.pred_days, 120)
    lookback = req.lookback if req.lookback > 0 else min(max(100, pred_len * 3), 400, len(df) - pred_len)
    lookback = max(50, min(lookback, len(df) - 1))

    # 3. Load model
    try:
        predictor = get_predictor(req.model_size)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))

    # 4. Prepare input
    tail = df.tail(lookback).reset_index(drop=True)
    x_df = tail[['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_ts = tail['timestamps']

    last_date = df['timestamps'].iloc[-1]
    future_dates = generate_future_dates(last_date, pred_len)

    # 5. Predict
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=pd.Series(future_dates),
            pred_len=pred_len,
            T=req.temperature,
            top_p=req.top_p,
            sample_count=1,
            verbose=False,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {e}")

    # 6. Build response
    current_price = float(df['close'].iloc[-1])
    end_price = float(pred_df['close'].iloc[-1])
    change_pct = (end_price / current_price - 1) * 100

    rows = []
    for i, d in enumerate(future_dates[:len(pred_df)]):
        row = {"date": d.strftime("%Y-%m-%d")}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in pred_df.columns:
                row[col] = round(float(pred_df[col].iloc[i]), 4)
        rows.append(row)

    direction = "上涨" if change_pct > 1 else ("下跌" if change_pct < -1 else "震荡")
    summary = (
        f"基于Kronos-{req.model_size}模型预测，"
        f"当前价{current_price:.2f}元，"
        f"未来{pred_len}个交易日预计{direction}至{end_price:.2f}元（{change_pct:+.2f}%）。"
    )

    return PredictResponse(
        stock_code=req.stock_code,
        model_used=f"Kronos-{req.model_size}",
        device=DEVICE,
        lookback=lookback,
        pred_days=pred_len,
        current_price=current_price,
        predicted_end_price=end_price,
        change_pct=round(change_pct, 2),
        predictions=rows,
        summary=summary,
    )


@app.get("/models")
def list_models():
    """List available model variants."""
    variants = []
    for size in ["mini", "small", "base"]:
        tok_name = "Kronos-Tokenizer-2k" if size == "mini" else "Kronos-Tokenizer-base"
        model_path = os.path.join(MODEL_DIR, f"Kronos-{size}")
        tok_path = os.path.join(MODEL_DIR, tok_name)
        variants.append({
            "name": f"Kronos-{size}",
            "available": os.path.isdir(model_path) and os.path.isdir(tok_path),
            "loaded": size in _models,
            "context_length": 2048 if size == "mini" else 512,
        })
    return {"models": variants, "device": DEVICE}


# ---------------------------------------------------------------------------
# Startup - preload base model
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    print(f"[Kronos API] Device: {DEVICE}")
    print(f"[Kronos API] Model dir: {MODEL_DIR}")
    try:
        get_predictor("base")
        print("[Kronos API] Kronos-base model loaded successfully")
    except Exception as e:
        print(f"[Kronos API] Warning: could not preload base model: {e}")


if __name__ == "__main__":
    uvicorn.run("kronos_api:app", host="0.0.0.0", port=8100, reload=False)
