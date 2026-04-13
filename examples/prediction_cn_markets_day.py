# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Description:
    Predicts future daily K-line (1D) data for A-share markets using Kronos model and akshare.
    The script automatically downloads the latest historical data, cleans it, and runs model inference.

Usage:
    python prediction_cn_markets_day.py --symbol 000001

Arguments:
    --symbol     Stock code (e.g. 002594 for BYD, 000001 for SSE Index)

Output:
    - Saves the prediction results to ./outputs/pred_<symbol>_data.csv and ./outputs/pred_<symbol>_chart.png
    - Logs and progress are printed to console

Example:
    bash> python prediction_cn_markets_day.py --symbol 000001
    python3 prediction_cn_markets_day.py --symbol 002594
"""

import os
import argparse
import time
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
import sys
from model import Kronos, KronosTokenizer, KronosPredictor

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# Setting
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 400
PRED_LEN = 120
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1

def load_data(symbol: str) -> pd.DataFrame:
    print(f"📥 Fetching {symbol} daily data from akshare ...")

    max_retries = 3
    df = None

    # Retry mechanism
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(1.5)

    # If still empty after retries
    if df is None or df.empty:
        print(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts. Exiting.")
        sys.exit(1)
    
    df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    }, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"--": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix invalid open values
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        print(f"⚠️  Fixed {open_bad.sum()} invalid open values.")
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # Fix missing amount
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    print(f"✅ Data loaded: {len(df)} rows, range: {df['date'].min()} ~ {df['date'].max()}")

    print("Data Head:")
    print(df.head())

    return df


def prepare_inputs(df):
    x_df = df.iloc[-LOOKBACK:][["open","high","low","close","volume","amount"]]
    x_timestamp = df.iloc[-LOOKBACK:]["date"]
    y_timestamp = pd.bdate_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN)
    return x_df, pd.Series(x_timestamp), pd.Series(y_timestamp)

def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    print(f"🔒 Applying ±{limit_rate*100:.0f}% price limit ...")

    # Ensure integer index
    pred_df = pred_df.reset_index(drop=True)

    # Ensure float64 dtype for safe assignment
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)

        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                clipped = max(min(value, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)

        last_close = float(pred_df.at[i, "close"])  # ensure float type

    return pred_df


def plot_result(df_hist, df_pred, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df_hist["date"], df_hist["close"], label="Historical", color="blue")
    plt.plot(df_pred["date"], df_pred["close"], label="Predicted", color="red", linestyle="--")
    plt.title(f"Kronos Prediction for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_chart.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Chart saved: {plot_path}")


def predict_future(symbol):
    print(f"🚀 Loading Kronos tokenizer:{TOKENIZER_PRETRAINED} model:{MODEL_PRETRAINED} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
    model = Kronos.from_pretrained(MODEL_PRETRAINED)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    df = load_data(symbol)
    x_df, x_timestamp, y_timestamp = prepare_inputs(df)

    print("🔮 Generating predictions ...")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )

    pred_df["date"] = y_timestamp.values

    # Apply ±10% price limit
    last_close = df["close"].iloc[-1]
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # Merge historical and predicted data
    df_out = pd.concat([
        df[["date", "open", "high", "low", "close", "volume", "amount"]],
        pred_df[["date", "open", "high", "low", "close", "volume", "amount"]]
    ]).reset_index(drop=True)

    # Save CSV
    out_file = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction completed and saved: {out_file}")

    # Plot
    plot_result(df, pred_df, symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos stock prediction script")
    parser.add_argument("--symbol", type=str, default="000001", help="Stock code")
    args = parser.parse_args()

    predict_future(
        symbol=args.symbol,
    )
