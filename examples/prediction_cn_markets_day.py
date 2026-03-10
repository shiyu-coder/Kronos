# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Two modes are supported:
1. forward: predict the next unknown window from the latest available history.
2. eval: replay a historical cutoff and compare the forecast against known future data.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import OUTPUT_ROOT, detect_device, get_model_spec, save_json
from model import Kronos, KronosPredictor, KronosTokenizer

try:
    import akshare as ak
except ImportError as exc:
    raise ImportError(
        "akshare is required for prediction_cn_markets_day.py. Install it with "
        "`uv pip install --python .venv/bin/python akshare`."
    ) from exc

MAX_CONTEXT = 512
LOOKBACK = 400
PRED_LEN = 120
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1
PLOT_COLUMNS = ["close"]
METRIC_COLUMNS = ["open", "high", "low", "close"]


def load_data(symbol: str) -> pd.DataFrame:
    print(f"Fetching {symbol} daily data from akshare...")

    max_retries = 3
    df = None
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
            if df is not None and not df.empty:
                break
        except Exception as exc:
            print(f"Attempt {attempt}/{max_retries} failed: {exc}")
        time.sleep(1.5)

    if df is None or df.empty:
        raise RuntimeError(f"Failed to fetch data for {symbol} after {max_retries} attempts.")

    df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False).replace({"--": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        print(f"Fixed {open_bad.sum()} invalid open values.")
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"] = df["open"].fillna(df["close"])

    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    print(f"Loaded {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
    print(df.head())
    return df


def resolve_eval_end_index(df: pd.DataFrame, as_of_date: str | None) -> int:
    min_end_idx = LOOKBACK - 1
    max_end_idx = len(df) - PRED_LEN - 1
    if max_end_idx < min_end_idx:
        raise ValueError("Not enough rows to run historical evaluation with the requested lookback and pred_len.")

    if as_of_date is None:
        return max_end_idx

    target = pd.Timestamp(as_of_date)
    eligible = df.index[df["date"] <= target]
    if len(eligible) == 0:
        raise ValueError(f"No data is available on or before {target.date()}.")

    end_idx = int(eligible[-1])
    if end_idx < min_end_idx:
        raise ValueError(f"Evaluation cutoff {df.loc[end_idx, 'date'].date()} is before the required lookback.")
    if end_idx > max_end_idx:
        raise ValueError(
            f"Evaluation cutoff {df.loc[end_idx, 'date'].date()} leaves no future truth window of length {PRED_LEN}."
        )
    return end_idx


def prepare_forward_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = pd.Series(df.iloc[-LOOKBACK:]["date"])
    y_timestamp = pd.Series(
        pd.bdate_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN)
    )
    return x_df, x_timestamp, y_timestamp


def prepare_eval_inputs(df: pd.DataFrame, as_of_date: str | None) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Timestamp]:
    end_idx = resolve_eval_end_index(df, as_of_date)
    x_df = df.iloc[end_idx - LOOKBACK + 1 : end_idx + 1][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = pd.Series(df.iloc[end_idx - LOOKBACK + 1 : end_idx + 1]["date"])
    actual_df = df.iloc[end_idx + 1 : end_idx + 1 + PRED_LEN][["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    y_timestamp = pd.Series(actual_df["date"])
    return x_df, x_timestamp, y_timestamp, actual_df, df.loc[end_idx, "date"]


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float = 0.1) -> pd.DataFrame:
    clipped_df = pred_df.reset_index(drop=True).copy()
    cols = ["open", "high", "low", "close"]
    clipped_df[cols] = clipped_df[cols].astype("float64")

    for idx in range(len(clipped_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)
        for col in cols:
            value = clipped_df.at[idx, col]
            if pd.notna(value):
                clipped_df.at[idx, col] = float(max(min(value, limit_up), limit_down))
        last_close = float(clipped_df.at[idx, "close"])

    return clipped_df


def compute_eval_metrics(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for col in METRIC_COLUMNS:
        diff = pred_df[col] - actual_df[col]
        metrics[f"{col}_mae"] = float(diff.abs().mean())
        metrics[f"{col}_mse"] = float((diff ** 2).mean())
    return metrics


def plot_result(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    symbol: str,
    plot_path: Path,
    actual_df: pd.DataFrame | None = None,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df_hist["date"], df_hist["close"], label="Historical", color="blue")
    plt.plot(df_pred["date"], df_pred["close"], label="Predicted", color="red", linestyle="--")
    if actual_df is not None:
        plt.plot(actual_df["date"], actual_df["close"], label="Actual Future", color="green", linestyle=":")
    plt.title(f"Kronos Prediction for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def run_prediction(symbol: str, model_key: str, mode: str, as_of_date: str | None) -> None:
    spec = get_model_spec(model_key)
    device = detect_device()
    prefix = f"pred_{symbol.replace('.', '_')}_{model_key}_{mode}"
    csv_path = OUTPUT_ROOT / f"{prefix}_data.csv"
    plot_path = OUTPUT_ROOT / f"{prefix}_chart.png"
    summary_path = OUTPUT_ROOT / f"{prefix}_summary.json"

    print(f"Loading {spec['name']} on device: {device}")
    start_time = time.perf_counter()
    tokenizer = KronosTokenizer.from_pretrained(spec["tokenizer_id"])
    model = Kronos.from_pretrained(spec["model_id"])
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)

    df = load_data(symbol)
    actual_df = None
    eval_cutoff = None

    if mode == "forward":
        x_df, x_timestamp, y_timestamp = prepare_forward_inputs(df)
        history_df = df.copy()
    else:
        x_df, x_timestamp, y_timestamp, actual_df, eval_cutoff = prepare_eval_inputs(df, as_of_date)
        history_df = df[df["date"] <= eval_cutoff].copy()

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
    )
    elapsed_seconds = time.perf_counter() - start_time

    pred_df["date"] = y_timestamp.values
    pred_df = apply_price_limits(pred_df, float(x_df["close"].iloc[-1]), limit_rate=0.1)

    if actual_df is None:
        output_df = pd.concat(
            [
                history_df[["date", "open", "high", "low", "close", "volume", "amount"]],
                pred_df[["date", "open", "high", "low", "close", "volume", "amount"]],
            ]
        ).reset_index(drop=True)
        metrics = None
    else:
        comparison_df = pred_df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            comparison_df[f"actual_{col}"] = actual_df[col]
        output_df = comparison_df
        metrics = compute_eval_metrics(pred_df, actual_df)

    output_df.to_csv(csv_path, index=False)
    plot_result(history_df, pred_df, symbol, plot_path, actual_df=actual_df)

    save_json(
        summary_path,
        {
            "experiment": "prediction_cn_markets_day",
            "description": (
                "Forward mode predicts unseen future dates. Eval mode replays a historical cutoff and compares "
                "the forecast against known future truth."
            ),
            "symbol": symbol,
            "mode": mode,
            "has_ground_truth": actual_df is not None,
            "evaluation_cutoff": eval_cutoff,
            "model_key": model_key,
            "model_name": spec["name"],
            "model_id": spec["model_id"],
            "tokenizer_id": spec["tokenizer_id"],
            "device": device,
            "elapsed_seconds": elapsed_seconds,
            "lookback": LOOKBACK,
            "pred_len": PRED_LEN,
            "history_start": x_timestamp.iloc[0],
            "history_end": x_timestamp.iloc[-1],
            "prediction_start": y_timestamp.iloc[0],
            "prediction_end": y_timestamp.iloc[-1],
            "output_csv": csv_path,
            "plot_file": plot_path,
            "metrics": metrics,
            "prediction_head": pred_df.head(),
            "actual_head": actual_df.head() if actual_df is not None else None,
        },
    )

    print(f"Saved forecast CSV: {csv_path}")
    print(f"Saved forecast plot: {plot_path}")
    print(f"Saved summary JSON: {summary_path}")
    if metrics is not None:
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos stock prediction script")
    parser.add_argument("--symbol", type=str, default="000001", help="Stock code")
    parser.add_argument("--model", choices=["small", "base"], default="small", help="Model size to run.")
    parser.add_argument(
        "--mode",
        choices=["forward", "eval"],
        default="forward",
        help="forward predicts unknown future dates, eval compares against known future truth.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Historical cutoff date for eval mode, e.g. 2020-06-30. Defaults to the latest eligible date.",
    )
    args = parser.parse_args()

    run_prediction(symbol=args.symbol, model_key=args.model, mode=args.mode, as_of_date=args.as_of_date)
