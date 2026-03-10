import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import data_path, detect_device, get_model_spec, output_path, save_json
from model import Kronos, KronosPredictor, KronosTokenizer

LOOKBACK = 400
PRED_LEN = 120


def plot_prediction(kline_df: pd.DataFrame, pred_df: pd.DataFrame, plot_file: Path) -> None:
    plot_df = pred_df.copy()
    plot_df.index = kline_df.index[-plot_df.shape[0]:]

    close_df = pd.concat(
        [
            kline_df["close"].rename("Ground Truth"),
            plot_df["close"].rename("Prediction"),
        ],
        axis=1,
    )
    volume_df = pd.concat(
        [
            kline_df["volume"].rename("Ground Truth"),
            plot_df["volume"].rename("Prediction"),
        ],
        axis=1,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-series Kronos prediction example.")
    parser.add_argument("--model", choices=["small", "base"], default="small", help="Model size to run.")
    args = parser.parse_args()

    spec = get_model_spec(args.model)
    device = detect_device()
    data_file = data_path("XSHG_5min_600977.csv")
    csv_file = output_path(f"prediction_example_{args.model}_forecast.csv")
    plot_file = output_path(f"prediction_example_{args.model}_plot.png")
    summary_file = output_path(f"prediction_example_{args.model}_summary.json")

    print(f"Loading {spec['name']} on device: {device}")
    start_time = time.perf_counter()
    tokenizer = KronosTokenizer.from_pretrained(spec["tokenizer_id"])
    model = Kronos.from_pretrained(spec["model_id"])
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=spec["max_context"])

    df = pd.read_csv(data_file)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    x_df = df.loc[: LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: LOOKBACK - 1, "timestamps"]
    y_timestamp = df.loc[LOOKBACK : LOOKBACK + PRED_LEN - 1, "timestamps"]

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    elapsed_seconds = time.perf_counter() - start_time

    print("Forecasted Data Head:")
    print(pred_df.head())

    pred_df.to_csv(csv_file, index_label="timestamps")
    kline_df = df.loc[: LOOKBACK + PRED_LEN - 1]
    plot_prediction(kline_df, pred_df, plot_file)

    save_json(
        summary_file,
        {
            "experiment": "prediction_example",
            "description": "Single-series OHLCV prediction on one historical window.",
            "model_key": args.model,
            "model_name": spec["name"],
            "model_id": spec["model_id"],
            "tokenizer_id": spec["tokenizer_id"],
            "device": device,
            "elapsed_seconds": elapsed_seconds,
            "lookback": LOOKBACK,
            "pred_len": PRED_LEN,
            "input_window_start": x_timestamp.iloc[0],
            "input_window_end": x_timestamp.iloc[-1],
            "prediction_window_start": y_timestamp.iloc[0],
            "prediction_window_end": y_timestamp.iloc[-1],
            "data_file": data_file,
            "forecast_csv": csv_file,
            "plot_file": plot_file,
            "prediction_head": pred_df.head(),
        },
    )

    print(f"Saved forecast CSV: {csv_file}")
    print(f"Saved forecast plot: {plot_file}")
    print(f"Saved summary JSON: {summary_file}")


if __name__ == "__main__":
    main()
