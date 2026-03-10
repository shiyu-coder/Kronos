import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import data_path, detect_device, get_model_spec, output_path, save_json
from model import Kronos, KronosPredictor, KronosTokenizer

LOOKBACK = 400
PRED_LEN = 120
DEFAULT_BATCH_SIZE = 5


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
    parser = argparse.ArgumentParser(
        description="Run batch prediction on multiple time windows from the same instrument."
    )
    parser.add_argument("--model", choices=["small", "base"], default="small", help="Model size to run.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of sequential time windows to batch together.",
    )
    parser.add_argument(
        "--plot-series",
        type=int,
        default=0,
        help="Which window index to render as the representative plot.",
    )
    args = parser.parse_args()

    spec = get_model_spec(args.model)
    device = detect_device()
    data_file = data_path("XSHG_5min_600977.csv")
    summary_file = output_path(f"prediction_batch_example_{args.model}_summary.json")

    print(
        "Running batch prediction on multiple sequential windows from the same source file, "
        "not on multiple instruments."
    )
    print(f"Loading {spec['name']} on device: {device}")
    start_time = time.perf_counter()
    tokenizer = KronosTokenizer.from_pretrained(spec["tokenizer_id"])
    model = Kronos.from_pretrained(spec["model_id"])
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=spec["max_context"])

    df = pd.read_csv(data_file)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    df_list = []
    x_timestamp_list = []
    y_timestamp_list = []
    window_summaries = []

    for idx in range(args.batch_size):
        start = idx * LOOKBACK
        end = start + LOOKBACK - 1
        pred_start = end + 1
        pred_end = pred_start + PRED_LEN - 1

        df_list.append(df.loc[start:end, ["open", "high", "low", "close", "volume", "amount"]])
        x_timestamp_list.append(df.loc[start:end, "timestamps"])
        y_timestamp_list.append(df.loc[pred_start:pred_end, "timestamps"])
        window_summaries.append(
            {
                "series_index": idx,
                "instrument_note": "Same instrument, different historical time window.",
                "history_row_start": start,
                "history_row_end": end,
                "forecast_row_start": pred_start,
                "forecast_row_end": pred_end,
                "history_start": df.loc[start, "timestamps"],
                "history_end": df.loc[end, "timestamps"],
                "forecast_start": df.loc[pred_start, "timestamps"],
                "forecast_end": df.loc[pred_end, "timestamps"],
            }
        )

    pred_df_list = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_timestamp_list,
        y_timestamp_list=y_timestamp_list,
        pred_len=PRED_LEN,
        verbose=False,
    )
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Generated batch predictions: {len(pred_df_list)} windows")
    representative_plot = None

    for idx, pred_df in enumerate(pred_df_list):
        csv_file = output_path(f"prediction_batch_example_{args.model}_series_{idx}.csv")
        pred_df.to_csv(csv_file, index_label="timestamps")
        window_summaries[idx]["forecast_csv"] = csv_file

        print(f"Series {idx} forecast head:")
        print(pred_df.head())
        print(f"Saved series {idx} CSV: {csv_file}")

        if idx == args.plot_series:
            plot_file = output_path(f"prediction_batch_example_{args.model}_representative_series_{idx}.png")
            kline_df = df.loc[idx * LOOKBACK : idx * LOOKBACK + LOOKBACK + PRED_LEN - 1]
            plot_prediction(kline_df, pred_df, plot_file)
            representative_plot = plot_file
            window_summaries[idx]["representative_plot"] = plot_file
            print(f"Saved representative plot for series {idx}: {plot_file}")

    save_json(
        summary_file,
        {
            "experiment": "prediction_batch_example",
            "description": (
                "Batch prediction on multiple sequential time windows from a single instrument. "
                "This example demonstrates parallel inference, not cross-sectional multi-asset prediction."
            ),
            "model_key": args.model,
            "model_name": spec["name"],
            "model_id": spec["model_id"],
            "tokenizer_id": spec["tokenizer_id"],
            "device": device,
            "elapsed_seconds": elapsed_seconds,
            "lookback": LOOKBACK,
            "pred_len": PRED_LEN,
            "batch_size": args.batch_size,
            "plot_series": args.plot_series,
            "data_file": data_file,
            "representative_plot": representative_plot,
            "windows": window_summaries,
        },
    )

    print(f"Saved summary JSON: {summary_file}")


if __name__ == "__main__":
    main()
