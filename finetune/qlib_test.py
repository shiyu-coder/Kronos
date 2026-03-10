import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import qlib
from qlib.backtest import backtest, executor
from qlib.config import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils.time import Freq

# Ensure project root is in the Python path
sys.path.append("../")
from config import Config
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def analysis_frame_to_dict(frame: pd.DataFrame) -> dict[str, float]:
    if "risk" in frame.columns:
        series = frame["risk"]
    else:
        series = frame.iloc[:, 0]
    return {str(key): float(value) for key, value in series.items()}


class QlibTestDataset(Dataset):
    """
    PyTorch Dataset for handling Qlib test data, specifically for inference.
    """

    def __init__(self, data: dict, config: Config):
        self.data = data
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.symbols = list(self.data.keys())
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.indices = []

        print("Preprocessing and building indices for test dataset...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            df["minute"] = df["datetime"].dt.minute
            df["hour"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            self.data[symbol] = df

            num_samples = len(df) - self.window_size + 1
            if num_samples > 0:
                for idx in range(num_samples):
                    timestamp = df.iloc[idx + self.config.lookback_window - 1]["datetime"]
                    self.indices.append((symbol, idx, timestamp))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, start_idx, timestamp = self.indices[idx]
        df = self.data[symbol]

        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window

        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]

        x = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp), symbol, timestamp


class QlibBacktest:
    """
    A wrapper class for conducting Qlib backtests and saving structured results.
    """

    def __init__(self, config: Config, save_dir: Path, show_plot: bool):
        self.config = config
        self.save_dir = save_dir
        self.show_plot = show_plot
        self.initialize_qlib()

    def initialize_qlib(self):
        print("Initializing Qlib for backtesting...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def run_single_backtest(self, signal_series: pd.Series) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        strategy = TopkDropoutStrategy(
            topk=self.config.backtest_n_symbol_hold,
            n_drop=self.config.backtest_n_symbol_drop,
            hold_thresh=self.config.backtest_hold_thresh,
            signal=signal_series,
        )
        executor_config = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "delay_execution": True,
        }
        backtest_config = {
            "start_time": self.config.backtest_time_range[0],
            "end_time": self.config.backtest_time_range[1],
            "account": 100_000_000,
            "benchmark": self.config.backtest_benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "open",
                "open_cost": 0.001,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
            "executor": executor.SimulatorExecutor(**executor_config),
        }

        portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse("day"))
        report, _ = portfolio_metric_dict.get(analysis_freq)

        benchmark_metrics = risk_analysis(report["bench"], freq=analysis_freq)
        excess_without_cost = risk_analysis(report["return"] - report["bench"], freq=analysis_freq)
        excess_with_cost = risk_analysis(report["return"] - report["bench"] - report["cost"], freq=analysis_freq)

        print("\n--- Backtest Analysis ---")
        print("Benchmark Return:", benchmark_metrics, sep="\n")
        print("\nExcess Return (w/o cost):", excess_without_cost, sep="\n")
        print("\nExcess Return (w/ cost):", excess_with_cost, sep="\n")

        report_df = pd.DataFrame(
            {
                "cum_bench": report["bench"].cumsum(),
                "cum_return_w_cost": (report["return"] - report["cost"]).cumsum(),
                "cum_ex_return_w_cost": (report["return"] - report["bench"] - report["cost"]).cumsum(),
            }
        )
        analysis = {
            "benchmark": analysis_frame_to_dict(benchmark_metrics),
            "excess_return_without_cost": analysis_frame_to_dict(excess_without_cost),
            "excess_return_with_cost": analysis_frame_to_dict(excess_with_cost),
        }
        return report_df, analysis

    def run_and_plot_results(self, signals: dict[str, pd.DataFrame]):
        return_df, ex_return_df, bench_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        summary_rows = []

        for signal_name, pred_df in signals.items():
            print(f"\nBacktesting signal: {signal_name}...")
            pred_series = pred_df.stack()
            pred_series.index.names = ["datetime", "instrument"]
            pred_series = pred_series.swaplevel().sort_index()
            report_df, analysis = self.run_single_backtest(pred_series)

            report_df.to_csv(self.save_dir / f"{signal_name}_curve.csv", index_label="datetime")
            save_json(self.save_dir / f"{signal_name}_metrics.json", analysis)

            return_df[signal_name] = report_df["cum_return_w_cost"]
            ex_return_df[signal_name] = report_df["cum_ex_return_w_cost"]
            if "return" not in bench_df:
                bench_df["return"] = report_df["cum_bench"]

            summary_rows.append(
                {
                    "signal": signal_name,
                    "benchmark_annualized_return": analysis["benchmark"]["annualized_return"],
                    "benchmark_information_ratio": analysis["benchmark"]["information_ratio"],
                    "benchmark_max_drawdown": analysis["benchmark"]["max_drawdown"],
                    "excess_annualized_return_without_cost": analysis["excess_return_without_cost"]["annualized_return"],
                    "excess_information_ratio_without_cost": analysis["excess_return_without_cost"]["information_ratio"],
                    "excess_max_drawdown_without_cost": analysis["excess_return_without_cost"]["max_drawdown"],
                    "excess_annualized_return_with_cost": analysis["excess_return_with_cost"]["annualized_return"],
                    "excess_information_ratio_with_cost": analysis["excess_return_with_cost"]["information_ratio"],
                    "excess_max_drawdown_with_cost": analysis["excess_return_with_cost"]["max_drawdown"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.save_dir / "backtest_summary.csv", index=False)
        save_json(self.save_dir / "backtest_summary.json", {"signals": summary_rows})

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        return_df.plot(ax=axes[0], title="Cumulative Return with Cost", grid=True)
        axes[0].plot(bench_df["return"], label=self.config.instrument.upper(), color="black", linestyle="--")
        axes[0].legend()
        axes[0].set_ylabel("Cumulative Return")

        ex_return_df.plot(ax=axes[1], title="Cumulative Excess Return with Cost", grid=True)
        axes[1].legend()
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Cumulative Excess Return")

        plt.tight_layout()
        plot_path = self.save_dir / "backtest_curves.png"
        plt.savefig(plot_path, dpi=200)
        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)

        print(f"Saved backtest summary CSV: {self.save_dir / 'backtest_summary.csv'}")
        print(f"Saved backtest plot: {plot_path}")


def load_models(config: dict) -> tuple[KronosTokenizer, Kronos]:
    device = torch.device(config["device"])
    print(f"Loading models onto device: {device}...")
    tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_path"]).to(device).eval()
    model = Kronos.from_pretrained(config["model_path"]).to(device).eval()
    return tokenizer, model


def collate_fn_for_inference(batch):
    x, x_stamp, y_stamp, symbols, timestamps = zip(*batch)
    x_batch = torch.stack(x, dim=0)
    x_stamp_batch = torch.stack(x_stamp, dim=0)
    y_stamp_batch = torch.stack(y_stamp, dim=0)
    return x_batch, x_stamp_batch, y_stamp_batch, list(symbols), list(timestamps)


def generate_predictions(run_config: dict, test_data: dict) -> dict[str, pd.DataFrame]:
    tokenizer, model = load_models(run_config)
    device = torch.device(run_config["device"])

    dataset = QlibTestDataset(data=test_data, config=run_config["config_obj"])
    loader = DataLoader(
        dataset,
        batch_size=max(1, run_config["batch_size"] // run_config["sample_count"]),
        shuffle=False,
        num_workers=run_config["num_workers"],
        collate_fn=collate_fn_for_inference,
    )

    results = defaultdict(list)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps in tqdm(loader, desc="Inference"):
            preds = auto_regressive_inference(
                tokenizer,
                model,
                x.to(device),
                x_stamp.to(device),
                y_stamp.to(device),
                max_context=run_config["max_context"],
                pred_len=run_config["pred_len"],
                clip=run_config["clip"],
                T=run_config["T"],
                top_k=run_config["top_k"],
                top_p=run_config["top_p"],
                sample_count=run_config["sample_count"],
            )
            preds = preds[:, -run_config["pred_len"] :, :]
            last_day_close = x[:, -1, 3].numpy()
            signals = {
                "last": preds[:, -1, 3] - last_day_close,
                "mean": np.mean(preds[:, :, 3], axis=1) - last_day_close,
                "max": np.max(preds[:, :, 3], axis=1) - last_day_close,
                "min": np.min(preds[:, :, 3], axis=1) - last_day_close,
            }

            for idx in range(len(symbols)):
                for sig_type, sig_values in signals.items():
                    results[sig_type].append((timestamps[idx], symbols[idx], sig_values[idx]))

    print("Post-processing predictions into DataFrames...")
    prediction_dfs = {}
    for sig_type, records in results.items():
        df = pd.DataFrame(records, columns=["datetime", "instrument", "score"])
        pivot_df = df.pivot_table(index="datetime", columns="instrument", values="score")
        prediction_dfs[sig_type] = pivot_df.sort_index()

    return prediction_dfs


def main():
    parser = argparse.ArgumentParser(description="Run Kronos inference and Qlib backtesting.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (e.g. cuda:0, mps, cpu).")
    parser.add_argument("--config-file", type=str, default=None, help="Optional JSON config override file.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Explicit tokenizer path or Hugging Face model ID.")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit predictor path or Hugging Face model ID.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional override for the backtest result folder.")
    parser.add_argument("--save-only", action="store_true", help="Save plots to disk without opening a GUI window.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for inference.")
    args = parser.parse_args()

    overrides = {}
    if args.run_name:
        overrides["backtest_save_folder_name"] = args.run_name
    if args.tokenizer_path:
        overrides["finetuned_tokenizer_path"] = args.tokenizer_path
    if args.model_path:
        overrides["finetuned_predictor_path"] = args.model_path

    base_config = Config(config_file=args.config_file, overrides=overrides if overrides else None)
    run_config = {
        "config_obj": base_config,
        "device": args.device,
        "data_path": base_config.dataset_path,
        "result_save_path": base_config.backtest_result_path,
        "result_name": base_config.backtest_save_folder_name,
        "tokenizer_path": base_config.finetuned_tokenizer_path,
        "model_path": base_config.finetuned_predictor_path,
        "max_context": base_config.max_context,
        "pred_len": base_config.predict_window,
        "clip": base_config.clip,
        "T": base_config.inference_T,
        "top_k": base_config.inference_top_k,
        "top_p": base_config.inference_top_p,
        "sample_count": base_config.inference_sample_count,
        "batch_size": base_config.backtest_batch_size,
        "num_workers": args.num_workers,
    }

    print("--- Running with Configuration ---")
    for key, val in run_config.items():
        if key == "config_obj":
            continue
        print(f"{key:>20}: {val}")
    print("-" * 35)

    test_data_path = Path(run_config["data_path"]) / "test_data.pkl"
    print(f"Loading test data from {test_data_path}...")
    with test_data_path.open("rb") as handle:
        test_data = pickle.load(handle)
    non_empty_symbols = sum(1 for df in test_data.values() if not df.empty)
    print(f"Loaded {len(test_data)} symbols from test data, {non_empty_symbols} non-empty.")
    if non_empty_symbols == 0:
        raise RuntimeError(
            "test_data.pkl contains no non-empty symbols. Adjust the config ranges or local Qlib dataset coverage."
        )

    model_preds = generate_predictions(run_config, test_data)

    save_dir = Path(run_config["result_save_path"]) / run_config["result_name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = save_dir / "predictions.pkl"
    print(f"Saving prediction signals to {predictions_file}...")
    with predictions_file.open("wb") as handle:
        pickle.dump(model_preds, handle)

    save_json(
        save_dir / "run_config.json",
        {
            "device": args.device,
            "config_file": args.config_file,
            "tokenizer_path": run_config["tokenizer_path"],
            "model_path": run_config["model_path"],
            "result_save_path": str(save_dir),
            "save_only": args.save_only,
            "num_workers": args.num_workers,
            "config": base_config.to_dict(),
            "non_empty_symbols": non_empty_symbols,
        },
    )

    backtester = QlibBacktest(base_config, save_dir=save_dir, show_plot=not args.save_only)
    backtester.run_and_plot_results(model_preds)


if __name__ == "__main__":
    main()
