import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from dataset import restricted_load
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from matplotlib import pyplot as plt

import qlib
from qlib.config import REG_CN
from qlib.backtest import backtest, executor, CommonInfrastructure
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils import flatten_dict
from qlib.utils.time import Freq

# Ensure project root is in the Python path
sys.path.append("../")
from config import Config
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference


# =================================================================================
# 1. Data Loading and Processing for Inference
# =================================================================================

class QlibTestDataset(Dataset):
    """
    PyTorch Dataset for handling Qlib test data, specifically for inference.

    This dataset iterates through all possible sliding windows sequentially. It also
    yields metadata like symbol and timestamp, which are crucial for mapping
    predictions back to the original time series.
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
            # Generate time features on-the-fly
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df  # Store preprocessed dataframe

            num_samples = len(df) - self.window_size + 1
            if num_samples > 0:
                for i in range(num_samples):
                    timestamp = df.iloc[i + self.config.lookback_window - 1]['datetime']
                    self.indices.append((symbol, i, timestamp))

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

        # Instance-level normalization, consistent with training
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp), symbol, timestamp


# =================================================================================
# 2. Backtesting Logic
# =================================================================================

class QlibBacktest:
    """
    A wrapper class for conducting backtesting experiments using Qlib.
    """

    def __init__(self, config: Config):
        self.config = config
        self.initialize_qlib()

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib for backtesting...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def run_single_backtest(self, signal_series: pd.Series) -> pd.DataFrame:
        """
        Runs a single backtest for a given prediction signal.

        Args:
            signal_series (pd.Series): A pandas Series with a MultiIndex
                                       (instrument, datetime) and prediction scores.
        Returns:
            pd.DataFrame: A DataFrame containing the performance report.
        """
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
                "freq": "day", "limit_threshold": 0.095, "deal_price": "open",
                "open_cost": 0.001, "close_cost": 0.0015, "min_cost": 5,
            },
            "executor": executor.SimulatorExecutor(**executor_config),
        }

        portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse("day"))
        report, _ = portfolio_metric_dict.get(analysis_freq)

        # --- Analysis and Reporting ---
        analysis = {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"], freq=analysis_freq),
            "excess_return_with_cost": risk_analysis(report["return"] - report["bench"] - report["cost"], freq=analysis_freq),
        }
        print("\n--- Backtest Analysis ---")
        print("Benchmark Return:", risk_analysis(report["bench"], freq=analysis_freq), sep='\n')
        print("\nExcess Return (w/o cost):", analysis["excess_return_without_cost"], sep='\n')
        print("\nExcess Return (w/ cost):", analysis["excess_return_with_cost"], sep='\n')

        report_df = pd.DataFrame({
            "cum_bench": report["bench"].cumsum(),
            "cum_return_w_cost": (report["return"] - report["cost"]).cumsum(),
            "cum_ex_return_w_cost": (report["return"] - report["bench"] - report["cost"]).cumsum(),
        })
        return report_df

    def run_and_plot_results(self, signals: dict[str, pd.DataFrame]):
        """
        Runs backtests for multiple signals and plots the cumulative return curves.

        Args:
            signals (dict[str, pd.DataFrame]): A dictionary where keys are signal names
                                               and values are prediction DataFrames.
        """
        return_df, ex_return_df, bench_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for signal_name, pred_df in signals.items():
            print(f"\nBacktesting signal: {signal_name}...")
            pred_series = pred_df.stack()
            pred_series.index.names = ['datetime', 'instrument']
            pred_series = pred_series.swaplevel().sort_index()
            report_df = self.run_single_backtest(pred_series)

            return_df[signal_name] = report_df['cum_return_w_cost']
            ex_return_df[signal_name] = report_df['cum_ex_return_w_cost']
            if 'return' not in bench_df:
                bench_df['return'] = report_df['cum_bench']

        # Plotting results
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        return_df.plot(ax=axes[0], title='Cumulative Return with Cost', grid=True)
        axes[0].plot(bench_df['return'], label=self.config.instrument.upper(), color='black', linestyle='--')
        axes[0].legend()
        axes[0].set_ylabel("Cumulative Return")

        ex_return_df.plot(ax=axes[1], title='Cumulative Excess Return with Cost', grid=True)
        axes[1].legend()
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Cumulative Excess Return")

        plt.tight_layout()
        plt.savefig("../figures/backtest_result_example.png", dpi=200)
        plt.show()


# =================================================================================
# 3. Inference Logic
# =================================================================================

def load_models(config: dict) -> tuple[KronosTokenizer, Kronos]:
    """Loads the fine-tuned tokenizer and predictor model."""
    device = torch.device(config['device'])
    print(f"Loading models onto device: {device}...")
    tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_path']).to(device).eval()
    model = Kronos.from_pretrained(config['model_path']).to(device).eval()
    return tokenizer, model


def collate_fn_for_inference(batch):
    """
    Custom collate function to handle batches containing Tensors, strings, and Timestamps.

    Args:
        batch (list): A list of samples, where each sample is the tuple returned by
                      QlibTestDataset.__getitem__.

    Returns:
        A single tuple containing the batched data.
    """
    # Unzip the list of samples into separate lists for each data type
    x, x_stamp, y_stamp, symbols, timestamps = zip(*batch)

    # Stack the tensors to create a batch
    x_batch = torch.stack(x, dim=0)
    x_stamp_batch = torch.stack(x_stamp, dim=0)
    y_stamp_batch = torch.stack(y_stamp, dim=0)

    # Return the strings and timestamps as lists
    return x_batch, x_stamp_batch, y_stamp_batch, list(symbols), list(timestamps)


def generate_predictions(config: dict, test_data: dict) -> dict[str, pd.DataFrame]:
    """
    Runs inference on the test dataset to generate prediction signals.

    Args:
        config (dict): A dictionary containing inference parameters.
        test_data (dict): The raw test data loaded from a pickle file.

    Returns:
        A dictionary where keys are signal types (e.g., 'mean', 'last') and
        values are DataFrames of predictions (datetime index, symbol columns).
    """
    tokenizer, model = load_models(config)
    device = torch.device(config['device'])

    # Use the Dataset and DataLoader for efficient batching and processing
    dataset = QlibTestDataset(data=test_data, config=Config())
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'] // config['sample_count'],
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn_for_inference
    )

    results = defaultdict(list)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps in tqdm(loader, desc="Inference"):
            preds = auto_regressive_inference(
                tokenizer, model, x.to(device), x_stamp.to(device), y_stamp.to(device),
                max_context=config['max_context'], pred_len=config['pred_len'], clip=config['clip'],
                T=config['T'], top_k=config['top_k'], top_p=config['top_p'], sample_count=config['sample_count']
            )
            # You can try commenting on this line to keep the history data
            preds = preds[:, -config['pred_len']:, :]

            # The 'close' price is at index 3 in `feature_list`
            last_day_close = x[:, -1, 3].numpy()
            signals = {
                'last': preds[:, -1, 3] - last_day_close,
                'mean': np.mean(preds[:, :, 3], axis=1) - last_day_close,
                'max': np.max(preds[:, :, 3], axis=1) - last_day_close,
                'min': np.min(preds[:, :, 3], axis=1) - last_day_close,
            }

            for i in range(len(symbols)):
                for sig_type, sig_values in signals.items():
                    results[sig_type].append((timestamps[i], symbols[i], sig_values[i]))

    print("Post-processing predictions into DataFrames...")
    prediction_dfs = {}
    for sig_type, records in results.items():
        df = pd.DataFrame(records, columns=['datetime', 'instrument', 'score'])
        pivot_df = df.pivot_table(index='datetime', columns='instrument', values='score')
        prediction_dfs[sig_type] = pivot_df.sort_index()

    return prediction_dfs


# =================================================================================
# 4. Main Execution
# =================================================================================

def main():
    """Main function to set up config, run inference, and execute backtesting."""
    parser = argparse.ArgumentParser(description="Run Kronos Inference and Backtesting")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for inference (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()

    # --- 1. Configuration Setup ---
    base_config = Config()

    # Create a dedicated dictionary for this run's configuration
    run_config = {
        'device': args.device,
        'data_path': base_config.dataset_path,
        'result_save_path': base_config.backtest_result_path,
        'result_name': base_config.backtest_save_folder_name,
        'tokenizer_path': base_config.finetuned_tokenizer_path,
        'model_path': base_config.finetuned_predictor_path,
        'max_context': base_config.max_context,
        'pred_len': base_config.predict_window,
        'clip': base_config.clip,
        'T': base_config.inference_T,
        'top_k': base_config.inference_top_k,
        'top_p': base_config.inference_top_p,
        'sample_count': base_config.inference_sample_count,
        'batch_size': base_config.backtest_batch_size,
    }

    print("--- Running with Configuration ---")
    for key, val in run_config.items():
        print(f"{key:>20}: {val}")
    print("-" * 35)

    # --- 2. Load Data ---
    test_data_path = os.path.join(run_config['data_path'], "test_data.pkl")
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'rb') as f:
        test_data = restricted_load(f)
    print(test_data)
    # --- 3. Generate Predictions ---
    model_preds = generate_predictions(run_config, test_data)

    # --- 4. Save Predictions ---
    save_dir = os.path.join(run_config['result_save_path'], run_config['result_name'])
    os.makedirs(save_dir, exist_ok=True)
    predictions_file = os.path.join(save_dir, "predictions.pkl")
    print(f"Saving prediction signals to {predictions_file}...")
    with open(predictions_file, 'wb') as f:
        pickle.dump(model_preds, f)

    # --- 5. Run Backtesting ---
    with open(predictions_file, 'rb') as f:
        model_preds = restricted_load(f)

    backtester = QlibBacktest(base_config)
    backtester.run_and_plot_results(model_preds)


if __name__ == '__main__':
    main()


