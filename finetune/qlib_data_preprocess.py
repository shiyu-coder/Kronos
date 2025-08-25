import os
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from tqdm import trange

from config import Config


class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.data = {}  # A dictionary to store processed data for each symbol.

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def load_qlib_data(self):
        """
        Loads raw data from Qlib, processes it symbol by symbol, and stores
        it in the `self.data` attribute.
        """
        print("Loading and processing data from Qlib...")
        data_fields_qlib = ['$' + f for f in self.data_fields]
        cal: np.ndarray = D.calendar()

        # Determine the actual start and end times to load, including buffer for lookback and predict windows.
        start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
        end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

        # 检查start_index - lookback_window是否会导致负数索引
        adjusted_start_index = max(start_index - self.config.lookback_window, 0)
        real_start_time = cal[adjusted_start_index]

        # 检查end_index是否超出数组范围
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
            end_index -= 1

        # 检查end_index + predict_window是否会超出数组范围
        adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
        real_end_time = cal[adjusted_end_index]

        # Load data using Qlib's data loader.
        data_df = QlibDataLoader(config=data_fields_qlib).load(
            self.config.instrument, real_start_time, real_end_time
        )
        data_df = data_df.stack().unstack(level=1)  # Reshape for easier access.

        symbol_list = list(data_df.columns)
        for i in trange(len(symbol_list), desc="Processing Symbols"):
            symbol = symbol_list[i]
            symbol_df = data_df[symbol]

            # Pivot the table to have features as columns and datetime as index.
            symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
            symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
            symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})

            # Calculate amount and select final features.
            symbol_df['vol'] = symbol_df['volume']
            symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            symbol_df = symbol_df[self.config.feature_list]

            # Filter out symbols with insufficient data.
            symbol_df = symbol_df.dropna()
            if len(symbol_df) < self.config.lookback_window + self.config.predict_window + 1:
                continue

            self.data[symbol] = symbol_df

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # Define time ranges from config.
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range

            # Create boolean masks for each dataset split.
            train_mask = (symbol_df.index >= train_start) & (symbol_df.index <= train_end)
            val_mask = (symbol_df.index >= val_start) & (symbol_df.index <= val_end)
            test_mask = (symbol_df.index >= test_start) & (symbol_df.index <= test_end)

            # Apply masks to create the final datasets.
            train_data[symbol] = symbol_df[train_mask]
            val_data[symbol] = symbol_df[val_mask]
            test_data[symbol] = symbol_df[test_mask]

        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("Datasets prepared and saved successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()
