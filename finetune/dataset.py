import io
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config


# Allowlist of types that are safe to unpickle from training data.
_SAFE_MODULES = {
    'numpy': {'ndarray', 'dtype', 'float64', 'float32', 'int64', 'int32', 'bool_'},
    'numpy.core.multiarray': {'scalar', '_reconstruct'},
    'numpy.core.numeric': {'*'},
    'pandas.core.frame': {'DataFrame'},
    'pandas.core.series': {'Series'},
    'pandas.core.indexes.base': {'_new_Index'},
    'pandas.core.indexes.range': {'RangeIndex'},
    'pandas.core.indexes.datetimes': {'DatetimeIndex', '_new_DatetimeIndex'},
    'pandas.core.internals.blocks': {'new_block', 'DatetimeTZBlock', 'FloatBlock', 'IntBlock', 'ObjectBlock'},
    'pandas.core.internals.managers': {'BlockManager'},
    'pandas._libs.tslibs.timestamps': {'Timestamp'},
    'pandas._libs.tslibs.nattype': {'NaTType'},
    'pandas._libs.internals': {'BlockPlacement'},
    'builtins': {'dict', 'list', 'tuple', 'set', 'frozenset', 'str', 'int', 'float', 'bool', 'bytes', 'complex', 'slice', 'range', 'type'},
    'collections': {'OrderedDict', 'defaultdict'},
    'datetime': {'datetime', 'date', 'timedelta', 'timezone'},
    'copy_reg': {'_reconstructor'},
    'copyreg': {'_reconstructor'},
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = _SAFE_MODULES.get(module)
        if allowed is not None and ('*' in allowed or name in allowed):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Restricted unpickler refused to load {module}.{name}. "
            f"If this type is expected in your data, add it to _SAFE_MODULES in dataset.py."
        )


def restricted_loads(data: bytes):
    return RestrictedUnpickler(io.BytesIO(data)).load()


def restricted_load(f):
    return RestrictedUnpickler(f).load()


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = restricted_load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
