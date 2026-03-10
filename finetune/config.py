import copy
import json
from pathlib import Path


class Config:
    """
    Configuration class for finetuning and Qlib experiments.

    Defaults preserve the repository's original intended experiment setup.
    Use a config file or explicit overrides for local smoke tests and other
    environment-specific runs.
    """

    DEFAULTS = {
        "qlib_data_path": "~/.qlib/qlib_data/cn_data",
        "instrument": "csi300",
        "dataset_begin_time": "2011-01-01",
        "dataset_end_time": "2025-06-05",
        "lookback_window": 90,
        "predict_window": 10,
        "max_context": 512,
        "feature_list": ["open", "high", "low", "close", "vol", "amt"],
        "time_feature_list": ["minute", "hour", "weekday", "day", "month"],
        "train_time_range": ["2011-01-01", "2022-12-31"],
        "val_time_range": ["2022-09-01", "2024-06-30"],
        "test_time_range": ["2024-04-01", "2025-06-05"],
        "backtest_time_range": ["2024-07-01", "2025-06-05"],
        "dataset_path": "./data/processed_datasets",
        "clip": 5.0,
        "epochs": 30,
        "log_interval": 100,
        "batch_size": 50,
        "n_train_iter": 2000 * 50,
        "n_val_iter": 400 * 50,
        "tokenizer_learning_rate": 2e-4,
        "predictor_learning_rate": 4e-5,
        "accumulation_steps": 1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_weight_decay": 0.1,
        "seed": 100,
        "use_comet": True,
        "comet_config": {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace",
        },
        "comet_tag": "finetune_demo",
        "comet_name": "finetune_demo",
        "save_path": "./outputs/models",
        "tokenizer_save_folder_name": "finetune_tokenizer_demo",
        "predictor_save_folder_name": "finetune_predictor_demo",
        "backtest_save_folder_name": "finetune_backtest_demo",
        "backtest_result_path": "./outputs/backtest_results",
        "pretrained_tokenizer_path": "path/to/your/Kronos-Tokenizer-base",
        "pretrained_predictor_path": "path/to/your/Kronos-small",
        "finetuned_tokenizer_path": "__AUTO__",
        "finetuned_predictor_path": "__AUTO__",
        "backtest_n_symbol_hold": 50,
        "backtest_n_symbol_drop": 5,
        "backtest_hold_thresh": 5,
        "inference_T": 0.6,
        "inference_top_p": 0.9,
        "inference_top_k": 0,
        "inference_sample_count": 5,
        "backtest_batch_size": 1000,
    }

    def __init__(self, config_file: str | None = None, overrides: dict | None = None):
        defaults = copy.deepcopy(self.DEFAULTS)
        for key, value in defaults.items():
            setattr(self, key, value)
        self._explicit_finetuned_tokenizer_path = False
        self._explicit_finetuned_predictor_path = False

        if config_file is not None:
            self.apply_overrides(self._load_config_file(config_file))
        if overrides:
            self.apply_overrides(overrides)

        self._refresh_derived_fields()

    @staticmethod
    def _load_config_file(config_file: str) -> dict:
        path = Path(config_file).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def apply_overrides(self, overrides: dict) -> None:
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config key: {key}")
            if key == "finetuned_tokenizer_path":
                self._explicit_finetuned_tokenizer_path = True
            if key == "finetuned_predictor_path":
                self._explicit_finetuned_predictor_path = True
            setattr(self, key, copy.deepcopy(value))
        self._refresh_derived_fields()

    def _refresh_derived_fields(self) -> None:
        default_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        default_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        if not self._explicit_finetuned_tokenizer_path:
            self.finetuned_tokenizer_path = default_tokenizer_path
        if not self._explicit_finetuned_predictor_path:
            self.finetuned_predictor_path = default_predictor_path

        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def to_dict(self) -> dict:
        return {
            key: copy.deepcopy(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            "csi800": "SH000906",
            "csi1000": "SH000852",
            "csi300": "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        raise ValueError(f"Benchmark not defined for instrument: {instrument}")
