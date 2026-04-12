import os
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, config_path: str):

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")

        with open(self.config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config = self._resolve_dynamic_paths(config)

        return config

    def _resolve_dynamic_paths(self, config: dict[str, Any]) -> dict[str, Any]:

        exp_name = config.get("model_paths", {}).get("exp_name", "")
        if not exp_name:
            return config

        base_path = config.get("model_paths", {}).get("base_path", "")
        path_templates = {
            "base_save_path": f"{base_path}/{exp_name}",
            "finetuned_tokenizer": f"{base_path}/{exp_name}/tokenizer/best_model",
        }

        if "model_paths" in config:
            for key, template in path_templates.items():
                if key in config["model_paths"]:
                    # only use template when the original value is empty string
                    current_value = config["model_paths"][key]
                    if current_value == "" or current_value is None:
                        config["model_paths"][key] = template
                    else:
                        # if the original value is not empty, use template to replace the {exp_name} placeholder
                        if (
                            isinstance(current_value, str)
                            and "{exp_name}" in current_value
                        ):
                            config["model_paths"][key] = current_value.format(
                                exp_name=exp_name
                            )

        return config

    def get(self, key: str, default=None):

        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_config(self) -> dict[str, Any]:
        return self.config.get("data", {})

    def get_training_config(self) -> dict[str, Any]:
        return self.config.get("training", {})

    def get_model_paths(self) -> dict[str, str]:
        return self.config.get("model_paths", {})

    def get_experiment_config(self) -> dict[str, Any]:
        return self.config.get("experiment", {})

    def get_device_config(self) -> dict[str, Any]:
        return self.config.get("device", {})

    def get_distributed_config(self) -> dict[str, Any]:
        return self.config.get("distributed", {})

    def update_config(self, updates: dict[str, Any]):

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_nested_dict(self.config, updates)

    def save_config(self, save_path: str = None):

        if save_path is None:
            save_path = self.config_path

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.config, f, default_flow_style=False, allow_unicode=True, indent=2
            )

    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        yaml.dump(self.config, default_flow_style=False, allow_unicode=True, indent=2)
        print("=" * 50)


class CustomFinetuneConfig:
    def __init__(self, config_path: str = None):

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        self.loader = ConfigLoader(config_path)
        self._load_all_configs()

    def _load_all_configs(self):

        data_config = self.loader.get_data_config()
        self.data_path = data_config.get("data_path")
        self.lookback_window = data_config.get("lookback_window", 512)
        self.predict_window = data_config.get("predict_window", 48)
        self.max_context = data_config.get("max_context", 512)
        self.clip = data_config.get("clip", 5.0)
        self.train_ratio = data_config.get("train_ratio", 0.9)
        self.val_ratio = data_config.get("val_ratio", 0.1)
        self.test_ratio = data_config.get("test_ratio", 0.0)

        # training configuration
        training_config = self.loader.get_training_config()
        # support training epochs of tokenizer and basemodel separately
        self.tokenizer_epochs = training_config.get("tokenizer_epochs", 30)
        self.basemodel_epochs = training_config.get("basemodel_epochs", 30)

        if "epochs" in training_config and "tokenizer_epochs" not in training_config:
            self.tokenizer_epochs = training_config.get("epochs", 30)
        if "epochs" in training_config and "basemodel_epochs" not in training_config:
            self.basemodel_epochs = training_config.get("epochs", 30)

        self.batch_size = training_config.get("batch_size", 160)
        self.log_interval = training_config.get("log_interval", 50)
        self.num_workers = training_config.get("num_workers", 6)
        self.seed = training_config.get("seed", 100)
        self.tokenizer_learning_rate = training_config.get(
            "tokenizer_learning_rate", 2e-4
        )
        self.predictor_learning_rate = training_config.get(
            "predictor_learning_rate", 4e-5
        )
        self.adam_beta1 = training_config.get("adam_beta1", 0.9)
        self.adam_beta2 = training_config.get("adam_beta2", 0.95)
        self.adam_weight_decay = training_config.get("adam_weight_decay", 0.1)
        self.accumulation_steps = training_config.get("accumulation_steps", 1)

        model_paths = self.loader.get_model_paths()
        self.exp_name = model_paths.get("exp_name", "default_experiment")
        self.pretrained_tokenizer_path = model_paths.get("pretrained_tokenizer")
        self.pretrained_predictor_path = model_paths.get("pretrained_predictor")
        self.base_save_path = model_paths.get("base_save_path")
        self.tokenizer_save_name = model_paths.get("tokenizer_save_name", "tokenizer")
        self.basemodel_save_name = model_paths.get("basemodel_save_name", "basemodel")
        self.finetuned_tokenizer_path = model_paths.get("finetuned_tokenizer")

        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get("name", "kronos_custom_finetune")
        self.experiment_description = experiment_config.get("description", "")
        self.use_comet = experiment_config.get("use_comet", False)
        self.train_tokenizer = experiment_config.get("train_tokenizer", True)
        self.train_basemodel = experiment_config.get("train_basemodel", True)
        self.skip_existing = experiment_config.get("skip_existing", False)

        unified_pretrained = experiment_config.get("pre_trained", None)
        self.pre_trained_tokenizer = experiment_config.get(
            "pre_trained_tokenizer",
            unified_pretrained if unified_pretrained is not None else True,
        )
        self.pre_trained_predictor = experiment_config.get(
            "pre_trained_predictor",
            unified_pretrained if unified_pretrained is not None else True,
        )

        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get("use_cuda", True)
        self.device_id = device_config.get("device_id", 0)

        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get("use_ddp", False)
        self.ddp_backend = distributed_config.get("backend", "nccl")

        self._compute_full_paths()

    def _compute_full_paths(self):

        self.tokenizer_save_path = os.path.join(
            self.base_save_path, self.tokenizer_save_name
        )
        self.tokenizer_best_model_path = os.path.join(
            self.tokenizer_save_path, "best_model"
        )

        self.basemodel_save_path = os.path.join(
            self.base_save_path, self.basemodel_save_name
        )
        self.basemodel_best_model_path = os.path.join(
            self.basemodel_save_path, "best_model"
        )

    def get_tokenizer_config(self):

        return {
            "data_path": self.data_path,
            "lookback_window": self.lookback_window,
            "predict_window": self.predict_window,
            "max_context": self.max_context,
            "clip": self.clip,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "epochs": self.tokenizer_epochs,
            "batch_size": self.batch_size,
            "log_interval": self.log_interval,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "learning_rate": self.tokenizer_learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_weight_decay": self.adam_weight_decay,
            "accumulation_steps": self.accumulation_steps,
            "pretrained_model_path": self.pretrained_tokenizer_path,
            "save_path": self.tokenizer_save_path,
            "use_comet": self.use_comet,
        }

    def get_basemodel_config(self):

        return {
            "data_path": self.data_path,
            "lookback_window": self.lookback_window,
            "predict_window": self.predict_window,
            "max_context": self.max_context,
            "clip": self.clip,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "epochs": self.basemodel_epochs,
            "batch_size": self.batch_size,
            "log_interval": self.log_interval,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "predictor_learning_rate": self.predictor_learning_rate,
            "tokenizer_learning_rate": self.tokenizer_learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_weight_decay": self.adam_weight_decay,
            "pretrained_tokenizer_path": self.finetuned_tokenizer_path,
            "pretrained_predictor_path": self.pretrained_predictor_path,
            "save_path": self.basemodel_save_path,
            "use_comet": self.use_comet,
        }

    def print_config_summary(self):

        print("=" * 60)
        print("Kronos finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name: {self.exp_name}")
        print(f"Data path: {self.data_path}")
        print(f"Lookback window: {self.lookback_window}")
        print(f"Predict window: {self.predict_window}")
        print(f"Tokenizer training epochs: {self.tokenizer_epochs}")
        print(f"Basemodel training epochs: {self.basemodel_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tokenizer learning rate: {self.tokenizer_learning_rate}")
        print(f"Predictor learning rate: {self.predictor_learning_rate}")
        print(f"Train tokenizer: {self.train_tokenizer}")
        print(f"Train basemodel: {self.train_basemodel}")
        print(f"Skip existing: {self.skip_existing}")
        print(f"Use pre-trained tokenizer: {self.pre_trained_tokenizer}")
        print(f"Use pre-trained predictor: {self.pre_trained_predictor}")
        print(f"Base save path: {self.base_save_path}")
        print(f"Tokenizer save path: {self.tokenizer_save_path}")
        print(f"Basemodel save path: {self.basemodel_save_path}")
        print("=" * 60)
