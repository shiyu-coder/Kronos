from pathlib import Path

import pytest
import yaml

from finetune_csv.config_loader import CustomFinetuneConfig


@pytest.mark.parametrize("automatic_path", ["", None])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"tokenizer_save_name": "custom_tokenizer"},
        {"base_save_path": "custom_output"},
        {"base_save_path": "custom_output", "tokenizer_save_name": "custom_tokenizer"},
        {"base_save_path": "runs/{exp_name}", "tokenizer_save_name": "custom_tokenizer"},
    ],
)
def test_automatic_tokenizer_path_matches_saved_checkpoint(tmp_path, automatic_path, overrides):
    model_paths = {
        "exp_name": "test_run",
        "base_path": str(tmp_path / "outputs"),
        "base_save_path": automatic_path,
        "finetuned_tokenizer": automatic_path,
        **overrides,
    }
    if model_paths["base_save_path"]:
        model_paths["base_save_path"] = str(tmp_path / model_paths["base_save_path"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model_paths": model_paths}), encoding="utf-8")

    config = CustomFinetuneConfig(str(config_path))
    saved_checkpoint = Path(config.tokenizer_best_model_path)
    saved_checkpoint.mkdir(parents=True)
    (saved_checkpoint / "config.json").write_text("{}", encoding="utf-8")

    assert Path(config.finetuned_tokenizer_path) == saved_checkpoint
    assert (Path(config.get_basemodel_config()["pretrained_tokenizer_path"]) / "config.json").is_file()


@pytest.mark.parametrize("explicit_path", ["external/tokenizer", "external/{exp_name}/tokenizer"])
def test_explicit_tokenizer_path_is_preserved(tmp_path, explicit_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"model_paths": {
            "exp_name": "test_run",
            "base_path": str(tmp_path / "outputs"),
            "base_save_path": str(tmp_path / "custom_output"),
            "tokenizer_save_name": "custom_tokenizer",
            "finetuned_tokenizer": explicit_path,
        }}),
        encoding="utf-8",
    )

    config = CustomFinetuneConfig(str(config_path))

    assert config.finetuned_tokenizer_path == explicit_path.format(exp_name="test_run")
