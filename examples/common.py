import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

EXAMPLES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_ROOT.parent
DATA_ROOT = EXAMPLES_ROOT / "data"
OUTPUT_ROOT = EXAMPLES_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

MODEL_SPECS = {
    "small": {
        "name": "Kronos-small",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "base": {
        "name": "Kronos-base",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def data_path(filename: str) -> Path:
    return DATA_ROOT / filename


def output_path(filename: str) -> Path:
    return OUTPUT_ROOT / filename


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model_spec(model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unsupported model '{model_key}'. Expected one of: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[model_key]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [{key: to_jsonable(item) for key, item in row.items()} for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
