import importlib
import json
import datetime as dt
from pathlib import Path

import pandas as pd


webui_app = importlib.import_module("webui.app")


def _write_price_csv(path: Path) -> None:
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 09:30:00,10,11,9,10.5,1000\n"
        "2026-01-01 10:00:00,10.5,12,10,11.5,1100\n",
        encoding="utf-8",
    )


def test_load_data_file_rejects_paths_outside_configured_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_csv = tmp_path / "outside.csv"
    _write_price_csv(outside_csv)
    monkeypatch.setattr(webui_app, "DATA_DIR", data_dir, raising=False)

    df, error = webui_app.load_data_file(str(outside_csv))

    assert df is None
    assert "outside allowed data directory" in error


def test_load_data_file_allows_paths_inside_configured_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    inside_csv = data_dir / "prices.csv"
    _write_price_csv(inside_csv)
    monkeypatch.setattr(webui_app, "DATA_DIR", data_dir, raising=False)

    df, error = webui_app.load_data_file(str(inside_csv))

    assert error is None
    assert len(df) == 2
    assert list(df[["open", "high", "low", "close"]].columns) == ["open", "high", "low", "close"]


def test_webui_server_defaults_to_local_non_debug(monkeypatch):
    monkeypatch.delenv("KRONOS_WEBUI_HOST", raising=False)
    monkeypatch.delenv("KRONOS_WEBUI_PORT", raising=False)
    monkeypatch.delenv("KRONOS_WEBUI_DEBUG", raising=False)

    config = webui_app.get_server_config()

    assert config == {"host": "127.0.0.1", "port": 7070, "debug": False}


def test_cors_origins_follow_custom_server_port_when_not_overridden(monkeypatch):
    monkeypatch.setenv("KRONOS_WEBUI_HOST", "127.0.0.1")
    monkeypatch.setenv("KRONOS_WEBUI_PORT", "9090")
    monkeypatch.delenv("KRONOS_WEBUI_CORS_ORIGINS", raising=False)

    origins = webui_app.get_cors_origins()

    assert "http://127.0.0.1:9090" in origins
    assert "http://localhost:9090" in origins


def test_save_prediction_results_handles_empty_predictions_with_actual_data(monkeypatch, tmp_path):
    monkeypatch.setattr(webui_app, "PREDICTION_RESULTS_DIR", tmp_path, raising=False)
    input_data = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
        }
    )
    actual_data = [
        {
            "timestamp": "2026-01-02T09:30:00",
            "open": 10.5,
            "high": 12.0,
            "low": 10.0,
            "close": 11.5,
        }
    ]

    saved_path = webui_app.save_prediction_results(
        file_path=str(tmp_path / "prices.csv"),
        prediction_type="test",
        prediction_results=[],
        actual_data=actual_data,
        input_data=input_data,
        prediction_params={"lookback": 1, "pred_len": 1},
    )

    assert saved_path is not None
    payload = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert payload["actual_data"] == actual_data
    assert "continuity" not in payload["analysis"]


def test_save_prediction_results_does_not_overwrite_same_second_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(webui_app, "PREDICTION_RESULTS_DIR", tmp_path, raising=False)

    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 5, 28, 12, 0, 1, tzinfo=tz)

    monkeypatch.setattr(webui_app.datetime, "datetime", FixedDateTime)
    input_data = pd.DataFrame({"open": [1.0], "high": [1.2], "low": [0.9], "close": [1.1]})
    prediction = [{"timestamp": "2026-05-29T12:00:00", "open": 1.1, "high": 1.3, "low": 1.0, "close": 1.2}]

    first = webui_app.save_prediction_results("prices.csv", "test", prediction, [], input_data, {})
    second = webui_app.save_prediction_results("prices.csv", "test", prediction, [], input_data, {})

    assert first != second
    assert len(list(tmp_path.glob("prediction_*.json"))) == 2
