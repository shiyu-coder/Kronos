import importlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_cors")
pytest.importorskip("plotly")
webui = importlib.import_module("webui.app")


@pytest.mark.parametrize("invalid_rows", [[], [0], [0, 1]])
def test_custom_date_chart_matches_predictor_input_after_cleaning(tmp_path, monkeypatch, invalid_rows):
    timestamps = pd.date_range("2026-01-01", periods=8, freq="D")
    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": range(100, 108),
        "high": range(110, 118),
        "low": range(90, 98),
        "close": range(105, 113),
    })
    df.loc[invalid_rows, "high"] = float("nan")
    csv_path = tmp_path / "candles.csv"
    df.to_csv(csv_path, index=False)
    captured = {}

    def predict(**kwargs):
        captured.update(kwargs)
        prediction = kwargs["df"].iloc[:kwargs["pred_len"]].copy()
        prediction.index = kwargs["y_timestamp"]
        return prediction

    monkeypatch.setattr(webui, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(webui, "predictor", SimpleNamespace(predict=predict))
    monkeypatch.setattr(webui, "save_prediction_results", lambda **kwargs: None)

    response = webui.app.test_client().post("/api/predict", json={
        "file_path": str(csv_path),
        "start_date": "2026-01-03",
        "lookback": 2,
        "pred_len": 2,
    })

    assert response.status_code == 200, response.get_json()
    chart = json.loads(response.get_json()["chart"])
    history, prediction, actual = chart["data"]
    expected_history = list(captured["x_timestamp"])
    expected_prediction = list(captured["y_timestamp"])

    assert list(pd.to_datetime(history["x"])) == expected_history
    assert history["close"] == captured["df"]["close"].tolist()
    assert list(pd.to_datetime(prediction["x"])) == expected_prediction
    assert list(pd.to_datetime(actual["x"])) == expected_prediction
