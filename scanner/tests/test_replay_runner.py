import json
from pathlib import Path

import pandas as pd

from scanner.learning.replay_runner import run_replay_eval


def _replay_bars():
    rows = []
    for i in range(45):
        if i < 29:
            high, low, close, volume = 104.0, 96.0, 100.0 + (0.2 if i % 2 == 0 else -0.2), 1000
        elif i < 44:
            high, low, close, volume = 101.0, 99.0, 100.0 + (0.05 if i % 2 == 0 else -0.05), 1200
        else:
            high, low, close, volume = 104.0, 100.5, 103.0, 2600
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                "Open": 100.0 if i < 44 else 101.0,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
    return [{**row, "timestamp": row["timestamp"].isoformat()} for row in rows]


def test_sample_replay_dataset_has_enough_bars_for_potter_window():
    sample_path = Path(__file__).resolve().parents[1] / "replay" / "sample_replay_dataset.json"
    records = json.loads(sample_path.read_text(encoding="utf-8-sig"))

    assert len(records[0]["synthetic_bars"]) >= 40


def test_replay_eval_includes_stage_details(tmp_path):
    dataset = tmp_path / "replay.json"
    dataset.write_text(
        json.dumps([{"ticker": "EXAMPLE", "label_win": True, "synthetic_bars": _replay_bars()}]),
        encoding="utf-8",
    )

    payload = run_replay_eval(str(dataset), logger=type("Logger", (), {"info": lambda *args, **kwargs: None})())

    assert payload["samples"] == 1
    assert payload["details"][0]["stage"] in {"called", "potter_box", "empty_space"}
    assert "reason" in payload["details"][0]
    assert "potter_passed" in payload["details"][0]
    report = tmp_path / "reports" / "replay_eval_report.json"
    assert json.loads(report.read_text(encoding="utf-8"))["dataset"] == str(dataset)
