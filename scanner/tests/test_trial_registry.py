import json

from scanner.learning import trial_registry


def test_load_trials_filters_kind_and_skips_torn_lines(tmp_path, monkeypatch):
    path = tmp_path / "trial_registry.jsonl"
    monkeypatch.setattr(trial_registry, "TRIAL_REGISTRY_PATH", path)
    rows = [
        {"kind": "calibration_trial", "objective": "p_win", "acceptance": {"passed": False}},
        {"kind": "exit_geometry_trial", "variant": "none"},
        {"kind": "calibration_trial", "objective": "expected_r", "acceptance": {"passed": True}},
    ]
    text = "\n".join(json.dumps(r) for r in rows) + "\n" + '{"torn'
    path.write_text(text, encoding="utf-8")

    all_rows = trial_registry.load_trials()
    calibration = trial_registry.load_trials("calibration_trial")
    limited = trial_registry.load_trials("calibration_trial", limit=1)

    assert len(all_rows) == 3
    assert [r["objective"] for r in calibration] == ["p_win", "expected_r"]
    assert limited[0]["objective"] == "expected_r"


def test_load_trials_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(trial_registry, "TRIAL_REGISTRY_PATH", tmp_path / "nope.jsonl")
    assert trial_registry.load_trials() == []


def test_record_then_load_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "trial_registry.jsonl"
    monkeypatch.setattr(trial_registry, "TRIAL_REGISTRY_PATH", path)
    monkeypatch.setattr(trial_registry, "REPORT_DIR", tmp_path)

    trial_registry.record_trial("calibration_trial", {"objective": "tail_prob", "acceptance": {"passed": True}})

    rows = trial_registry.load_trials("calibration_trial")
    assert len(rows) == 1
    assert rows[0]["objective"] == "tail_prob"
    assert "recorded_at" in rows[0]
