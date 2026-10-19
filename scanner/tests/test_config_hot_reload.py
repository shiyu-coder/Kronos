import json

import pandas as pd

from scanner import config as scanner_config
from scanner.learning.autotuner import propose_overrides
from scanner.strategy.empty_space import score_empty_space

TUNABLES = [
    "MIN_RR",
    "MIN_KRONOS_AGREEMENT",
    "MIN_EMPTY_SPACE_SCORE",
    "MAX_ATM_BID_ASK_SPREAD_PCT",
    "MIN_ATM_OPEN_INTEREST",
    "ATR_COMPRESSION",
    "RANGE_COMPRESSION",
    "NO_TREND_SLOPE_ABS_MAX",
    "RESEARCH_CANDIDATE_MIN_SCORE",
    "DOCTRINE_V2_SCORE_BASELINE",
]


def test_reload_overrides_updates_every_tunable(tmp_path, monkeypatch):
    snapshot = {name: getattr(scanner_config, name) for name in TUNABLES}
    overrides = {
        "MIN_RR": 1.31,
        "MIN_KRONOS_AGREEMENT": 0.71,
        "MIN_EMPTY_SPACE_SCORE": 1,
        "MAX_ATM_BID_ASK_SPREAD_PCT": 0.21,
        "MIN_ATM_OPEN_INTEREST": 321,
        "ATR_COMPRESSION": 0.91,
        "RANGE_COMPRESSION": 0.81,
        "NO_TREND_SLOPE_ABS_MAX": 0.0031,
        "RESEARCH_CANDIDATE_MIN_SCORE": 51,
        "DOCTRINE_V2_SCORE_BASELINE": 81,
    }
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps(overrides), encoding="utf-8")
    monkeypatch.setattr(scanner_config, "OVERRIDES_PATH", overrides_path)

    try:
        scanner_config.reload_overrides()
        for name, expected in overrides.items():
            assert getattr(scanner_config, name) == expected, name

        # Removing a key from the overrides file must reset it to the
        # module default on the next reload, not keep the stale value.
        overrides_path.write_text(json.dumps({}), encoding="utf-8")
        scanner_config.reload_overrides()
        for name, (default, _) in scanner_config._TUNABLES.items():
            assert getattr(scanner_config, name) == default, name
    finally:
        for name, value in snapshot.items():
            setattr(scanner_config, name, value)


def _bars():
    rows = []
    for i in range(60):
        base = 100 + (i % 3) * 0.4
        rows.append([base, base + 4.0, base - 4.0, base + 0.2, 1000])
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def test_empty_space_gate_reads_live_config(monkeypatch):
    bars = _bars()

    monkeypatch.setattr(scanner_config, "MIN_RR", 999.0)
    blocked = score_empty_space(bars, "bullish", 103.0, 100.0)
    assert blocked.passed is False

    monkeypatch.setattr(scanner_config, "MIN_RR", 0.0)
    monkeypatch.setattr(scanner_config, "MIN_EMPTY_SPACE_SCORE", 0)
    allowed = score_empty_space(bars, "bullish", 103.0, 100.0)
    assert allowed.passed is True


def test_autotune_proposes_from_effective_values_not_import_defaults(monkeypatch):
    monkeypatch.setattr(scanner_config, "MIN_RR", 2.0)
    records = []
    for i in range(20):
        label = "win" if i < 15 else "loss"
        records.append(
            {
                "ticker": f"T{i}",
                "decision_ts": f"2026-06-{(i % 27) + 1:02d}T10:00:00-04:00",
                "outcome_status": "resolved",
                "outcome_label": label,
                "final_pass": False,
                "stage_failed": "potter_box",
                "entry_price": 10.0 + i,
                "direction": "bullish",
            }
        )

    proposal = propose_overrides(records)

    # Loosening step from the EFFECTIVE 2.0, not the import-time default 1.5.
    assert proposal["overrides"]["MIN_RR"] == 1.95
