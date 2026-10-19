"""Append-only registry of every tuning trial the system evaluates or applies.

Every threshold change is one more draw from the multiple-testing urn; the
deflated-Sharpe haircut needs the true trial count, so it is recorded as a
fact instead of reconstructed from memory.
"""

from __future__ import annotations

import json

import pandas as pd

from ..config import REPORT_DIR

TRIAL_REGISTRY_PATH = REPORT_DIR / "trial_registry.jsonl"


def record_trial(kind: str, payload: dict) -> None:
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        row = {"recorded_at": pd.Timestamp.utcnow().isoformat(), "kind": kind, **payload}
        with TRIAL_REGISTRY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # The registry is telemetry; never let it break a tuning run.
        pass


def load_trials(kind: str | None = None, limit: int | None = None) -> list[dict]:
    """Read registered trials (newest last). Unparseable lines are skipped -
    the registry is telemetry, and a torn tail must not break the loop that
    reads its own history to gate self-improvement."""
    if not TRIAL_REGISTRY_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        for line in TRIAL_REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and (kind is None or row.get("kind") == kind):
                rows.append(row)
    except Exception:
        return []
    if limit is not None and limit >= 0:
        rows = rows[-limit:]
    return rows
