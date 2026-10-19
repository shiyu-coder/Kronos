from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import REPORT_DIR
from ..strategy.empty_space import score_empty_space
from ..strategy.potter_box import detect_potter_box


def run_replay_eval(dataset_path: str, logger) -> dict:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Replay dataset not found: {dataset_path}")

    # Accept both plain UTF-8 and UTF-8 BOM exports from external tooling.
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("Replay dataset must be a list")

    tp = fp = tn = fn = 0
    details = []

    for r in records:
        ticker = r.get("ticker", "TEST")
        label = bool(r.get("label_win", False))
        bars = pd.DataFrame(r.get("synthetic_bars", []))
        if bars.empty:
            continue
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        bars = bars.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

        pb = detect_potter_box(ticker, bars)
        called = False
        stage = "potter_box"
        reason = pb.skip_reason or "potter_box_failed"
        if pb.passed and pb.direction:
            es = score_empty_space(bars, pb.direction, pb.breakout_close, pb.cost_basis)
            called = bool(es.passed)
            stage = "called" if called else "empty_space"
            reason = None if called else es.skip_reason or "empty_space_failed"

        if called and label:
            tp += 1
        elif called and not label:
            fp += 1
        elif (not called) and label:
            fn += 1
        else:
            tn += 1

        details.append(
            {
                "ticker": ticker,
                "called": called,
                "label_win": label,
                "stage": stage,
                "reason": reason,
                "potter_passed": bool(pb.passed),
                "direction": pb.direction,
                "edge_score": None,
            }
        )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    payload = {
        "mode": "replay_eval",
        "dataset": str(path),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "samples": len(details),
        "details": details,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "replay_eval_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("REPLAY_EVAL_REPORT: %s", json.dumps(payload))
    return payload
