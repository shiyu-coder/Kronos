"""Pre-registered post-breakout retest entry experiment.

The script prints results as JSON and deliberately does not mutate production
reports, tuning, or source. See preregistration.json before reading output.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import EDGE_COST_BPS_PER_SIDE, EDGE_INDEX_PATH
from scanner.edge.outcomes import walk_triple_barrier
from scanner.edge.stats import _hac_day_mean_summary


WAIT_BARS = 3
HORIZON_BARS = 5
DEPENDENCE_LAGS = WAIT_BARS + HORIZON_BARS
ROUND_TRIP_COST_PCT = 2.0 * EDGE_COST_BPS_PER_SIDE / 100.0
R_CAP = 10.0
CACHE_PATH = (
    REPO_ROOT
    / "scanner"
    / "research"
    / "experiments"
    / "20260813_horizon_sweep"
    / "bars_cache.pkl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _net_r(return_pct: float, risk_pct: float) -> float:
    value = (float(return_pct) - ROUND_TRIP_COST_PCT) / float(risk_pct)
    return float(np.clip(value, -R_CAP, R_CAP))


def _stats(rows: list[dict], *, day_key: str) -> dict:
    if not rows:
        return {
            "rows": 0,
            "days": 0,
            "row_mean_net_r": 0.0,
            "day_mean_net_r": 0.0,
            "hac_t": 0.0,
            "win_rate": 0.0,
        }
    values = [float(row["net_r"]) for row in rows]
    days = [str(row[day_key]) for row in rows]
    hac = _hac_day_mean_summary(values, days, hac_lags=DEPENDENCE_LAGS)
    return {
        "rows": len(rows),
        "days": int(hac["n_days"]),
        "row_mean_net_r": round(float(np.mean(values)), 6),
        "day_mean_net_r": round(float(hac["mean_of_day_means"]), 6),
        "hac_t": round(float(hac["t_stat"]), 4),
        "win_rate": round(float(np.mean([value > 0 for value in values])), 6),
        "dependence_method": hac["method"],
    }


def _paired_margin(rows: list[dict], left_key: str, right_key: str, *, day_key: str) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[day_key])].append(float(row[left_key]) - float(row[right_key]))
    days = sorted(grouped)
    if not days:
        return {"days": 0, "day_mean_margin_r": 0.0, "hac_t": 0.0}
    margins = [float(np.mean(grouped[day])) for day in days]
    hac = _hac_day_mean_summary(margins, days, hac_lags=DEPENDENCE_LAGS)
    return {
        "days": int(hac["n_days"]),
        "day_mean_margin_r": round(float(hac["mean_of_day_means"]), 6),
        "hac_t": round(float(hac["t_stat"]), 4),
        "dependence_method": hac["method"],
    }


def _day_margin(left_rows: list[dict], right_rows: list[dict]) -> dict:
    def day_means(rows: list[dict]) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["entry_day"])].append(float(row["net_r"]))
        return {day: float(np.mean(values)) for day, values in grouped.items()}

    left = day_means(left_rows)
    right = day_means(right_rows)
    days = sorted(set(left) & set(right))
    if not days:
        return {"days": 0, "day_mean_margin_r": 0.0, "hac_t": 0.0}
    margins = [left[day] - right[day] for day in days]
    hac = _hac_day_mean_summary(margins, days, hac_lags=DEPENDENCE_LAGS)
    return {
        "days": int(hac["n_days"]),
        "day_mean_margin_r": round(float(hac["mean_of_day_means"]), 6),
        "hac_t": round(float(hac["t_stat"]), 4),
        "dependence_method": hac["method"],
    }


def _outcome(
    frame: pd.DataFrame,
    entry_pos: int,
    direction: str,
    entry: float,
    risk_pct: float,
) -> dict | None:
    future = frame.iloc[entry_pos + 1 : entry_pos + 1 + HORIZON_BARS]
    if len(future) < HORIZON_BARS or entry <= 0 or risk_pct <= 0:
        return None
    return walk_triple_barrier(future, direction, entry, risk_pct, None)


def _confirmation(
    frame: pd.DataFrame,
    signal_pos: int,
    direction: str,
    level: float,
    tolerance: float,
    original_entry: float,
) -> tuple[int, int] | None:
    for delay in range(1, WAIT_BARS + 1):
        pos = signal_pos + delay
        bar = frame.iloc[pos]
        low = float(bar["Low"])
        high = float(bar["High"])
        close = float(bar["Close"])
        intersects = low <= level + tolerance and high >= level - tolerance
        if direction == "bullish":
            confirmed = intersects and close > level and close <= original_entry
        else:
            confirmed = intersects and close < level and close >= original_entry
        if confirmed:
            return pos, delay
    return None


def main() -> int:
    if not CACHE_PATH.exists():
        raise SystemExit(f"missing immutable experiment cache: {CACHE_PATH}")
    index_path = Path(EDGE_INDEX_PATH)
    records = json.loads(index_path.read_text(encoding="utf-8"))
    bars: dict[str, pd.DataFrame] = pickle.loads(CACHE_PATH.read_bytes())["bars"]

    positions: dict[str, dict[pd.Timestamp, int]] = {}
    positions_by_day: dict[str, dict[str, int]] = {}
    for ticker, frame in bars.items():
        ordered = frame.sort_index()
        bars[ticker] = ordered
        positions[ticker] = {pd.Timestamp(ts): pos for pos, ts in enumerate(ordered.index)}
        positions_by_day[ticker] = {str(pd.Timestamp(ts).date()): pos for pos, ts in enumerate(ordered.index)}

    fixed: list[dict] = []
    exclusion_counts: Counter[str] = Counter()
    embargo_until: dict[tuple[str, str], int] = {}
    ordered_records = sorted(records, key=lambda row: (row["ticker"], row["direction"], row["timestamp"]))
    for record in ordered_records:
        features = record.get("features") or {}
        if not bool(features.get("potter_passed")):
            exclusion_counts["not_source_valid_setup"] += 1
            continue
        ticker = str(record["ticker"])
        direction = str(record["direction"])
        if ticker not in bars or direction not in {"bullish", "bearish"}:
            exclusion_counts["missing_bars_or_direction"] += 1
            continue
        signal_pos = positions[ticker].get(pd.Timestamp(record["timestamp"]))
        if signal_pos is None:
            exclusion_counts["timestamp_not_in_cache"] += 1
            continue
        if len(bars[ticker]) - signal_pos - 1 < WAIT_BARS + HORIZON_BARS:
            exclusion_counts["incomplete_wait_or_outcome_window"] += 1
            continue
        risk_pct = float(record.get("risk_pct_used") or 0.0)
        if risk_pct <= 0:
            exclusion_counts["invalid_original_risk"] += 1
            continue
        key = (ticker, direction)
        if signal_pos <= embargo_until.get(key, -1):
            exclusion_counts["overlapping_trigger"] += 1
            continue
        embargo_until[key] = signal_pos + WAIT_BARS + HORIZON_BARS

        frame = bars[ticker]
        original_entry = float(frame["Close"].iloc[signal_pos])
        level_key = "box_top" if direction == "bullish" else "box_bottom"
        level = float(features.get(level_key) or 0.0)
        tolerance = float(features.get("touch_tolerance") or 0.0)
        if original_entry <= 0 or level <= 0 or tolerance < 0:
            exclusion_counts["invalid_control_geometry"] += 1
            continue
        fixed.append(
            {
                "ticker": ticker,
                "direction": direction,
                "signal_pos": signal_pos,
                "signal_day": str(pd.Timestamp(frame.index[signal_pos]).date()),
                "original_entry": original_entry,
                "original_risk_pct": risk_pct,
                "level": level,
                "tolerance": tolerance,
            }
        )

    event_rows: list[dict] = []
    no_confirmation: Counter[str] = Counter()
    invalid_retest_risk: Counter[str] = Counter()
    for trigger in fixed:
        ticker = trigger["ticker"]
        direction = trigger["direction"]
        frame = bars[ticker]
        found = _confirmation(
            frame,
            trigger["signal_pos"],
            direction,
            trigger["level"],
            trigger["tolerance"],
            trigger["original_entry"],
        )
        if found is None:
            no_confirmation[direction] += 1
            continue
        entry_pos, delay = found
        retest_entry = float(frame["Close"].iloc[entry_pos])
        sign = 1.0 if direction == "bullish" else -1.0
        stop_price = trigger["original_entry"] * (
            1.0 - sign * trigger["original_risk_pct"] / 100.0
        )
        retest_risk_pct = sign * (retest_entry - stop_price) / retest_entry * 100.0
        if retest_risk_pct <= 0:
            invalid_retest_risk[direction] += 1
            continue
        retest_outcome = _outcome(frame, entry_pos, direction, retest_entry, retest_risk_pct)
        immediate_outcome = _outcome(
            frame,
            trigger["signal_pos"],
            direction,
            trigger["original_entry"],
            trigger["original_risk_pct"],
        )
        if retest_outcome is None or immediate_outcome is None:
            exclusion_counts["unexpected_incomplete_outcome"] += 1
            continue
        improvement_pct = sign * (trigger["original_entry"] - retest_entry) / trigger["original_entry"] * 100.0
        event_rows.append(
            {
                **trigger,
                "entry_pos": entry_pos,
                "entry_day": str(pd.Timestamp(frame.index[entry_pos]).date()),
                "delay_bars": delay,
                "retest_entry": retest_entry,
                "retest_risk_pct": retest_risk_pct,
                "entry_improvement_pct": improvement_pct,
                "net_r": _net_r(retest_outcome["return_pct"], retest_risk_pct),
                "immediate_net_r": _net_r(
                    immediate_outcome["return_pct"], trigger["original_risk_pct"]
                ),
                "retest_exit_reason": retest_outcome["exit_reason"],
                "immediate_exit_reason": immediate_outcome["exit_reason"],
            }
        )

    placebo_rows: list[dict] = []
    for direction in ("bullish", "bearish"):
        cohort = [row for row in event_rows if row["direction"] == direction]
        entry_days = sorted({row["entry_day"] for row in cohort})
        risks_by_ticker: dict[str, float] = {}
        for ticker in sorted({row["ticker"] for row in cohort}):
            risks = [row["retest_risk_pct"] for row in cohort if row["ticker"] == ticker]
            risks_by_ticker[ticker] = float(median(risks))
        for day in entry_days:
            for ticker, risk_pct in risks_by_ticker.items():
                pos = positions_by_day[ticker].get(day)
                if pos is None:
                    continue
                frame = bars[ticker]
                entry = float(frame["Close"].iloc[pos])
                outcome = _outcome(frame, pos, direction, entry, risk_pct)
                if outcome is None:
                    continue
                placebo_rows.append(
                    {
                        "ticker": ticker,
                        "direction": direction,
                        "entry_day": day,
                        "net_r": _net_r(outcome["return_pct"], risk_pct),
                    }
                )

    by_direction: dict[str, dict] = {}
    for direction in ("bullish", "bearish"):
        triggers = [row for row in fixed if row["direction"] == direction]
        events = [row for row in event_rows if row["direction"] == direction]
        placebo = [row for row in placebo_rows if row["direction"] == direction]
        delay_counts = Counter(int(row["delay_bars"]) for row in events)
        by_direction[direction] = {
            "eligible_triggers": len(triggers),
            "confirmed_entries": len(events),
            "confirmation_rate": round(len(events) / len(triggers), 6) if triggers else 0.0,
            "no_confirmation": int(no_confirmation[direction]),
            "invalid_retest_risk": int(invalid_retest_risk[direction]),
            "delay_counts": {str(key): value for key, value in sorted(delay_counts.items())},
            "median_entry_improvement_pct": round(
                float(median([row["entry_improvement_pct"] for row in events])), 6
            ) if events else 0.0,
            "retest": _stats(events, day_key="entry_day"),
            "immediate_same_setups": _stats(
                [{**row, "net_r": row["immediate_net_r"]} for row in events],
                day_key="signal_day",
            ),
            "retest_minus_immediate": _paired_margin(
                events, "net_r", "immediate_net_r", day_key="signal_day"
            ),
            "drift_placebo": _stats(placebo, day_key="entry_day"),
            "retest_minus_drift": _day_margin(events, placebo),
        }

    primary = by_direction["bullish"]
    gates = {
        "coverage": (
            primary["confirmed_entries"] >= 100
            and primary["retest"]["days"] >= 40
        ),
        "absolute": (
            primary["retest"]["day_mean_net_r"] > 0
            and primary["retest"]["hac_t"] >= 2.0
        ),
        "same_setup": (
            primary["retest_minus_immediate"]["day_mean_margin_r"] > 0
            and primary["retest_minus_immediate"]["hac_t"] >= 2.0
        ),
        "drift": (
            primary["retest_minus_drift"]["day_mean_margin_r"] > 0
            and primary["retest_minus_drift"]["hac_t"] >= 2.0
        ),
    }
    result = {
        "experiment": "post_breakout_retest_entry",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "verdict": "pass_for_shadow_confirmation" if all(gates.values()) else "falsified_no_change",
        "production_change_authorized": False,
        "source": {
            "edge_index_path": str(index_path.resolve()),
            "edge_index_sha256": _sha256(index_path),
            "edge_index_records": len(records),
            "bars_cache_path": str(CACHE_PATH.resolve()),
            "bars_cache_sha256": _sha256(CACHE_PATH),
            "bars_cache_tickers": len(bars),
        },
        "config": {
            "wait_bars": WAIT_BARS,
            "horizon_bars": HORIZON_BARS,
            "dependence_lags": DEPENDENCE_LAGS,
            "cost_bps_per_side": EDGE_COST_BPS_PER_SIDE,
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "target_mode": "none",
        },
        "cohort": {
            "retained_non_overlapping_triggers": len(fixed),
            "exclusion_counts": dict(exclusion_counts),
        },
        "by_direction": by_direction,
        "primary_gates": gates,
        "all_primary_gates_passed": all(gates.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
