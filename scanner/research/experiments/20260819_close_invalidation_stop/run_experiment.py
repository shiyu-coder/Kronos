"""Pre-registered Potter cost-basis close-invalidation experiment.

Prints JSON only and deliberately does not mutate production reports, tuning,
alerts, or evidence. The rule and gates are pinned in preregistration.json.
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


HERE = Path(__file__).parent
CACHE_PATH = (
    REPO_ROOT
    / "scanner"
    / "research"
    / "experiments"
    / "20260813_horizon_sweep"
    / "bars_cache.pkl"
)
HORIZON_BARS = 5
ROUND_TRIP_COST_PCT = 2.0 * EDGE_COST_BPS_PER_SIDE / 100.0
R_CAP = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _net_r(return_pct: float, risk_pct: float) -> float:
    return float(np.clip((float(return_pct) - ROUND_TRIP_COST_PCT) / float(risk_pct), -R_CAP, R_CAP))


def _close_invalidation(
    path: pd.DataFrame,
    direction: str,
    entry: float,
    risk_pct: float,
) -> dict:
    sign = 1.0 if direction == "bullish" else -1.0
    stop_price = entry * (1.0 - sign * risk_pct / 100.0)
    exit_reason = "horizon"
    exit_price = float(path["Close"].iloc[-1])
    exit_pos = len(path) - 1
    for pos, close in enumerate(path["Close"].astype(float)):
        invalidated = close <= stop_price if direction == "bullish" else close >= stop_price
        if invalidated:
            exit_reason = "close_invalidation"
            exit_price = float(close)
            exit_pos = pos
            break
    ret_pct = sign * ((exit_price - entry) / entry) * 100.0
    return {
        "return_pct": ret_pct,
        "r_multiple": float(np.clip(ret_pct / risk_pct, -R_CAP, R_CAP)),
        "exit_reason": exit_reason,
        "exit_pos": exit_pos,
    }


def _stats(rows: list[dict], value_key: str = "treatment_net_r") -> dict:
    if not rows:
        return {"rows": 0, "days": 0, "row_mean_net_r": 0.0, "day_mean_net_r": 0.0, "hac_t": 0.0}
    values = [float(row[value_key]) for row in rows]
    days = [str(row["entry_day"]) for row in rows]
    hac = _hac_day_mean_summary(values, days, hac_lags=HORIZON_BARS)
    return {
        "rows": len(rows),
        "days": int(hac["n_days"]),
        "row_mean_net_r": round(float(np.mean(values)), 6),
        "day_mean_net_r": round(float(hac["mean_of_day_means"]), 6),
        "hac_t": round(float(hac["t_stat"]), 4),
        "win_rate": round(float(np.mean([value > 0 for value in values])), 6),
        "dependence_method": hac["method"],
    }


def _paired_margin(rows: list[dict], left_key: str, right_key: str) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["entry_day"])].append(float(row[left_key]) - float(row[right_key]))
    days = sorted(grouped)
    values = [float(np.mean(grouped[day])) for day in days]
    if not values:
        return {"days": 0, "day_mean_margin_r": 0.0, "hac_t": 0.0}
    hac = _hac_day_mean_summary(values, days, hac_lags=HORIZON_BARS)
    return {
        "days": int(hac["n_days"]),
        "day_mean_margin_r": round(float(hac["mean_of_day_means"]), 6),
        "hac_t": round(float(hac["t_stat"]), 4),
        "dependence_method": hac["method"],
    }


def _day_margin(left_rows: list[dict], right_rows: list[dict]) -> dict:
    def means(rows: list[dict]) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["entry_day"])].append(float(row["treatment_net_r"]))
        return {day: float(np.mean(values)) for day, values in grouped.items()}

    left = means(left_rows)
    right = means(right_rows)
    days = sorted(set(left) & set(right))
    values = [left[day] - right[day] for day in days]
    if not values:
        return {"days": 0, "day_mean_margin_r": 0.0, "hac_t": 0.0}
    hac = _hac_day_mean_summary(values, days, hac_lags=HORIZON_BARS)
    return {
        "days": int(hac["n_days"]),
        "day_mean_margin_r": round(float(hac["mean_of_day_means"]), 6),
        "hac_t": round(float(hac["t_stat"]), 4),
        "dependence_method": hac["method"],
    }


def main() -> int:
    prereg = json.loads((HERE / "preregistration.json").read_text(encoding="utf-8"))
    index_path = Path(EDGE_INDEX_PATH)
    expected_index_hash = prereg["pinned_inputs"]["edge_index_sha256"]
    expected_cache_hash = prereg["pinned_inputs"]["bars_cache_sha256"]
    actual_index_hash = _sha256(index_path)
    actual_cache_hash = _sha256(CACHE_PATH)
    if actual_index_hash != expected_index_hash or actual_cache_hash != expected_cache_hash:
        raise SystemExit(
            "pinned input hash mismatch; refusing to run: "
            f"index={actual_index_hash} cache={actual_cache_hash}"
        )

    records = json.loads(index_path.read_text(encoding="utf-8"))
    bars: dict[str, pd.DataFrame] = pickle.loads(CACHE_PATH.read_bytes())["bars"]
    positions: dict[str, dict[pd.Timestamp, int]] = {}
    positions_by_day: dict[str, dict[str, int]] = {}
    for ticker, frame in bars.items():
        ordered = frame.sort_index()
        bars[ticker] = ordered
        positions[ticker] = {pd.Timestamp(ts): pos for pos, ts in enumerate(ordered.index)}
        positions_by_day[ticker] = {str(pd.Timestamp(ts).date()): pos for pos, ts in enumerate(ordered.index)}

    prepared: list[dict] = []
    exclusions: Counter[str] = Counter()
    embargo_until: dict[tuple[str, str], int] = {}
    ordered_records = sorted(records, key=lambda row: (row["ticker"], row["direction"], row["timestamp"]))
    for record in ordered_records:
        features = record.get("features") or {}
        if int(features.get("feature_version") or 0) != 4:
            exclusions["wrong_feature_version"] += 1
            continue
        if not bool(features.get("potter_passed")):
            exclusions["not_source_valid_setup"] += 1
            continue
        ticker = str(record.get("ticker"))
        direction = str(record.get("direction"))
        if ticker not in bars or direction not in {"bullish", "bearish"}:
            exclusions["missing_bars_or_direction"] += 1
            continue
        signal_pos = positions[ticker].get(pd.Timestamp(record["timestamp"]))
        risk_pct = float(record.get("risk_pct_used") or 0.0)
        if signal_pos is None:
            exclusions["timestamp_not_in_cache"] += 1
            continue
        if len(bars[ticker]) - signal_pos - 1 < HORIZON_BARS:
            exclusions["incomplete_outcome_window"] += 1
            continue
        if risk_pct <= 0:
            exclusions["invalid_risk"] += 1
            continue
        key = (ticker, direction)
        if signal_pos <= embargo_until.get(key, -1):
            exclusions["overlapping_trigger"] += 1
            continue
        embargo_until[key] = signal_pos + HORIZON_BARS

        frame = bars[ticker]
        entry = float(frame["Close"].iloc[signal_pos])
        future = frame.iloc[signal_pos + 1 : signal_pos + 1 + HORIZON_BARS]
        production = walk_triple_barrier(future, direction, entry, risk_pct, None)
        treatment = _close_invalidation(future, direction, entry, risk_pct)
        prepared.append(
            {
                "ticker": ticker,
                "direction": direction,
                "entry_day": str(pd.Timestamp(frame.index[signal_pos]).date()),
                "entry": entry,
                "risk_pct": risk_pct,
                "stored_gross_r": float(record.get("r_multiple") or 0.0),
                "rewalk_gross_r": float(production["r_multiple"]),
                "production_net_r": _net_r(production["return_pct"], risk_pct),
                "treatment_net_r": _net_r(treatment["return_pct"], risk_pct),
                "production_exit": str(production["exit_reason"]),
                "treatment_exit": str(treatment["exit_reason"]),
            }
        )

    stored = np.asarray([row["stored_gross_r"] for row in prepared], dtype=float)
    rewalk = np.asarray([row["rewalk_gross_r"] for row in prepared], dtype=float)
    delta_mean = float(np.mean(rewalk) - np.mean(stored))
    correlation = float(np.corrcoef(rewalk, stored)[0, 1])
    baseline_control = {
        "rows": len(prepared),
        "delta_mean_r": round(delta_mean, 8),
        "correlation": round(correlation, 8),
        "passed": abs(delta_mean) <= 0.005 and correlation >= 0.99,
    }
    if not baseline_control["passed"]:
        raise SystemExit(f"baseline reproduction failed before treatment read: {baseline_control}")

    placebo_rows: list[dict] = []
    for direction in ("bullish", "bearish"):
        cohort = [row for row in prepared if row["direction"] == direction]
        entry_days = sorted({row["entry_day"] for row in cohort})
        risks_by_ticker = {
            ticker: float(median([row["risk_pct"] for row in cohort if row["ticker"] == ticker]))
            for ticker in sorted({row["ticker"] for row in cohort})
        }
        for day in entry_days:
            for ticker, risk_pct in risks_by_ticker.items():
                pos = positions_by_day[ticker].get(day)
                if pos is None or len(bars[ticker]) - pos - 1 < HORIZON_BARS:
                    continue
                frame = bars[ticker]
                entry = float(frame["Close"].iloc[pos])
                future = frame.iloc[pos + 1 : pos + 1 + HORIZON_BARS]
                treatment = _close_invalidation(future, direction, entry, risk_pct)
                placebo_rows.append(
                    {
                        "ticker": ticker,
                        "direction": direction,
                        "entry_day": day,
                        "treatment_net_r": _net_r(treatment["return_pct"], risk_pct),
                    }
                )

    by_direction: dict[str, dict] = {}
    for direction in ("bullish", "bearish"):
        cohort = [row for row in prepared if row["direction"] == direction]
        placebo = [row for row in placebo_rows if row["direction"] == direction]
        exit_transitions = Counter(
            f"{row['production_exit']}->{row['treatment_exit']}" for row in cohort
        )
        by_direction[direction] = {
            "treatment": _stats(cohort),
            "production_same_setups": _stats(cohort, value_key="production_net_r"),
            "treatment_minus_production": _paired_margin(
                cohort, "treatment_net_r", "production_net_r"
            ),
            "drift_placebo": _stats(placebo),
            "treatment_minus_drift": _day_margin(cohort, placebo),
            "median_risk_pct": round(float(median([row["risk_pct"] for row in cohort])), 6)
            if cohort
            else 0.0,
            "exit_transitions": dict(sorted(exit_transitions.items())),
        }

    primary = by_direction["bullish"]
    gates = {
        "coverage": primary["treatment"]["rows"] >= 100 and primary["treatment"]["days"] >= 40,
        "absolute": primary["treatment"]["day_mean_net_r"] > 0 and primary["treatment"]["hac_t"] >= 2.0,
        "same_setup": primary["treatment_minus_production"]["day_mean_margin_r"] > 0
        and primary["treatment_minus_production"]["hac_t"] >= 2.0,
        "drift": primary["treatment_minus_drift"]["day_mean_margin_r"] > 0
        and primary["treatment_minus_drift"]["hac_t"] >= 2.0,
    }
    result = {
        "experiment": "close_based_cost_basis_invalidation",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "verdict": "pass_for_shadow_confirmation" if all(gates.values()) else "falsified_no_change",
        "production_change_authorized": False,
        "source": {
            "edge_index_sha256": actual_index_hash,
            "edge_index_records": len(records),
            "bars_cache_sha256": actual_cache_hash,
            "bars_cache_tickers": len(bars),
        },
        "config": {
            "horizon_bars": HORIZON_BARS,
            "cost_bps_per_side": EDGE_COST_BPS_PER_SIDE,
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "target_mode": "none",
            "stop_level": "potter_cost_basis",
            "invalidation_observation": "daily_close",
        },
        "baseline_reproduction": baseline_control,
        "cohort": {"retained_non_overlapping_rows": len(prepared), "exclusion_counts": dict(exclusions)},
        "by_direction": by_direction,
        "primary_gates": gates,
        "all_primary_gates_passed": all(gates.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
