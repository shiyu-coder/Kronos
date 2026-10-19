"""W2 compact5 robustness battery.

Stress-tests the E3 finding (scanner/research/experiments/20260710_sprint/
e3_features/): dropping the box-geometry/compression cluster down to a
compact5 feature set cuts tail_prob OOF rank IC from -0.0664 (10-key
control) to -0.0233 (still fails gates). This script does NOT tune for a
better number - it checks whether that improvement survives lambda
variation, an objective swap, a time split, and a purge-window change.

Read-only against scanner/edge/*. Reuses _standardize_train,
_fit_logistic_irls, _feature_matrix, predict_score straight from
scanner.edge.calibration, same as E3's harness. The walk-forward loop is
split into a prediction pass (_walk_forward_predict) and a metrics pass
(_compute_metrics) so the same OOF run can be sliced into temporal halves
without refitting anything.

Writes only inside this experiment folder: results.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scanner.config import EDGE_INDEX_PATH
from scanner.edge.calibration import (
    META_ACCEPT_MIN_N,
    META_L2_LAMBDA,
    META_MIN_CLASS_EVENTS,
    META_MIN_TRAIN,
    META_PURGE_DAYS,
    META_REFIT_EVERY_DAYS,
    META_TAIL_R,
    _feature_matrix,
    _fit_logistic_irls,
    _finite,
    _standardize_train,
    predict_score,
)
from scanner.edge.retrieval import load_edge_index
from scanner.edge.stats import spearman_rank_ic, tail_retention, tercile_lift

EXPERIMENT_DIR = Path(__file__).resolve().parent

STANDARD_10 = (
    "volume_expansion",
    "volume_percentile",
    "breakout_strength_pct",
    "close_position_in_box",
    "box_width_pct",
    "range_compression_ratio",
    "realized_volatility_pct",
    "no_trend_score",
    "doctrine_v2_score",
    "recent_return_pct",
)

COMPACT5 = (
    "volume_expansion",
    "volume_percentile",
    "breakout_strength_pct",
    "realized_volatility_pct",
    "recent_return_pct",
)

GEOMETRY5 = (
    "close_position_in_box",
    "box_width_pct",
    "range_compression_ratio",
    "no_trend_score",
    "doctrine_v2_score",
)

FEATURE_SETS = {"standard_10": STANDARD_10, "compact5": COMPACT5, "geometry5": GEOMETRY5}


def _brier(probabilities: list[float], labels: list[int]) -> float:
    pairs = [(p, y) for p, y in zip(probabilities, labels, strict=False) if p is not None]
    if not pairs:
        return 1.0
    return float(np.mean([(p - y) ** 2 for p, y in pairs]))


def fit_model_generic(
    feature_dicts: list[dict],
    r_multiples: list[float],
    feature_keys: tuple[str, ...],
    objective: str = "tail_prob",
    tail_r: float = META_TAIL_R,
    l2_lambda: float = META_L2_LAMBDA,
    min_train: int = META_MIN_TRAIN,
    min_class_events: int = META_MIN_CLASS_EVENTS,
) -> dict | None:
    """tail_prob or p_win objective. Mirrors calibration.fit_model's logistic
    branch, generalized over feature_keys/objective/lambda."""
    raw = _feature_matrix(feature_dicts, feature_keys)
    r_values = np.array([_finite(r) for r in r_multiples], dtype=float)
    usable = np.isfinite(r_values)
    raw = raw[usable]
    r_values = r_values[usable]
    n = len(r_values)
    if n < min_train:
        return None

    if objective == "tail_prob":
        y = (r_values >= tail_r).astype(float)
    elif objective == "p_win":
        y = (r_values > 0.0).astype(float)
    else:
        raise ValueError(f"unsupported objective: {objective}")

    positives = float(y.sum())
    if positives < min_class_events or (n - positives) < min_class_events:
        return None

    transform = _standardize_train(raw)
    if transform is None:
        return None
    design = np.hstack([np.ones((n, 1)), transform["x"]])
    weights = _fit_logistic_irls(design, y, l2_lambda)
    if not np.isfinite(weights).all():
        return None

    return {
        "objective": objective,
        "feature_keys": list(feature_keys),
        "l2_lambda": float(l2_lambda),
        "intercept": float(weights[0]),
        "coefficients": [float(w) for w in weights[1:]],
        "winsor_low": [float(v) for v in transform["lo"]],
        "winsor_high": [float(v) for v in transform["hi"]],
        "medians": [float(v) for v in transform["medians"]],
        "means": [float(v) for v in transform["mean"]],
        "stds": [float(v) for v in transform["std"]],
        "n_train": int(n),
    }


def _walk_forward_predict(
    records,
    direction: str,
    feature_keys: tuple[str, ...],
    objective: str = "tail_prob",
    tail_r: float = META_TAIL_R,
    l2_lambda: float = META_L2_LAMBDA,
    refit_every_days: int = META_REFIT_EVERY_DAYS,
    purge_days: int = META_PURGE_DAYS,
    min_train: int = META_MIN_TRAIN,
    min_class_events: int = META_MIN_CLASS_EVENTS,
) -> dict:
    """Expanding-window OOF prediction pass only - no metrics. Direct
    generalization of calibration.walk_forward_calibration's loop. Returns
    raw per-row arrays so the same run can be sliced (e.g. by day, for the
    temporal-stability cells) without refitting."""
    rows = []
    for record in records:
        rec_direction = getattr(record, "direction", None) if not isinstance(record, dict) else record.get("direction")
        if str(rec_direction) != direction:
            continue
        features = getattr(record, "features", None) if not isinstance(record, dict) else record.get("features")
        timestamp = getattr(record, "timestamp", None) if not isinstance(record, dict) else record.get("timestamp")
        r_multiple = getattr(record, "r_multiple", None) if not isinstance(record, dict) else record.get("r_multiple")
        ticker = getattr(record, "ticker", "") if not isinstance(record, dict) else record.get("ticker", "")
        ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
        r_value = _finite(r_multiple)
        if pd.isna(ts) or not isinstance(features, dict) or not np.isfinite(r_value):
            continue
        rows.append({"ts": ts, "ticker": str(ticker), "timestamp": str(timestamp), "features": features, "r": r_value})

    rows.sort(key=lambda row: row["ts"])
    if len(rows) < min_train + 50:
        return {"insufficient": True, "reason": "insufficient_records", "n_records": len(rows)}

    purge = pd.Timedelta(days=purge_days)
    refit_interval = pd.Timedelta(days=refit_every_days)
    model = None
    model_fit_ts = None
    predictions: list[float] = []
    outcomes: list[float] = []
    tail_labels: list[int] = []
    win_labels: list[int] = []
    day_keys: list[str] = []
    row_ids: list[str] = []

    for row in rows:
        needs_refit = model_fit_ts is None or (row["ts"] - model_fit_ts) >= refit_interval
        if needs_refit:
            train = [r for r in rows if r["ts"] <= row["ts"] - purge]
            if len(train) >= min_train:
                candidate = fit_model_generic(
                    [r["features"] for r in train],
                    [r["r"] for r in train],
                    feature_keys,
                    objective=objective,
                    tail_r=tail_r,
                    l2_lambda=l2_lambda,
                    min_train=min_train,
                    min_class_events=min_class_events,
                )
                if candidate is not None:
                    model = candidate
                    model_fit_ts = row["ts"]
        if model is None:
            continue
        score = predict_score(model, row["features"])
        if score is None:
            continue
        predictions.append(score)
        outcomes.append(row["r"])
        tail_labels.append(1 if row["r"] >= tail_r else 0)
        win_labels.append(1 if row["r"] > 0 else 0)
        day_keys.append(row["ts"].strftime("%Y-%m-%d"))
        row_ids.append(f"{row['ticker']}|{row['timestamp']}")

    return {
        "insufficient": False,
        "n_records": len(rows),
        "predictions": predictions,
        "outcomes": outcomes,
        "tail_labels": tail_labels,
        "win_labels": win_labels,
        "day_keys": day_keys,
        "row_ids": row_ids,
    }


def _compute_metrics(
    predictions: list[float],
    outcomes: list[float],
    day_keys: list[str],
    row_ids: list[str],
    tail_labels: list[int],
    win_labels: list[int],
    objective: str,
    tail_r: float = META_TAIL_R,
    min_accept_n: int = META_ACCEPT_MIN_N,
) -> dict:
    """Same metric/acceptance computation as calibration.walk_forward_calibration,
    factored out so it can run on a full OOF set or a temporal slice of one."""
    n_eval = len(predictions)
    if n_eval < min_accept_n:
        return {
            "n_evaluated": n_eval,
            "metrics": {"insufficient": True, "n_evaluated": n_eval},
            "acceptance": {"passed": False, "reason": "insufficient_out_of_fold_predictions"},
        }

    ic = spearman_rank_ic(predictions, outcomes, day_keys=day_keys)
    lift = tercile_lift(predictions, outcomes, day_keys, row_ids=row_ids)
    tail = tail_retention(predictions, outcomes, row_ids=row_ids, tail_r=tail_r)

    labels = tail_labels if objective == "tail_prob" else win_labels
    base_rate = float(np.mean(labels))
    oof_loss = _brier(predictions, labels)
    naive_loss = float(np.mean([(base_rate - y) ** 2 for y in labels]))
    beats_naive = oof_loss < naive_loss

    metrics = {
        "insufficient": False,
        "n_evaluated": n_eval,
        "rank_ic_r": ic,
        "tercile_lift": lift,
        "tail_retention": tail,
        "oof_loss": round(oof_loss, 6),
        "naive_loss": round(naive_loss, 6),
        "beats_naive": beats_naive,
        "mean_score": round(float(np.mean(predictions)), 4),
        "realized_win_rate": round(float(np.mean(win_labels)), 4),
        "realized_tail_rate": round(float(np.mean(tail_labels)), 4),
    }

    from scanner.edge.calibration import META_ACCEPT_MAX_P_DAY, META_ACCEPT_MIN_IC

    criteria = {
        "ic_at_least_0.07": float(ic.get("ic", 0.0)) >= META_ACCEPT_MIN_IC,
        "day_clustered_p_at_most_0.05": float(ic.get("p_value_day_clustered", 1.0)) <= META_ACCEPT_MAX_P_DAY,
        "n_at_least_300": n_eval >= META_ACCEPT_MIN_N,
        "tercile_spread_ci_low_positive": bool(
            not lift.get("insufficient") and lift.get("spread_ci_low") is not None and lift["spread_ci_low"] > 0
        ),
        "tail_retention_at_least_pro_rata": bool(
            tail.get("insufficient")
            or tail.get("observed_share") is None
            or tail["observed_share"] >= tail["expected_share"]
        ),
        "beats_naive": beats_naive,
    }
    return {
        "n_evaluated": n_eval,
        "metrics": metrics,
        "acceptance": {"passed": all(criteria.values()), "criteria": criteria},
    }


def walk_forward_generic(
    records,
    direction: str,
    feature_keys: tuple[str, ...],
    objective: str = "tail_prob",
    tail_r: float = META_TAIL_R,
    l2_lambda: float = META_L2_LAMBDA,
    refit_every_days: int = META_REFIT_EVERY_DAYS,
    purge_days: int = META_PURGE_DAYS,
    min_train: int = META_MIN_TRAIN,
    min_class_events: int = META_MIN_CLASS_EVENTS,
    min_accept_n: int = META_ACCEPT_MIN_N,
) -> dict:
    """Full run: predict pass + metrics pass on the whole OOF set."""
    pred = _walk_forward_predict(
        records,
        direction,
        feature_keys,
        objective=objective,
        tail_r=tail_r,
        l2_lambda=l2_lambda,
        refit_every_days=refit_every_days,
        purge_days=purge_days,
        min_train=min_train,
        min_class_events=min_class_events,
    )
    result: dict[str, Any] = {
        "direction": direction,
        "objective": objective,
        "feature_keys": list(feature_keys),
        "config": {
            "l2_lambda": l2_lambda,
            "refit_every_days": refit_every_days,
            "purge_days": purge_days,
            "min_train": min_train,
            "tail_r": tail_r,
        },
        "n_records": pred.get("n_records", 0),
    }
    if pred.get("insufficient"):
        result["n_evaluated"] = 0
        result["metrics"] = {"insufficient": True}
        result["acceptance"] = {"passed": False, "reason": pred["reason"]}
        return result

    computed = _compute_metrics(
        pred["predictions"],
        pred["outcomes"],
        pred["day_keys"],
        pred["row_ids"],
        pred["tail_labels"],
        pred["win_labels"],
        objective=objective,
        tail_r=tail_r,
        min_accept_n=min_accept_n,
    )
    result.update(computed)
    result["_raw"] = pred  # stripped before writing to disk; used for temporal split
    return result


def _slice_by_days(pred: dict, wanted_days: set[str]) -> dict:
    idx = [i for i, d in enumerate(pred["day_keys"]) if d in wanted_days]
    return {
        "predictions": [pred["predictions"][i] for i in idx],
        "outcomes": [pred["outcomes"][i] for i in idx],
        "tail_labels": [pred["tail_labels"][i] for i in idx],
        "win_labels": [pred["win_labels"][i] for i in idx],
        "day_keys": [pred["day_keys"][i] for i in idx],
        "row_ids": [pred["row_ids"][i] for i in idx],
    }


def temporal_split_cells(pred: dict, objective: str, tail_r: float) -> tuple[dict, dict, dict]:
    """Split one OOF run's predictions by median OOF day into first/second
    half evaluation windows. Returns (first_half_result, second_half_result,
    split_meta). No refit - same predictions, two disjoint day sets."""
    unique_days = sorted(set(pred["day_keys"]))
    n_days = len(unique_days)
    median_idx = n_days // 2
    median_day = unique_days[median_idx]
    first_days = set(unique_days[: median_idx + 1])  # inclusive of median day
    second_days = set(unique_days[median_idx + 1 :])

    first_slice = _slice_by_days(pred, first_days)
    second_slice = _slice_by_days(pred, second_days)

    first_result = _compute_metrics(
        first_slice["predictions"], first_slice["outcomes"], first_slice["day_keys"],
        first_slice["row_ids"], first_slice["tail_labels"], first_slice["win_labels"],
        objective=objective, tail_r=tail_r,
    )
    second_result = _compute_metrics(
        second_slice["predictions"], second_slice["outcomes"], second_slice["day_keys"],
        second_slice["row_ids"], second_slice["tail_labels"], second_slice["win_labels"],
        objective=objective, tail_r=tail_r,
    )
    split_meta = {
        "n_unique_days_total": n_days,
        "median_day": median_day,
        "first_half_days": len(first_days),
        "second_half_days": len(second_days),
        "first_half_date_range": [unique_days[0], unique_days[median_idx]] if n_days else None,
        "second_half_date_range": [unique_days[median_idx + 1], unique_days[-1]] if median_idx + 1 < n_days else None,
    }
    return first_result, second_result, split_meta


def _strip_raw(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_raw"}


def main() -> None:
    prereg = json.loads((EXPERIMENT_DIR / "preregistration.json").read_text(encoding="utf-8"))
    records = list(load_edge_index(EDGE_INDEX_PATH))

    # ---- Step 1: harness sanity check (both E3 reference points) ----
    control_10key = walk_forward_generic(records, direction="bullish", feature_keys=STANDARD_10, objective="tail_prob", l2_lambda=1.0, purge_days=9)
    control_10key_ic = float(control_10key["metrics"]["rank_ic_r"]["ic"])

    control_compact5 = walk_forward_generic(records, direction="bullish", feature_keys=COMPACT5, objective="tail_prob", l2_lambda=1.0, purge_days=9)
    control_compact5_ic = float(control_compact5["metrics"]["rank_ic_r"]["ic"])

    sanity = {
        "10key_ic": control_10key_ic,
        "10key_n_evaluated": control_10key["n_evaluated"],
        "10key_expected_ic": -0.0664,
        "10key_within_tolerance_0.005": abs(control_10key_ic - (-0.0664)) <= 0.005,
        "compact5_ic": control_compact5_ic,
        "compact5_n_evaluated": control_compact5["n_evaluated"],
        "compact5_expected_ic": -0.0233,
        "compact5_within_tolerance_0.005": abs(control_compact5_ic - (-0.0233)) <= 0.005,
    }
    print("SANITY CHECK:", json.dumps(sanity, indent=2))
    if not sanity["10key_within_tolerance_0.005"]:
        raise SystemExit("Harness sanity check FAILED - 10-key control does not match E3 (-0.0664). Aborting.")
    if not sanity["compact5_within_tolerance_0.005"]:
        raise SystemExit("Harness sanity check FAILED - compact5 control does not match E3 (-0.0233). Aborting.")

    results: dict[str, Any] = {
        "sanity_check": sanity,
        "control_10key": _strip_raw(control_10key),
        "control_compact5": _strip_raw(control_compact5),
        "cells": {},
    }

    def delta(ic_value: float) -> float:
        return round(ic_value - control_10key_ic, 4)

    # ---- Cells 1-3: compact5, tail_prob, lambda sweep ----
    for lam in (0.3, 3.0, 10.0):
        cell_id = {0.3: 1, 3.0: 2, 10.0: 3}[lam]
        name = f"compact5_tail_prob_lambda_{lam}"
        print(f"\n--- Cell {cell_id}: {name} ---")
        outcome = walk_forward_generic(records, direction="bullish", feature_keys=COMPACT5, objective="tail_prob", l2_lambda=lam, purge_days=9)
        outcome = _strip_raw(outcome)
        if not outcome["metrics"].get("insufficient"):
            ic = float(outcome["metrics"]["rank_ic_r"]["ic"])
            outcome["delta_ic_vs_10key_control"] = delta(ic)
            print(f"  n={outcome['n_evaluated']} ic={ic} delta_vs_10key={outcome['delta_ic_vs_10key_control']} passed={outcome['acceptance']['passed']}")
        else:
            outcome["delta_ic_vs_10key_control"] = None
            print(f"  INSUFFICIENT")
        results["cells"][f"cell_{cell_id}_{name}"] = outcome

    # ---- Cell 4: compact5, p_win objective, lambda=1.0 ----
    print("\n--- Cell 4: compact5_p_win_lambda_1.0 ---")
    outcome4 = walk_forward_generic(records, direction="bullish", feature_keys=COMPACT5, objective="p_win", l2_lambda=1.0, purge_days=9)
    outcome4 = _strip_raw(outcome4)
    outcome4_ic = None
    if not outcome4["metrics"].get("insufficient"):
        outcome4_ic = float(outcome4["metrics"]["rank_ic_r"]["ic"])
        outcome4["note"] = "p_win objective is NOT comparable in absolute IC to the tail_prob 10key control (different y). See diagnostics.10key_p_win_control for the same-objective baseline."
        print(f"  n={outcome4['n_evaluated']} ic={outcome4_ic} passed={outcome4['acceptance']['passed']}")
    else:
        print("  INSUFFICIENT")
    results["cells"]["cell_4_compact5_p_win_lambda_1.0"] = outcome4

    # ---- Diagnostic (post-hoc, not one of the 8 pre-registered cells): the
    # tail_prob 10-key control above is NOT the right baseline for cell 4
    # (different objective/y). To answer "does dropping geometry help p_win
    # too" we need the p_win 10-key control on the same objective. ----
    print("\n--- Diagnostic: 10-key p_win control (same-objective baseline for cell 4) ---")
    diag_10key_pwin = walk_forward_generic(records, direction="bullish", feature_keys=STANDARD_10, objective="p_win", l2_lambda=1.0, purge_days=9)
    diag_10key_pwin = _strip_raw(diag_10key_pwin)
    diag_10key_pwin_ic = None
    if not diag_10key_pwin["metrics"].get("insufficient"):
        diag_10key_pwin_ic = float(diag_10key_pwin["metrics"]["rank_ic_r"]["ic"])
        print(f"  n={diag_10key_pwin['n_evaluated']} ic={diag_10key_pwin_ic}")
    if outcome4_ic is not None and diag_10key_pwin_ic is not None:
        outcome4["delta_ic_vs_10key_p_win_control"] = round(outcome4_ic - diag_10key_pwin_ic, 4)
        print(f"  cell4 compact5 p_win delta vs 10key p_win control: {outcome4['delta_ic_vs_10key_p_win_control']}")
    results["diagnostics"] = {
        "10key_p_win_control": diag_10key_pwin,
        "note": (
            "Post-hoc, not one of the 8 pre-registered cells. Added because the pre-registered "
            "control_10key in this results file is objective=tail_prob and is not a valid baseline "
            "for cell 4's p_win compact5 result. This runs the same 10-key set under objective=p_win "
            "so cell 4 has an apples-to-apples comparison."
        ),
    }

    # ---- Cell 5: geometry5-only, tail_prob, lambda=1.0 ----
    print("\n--- Cell 5: geometry5_tail_prob_lambda_1.0 ---")
    outcome5 = walk_forward_generic(records, direction="bullish", feature_keys=GEOMETRY5, objective="tail_prob", l2_lambda=1.0, purge_days=9)
    outcome5 = _strip_raw(outcome5)
    if not outcome5["metrics"].get("insufficient"):
        ic = float(outcome5["metrics"]["rank_ic_r"]["ic"])
        outcome5["delta_ic_vs_10key_control"] = delta(ic)
        print(f"  n={outcome5['n_evaluated']} ic={ic} delta_vs_10key={outcome5['delta_ic_vs_10key_control']} passed={outcome5['acceptance']['passed']}")
    else:
        outcome5["delta_ic_vs_10key_control"] = None
        print("  INSUFFICIENT")
    results["cells"]["cell_5_geometry5_tail_prob_lambda_1.0"] = outcome5

    # ---- Cells 6-7: temporal stability split of the compact5/tail_prob/lambda=1/purge=9 run ----
    print("\n--- Cells 6-7: temporal split of compact5_tail_prob_lambda_1.0 (purge=9) ---")
    raw = control_compact5["_raw"]
    first_result, second_result, split_meta = temporal_split_cells(raw, objective="tail_prob", tail_r=META_TAIL_R)
    for cell_id, name, res in ((6, "compact5_tail_prob_first_half", first_result), (7, "compact5_tail_prob_second_half", second_result)):
        res["split_meta"] = split_meta
        res["config"] = {"l2_lambda": 1.0, "purge_days": 9, "feature_keys": list(COMPACT5), "objective": "tail_prob"}
        if not res["metrics"].get("insufficient"):
            ic = float(res["metrics"]["rank_ic_r"]["ic"])
            res["delta_ic_vs_10key_control"] = delta(ic)
            print(f"  Cell {cell_id} ({name}): n={res['n_evaluated']} ic={ic} delta_vs_10key={res['delta_ic_vs_10key_control']} passed={res['acceptance']['passed']}")
        else:
            res["delta_ic_vs_10key_control"] = None
            print(f"  Cell {cell_id} ({name}): INSUFFICIENT")
        results["cells"][f"cell_{cell_id}_{name}"] = res

    # ---- Cell 8: purge robustness, compact5/tail_prob/lambda=1.0, purge=14 ----
    print("\n--- Cell 8: compact5_tail_prob_purge14 ---")
    outcome8 = walk_forward_generic(records, direction="bullish", feature_keys=COMPACT5, objective="tail_prob", l2_lambda=1.0, purge_days=14)
    outcome8 = _strip_raw(outcome8)
    if not outcome8["metrics"].get("insufficient"):
        ic = float(outcome8["metrics"]["rank_ic_r"]["ic"])
        outcome8["delta_ic_vs_10key_control"] = delta(ic)
        outcome8["delta_ic_vs_compact5_purge9"] = round(ic - control_compact5_ic, 4)
        print(f"  n={outcome8['n_evaluated']} ic={ic} delta_vs_10key={outcome8['delta_ic_vs_10key_control']} delta_vs_compact5_purge9={outcome8['delta_ic_vs_compact5_purge9']} passed={outcome8['acceptance']['passed']}")
    else:
        outcome8["delta_ic_vs_10key_control"] = None
        print("  INSUFFICIENT")
    results["cells"]["cell_8_compact5_tail_prob_purge14"] = outcome8

    # strip _raw from controls before writing
    results["control_10key"] = _strip_raw(results["control_10key"])
    results["control_compact5"] = _strip_raw(results["control_compact5"])

    # ---- Verdict ----
    def ic_of(cell_key: str) -> float | None:
        m = results["cells"][cell_key]["metrics"]
        if m.get("insufficient"):
            return None
        return float(m["rank_ic_r"]["ic"])

    lambda_ics = {
        "0.3": ic_of("cell_1_compact5_tail_prob_lambda_0.3"),
        "1.0": control_compact5_ic,
        "3.0": ic_of("cell_2_compact5_tail_prob_lambda_3.0"),
        "10.0": ic_of("cell_3_compact5_tail_prob_lambda_10.0"),
    }
    p_win_ic = ic_of("cell_4_compact5_p_win_lambda_1.0")
    geometry5_ic = ic_of("cell_5_geometry5_tail_prob_lambda_1.0")
    first_half_ic = ic_of("cell_6_compact5_tail_prob_first_half")
    second_half_ic = ic_of("cell_7_compact5_tail_prob_second_half")
    purge14_ic = ic_of("cell_8_compact5_tail_prob_purge14")

    lambda_stable = all(v is not None and v > control_10key_ic for v in lambda_ics.values())
    geometry_confirms_noise_source = geometry5_ic is not None and geometry5_ic <= control_10key_ic
    temporal_stable = (
        first_half_ic is not None and second_half_ic is not None
        and first_half_ic > -0.15 and second_half_ic > -0.15  # both not catastrophically worse than control
        and (first_half_ic > control_10key_ic) == (second_half_ic > control_10key_ic)  # same sign of improvement
    )
    purge_stable = purge14_ic is not None and abs(purge14_ic - control_compact5_ic) <= 0.02

    p_win_effect_reverses = (
        p_win_ic is not None and diag_10key_pwin_ic is not None and p_win_ic < diag_10key_pwin_ic
    )
    summary = {
        "control_10key_tail_prob_ic": control_10key_ic,
        "control_compact5_tail_prob_ic": control_compact5_ic,
        "lambda_sweep_ic": lambda_ics,
        "lambda_lead_holds_at_every_lambda": lambda_stable,
        "p_win_objective_compact5_ic": p_win_ic,
        "p_win_objective_10key_control_ic": diag_10key_pwin_ic,
        "p_win_effect_reverses_vs_tail_prob": p_win_effect_reverses,
        "geometry5_only_ic": geometry5_ic,
        "geometry5_confirms_dominant_noise_source_claim": geometry_confirms_noise_source,
        "temporal_first_half_ic": first_half_ic,
        "temporal_second_half_ic": second_half_ic,
        "temporal_split_meta": split_meta,
        "temporal_stable": temporal_stable,
        "purge9_compact5_ic": control_compact5_ic,
        "purge14_compact5_ic": purge14_ic,
        "purge_delta": round(purge14_ic - control_compact5_ic, 4) if purge14_ic is not None else None,
        "purge_stable": purge_stable,
        "overall_verdict": (
            "STABLE_BUT_TAIL_SPECIFIC"
            if (lambda_stable and temporal_stable and purge_stable and p_win_effect_reverses)
            else ("STABLE" if (lambda_stable and temporal_stable and purge_stable) else "FRAGILE")
        ),
    }
    results["summary"] = summary
    print("\nSUMMARY:", json.dumps(summary, indent=2))

    out_path = EXPERIMENT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
