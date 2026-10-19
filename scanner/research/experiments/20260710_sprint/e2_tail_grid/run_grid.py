"""E2: pre-registered tail_r x l2_lambda grid for the tail_prob objective.

Read-only against scanner/. Reuses scanner.edge.calibration's standardize /
IRLS-logistic / feature-matrix primitives and scanner.edge.stats's metrics.
fit_model() is not reused for label construction because it hardcodes
tail_r=2.0 (META_TAIL_R module constant) -- everything else in the logistic
path (standardize -> intercept-unshrunk L2 IRLS logistic) is copied exactly.

Run: .\\venv\\Scripts\\python.exe scanner\\research\\experiments\\20260710_sprint\\e2_tail_grid\\run_grid.py
(from repo root, so `scanner` imports resolve)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scanner.config import EDGE_INDEX_PATH
from scanner.edge.retrieval import load_edge_index
from scanner.edge.calibration import (
    META_FEATURE_KEYS,
    META_MIN_TRAIN,
    META_MIN_CLASS_EVENTS,
    META_REFIT_EVERY_DAYS,
    META_PURGE_DAYS,
    META_ACCEPT_MIN_IC,
    META_ACCEPT_MAX_P_DAY,
    META_ACCEPT_MIN_N,
    _standardize_train,
    _fit_logistic_irls,
    _feature_matrix,
    predict_score,
    walk_forward_calibration,
)
from scanner.edge.stats import spearman_rank_ic, tercile_lift, tail_retention

OUT_DIR = Path(__file__).parent
TAIL_R_GRID = [1.5, 2.0, 2.5, 3.0]
LAMBDA_GRID = [0.3, 1.0, 3.0, 10.0]


def _finite(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def load_bullish_rows() -> list[dict]:
    records = load_edge_index(EDGE_INDEX_PATH)
    rows = []
    for record in records:
        if str(getattr(record, "direction", None)) != "bullish":
            continue
        features = getattr(record, "features", None)
        timestamp = getattr(record, "timestamp", None)
        r_multiple = getattr(record, "r_multiple", None)
        ticker = getattr(record, "ticker", "")
        ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
        r_value = _finite(r_multiple)
        if pd.isna(ts) or not isinstance(features, dict) or not math.isfinite(r_value):
            continue
        rows.append(
            {
                "ts": ts,
                "ticker": str(ticker),
                "timestamp": str(timestamp),
                "features": features,
                "r": r_value,
            }
        )
    rows.sort(key=lambda row: row["ts"])
    return rows


def fit_tail_model(train_rows: list[dict], tail_r: float, l2_lambda: float):
    """Mirror fit_model()'s logistic path exactly, but with a caller-chosen
    tail_r and l2_lambda instead of the module-hardcoded META_TAIL_R/1.0.
    Returns (model_dict_or_None, reason)."""
    raw = _feature_matrix([r["features"] for r in train_rows], META_FEATURE_KEYS)
    r_values = np.array([r["r"] for r in train_rows], dtype=float)
    n = len(r_values)
    if n < META_MIN_TRAIN:
        return None, "insufficient_train"

    y = (r_values >= tail_r).astype(float)
    positives = float(y.sum())
    negatives = n - positives
    if positives < META_MIN_CLASS_EVENTS or negatives < META_MIN_CLASS_EVENTS:
        return None, "class_min"

    transform = _standardize_train(raw)
    if transform is None:
        return None, "standardize_fail"
    design = np.hstack([np.ones((n, 1)), transform["x"]])
    weights = _fit_logistic_irls(design, y, l2_lambda)
    if not np.isfinite(weights).all():
        return None, "non_finite_weights"

    model = {
        "objective": "tail_prob",
        "feature_keys": list(META_FEATURE_KEYS),
        "l2_lambda": float(l2_lambda),
        "tail_r": float(tail_r),
        "intercept": float(weights[0]),
        "coefficients": [float(w) for w in weights[1:]],
        "winsor_low": [float(v) for v in transform["lo"]],
        "winsor_high": [float(v) for v in transform["hi"]],
        "medians": [float(v) for v in transform["medians"]],
        "means": [float(v) for v in transform["mean"]],
        "stds": [float(v) for v in transform["std"]],
        "n_train": int(n),
    }
    return model, "ok"


def _brier(probabilities: list[float], labels: list[int]) -> float:
    pairs = [(p, y) for p, y in zip(probabilities, labels) if p is not None]
    if not pairs:
        return 1.0
    return float(np.mean([(p - y) ** 2 for p, y in pairs]))


def run_cell(rows: list[dict], tail_r: float, l2_lambda: float) -> dict:
    purge = pd.Timedelta(days=META_PURGE_DAYS)
    refit_interval = pd.Timedelta(days=META_REFIT_EVERY_DAYS)

    model = None
    model_fit_ts = None
    predictions: list[float] = []
    outcomes: list[float] = []
    day_keys: list[str] = []
    row_ids: list[str] = []

    refit_checkpoints = 0
    refit_succeeded = 0
    refit_skipped_class_min = 0
    refit_skipped_insufficient_train = 0
    refit_skipped_other = 0

    for row in rows:
        needs_refit = model_fit_ts is None or (row["ts"] - model_fit_ts) >= refit_interval
        if needs_refit:
            refit_checkpoints += 1
            train = [r for r in rows if r["ts"] <= row["ts"] - purge]
            candidate_model, reason = fit_tail_model(train, tail_r, l2_lambda)
            if candidate_model is not None:
                model = candidate_model
                model_fit_ts = row["ts"]
                refit_succeeded += 1
            elif reason == "class_min":
                refit_skipped_class_min += 1
            elif reason == "insufficient_train":
                refit_skipped_insufficient_train += 1
            else:
                refit_skipped_other += 1
        if model is None:
            continue
        score = predict_score(model, row["features"])
        if score is None:
            continue
        predictions.append(score)
        outcomes.append(row["r"])
        day_keys.append(row["ts"].strftime("%Y-%m-%d"))
        row_ids.append(f"{row['ticker']}|{row['timestamp']}")

    n_eval = len(predictions)
    refit_stats = {
        "refit_checkpoints": refit_checkpoints,
        "refit_succeeded": refit_succeeded,
        "refit_skipped_class_min": refit_skipped_class_min,
        "refit_skipped_insufficient_train": refit_skipped_insufficient_train,
        "refit_skipped_other": refit_skipped_other,
    }

    result = {
        "tail_r": tail_r,
        "l2_lambda": l2_lambda,
        "n_evaluated": n_eval,
        "refit_stats": refit_stats,
    }

    if n_eval < META_ACCEPT_MIN_N:
        result["insufficient"] = True
        return result

    result["insufficient"] = False

    # Primary: rank vs RAW R outcomes (same target the repo evaluates against).
    ic = spearman_rank_ic(predictions, outcomes, day_keys=day_keys)
    lift = tercile_lift(predictions, outcomes, day_keys, row_ids=row_ids)
    tail_fixed = tail_retention(predictions, outcomes, row_ids=row_ids, tail_r=2.0)
    tail_own = tail_retention(predictions, outcomes, row_ids=row_ids, tail_r=tail_r)

    # Brier vs base-rate, on labels at the CELL's own tail_r (what it was trained for).
    labels_own = [1 if r >= tail_r else 0 for r in outcomes]
    base_rate_own = float(np.mean(labels_own))
    oof_loss = _brier(predictions, labels_own)
    naive_loss = float(np.mean([(base_rate_own - y) ** 2 for y in labels_own]))
    beats_naive = oof_loss < naive_loss

    result["rank_ic_r"] = ic
    result["tercile_lift"] = lift
    result["tail_retention_at_2.0_fixed"] = tail_fixed
    result["tail_retention_at_own_tail_r"] = tail_own
    result["brier"] = {
        "base_rate_own_tail_r": round(base_rate_own, 4),
        "oof_brier": round(oof_loss, 6),
        "naive_brier": round(naive_loss, 6),
        "beats_naive": beats_naive,
    }

    criteria = {
        "ic_at_least_0.07": float(ic.get("ic", 0.0)) >= META_ACCEPT_MIN_IC,
        "day_clustered_p_at_most_0.05": float(ic.get("p_value_day_clustered", 1.0)) <= META_ACCEPT_MAX_P_DAY,
        "n_at_least_300": n_eval >= META_ACCEPT_MIN_N,
        "tercile_spread_ci_low_positive": bool(
            not lift.get("insufficient") and lift.get("spread_ci_low") is not None and lift["spread_ci_low"] > 0
        ),
        "tail_retention_at_least_pro_rata_FIXED_2.0": bool(
            tail_fixed.get("insufficient")
            or tail_fixed.get("observed_share") is None
            or tail_fixed["observed_share"] >= tail_fixed["expected_share"]
        ),
        "beats_naive": beats_naive,
    }
    result["gates"] = criteria
    result["passed_all_6_gates"] = all(criteria.values())
    return result


def main():
    print("Loading edge index...")
    rows = load_bullish_rows()
    print(f"Loaded {len(rows)} bullish rows with finite R and features.")

    # ---- MANDATORY sanity check ----
    print("\n=== SANITY CHECK 1: repo's own walk_forward_calibration(tail_prob) ===")
    records = load_edge_index(EDGE_INDEX_PATH)
    repo_result = walk_forward_calibration(records, direction="bullish", objective="tail_prob")
    repo_ic = repo_result.get("metrics", {}).get("rank_ic_r", {})
    print(f"repo n_evaluated={repo_result.get('n_evaluated')} ic={repo_ic.get('ic')} p_day={repo_ic.get('p_value_day_clustered')}")

    print("\n=== SANITY CHECK 2: this harness at (tail_r=2.0, lambda=1.0) control cell ===")
    control = run_cell(rows, tail_r=2.0, l2_lambda=1.0)
    control_ic = control.get("rank_ic_r", {})
    print(f"own  n_evaluated={control.get('n_evaluated')} ic={control_ic.get('ic')} p_day={control_ic.get('p_value_day_clustered')}")

    ic_diff = abs(float(repo_ic.get("ic", 0.0)) - float(control_ic.get("ic", 0.0)))
    n_diff = abs(int(repo_result.get("n_evaluated", 0)) - int(control.get("n_evaluated", 0)))
    sanity = {
        "repo_ic": repo_ic.get("ic"),
        "repo_n_evaluated": repo_result.get("n_evaluated"),
        "repo_p_day": repo_ic.get("p_value_day_clustered"),
        "own_harness_ic": control_ic.get("ic"),
        "own_harness_n_evaluated": control.get("n_evaluated"),
        "own_harness_p_day": control_ic.get("p_value_day_clustered"),
        "ic_diff": round(ic_diff, 6),
        "n_diff": n_diff,
        "harness_matches_repo_within_tolerance": bool(ic_diff <= 0.005 and n_diff == 0),
    }
    print(f"\nSanity summary: {json.dumps(sanity, indent=2)}")

    if not sanity["harness_matches_repo_within_tolerance"]:
        print("\n!!! HARNESS DOES NOT REPRODUCE REPO CONTROL WITHIN TOLERANCE -- STOPPING BEFORE GRID !!!")
        (OUT_DIR / "results.json").write_text(
            json.dumps({"sanity_check": sanity, "status": "HARNESS_MISMATCH_ABORTED", "grid": []}, indent=2, default=str),
            encoding="utf-8",
        )
        return

    # ---- PRE-REGISTERED grid, run in the order declared in preregistration.json ----
    print("\n=== RUNNING 16-CELL GRID ===")
    grid_results = []
    for tail_r in TAIL_R_GRID:
        for l2_lambda in LAMBDA_GRID:
            print(f"cell tail_r={tail_r} lambda={l2_lambda} ...")
            cell = run_cell(rows, tail_r=tail_r, l2_lambda=l2_lambda)
            grid_results.append(cell)
            ic_val = cell.get("rank_ic_r", {}).get("ic")
            print(f"  -> n={cell.get('n_evaluated')} ic={ic_val} passed_all_6={cell.get('passed_all_6_gates')} skipped_class_min={cell['refit_stats']['refit_skipped_class_min']}")

    best_by_ic = sorted(
        [c for c in grid_results if not c.get("insufficient")],
        key=lambda c: float(c.get("rank_ic_r", {}).get("ic", -999)),
        reverse=True,
    )
    passing_cells = [c for c in grid_results if c.get("passed_all_6_gates")]

    summary_lines = []
    summary_lines.append("# E2 tail_r x l2_lambda grid -- results\n")
    summary_lines.append(f"Sanity check: repo IC={sanity['repo_ic']} (n={sanity['repo_n_evaluated']}) vs own-harness IC={sanity['own_harness_ic']} (n={sanity['own_harness_n_evaluated']}); diff={sanity['ic_diff']} -> {'MATCH' if sanity['harness_matches_repo_within_tolerance'] else 'MISMATCH'}\n")
    summary_lines.append("\n| tail_r | lambda | n | IC | p_day | tercile_ci_low | tail_ret(2.0) obs/exp | beats_naive | gates_passed | 6/6 |")
    summary_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in grid_results:
        if c.get("insufficient"):
            summary_lines.append(f"| {c['tail_r']} | {c['l2_lambda']} | {c['n_evaluated']} | INSUFFICIENT n | - | - | - | - | - | NO |")
            continue
        ic = c["rank_ic_r"]
        lift = c["tercile_lift"]
        tail = c["tail_retention_at_2.0_fixed"]
        gates = c["gates"]
        n_passed = sum(1 for v in gates.values() if v)
        summary_lines.append(
            f"| {c['tail_r']} | {c['l2_lambda']} | {c['n_evaluated']} | {ic.get('ic')} | {ic.get('p_value_day_clustered')} | "
            f"{lift.get('spread_ci_low')} | {tail.get('observed_share')}/{tail.get('expected_share')} | "
            f"{c['brier']['beats_naive']} | {n_passed}/6 | {'YES' if c['passed_all_6_gates'] else 'no'} |"
        )
    summary_lines.append(f"\nBest cell by OOF rank IC: tail_r={best_by_ic[0]['tail_r']}, lambda={best_by_ic[0]['l2_lambda']}, IC={best_by_ic[0]['rank_ic_r']['ic']}" if best_by_ic else "\nNo cell had sufficient n.")
    summary_lines.append(f"\nCells passing all 6 gates: {len(passing_cells)} / 16")
    summary_lines.append("\nMultiplicity caveat: 16 cells at alpha=0.05 on the day-clustered p-value -> ~0.8 false positives expected under a global null. Any single passing cell not corroborated by its grid neighbors should be treated as noise, not discovery.")
    summary_text = "\n".join(summary_lines)

    output = {
        "sanity_check": sanity,
        "status": "OK",
        "grid": grid_results,
        "best_cell_by_ic": {"tail_r": best_by_ic[0]["tail_r"], "l2_lambda": best_by_ic[0]["l2_lambda"], "ic": best_by_ic[0]["rank_ic_r"]["ic"]} if best_by_ic else None,
        "n_cells_passing_all_6_gates": len(passing_cells),
        "passing_cells": [{"tail_r": c["tail_r"], "l2_lambda": c["l2_lambda"]} for c in passing_cells],
        "summary_markdown": summary_text,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    try:
        (OUT_DIR / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    except Exception as exc:
        print(f"WARNING: could not write SUMMARY.md ({exc}); summary is embedded in results.json under 'summary_markdown'.")

    print("\n" + summary_text)
    print("\nDone. Wrote results.json" + (" and SUMMARY.md" if (OUT_DIR / "SUMMARY.md").exists() else " (SUMMARY.md write failed, see results.json summary_markdown)"))


if __name__ == "__main__":
    main()
