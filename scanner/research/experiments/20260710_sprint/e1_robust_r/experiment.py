"""E1: robust/rank targets for the within-direction ranking model (20260710 sprint).

expected_r (L2 ridge on raw R) failed OOF: rank IC -0.086 on ~5.7-5.8k
bullish rows. Hypothesis: unbounded right-tail R dominates the squared
loss and pulls the fit away from the ranking-relevant bulk of the
distribution. This script tests four targets that keep magnitude/order
information without letting tail rows dominate the objective:
huber_r, quantile_r_75, winsor_ridge, rank_ridge.

The walk-forward skeleton (expanding window, 21-day refit, 9-day purge,
min_train 300, META_FEATURE_KEYS, _standardize_train) is copied exactly
from scanner.edge.calibration.walk_forward_calibration - only the target
transform and the fit function change per cell. READ-ONLY against
scanner/: everything here is additive under this experiment directory.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import EDGE_INDEX_PATH  # noqa: E402
from scanner.edge.calibration import (  # noqa: E402
    META_ACCEPT_MAX_P_DAY,
    META_ACCEPT_MIN_IC,
    META_ACCEPT_MIN_N,
    META_FEATURE_KEYS,
    META_L2_LAMBDA,
    META_MIN_TRAIN,
    META_PURGE_DAYS,
    META_REFIT_EVERY_DAYS,
    META_TAIL_R,
    _feature_matrix,
    _finite,
    _fit_ridge,
    _standardize_train,
    walk_forward_calibration,
)
from scanner.edge.retrieval import load_edge_index  # noqa: E402
from scanner.edge.stats import spearman_rank_ic, tail_retention, tercile_lift  # noqa: E402

OUT_DIR = Path(__file__).parent


# --------------------------------------------------------------------------
# Robust/rank fit functions (numpy-only, deterministic, no randomness)
# --------------------------------------------------------------------------


def _fit_huber(design: np.ndarray, y: np.ndarray, l2_lambda: float, delta: float = 1.0,
               max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """IRLS Huber M-estimator with an L2 ridge penalty on coefficients.

    Weighted normal equations each iteration: w_i = 1 if |r_i|<=delta else
    delta/|r_i|; (X^T W X + Lambda) beta = X^T W y. Standard Huber IRLS.
    """
    penalty = np.full(design.shape[1], float(l2_lambda))
    penalty[0] = 0.0
    gram0 = design.T @ design + np.diag(penalty + 1e-9)
    weights = np.linalg.solve(gram0, design.T @ y)
    for _ in range(max_iter):
        residual = y - design @ weights
        abs_r = np.abs(residual)
        w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-12))
        wx = design * w[:, None]
        gram = design.T @ wx + np.diag(penalty + 1e-9)
        rhs = design.T @ (w * y)
        try:
            new_weights = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            new_weights = np.linalg.pinv(gram) @ rhs
        step = float(np.max(np.abs(new_weights - weights)))
        weights = new_weights
        if step < tol:
            break
    return weights


def _fit_quantile(design: np.ndarray, y: np.ndarray, l2_lambda: float, tau: float = 0.75,
                   max_iter: int = 100, tol: float = 1e-8, eps: float = 1e-6) -> np.ndarray:
    """IRLS approximation to pinball-loss quantile regression with an L2 penalty.

    pinball(r) = c(r)*|r|, c(r) = tau if r>=0 else (1-tau). Majorize |r| by
    r^2/|r| (Schlossmacher-style IRLS, generalized asymmetrically for tau
    != 0.5): weighted normal equations with w_i = c(r_i) / max(|r_i|, eps).
    """
    penalty = np.full(design.shape[1], float(l2_lambda))
    penalty[0] = 0.0
    gram0 = design.T @ design + np.diag(penalty + 1e-9)
    weights = np.linalg.solve(gram0, design.T @ y)
    for _ in range(max_iter):
        residual = y - design @ weights
        abs_r = np.maximum(np.abs(residual), eps)
        c = np.where(residual >= 0, tau, 1.0 - tau)
        w = c / abs_r
        wx = design * w[:, None]
        gram = design.T @ wx + np.diag(penalty + 1e-9)
        rhs = design.T @ (w * y)
        try:
            new_weights = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            new_weights = np.linalg.pinv(gram) @ rhs
        step = float(np.max(np.abs(new_weights - weights)))
        weights = new_weights
        if step < tol:
            break
    return weights


# --------------------------------------------------------------------------
# Target transforms (frozen on the TRAIN WINDOW only, no look-ahead)
# --------------------------------------------------------------------------


def _target_raw(train_rows: list[dict]) -> tuple[np.ndarray, dict]:
    r = np.array([row["r"] for row in train_rows], dtype=float)
    return r, {}


def _target_winsor(train_rows: list[dict]) -> tuple[np.ndarray, dict]:
    r = np.array([row["r"] for row in train_rows], dtype=float)
    with np.errstate(all="ignore"):
        lo = float(np.nanpercentile(r, 1))
        hi = float(np.nanpercentile(r, 99))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return np.full_like(r, np.nan), {}
    y = np.clip(r, lo, hi)
    return y, {"winsor_lo": lo, "winsor_hi": hi}


def _target_rank(train_rows: list[dict]) -> tuple[np.ndarray, dict]:
    r = np.array([row["r"] for row in train_rows], dtype=float)
    n = len(r)
    ranks = pd.Series(r).rank(method="average").to_numpy()
    y = (ranks - 0.5) / n
    return y, {}


CELLS = {
    "huber_r": {
        "target_fn": _target_raw,
        "fit_fn": lambda design, y, lam: _fit_huber(design, y, lam, delta=1.0),
    },
    "quantile_r_75": {
        "target_fn": _target_raw,
        "fit_fn": lambda design, y, lam: _fit_quantile(design, y, lam, tau=0.75),
    },
    "winsor_ridge": {
        "target_fn": _target_winsor,
        "fit_fn": lambda design, y, lam: _fit_ridge(design, y, lam),
    },
    "rank_ridge": {
        "target_fn": _target_rank,
        "fit_fn": lambda design, y, lam: _fit_ridge(design, y, lam),
    },
}


# --------------------------------------------------------------------------
# Harness (copied skeleton from calibration.walk_forward_calibration)
# --------------------------------------------------------------------------


def _load_rows(records, direction: str = "bullish") -> list[dict]:
    rows = []
    for record in records:
        rec_direction = getattr(record, "direction", None)
        if str(rec_direction) != direction:
            continue
        features = getattr(record, "features", None)
        timestamp = getattr(record, "timestamp", None)
        r_multiple = getattr(record, "r_multiple", None)
        ticker = getattr(record, "ticker", "")
        ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
        r_value = _finite(r_multiple)
        if pd.isna(ts) or not isinstance(features, dict) or not math.isfinite(r_value):
            continue
        rows.append({"ts": ts, "ticker": str(ticker), "timestamp": str(timestamp), "features": features, "r": r_value})
    rows.sort(key=lambda row: row["ts"])
    return rows


def _fit_cell_model(train_rows: list[dict], target_fn, fit_fn, l2_lambda: float) -> dict | None:
    n = len(train_rows)
    if n < META_MIN_TRAIN:
        return None
    raw = _feature_matrix([r["features"] for r in train_rows], META_FEATURE_KEYS)
    y, _meta = target_fn(train_rows)
    if y is None or not np.isfinite(y).all():
        return None
    transform = _standardize_train(raw)
    if transform is None:
        return None
    design = np.hstack([np.ones((n, 1)), transform["x"]])
    weights = fit_fn(design, y, l2_lambda)
    if weights is None or not np.isfinite(weights).all():
        return None
    return {
        "intercept": float(weights[0]),
        "coefficients": np.asarray(weights[1:], dtype=float),
        "medians": transform["medians"],
        "winsor_low": transform["lo"],
        "winsor_high": transform["hi"],
        "means": transform["mean"],
        "stds": transform["std"],
    }


def _predict(model: dict, features: dict) -> float | None:
    values = np.array([_finite(features.get(k)) for k in META_FEATURE_KEYS], dtype=float)
    filled = np.where(np.isfinite(values), values, model["medians"])
    clipped = np.clip(filled, model["winsor_low"], model["winsor_high"])
    x = (clipped - model["means"]) / model["stds"]
    z = float(model["intercept"]) + float(np.dot(model["coefficients"], x))
    if not math.isfinite(z):
        return None
    return z


def walk_forward_generic(
    rows: list[dict],
    target_fn,
    fit_fn,
    l2_lambda: float = META_L2_LAMBDA,
    refit_every_days: int = META_REFIT_EVERY_DAYS,
    purge_days: int = META_PURGE_DAYS,
) -> dict:
    purge = pd.Timedelta(days=purge_days)
    refit_interval = pd.Timedelta(days=refit_every_days)
    model = None
    model_fit_ts = None
    predictions: list[float] = []
    outcomes: list[float] = []
    day_keys: list[str] = []
    row_ids: list[str] = []

    for row in rows:
        needs_refit = model_fit_ts is None or (row["ts"] - model_fit_ts) >= refit_interval
        if needs_refit:
            train = [r for r in rows if r["ts"] <= row["ts"] - purge]
            if len(train) >= META_MIN_TRAIN:
                candidate = _fit_cell_model(train, target_fn, fit_fn, l2_lambda)
                if candidate is not None:
                    model = candidate
                    model_fit_ts = row["ts"]
        if model is None:
            continue
        score = _predict(model, row["features"])
        if score is None:
            continue
        predictions.append(score)
        outcomes.append(row["r"])
        day_keys.append(row["ts"].strftime("%Y-%m-%d"))
        row_ids.append(f"{row['ticker']}|{row['timestamp']}")

    return {"predictions": predictions, "outcomes": outcomes, "day_keys": day_keys, "row_ids": row_ids}


def evaluate(oof: dict) -> dict:
    predictions, outcomes, day_keys, row_ids = oof["predictions"], oof["outcomes"], oof["day_keys"], oof["row_ids"]
    n_eval = len(predictions)
    if n_eval < META_ACCEPT_MIN_N:
        return {"metrics": {"insufficient": True, "n_evaluated": n_eval}, "acceptance": {"passed": False, "reason": "insufficient_out_of_fold_predictions"}}

    ic = spearman_rank_ic(predictions, outcomes, day_keys=day_keys)
    lift = tercile_lift(predictions, outcomes, day_keys, row_ids=row_ids)
    tail = tail_retention(predictions, outcomes, row_ids=row_ids, tail_r=META_TAIL_R)

    naive = float(np.mean(outcomes))
    oof_loss = float(np.mean([(p - r) ** 2 for p, r in zip(predictions, outcomes, strict=False)]))
    naive_loss = float(np.mean([(naive - r) ** 2 for r in outcomes]))
    beats_naive = oof_loss < naive_loss

    metrics = {
        "insufficient": False,
        "n_evaluated": n_eval,
        "rank_ic_r": ic,
        "tercile_lift": lift,
        "tail_retention": tail,
        "oof_mse_raw_r": round(oof_loss, 6),
        "naive_mse_raw_r": round(naive_loss, 6),
        "beats_naive": beats_naive,
    }

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
    return {"metrics": metrics, "acceptance": {"passed": all(criteria.values()), "criteria": criteria}}


def main() -> None:
    records = load_edge_index(EDGE_INDEX_PATH)
    bullish_count = sum(1 for r in records if getattr(r, "direction", None) == "bullish")

    results: dict = {
        "n_records_total": len(records),
        "n_records_bullish": bullish_count,
        "harness_sanity": {},
        "cells": {},
    }

    # --- MANDATORY sanity check ---------------------------------------
    repo_expected_r = walk_forward_calibration(records, direction="bullish", objective="expected_r")
    repo_ic = repo_expected_r["metrics"]["rank_ic_r"]["ic"]
    repo_n = repo_expected_r["n_evaluated"]

    rows = _load_rows(records, direction="bullish")
    our_ridge_oof = walk_forward_generic(rows, _target_raw, lambda d, y, lam: _fit_ridge(d, y, lam))
    our_ridge_eval = evaluate(our_ridge_oof)
    our_ic = our_ridge_eval["metrics"]["rank_ic_r"]["ic"] if not our_ridge_eval["metrics"].get("insufficient") else None
    our_n = our_ridge_eval["metrics"].get("n_evaluated")

    ic_match = our_ic is not None and abs(our_ic - repo_ic) <= 0.005
    results["harness_sanity"] = {
        "repo_expected_r_ic": repo_ic,
        "repo_expected_r_n_evaluated": repo_n,
        "our_harness_ridge_r_ic": our_ic,
        "our_harness_ridge_r_n_evaluated": our_n,
        "ic_match_within_0.005": ic_match,
        "harness_confirmed_ok": ic_match,
    }

    if not ic_match:
        results["harness_sanity"]["FATAL"] = "Harness does not reproduce repo expected_r IC within tolerance. Cells below are NOT trustworthy."

    # --- PRE-REGISTERED cells -------------------------------------------
    for cell_id, spec in CELLS.items():
        try:
            oof = walk_forward_generic(rows, spec["target_fn"], spec["fit_fn"])
            cell_eval = evaluate(oof)
            results["cells"][cell_id] = {"status": "ok", **cell_eval}
        except Exception as exc:  # fail closed: report as errored, never silently skip
            results["cells"][cell_id] = {
                "status": "errored",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(json.dumps(results["harness_sanity"], indent=2))
    print("---")
    for cell_id, res in results["cells"].items():
        if res["status"] == "errored":
            print(cell_id, "ERRORED:", res["error"])
            continue
        m = res["metrics"]
        if m.get("insufficient"):
            print(cell_id, "INSUFFICIENT", m)
            continue
        ic = m["rank_ic_r"]
        lift = m["tercile_lift"]
        print(
            cell_id,
            "n=", m["n_evaluated"],
            "ic=", ic.get("ic"),
            "p_day=", ic.get("p_value_day_clustered"),
            "spread_ci_low=", lift.get("spread_ci_low"),
            "beats_naive=", m["beats_naive"],
            "passed=", res["acceptance"]["passed"],
        )


if __name__ == "__main__":
    main()
