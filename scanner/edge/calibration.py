"""Within-direction ranking models (the calibration frontier), self-run.

The composite edge score is a fail-closed evidence accumulator, not a
ranker: gate caps and flat penalties collapse most walk-forward scores onto
a handful of tied values (51% of bullish rows score exactly 0), so its
within-direction rank IC is structurally ~0. Post-hoc calibration of that
score cannot help - calibration is monotone and preserves rank order.

This module therefore fits SECONDARY models on the walk-forward index
(meta-labeling: Lopez de Prado 2018, Joubert et al. JFDS 2022-2023) and
evaluates them purely out-of-fold. Three PRE-REGISTERED objectives run as a
suite every lab run, because the first full-history evaluation proved the
obvious objective wrong: P(win) is ANTI-informative for this right-tail
edge (its top tercile underperformed its bottom by 0.14R) - optimizing win
rate fights the tail exactly like profit targets did.

  p_win       L2 logistic on (R > 0)            - kept as the control arm
  expected_r  L2 ridge on R itself              - hunts magnitude
  tail_prob   L2 logistic on (R >= 2.0)         - hunts the tail directly

Everything is numpy-only (the venv has no sklearn/scipy), deterministic,
and strictly downstream of every gate - a model can only ever RANK inside
a direction, never create or promote a signal.

AUTONOMY CONTRACT (how self-improvement stays honest):
- every objective's out-of-fold evaluation is registered in the trial
  registry every run (kind=calibration_trial) - the multiple-testing ledger;
- a model ships as the live advisory ONLY if it passes the pre-registered
  acceptance on THIS run AND on its previous registered run (two-touch,
  mirroring the adaptive policy's loosen confirmation);
- if several objectives clear both touches, the highest current OOF rank IC
  ships - a deterministic, pre-registered choice, not a human pick;
- anything less ships nothing: take-all-bullish stays the standing policy.

PRE-REGISTRATION (2026-07-02/03): feature set, model classes, lambda, purge,
refit cadence, and acceptance thresholds below were fixed before seeing any
out-of-fold results for the respective objective.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ..config import EDGE_EMBARGO_DAYS
from .stats import spearman_rank_ic, tail_retention, tercile_lift

# Scale-free setup/context features present on every historical index record
# (feature_version=3). Chosen for literature support (relative volume and
# participation are the strongest published conditioners of short-horizon
# breakout outcomes; volatility regime and trend context next), NOT for
# in-sample performance. rr_ratio is deliberately excluded: degenerate
# cost-basis rows push it to astronomical values and its geometry is already
# spanned by box/breakout features.
META_FEATURE_KEYS: tuple[str, ...] = (
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

META_OBJECTIVES: tuple[str, ...] = ("expected_r", "tail_prob", "p_win")
META_TAIL_R = 2.0
META_L2_LAMBDA = 1.0
META_REFIT_EVERY_DAYS = 21
META_PURGE_DAYS = EDGE_EMBARGO_DAYS  # outcome must be resolved before scoring
META_MIN_TRAIN = 300
META_MIN_CLASS_EVENTS = 25
META_ACCEPT_MIN_IC = 0.07
META_ACCEPT_MAX_P_DAY = 0.05
META_ACCEPT_MIN_N = 300
META_MODEL_VERSION = 2


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _feature_matrix(feature_dicts: list[dict], keys: tuple[str, ...]) -> np.ndarray:
    matrix = np.full((len(feature_dicts), len(keys)), np.nan, dtype=float)
    for row, features in enumerate(feature_dicts):
        if not isinstance(features, dict):
            continue
        for col, key in enumerate(keys):
            matrix[row, col] = _finite(features.get(key))
    return matrix


def _standardize_train(raw: np.ndarray) -> dict | None:
    """Winsorize 1/99 + median-impute + standardize; frozen transform params.

    An all-NaN column (feature outage) yields NaN percentiles; clipping
    against NaN bounds would poison the whole fit, so dead columns degrade
    to no-op bounds instead.
    """
    with np.errstate(all="ignore"):
        lo = np.nanpercentile(raw, 1, axis=0)
        hi = np.nanpercentile(raw, 99, axis=0)
        medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    lo = np.where(np.isfinite(lo), lo, medians)
    hi = np.where(np.isfinite(hi), hi, medians)
    filled = np.where(np.isfinite(raw), raw, medians)
    clipped = np.clip(filled, lo, hi)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x = (clipped - mean) / std
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and np.isfinite(x).all()):
        return None
    return {"lo": lo, "hi": hi, "medians": medians, "mean": mean, "std": std, "x": x}


def _fit_logistic_irls(design: np.ndarray, y: np.ndarray, l2_lambda: float, max_iter: int = 50) -> np.ndarray:
    weights = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], float(l2_lambda))
    penalty[0] = 0.0  # never shrink the intercept toward 0
    for _ in range(max_iter):
        p = _sigmoid(design @ weights)
        w_diag = np.maximum(p * (1.0 - p), 1e-9)
        gradient = design.T @ (y - p) - penalty * weights
        hessian = (design.T * w_diag) @ design + np.diag(penalty + 1e-9)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights = weights + step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights


def _fit_ridge(design: np.ndarray, y: np.ndarray, l2_lambda: float) -> np.ndarray:
    penalty = np.full(design.shape[1], float(l2_lambda))
    penalty[0] = 0.0
    gram = design.T @ design + np.diag(penalty + 1e-9)
    try:
        return np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ (design.T @ y)


def fit_model(
    feature_dicts: list[dict],
    r_multiples: list[float],
    objective: str = "p_win",
    l2_lambda: float = META_L2_LAMBDA,
) -> dict | None:
    """Fit one objective's model. Returns a JSON-safe model dict or None."""
    if objective not in META_OBJECTIVES:
        return None
    raw = _feature_matrix(feature_dicts, META_FEATURE_KEYS)
    r_values = np.array([_finite(r) for r in r_multiples], dtype=float)
    usable = np.isfinite(r_values)
    raw = raw[usable]
    r_values = r_values[usable]
    n = len(r_values)
    if n < META_MIN_TRAIN:
        return None

    if objective == "p_win":
        y = (r_values > 0.0).astype(float)
    elif objective == "tail_prob":
        y = (r_values >= META_TAIL_R).astype(float)
    else:  # expected_r
        y = r_values

    if objective in {"p_win", "tail_prob"}:
        positives = float(y.sum())
        if positives < META_MIN_CLASS_EVENTS or (n - positives) < META_MIN_CLASS_EVENTS:
            return None

    transform = _standardize_train(raw)
    if transform is None:
        return None
    design = np.hstack([np.ones((n, 1)), transform["x"]])
    if objective == "expected_r":
        weights = _fit_ridge(design, y, l2_lambda)
    else:
        weights = _fit_logistic_irls(design, y, l2_lambda)
    if not np.isfinite(weights).all():
        # A poisoned fit must fail closed (no model), never ship NaN weights.
        return None

    wins = r_values[r_values > 0.0]
    losses = r_values[r_values <= 0.0]
    return {
        "version": META_MODEL_VERSION,
        "objective": objective,
        "feature_keys": list(META_FEATURE_KEYS),
        "l2_lambda": float(l2_lambda),
        "intercept": float(weights[0]),
        "coefficients": [float(w) for w in weights[1:]],
        "winsor_low": [float(v) for v in transform["lo"]],
        "winsor_high": [float(v) for v in transform["hi"]],
        "medians": [float(v) for v in transform["medians"]],
        "means": [float(v) for v in transform["mean"]],
        "stds": [float(v) for v in transform["std"]],
        "n_train": int(n),
        "base_rate": float((r_values > 0.0).mean()),
        "tail_rate": float((r_values >= META_TAIL_R).mean()),
        "e_r_win": float(wins.mean()) if wins.size else 0.0,
        "e_r_loss": float(losses.mean()) if losses.size else 0.0,
    }


def predict_score(model: dict, features: dict) -> float | None:
    """The model's ranking score: a probability for logistic objectives, an
    E[R] estimate for the ridge objective. Fails closed on any non-finite."""
    if not isinstance(model, dict) or not isinstance(features, dict):
        return None
    keys = model.get("feature_keys") or []
    required = ("medians", "winsor_low", "winsor_high", "means", "stds", "coefficients", "intercept")
    if not keys or any(model.get(field) is None for field in required):
        return None
    if any(len(model[field]) != len(keys) for field in required[:-1]):
        return None
    values = np.array([_finite(features.get(key)) for key in keys], dtype=float)
    medians = np.array(model["medians"], dtype=float)
    filled = np.where(np.isfinite(values), values, medians)
    clipped = np.clip(filled, np.array(model["winsor_low"]), np.array(model["winsor_high"]))
    x = (clipped - np.array(model["means"])) / np.array(model["stds"])
    z = float(model["intercept"]) + float(np.dot(np.array(model["coefficients"]), x))
    if not math.isfinite(z):
        return None
    if model.get("objective") == "expected_r":
        return z
    return float(_sigmoid(np.array([z]))[0])


def predict_expected_r(model: dict, features: dict) -> float | None:
    """E[R] in R units, per objective.

    expected_r: the score itself. p_win: p*E[R|win] + (1-p)*E[R|loss] with
    conditionals pooled from training (exit-geometry properties, so this is
    a monotone transform of p). tail_prob: no defensible E[R] decomposition -
    returns None; rank on the score instead.
    """
    score = predict_score(model, features)
    if score is None:
        return None
    objective = model.get("objective", "p_win")
    if objective == "expected_r":
        return float(score)
    if objective == "p_win":
        return float(score * float(model.get("e_r_win", 0.0)) + (1.0 - score) * float(model.get("e_r_loss", 0.0)))
    return None


# Backward-compatible wrappers (v1 API; p_win objective).
def fit_win_probability_model(feature_dicts: list[dict], r_multiples: list[float], l2_lambda: float = META_L2_LAMBDA, max_iter: int = 50) -> dict | None:
    return fit_model(feature_dicts, r_multiples, objective="p_win", l2_lambda=l2_lambda)


def predict_win_probability(model: dict, features: dict) -> float | None:
    return predict_score(model, features)


def _brier(probabilities: list[float], labels: list[int]) -> float:
    pairs = [(p, y) for p, y in zip(probabilities, labels, strict=False) if p is not None]
    if not pairs:
        return 1.0
    return float(np.mean([(p - y) ** 2 for p, y in pairs]))


def walk_forward_calibration(
    records: Iterable[Any],
    direction: str = "bullish",
    objective: str = "p_win",
    refit_every_days: int = META_REFIT_EVERY_DAYS,
    purge_days: int = META_PURGE_DAYS,
) -> dict:
    """Expanding-window out-of-fold evaluation of one objective.

    Each record is predicted by a model trained ONLY on records whose entry
    is at least `purge_days` calendar days older (their 5-bar outcomes are
    resolved before the prediction time), refit on a fixed calendar cadence.
    Returns OOF metrics, the pre-registered acceptance verdict, and the
    final model fit on all data (for live advisory scoring).
    """
    import pandas as pd

    rows = []
    for record in records:
        rec_direction = getattr(record, "direction", None) or (record.get("direction") if isinstance(record, dict) else None)
        if str(rec_direction) != direction:
            continue
        features = getattr(record, "features", None) if not isinstance(record, dict) else record.get("features")
        timestamp = getattr(record, "timestamp", None) if not isinstance(record, dict) else record.get("timestamp")
        r_multiple = getattr(record, "r_multiple", None) if not isinstance(record, dict) else record.get("r_multiple")
        ticker = getattr(record, "ticker", "") if not isinstance(record, dict) else record.get("ticker", "")
        ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
        r_value = _finite(r_multiple)
        if pd.isna(ts) or not isinstance(features, dict) or not math.isfinite(r_value):
            continue
        rows.append({"ts": ts, "ticker": str(ticker), "timestamp": str(timestamp), "features": features, "r": r_value})

    rows.sort(key=lambda row: row["ts"])
    result: dict[str, Any] = {
        "direction": direction,
        "objective": objective,
        "model_class": "l2_ridge" if objective == "expected_r" else "l2_logistic_irls",
        "model_version": META_MODEL_VERSION,
        "feature_keys": list(META_FEATURE_KEYS),
        "config": {
            "l2_lambda": META_L2_LAMBDA,
            "refit_every_days": refit_every_days,
            "purge_days": purge_days,
            "min_train": META_MIN_TRAIN,
            "tail_r": META_TAIL_R,
        },
        "n_records": len(rows),
        "n_evaluated": 0,
        "predictions": {},
        "final_model": None,
    }
    if len(rows) < META_MIN_TRAIN + 50:
        result["metrics"] = {"insufficient": True}
        result["acceptance"] = {"passed": False, "reason": "insufficient_records"}
        return result

    purge = pd.Timedelta(days=purge_days)
    refit_interval = pd.Timedelta(days=refit_every_days)
    model = None
    model_fit_ts = None
    predictions: list[float] = []
    outcomes: list[float] = []
    win_labels: list[int] = []
    tail_labels: list[int] = []
    day_keys: list[str] = []
    row_ids: list[str] = []
    prediction_map: dict[str, float] = {}

    for row in rows:
        needs_refit = model_fit_ts is None or (row["ts"] - model_fit_ts) >= refit_interval
        if needs_refit:
            train = [r for r in rows if r["ts"] <= row["ts"] - purge]
            if len(train) >= META_MIN_TRAIN:
                candidate_model = fit_model(
                    [r["features"] for r in train], [r["r"] for r in train], objective=objective
                )
                if candidate_model is not None:
                    model = candidate_model
                    model_fit_ts = row["ts"]
        if model is None:
            continue
        score = predict_score(model, row["features"])
        if score is None:
            continue
        predictions.append(score)
        outcomes.append(row["r"])
        win_labels.append(1 if row["r"] > 0 else 0)
        tail_labels.append(1 if row["r"] >= META_TAIL_R else 0)
        day_keys.append(row["ts"].strftime("%Y-%m-%d"))
        row_id = f"{row['ticker']}|{row['timestamp']}"
        row_ids.append(row_id)
        prediction_map[row_id] = score

    n_eval = len(predictions)
    result["n_evaluated"] = n_eval
    result["predictions"] = prediction_map
    if n_eval < META_ACCEPT_MIN_N:
        result["metrics"] = {"insufficient": True, "n_evaluated": n_eval}
        result["acceptance"] = {"passed": False, "reason": "insufficient_out_of_fold_predictions"}
        return result

    ic = spearman_rank_ic(predictions, outcomes, day_keys=day_keys)
    lift = tercile_lift(predictions, outcomes, day_keys, row_ids=row_ids)
    tail = tail_retention(predictions, outcomes, row_ids=row_ids)

    # "Fitted beats naive" per objective family: Brier vs base rate for the
    # classifiers, OOF MSE vs the constant mean for the regression.
    if objective == "expected_r":
        naive = float(np.mean(outcomes))
        oof_loss = float(np.mean([(p - r) ** 2 for p, r in zip(predictions, outcomes, strict=False)]))
        naive_loss = float(np.mean([(naive - r) ** 2 for r in outcomes]))
    else:
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
    result["metrics"] = metrics

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
    result["acceptance"] = {"passed": all(criteria.values()), "criteria": criteria}

    result["final_model"] = fit_model([row["features"] for row in rows], [row["r"] for row in rows], objective=objective)
    return result


def walk_forward_calibration_suite(records: Iterable[Any], direction: str = "bullish") -> dict[str, dict]:
    """Run every pre-registered objective over the same record set."""
    materialized = list(records)
    return {
        objective: walk_forward_calibration(materialized, direction=direction, objective=objective)
        for objective in META_OBJECTIVES
    }


def select_shippable_objective(
    suite: dict[str, dict],
    previous_pass_by_objective: dict[str, bool],
) -> str | None:
    """Deterministic ship rule: pass NOW and on the PREVIOUS registered run
    (two-touch), then highest current OOF rank IC wins. Returns None when
    nothing qualifies - take-all remains the standing policy."""
    qualified: list[tuple[float, str]] = []
    for objective, result in suite.items():
        if not isinstance(result, dict):
            continue
        passed_now = bool((result.get("acceptance") or {}).get("passed"))
        passed_before = bool(previous_pass_by_objective.get(objective))
        if passed_now and passed_before and result.get("final_model") is not None:
            ic = float(((result.get("metrics") or {}).get("rank_ic_r") or {}).get("ic", 0.0))
            qualified.append((ic, objective))
    if not qualified:
        return None
    qualified.sort(reverse=True)
    return qualified[0][1]
