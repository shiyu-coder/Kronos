"""E3 feature-set experiment: drop-one ablation + unused-field extensions on
tail_prob rank IC, bullish direction, walk-forward OOF only.

Read-only against scanner/edge/*. Reuses _standardize_train, _fit_logistic_irls,
_feature_matrix, predict_score straight from scanner.edge.calibration (no
duplication of the fitting math). The walk-forward loop below is a direct
copy of calibration.walk_forward_calibration's skeleton, generalized over an
arbitrary feature-key tuple and an optional list of per-row interaction pairs
instead of the hardcoded META_FEATURE_KEYS / p_win-or-tail_prob branch.

Writes only inside this experiment folder: results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scanner.config import EDGE_INDEX_PATH
from scanner.edge.calibration import (
    META_ACCEPT_MAX_P_DAY,
    META_ACCEPT_MIN_IC,
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


def _brier(probabilities: list[float], labels: list[int]) -> float:
    pairs = [(p, y) for p, y in zip(probabilities, labels, strict=False) if p is not None]
    if not pairs:
        return 1.0
    return float(np.mean([(p - y) ** 2 for p, y in pairs]))


def fit_model_generic(
    feature_dicts: list[dict],
    r_multiples: list[float],
    feature_keys: tuple[str, ...],
    tail_r: float = META_TAIL_R,
    l2_lambda: float = META_L2_LAMBDA,
    min_train: int = META_MIN_TRAIN,
    min_class_events: int = META_MIN_CLASS_EVENTS,
) -> dict | None:
    """tail_prob objective only. Mirrors calibration.fit_model's structure,
    generalized over feature_keys."""
    raw = _feature_matrix(feature_dicts, feature_keys)
    r_values = np.array([_finite(r) for r in r_multiples], dtype=float)
    usable = np.isfinite(r_values)
    raw = raw[usable]
    r_values = r_values[usable]
    n = len(r_values)
    if n < min_train:
        return None

    y = (r_values >= tail_r).astype(float)
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
        "objective": "tail_prob",
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


def _augment_with_interactions(features: dict, interaction_pairs: list[tuple[str, str]] | None) -> dict:
    if not interaction_pairs:
        return features
    out = dict(features)
    for a, b in interaction_pairs:
        va = _finite(features.get(a))
        vb = _finite(features.get(b))
        key = f"{a}*{b}"
        out[key] = va * vb if (math.isfinite(va) and math.isfinite(vb)) else math.nan
    return out


def walk_forward_generic(
    records,
    direction: str,
    feature_keys: tuple[str, ...],
    interaction_pairs: list[tuple[str, str]] | None = None,
    tail_r: float = META_TAIL_R,
    l2_lambda: float = META_L2_LAMBDA,
    refit_every_days: int = META_REFIT_EVERY_DAYS,
    purge_days: int = META_PURGE_DAYS,
    min_train: int = META_MIN_TRAIN,
    min_class_events: int = META_MIN_CLASS_EVENTS,
    min_accept_n: int = META_ACCEPT_MIN_N,
) -> dict:
    """Direct generalization of calibration.walk_forward_calibration:
    expanding window, fixed refit cadence, purge, min_train, min class
    events per fit, OOF-only evaluation. tail_prob objective only."""
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
        if pd.isna(ts) or not isinstance(features, dict) or not math.isfinite(r_value):
            continue
        feats = _augment_with_interactions(features, interaction_pairs)
        rows.append({"ts": ts, "ticker": str(ticker), "timestamp": str(timestamp), "features": feats, "r": r_value})

    rows.sort(key=lambda row: row["ts"])
    result: dict[str, Any] = {
        "direction": direction,
        "objective": "tail_prob",
        "feature_keys": list(feature_keys),
        "config": {
            "l2_lambda": l2_lambda,
            "refit_every_days": refit_every_days,
            "purge_days": purge_days,
            "min_train": min_train,
            "tail_r": tail_r,
        },
        "n_records": len(rows),
        "n_evaluated": 0,
    }
    if len(rows) < min_train + 50:
        result["metrics"] = {"insufficient": True}
        result["acceptance"] = {"passed": False, "reason": "insufficient_records"}
        return result

    purge = pd.Timedelta(days=purge_days)
    refit_interval = pd.Timedelta(days=refit_every_days)
    model = None
    model_fit_ts = None
    predictions: list[float] = []
    outcomes: list[float] = []
    tail_labels: list[int] = []
    day_keys: list[str] = []
    row_ids: list[str] = []

    for row in rows:
        needs_refit = model_fit_ts is None or (row["ts"] - model_fit_ts) >= refit_interval
        if needs_refit:
            train = [r for r in rows if r["ts"] <= row["ts"] - purge]
            if len(train) >= min_train:
                candidate_model = fit_model_generic(
                    [r["features"] for r in train],
                    [r["r"] for r in train],
                    feature_keys,
                    tail_r=tail_r,
                    l2_lambda=l2_lambda,
                    min_train=min_train,
                    min_class_events=min_class_events,
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
        tail_labels.append(1 if row["r"] >= tail_r else 0)
        day_keys.append(row["ts"].strftime("%Y-%m-%d"))
        row_ids.append(f"{row['ticker']}|{row['timestamp']}")

    n_eval = len(predictions)
    result["n_evaluated"] = n_eval
    if n_eval < min_accept_n:
        result["metrics"] = {"insufficient": True, "n_evaluated": n_eval}
        result["acceptance"] = {"passed": False, "reason": "insufficient_out_of_fold_predictions"}
        return result

    ic = spearman_rank_ic(predictions, outcomes, day_keys=day_keys)
    lift = tercile_lift(predictions, outcomes, day_keys, row_ids=row_ids)
    tail = tail_retention(predictions, outcomes, row_ids=row_ids, tail_r=tail_r)

    base_rate = float(np.mean(tail_labels))
    oof_loss = _brier(predictions, tail_labels)
    naive_loss = float(np.mean([(base_rate - y) ** 2 for y in tail_labels]))
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
        "realized_tail_rate": round(base_rate, 4),
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
    return result


def main() -> None:
    prereg = json.loads((EXPERIMENT_DIR / "preregistration.json").read_text(encoding="utf-8"))
    records = load_edge_index(EDGE_INDEX_PATH)

    # ---- Step 1: harness sanity check ----
    from scanner.edge.calibration import walk_forward_calibration

    repo_control = walk_forward_calibration(records, direction="bullish", objective="tail_prob")
    repo_ic = float(repo_control["metrics"]["rank_ic_r"]["ic"])

    own_control = walk_forward_generic(records, direction="bullish", feature_keys=STANDARD_10)
    own_ic = float(own_control["metrics"]["rank_ic_r"]["ic"])

    sanity = {
        "repo_control_ic": repo_ic,
        "repo_control_n_evaluated": repo_control["n_evaluated"],
        "own_harness_control_ic": own_ic,
        "own_harness_control_n_evaluated": own_control["n_evaluated"],
        "abs_diff": round(abs(repo_ic - own_ic), 6),
        "within_tolerance_0.005": abs(repo_ic - own_ic) <= 0.005,
        "matches_expected_minus_0.066_within_0.005": abs(repo_ic - (-0.066)) <= 0.005,
    }
    print("SANITY CHECK:", json.dumps(sanity, indent=2))
    if not sanity["within_tolerance_0.005"]:
        raise SystemExit("Harness sanity check FAILED - own harness does not match repo control. Aborting.")
    if not sanity["matches_expected_minus_0.066_within_0.005"]:
        raise SystemExit("Repo control does not match documented -0.066 IC within tolerance. Aborting.")

    control_ic = own_ic  # delta-IC baseline for ablations

    results: dict[str, Any] = {"sanity_check": sanity, "control": own_control, "cells": {}}

    # ---- Step 2: run every pre-registered cell ----
    for cell in prereg["cells"]:
        cell_id = cell["id"]
        name = cell["name"]
        if "keys_dropped" in cell:
            keys = tuple(k for k in STANDARD_10 if k not in cell["keys_dropped"])
        else:
            keys = tuple(cell["keys"])
        interaction_pairs = [tuple(p) for p in cell.get("interaction_pairs", [])] or None

        print(f"\n--- Cell {cell_id}: {name} ({len(keys)} keys) ---")
        outcome = walk_forward_generic(
            records,
            direction="bullish",
            feature_keys=keys,
            interaction_pairs=interaction_pairs,
        )
        metrics = outcome.get("metrics", {})
        if not metrics.get("insufficient"):
            outcome["delta_ic_vs_control"] = round(float(metrics["rank_ic_r"]["ic"]) - control_ic, 4)
        else:
            outcome["delta_ic_vs_control"] = None
        results["cells"][f"cell_{cell_id}_{name}"] = outcome

        if metrics.get("insufficient"):
            print(f"  INSUFFICIENT: {outcome['acceptance']['reason']}")
        else:
            ic = metrics["rank_ic_r"]
            print(f"  n_evaluated={outcome['n_evaluated']} ic={ic['ic']} p_day={ic.get('p_value_day_clustered')} "
                  f"delta_ic={outcome['delta_ic_vs_control']} passed={outcome['acceptance']['passed']}")

    out_path = EXPERIMENT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
