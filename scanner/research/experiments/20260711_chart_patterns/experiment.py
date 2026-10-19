"""E-chart-patterns: computable chart-shape features vs the 5-bar triple-
barrier R outcome (20260711 scaffold).

The goal is to test whether classic chart "shapes" carry within-direction
ranking signal, evaluated the same way every other ranking candidate in this
lab is evaluated: purged expanding-window walk-forward OOF, same 6-gate
acceptance as the production ranking model, no gate weakening, no peeking at
outcome correlations before this file (and preregistration.json) existed.

Data source: the edge_retrieval_index does NOT store raw per-bar OHLCV, only
the ~61 scalar fields extract_edge_features() computes per record (box
geometry, touch counts, volume/volatility summaries, doctrine scores, etc -
see scanner/edge/features.py). All 6 candidate pattern features below are
therefore pure functions of that already-stored, already-point-in-time
scalar dict - no new data dependency, no look-ahead (every input field is
itself computed from "bars up to and including the decision bar"). Two
classic shapes from the original brief - literal wick-rejection ratios and
gap state (open vs prior close) - need per-bar OHLC that is neither stored
in the index nor cached anywhere in this repo; fetching it live per
historical record would be slow, non-deterministic across runs, and violate
this lab's read-only/no-randomness experiment conventions, so they are
deliberately NOT built here. See preregistration.json's "excluded_patterns"
block and the top-level report for the full reasoning.

The walk-forward skeleton (expanding window, 21-day refit,
scanner.config.EDGE_EMBARGO_DAYS purge [11 calendar days as of the
2026-07-10 protocol hardening], min_train 300, ridge target=raw R,
_standardize_train) is copied from scanner.edge.calibration /
20260710_sprint/e1_robust_r/experiment.py, generalized over an arbitrary
feature-key tuple the way 20260710_sprint/e3_features/run_experiment.py
generalized its tail_prob harness - only the feature-key list changes per
cell, the fitting math is untouched and imported, not reimplemented.

Isolation: read-only against scanner/, additive only under this directory,
no writes to scanner/reports or scanner/models, deterministic (no
randomness anywhere in this file).

Run modes:
  python experiment.py               -> harness-sanity control ONLY.
                                         Reproduces the repo's own
                                         expected_r control IC within
                                         +/-0.005 using this file's
                                         generalized harness restricted to
                                         the standard 10 keys. Aborts loudly
                                         if it does not match. Pattern cells
                                         are NOT run.
  python experiment.py --run-cells   -> also runs the 13 pre-registered
                                         cells (6 solo + 1 combined + 6
                                         drop-one ablations) from
                                         preregistration.json. Do not pass
                                         this flag until the preregistration
                                         has been reviewed and locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
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
STANDARD_10: tuple[str, ...] = META_FEATURE_KEYS

# --------------------------------------------------------------------------
# Candidate pattern features: pure functions of one record's stored,
# point-in-time `features` dict. No bars, no randomness, no outcome fields.
# A small ratio floor keeps degenerate near-zero denominators from producing
# huge/unstable values; it does not change sign or ranking of well-formed
# rows.
# --------------------------------------------------------------------------

_RATIO_FLOOR = 0.05


def _pf_compression_breakout_thrust(f: dict) -> float:
    """Coiled-spring thrust: breakout strength scaled by how tight the prior
    range was. A given breakout_strength_pct out of a tighter
    range_compression_ratio should carry more continuation energy than the
    same strength out of a loose range - an interaction a linear model on
    the two features held separately cannot express."""
    breakout = _finite(f.get("breakout_strength_pct"))
    compression = _finite(f.get("range_compression_ratio"))
    if not (math.isfinite(breakout) and math.isfinite(compression)):
        return math.nan
    return breakout / max(compression, _RATIO_FLOOR)


def _pf_box_retest_asymmetry(f: dict) -> float:
    """Wyckoff-style base-defense asymmetry: (bottom_touches - top_touches)
    normalized to [-1, 1]-ish. A bullish base whose floor was tested/held
    more often than its ceiling before breaking out signals firmer
    accumulation than one where the ceiling saw more of the pre-breakout
    action. +1 smoothing avoids a 0/0 blowup when neither edge was
    touched."""
    bottom = _finite(f.get("bottom_touches"))
    top = _finite(f.get("top_touches"))
    if not (math.isfinite(bottom) and math.isfinite(top)):
        return math.nan
    return (bottom - top) / (bottom + top + 1.0)


def _pf_volume_thrust_interaction(f: dict) -> float:
    """Climax-volume shape: volume_expansion (vs recent average) times
    volume_percentile (vs recent history) - rows high on BOTH axes at once
    are a genuinely different shape than rows high on only one, which a
    linear model summing the two terms cannot isolate."""
    expansion = _finite(f.get("volume_expansion"))
    percentile = _finite(f.get("volume_percentile"))
    if not (math.isfinite(expansion) and math.isfinite(percentile)):
        return math.nan
    return expansion * percentile


def _pf_atr_relative_coil(f: dict) -> float:
    """NR7/inside-bar-style compression, but measured against the stock's
    own absolute volatility (ATR-as-pct-of-price) rather than against its
    own recent range (which is what range_compression_ratio already
    covers). A base that is tight relative to typical daily movement is a
    truer coil than one that is merely tight relative to its own
    (possibly already-quiet) recent history."""
    box_width = _finite(f.get("box_width_pct"))
    atr = _finite(f.get("atr_value"))
    close = _finite(f.get("latest_close"))
    if not (math.isfinite(box_width) and math.isfinite(atr) and math.isfinite(close)) or close <= 0:
        return math.nan
    atr_pct = (atr / close) * 100.0
    return box_width / max(atr_pct, _RATIO_FLOOR)


def _pf_box_stack_coil(f: dict) -> float:
    """Staircase-base shape: doctrine_v2_box_stack_score (how many prior
    bases already validated) times current tightness (1 - clipped
    range_compression_ratio). Isolates rows that are BOTH already-stacked
    AND currently coiling, a VCP/stage-2-staircase read that box_stack_score
    and range_compression_ratio alone (used elsewhere) don't capture
    jointly."""
    stack = _finite(f.get("doctrine_v2_box_stack_score"))
    compression = _finite(f.get("range_compression_ratio"))
    if not (math.isfinite(stack) and math.isfinite(compression)):
        return math.nan
    return stack * (1.0 - min(max(compression, 0.0), 1.0))


def _pf_breakout_snap_ratio(f: dict) -> float:
    """Measured-move overshoot: how large the breakout distance already is
    relative to the base's own height. A breakout that has already
    traveled a large fraction of its base height at signal time proxies for
    a partially pre-spent measured-move target - classic TA treats this as
    a chase-risk flag distinct from raw breakout_strength_pct alone."""
    distance = _finite(f.get("abs_breakout_distance_pct"))
    box_width = _finite(f.get("box_width_pct"))
    if not (math.isfinite(distance) and math.isfinite(box_width)):
        return math.nan
    return distance / max(box_width, _RATIO_FLOOR)


PATTERN_FEATURE_FNS = {
    "compression_breakout_thrust": _pf_compression_breakout_thrust,
    "box_retest_asymmetry": _pf_box_retest_asymmetry,
    "volume_thrust_interaction": _pf_volume_thrust_interaction,
    "atr_relative_coil": _pf_atr_relative_coil,
    "box_stack_coil": _pf_box_stack_coil,
    "breakout_snap_ratio": _pf_breakout_snap_ratio,
}
PATTERN_KEYS: tuple[str, ...] = tuple(PATTERN_FEATURE_FNS.keys())


def _augment_row_features(features: dict) -> dict:
    out = dict(features)
    for key, fn in PATTERN_FEATURE_FNS.items():
        out[key] = fn(features)
    return out


def _combined_keys(drop: str | None = None) -> tuple[str, ...]:
    pattern_part = [k for k in PATTERN_KEYS if k != drop]
    return tuple(list(STANDARD_10) + pattern_part)


CELLS: dict[str, tuple[str, ...]] = {}
for _pk in PATTERN_KEYS:
    CELLS[f"solo_{_pk}"] = (_pk,)
CELLS["combined_all_patterns"] = _combined_keys(drop=None)
for _pk in PATTERN_KEYS:
    CELLS[f"ablate_drop_{_pk}"] = _combined_keys(drop=_pk)


# --------------------------------------------------------------------------
# Harness (generalized skeleton copied from e1_robust_r / e3_features)
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
        augmented = _augment_row_features(features)
        rows.append({"ts": ts, "ticker": str(ticker), "timestamp": str(timestamp), "features": augmented, "r": r_value})
    rows.sort(key=lambda row: row["ts"])
    return rows


def _fit_cell_model(train_rows: list[dict], feature_keys: tuple[str, ...], l2_lambda: float) -> dict | None:
    n = len(train_rows)
    if n < META_MIN_TRAIN:
        return None
    raw = _feature_matrix([r["features"] for r in train_rows], feature_keys)
    y = np.array([r["r"] for r in train_rows], dtype=float)
    if not np.isfinite(y).all():
        return None
    transform = _standardize_train(raw)
    if transform is None:
        return None
    design = np.hstack([np.ones((n, 1)), transform["x"]])
    weights = _fit_ridge(design, y, l2_lambda)
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


def _predict(model: dict, features: dict, feature_keys: tuple[str, ...]) -> float | None:
    values = np.array([_finite(features.get(k)) for k in feature_keys], dtype=float)
    filled = np.where(np.isfinite(values), values, model["medians"])
    clipped = np.clip(filled, model["winsor_low"], model["winsor_high"])
    x = (clipped - model["means"]) / model["stds"]
    z = float(model["intercept"]) + float(np.dot(model["coefficients"], x))
    if not math.isfinite(z):
        return None
    return z


def walk_forward_generic(
    rows: list[dict],
    feature_keys: tuple[str, ...],
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
                candidate = _fit_cell_model(train, feature_keys, l2_lambda)
                if candidate is not None:
                    model = candidate
                    model_fit_ts = row["ts"]
        if model is None:
            continue
        score = _predict(model, row["features"], feature_keys)
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


def _index_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-cells",
        action="store_true",
        default=False,
        help="Run the 13 pre-registered pattern-feature cells. OFF by default - "
        "the preregistration must be reviewed/locked before scored cells run.",
    )
    args = parser.parse_args()

    records = load_edge_index(EDGE_INDEX_PATH)
    bullish_count = sum(1 for r in records if getattr(r, "direction", None) == "bullish")

    results: dict = {
        "n_records_total": len(records),
        "n_records_bullish": bullish_count,
        "index_sha256": _index_sha256(Path(EDGE_INDEX_PATH)),
        "run_cells_flag": args.run_cells,
        "harness_sanity": {},
        "cells": {},
    }

    # --- MANDATORY sanity check ------------------------------------------
    # Reproduce the repo's own expected_r control (no pattern features
    # involved at all) using this file's generalized harness restricted to
    # the standard 10 keys. This proves the wiring is correct before any
    # pattern cell is trusted - it is not a peek at pattern signal, since
    # zero pattern features participate in it.
    repo_expected_r = walk_forward_calibration(records, direction="bullish", objective="expected_r")
    repo_ic = repo_expected_r["metrics"]["rank_ic_r"]["ic"]
    repo_n = repo_expected_r["n_evaluated"]

    rows = _load_rows(records, direction="bullish")
    our_control_oof = walk_forward_generic(rows, STANDARD_10)
    our_control_eval = evaluate(our_control_oof)
    our_ic = our_control_eval["metrics"]["rank_ic_r"]["ic"] if not our_control_eval["metrics"].get("insufficient") else None
    our_n = our_control_eval["metrics"].get("n_evaluated")

    ic_match = our_ic is not None and abs(our_ic - repo_ic) <= 0.005
    results["harness_sanity"] = {
        "repo_expected_r_ic": repo_ic,
        "repo_expected_r_n_evaluated": repo_n,
        "our_harness_standard10_ic": our_ic,
        "our_harness_standard10_n_evaluated": our_n,
        "purge_days_used": META_PURGE_DAYS,
        "ic_match_within_0.005": ic_match,
        "harness_confirmed_ok": ic_match,
    }

    print(json.dumps(results["harness_sanity"], indent=2))

    if not ic_match:
        results["harness_sanity"]["FATAL"] = "Harness does not reproduce repo expected_r IC within tolerance. Aborting - no cells are trustworthy."
        out_path = OUT_DIR / "results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        raise SystemExit("HARNESS SANITY CHECK FAILED - see results.json. Aborting before any cell runs.")

    if not args.run_cells:
        print("---")
        print(
            "Harness sanity check PASSED. Pattern cells are pre-registered "
            f"({len(CELLS)} cells in preregistration.json) but NOT executed - "
            "pass --run-cells after the preregistration has been reviewed/locked."
        )
        results["cells"] = {"status": "not_run", "reason": "--run-cells not passed", "n_registered_cells": len(CELLS)}
        out_path = OUT_DIR / "results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        return

    # --- PRE-REGISTERED cells (only runs with --run-cells) ---------------
    control_ic = our_ic
    combined_ic: float | None = None
    for cell_id, feature_keys in CELLS.items():
        try:
            oof = walk_forward_generic(rows, feature_keys)
            cell_eval = evaluate(oof)
            entry = {"status": "ok", "feature_keys": list(feature_keys), **cell_eval}
            if not cell_eval["metrics"].get("insufficient"):
                cell_ic = float(cell_eval["metrics"]["rank_ic_r"]["ic"])
                entry["delta_ic_vs_standard10_control"] = round(cell_ic - control_ic, 4)
                if cell_id == "combined_all_patterns":
                    combined_ic = cell_ic
                if cell_id.startswith("ablate_drop_") and combined_ic is not None:
                    entry["delta_ic_vs_combined_all_patterns"] = round(cell_ic - combined_ic, 4)
            results["cells"][cell_id] = entry
        except Exception as exc:  # fail closed: report as errored, never silently skip
            results["cells"][cell_id] = {
                "status": "errored",
                "feature_keys": list(feature_keys),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

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
