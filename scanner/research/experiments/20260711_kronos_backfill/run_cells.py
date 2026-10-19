"""Score the pre-registered Kronos ranking cells against a built snapshot.

Loads scanner/research/experiments/20260711_kronos_backfill/kronos_index_snapshot.json
(a Kronos-populated rebuild of the edge index - see run_backfill.py), reproduces
the repo's own expected_r control on the SAME records as a live harness-fidelity
check (abort if it does not match within +/-0.005, zero kronos features involved),
then runs the 6 pre-registered cells from preregistration.json:

  control_standard10                        META_FEATURE_KEYS (10)               [control]
  combined_standard10_plus_kronos3          META_FEATURE_KEYS + 3 kronos (13)    [kronos test]
  solo_kronos3                              3 kronos only                        [kronos test]
  ablate_drop_kronos_directional_agreement  combined minus that field (12)       [ablation]
  ablate_drop_kronos_median_forecast_return_pct  combined minus that field (12)  [ablation]
  ablate_drop_kronos_worst_sampled_return_pct    combined minus that field (12)  [ablation]

The walk-forward harness (expanding window, 21-day refit,
scanner.config.EDGE_EMBARGO_DAYS purge, min_train 300, ridge on raw R,
_standardize_train) and the 6-gate acceptance are copied/imported from
scanner.edge.calibration exactly as 20260711_chart_patterns/experiment.py did -
only the feature-key list changes per cell, the fitting math is imported, not
reimplemented. No gate is softened. All cells are reported regardless of outcome;
an errored cell reports status:errored with its traceback.

Isolation: read-only against scanner/ and the snapshot; writes only results.json
under this directory. Deterministic (ridge closed-form; bootstrap uses the repo's
fixed seed). No scanner.main execution.

Run modes:
  python run_cells.py                 -> harness-sanity control ONLY. Aborts if the
                                         reproduction does not match. Cells NOT run.
  python run_cells.py --run-cells     -> also runs the 6 pre-registered cells.
                                         Do not pass until the prereg is locked.
  python run_cells.py --snapshot PATH -> score a different snapshot (e.g. a temp
                                         sanity snapshot). Default is the real one.
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

DIR = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = DIR / "kronos_index_snapshot.json"
STANDARD_10: tuple[str, ...] = META_FEATURE_KEYS
KRONOS_3: tuple[str, ...] = (
    "kronos_directional_agreement",
    "kronos_median_forecast_return_pct",
    "kronos_worst_sampled_return_pct",
)


def _combined_keys(drop: str | None = None) -> tuple[str, ...]:
    kron = [k for k in KRONOS_3 if k != drop]
    return tuple(list(STANDARD_10) + kron)


CELLS: dict[str, tuple[str, ...]] = {
    "control_standard10": STANDARD_10,
    "combined_standard10_plus_kronos3": _combined_keys(drop=None),
    "solo_kronos3": KRONOS_3,
    "ablate_drop_kronos_directional_agreement": _combined_keys(drop="kronos_directional_agreement"),
    "ablate_drop_kronos_median_forecast_return_pct": _combined_keys(drop="kronos_median_forecast_return_pct"),
    "ablate_drop_kronos_worst_sampled_return_pct": _combined_keys(drop="kronos_worst_sampled_return_pct"),
}


# --------------------------------------------------------------------------
# Harness (generalized skeleton, identical math to chart_patterns/e1)
# --------------------------------------------------------------------------


def _load_rows(records, direction: str = "bullish") -> list[dict]:
    rows = []
    for record in records:
        if str(getattr(record, "direction", None)) != direction:
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
        return {
            "metrics": {"insufficient": True, "n_evaluated": n_eval},
            "acceptance": {"passed": False, "reason": "insufficient_out_of_fold_predictions"},
        }

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


def _kronos_availability(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n_bullish_rows": n}
    for key in KRONOS_3:
        finite = sum(1 for r in rows if isinstance(r["features"].get(key), (int, float)) and math.isfinite(float(r["features"].get(key))))
        out[key] = {"finite": finite, "finite_rate": round(finite / n, 4) if n else 0.0}
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-cells", action="store_true", default=False, help="Run the 6 cells. OFF by default.")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--results", default=str(DIR / "results.json"), help="Where results.json is written (point at a temp path for smoke runs).")
    args = parser.parse_args()

    results_path = Path(args.results)

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise SystemExit(f"snapshot not found: {snapshot_path}. Build it first with run_backfill.py.")

    records = load_edge_index(snapshot_path)
    bullish = sum(1 for r in records if getattr(r, "direction", None) == "bullish")
    rows = _load_rows(records, direction="bullish")

    results: dict = {
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": _sha256(snapshot_path),
        "n_records_total": len(records),
        "n_records_bullish": bullish,
        "kronos_feature_availability": _kronos_availability(rows),
        "run_cells_flag": args.run_cells,
        "purge_days_used": META_PURGE_DAYS,
        "harness_sanity": {},
        "cells": {},
    }

    # --- MANDATORY live self-consistency control -------------------------
    # Reproduce the repo's own expected_r control (zero kronos features) with
    # this file's generalized harness restricted to the standard 10 keys, on
    # the SAME snapshot records. Proves harness fidelity before any cell is
    # trusted. Not a peek at kronos signal - no kronos feature participates.
    repo_expected_r = walk_forward_calibration(records, direction="bullish", objective="expected_r")
    repo_metrics = repo_expected_r.get("metrics", {})
    repo_ic = None if repo_metrics.get("insufficient") else repo_metrics.get("rank_ic_r", {}).get("ic")
    repo_n = repo_expected_r.get("n_evaluated")

    our_control_eval = evaluate(walk_forward_generic(rows, STANDARD_10))
    our_ic = None if our_control_eval["metrics"].get("insufficient") else our_control_eval["metrics"]["rank_ic_r"]["ic"]
    our_n = our_control_eval["metrics"].get("n_evaluated")

    ic_match = (repo_ic is not None) and (our_ic is not None) and abs(our_ic - repo_ic) <= 0.005
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

    def _write() -> None:
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    if not ic_match:
        results["harness_sanity"]["FATAL"] = (
            "Harness does not reproduce repo expected_r IC within +/-0.005 on this snapshot "
            "(or the snapshot is too small to reach sufficiency). Aborting - no cell is trustworthy."
        )
        _write()
        raise SystemExit("HARNESS SANITY CHECK FAILED - see results.json. Aborting before any cell runs.")

    if not args.run_cells:
        print("---")
        print(
            f"Harness sanity PASSED. {len(CELLS)} cells are pre-registered but NOT executed - "
            "pass --run-cells after the preregistration is reviewed/locked."
        )
        results["cells"] = {"status": "not_run", "reason": "--run-cells not passed", "n_registered_cells": len(CELLS)}
        _write()
        return 0

    # --- PRE-REGISTERED cells (only with --run-cells) --------------------
    control_ic = our_ic
    combined_ic: float | None = None
    for cell_id, feature_keys in CELLS.items():
        try:
            cell_eval = evaluate(walk_forward_generic(rows, feature_keys))
            entry = {"status": "ok", "feature_keys": list(feature_keys), **cell_eval}
            if not cell_eval["metrics"].get("insufficient"):
                cell_ic = float(cell_eval["metrics"]["rank_ic_r"]["ic"])
                entry["delta_ic_vs_standard10_control"] = round(cell_ic - control_ic, 4)
                if cell_id == "combined_standard10_plus_kronos3":
                    combined_ic = cell_ic
                if cell_id.startswith("ablate_drop_") and combined_ic is not None:
                    entry["delta_ic_vs_combined_standard10_plus_kronos3"] = round(cell_ic - combined_ic, 4)
            results["cells"][cell_id] = entry
        except Exception as exc:  # fail closed: report, never silently skip
            results["cells"][cell_id] = {
                "status": "errored",
                "feature_keys": list(feature_keys),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    _write()

    print("---")
    for cell_id, res in results["cells"].items():
        if res["status"] == "errored":
            print(f"{cell_id:48s} ERRORED: {res['error']}")
            continue
        m = res["metrics"]
        if m.get("insufficient"):
            print(f"{cell_id:48s} INSUFFICIENT n={m.get('n_evaluated')}")
            continue
        ic = m["rank_ic_r"]
        lift = m["tercile_lift"]
        print(
            f"{cell_id:48s} n={m['n_evaluated']:5d} ic={ic.get('ic'):+.4f} "
            f"p_day={ic.get('p_value_day_clustered')} spread_ci_low={lift.get('spread_ci_low')} "
            f"beats_naive={m['beats_naive']} d_ic={res.get('delta_ic_vs_standard10_control')} "
            f"PASS={res['acceptance']['passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
