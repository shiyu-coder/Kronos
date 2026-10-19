"""e4_decision: decision-layer statistical honesty analysis.

Read-only against scan_decisions.jsonl and the edge retrieval index.
Reuses scanner.edge.calibration.walk_forward_calibration for OOF scoring
(no re-implementation of CV). numpy + pandas only. Deterministic (seed 42
for the bootstrap). Writes results.json next to this script.

Run: .\\venv\\Scripts\\python.exe scanner\\research\\experiments\\20260710_sprint\\e4_decision\\run_analysis.py
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import EDGE_INDEX_PATH  # noqa: E402
from scanner.edge.calibration import walk_forward_calibration  # noqa: E402
from scanner.edge.retrieval import load_edge_index  # noqa: E402

JOURNAL_PATH = REPO_ROOT / "scanner" / "reports" / "scan_decisions.jsonl"
OUT_DIR = Path(__file__).resolve().parent
TODAY = pd.Timestamp("2026-07-10", tz="UTC")
SEED = 42
N_BOOT = 2000
BOOT_BLOCK_DAYS = 5
K_VALUES = (10, 25, 33, 50)
SLIPPAGE_BPS = (25, 50)


# ---------------------------------------------------------------------------
# shared stats helpers (kept local and simple -- classic textbook formulas)
# ---------------------------------------------------------------------------

def wilson_ci_95(wins: int, n: int) -> dict:
    """Two-sided 95% Wilson score interval, z=1.96."""
    if n <= 0:
        return {"n": 0, "wins": 0, "p_hat": None, "low": None, "high": None}
    z = 1.959963984540054
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return {"n": n, "wins": wins, "p_hat": round(p, 4), "low": round(low, 4), "high": round(high, 4)}


def one_sample_proportion_n(p0: float, p1: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Classic one-sample proportion sample size (unpooled variance), two-sided."""
    if not (0 < alpha < 1 and 0 < power < 1 and 0 < p0 < 1 and 0 < p1 < 1 and p0 != p1):
        raise ValueError("alpha, power, p0, and p1 must define a non-degenerate proportion test")
    normal = NormalDist()
    z_a = normal.inv_cdf(1.0 - alpha / 2.0)
    z_b = normal.inv_cdf(power)
    num = z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))
    n = (num / (p1 - p0)) ** 2
    return math.ceil(n)


def r_summary(r_values: np.ndarray) -> dict:
    r_values = r_values[np.isfinite(r_values)]
    n = len(r_values)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "win_rate": round(float(np.mean(r_values > 0)), 4),
        "avg_r": round(float(np.mean(r_values)), 4),
        "median_r": round(float(np.median(r_values)), 4),
        "tail_rate_r_ge_2": round(float(np.mean(r_values >= 2.0)), 4),
    }


# ---------------------------------------------------------------------------
# Task 1: journal honesty
# ---------------------------------------------------------------------------

def task1_journal_honesty() -> dict:
    lines = [json.loads(l) for l in JOURNAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    candidates = [r for r in lines if r.get("skip_reason") == "research_candidate"]
    resolved = [r for r in candidates if r.get("outcome_status") == "resolved"]
    pending = [r for r in candidates if r.get("outcome_status") == "pending"]
    wins = sum(1 for r in resolved if r.get("outcome_label") == "win")
    losses = sum(1 for r in resolved if r.get("outcome_label") == "loss")
    n = len(resolved)

    ci = wilson_ci_95(wins, n)
    p0 = wins / n if n else 0.0

    resolution_rate_per_week = 12.0  # given by task background
    targets = {}
    for label, delta in (("+5pp", 0.05), ("+10pp", 0.10)):
        p1 = p0 + delta
        n_needed = one_sample_proportion_n(p0, p1) if 0 < p1 < 1 else None
        if n_needed is None:
            targets[label] = {"p1": round(p1, 4), "n_needed": None, "weeks": None, "earliest_date": None}
            continue
        weeks = math.ceil(n_needed / resolution_rate_per_week)
        earliest = (TODAY + pd.Timedelta(weeks=weeks)).strftime("%Y-%m-%d")
        targets[label] = {
            "p1_target_wr": round(p1, 4),
            "n_resolved_needed": n_needed,
            "weeks_at_12_per_week": weeks,
            "assumption_bound_date": earliest,
            "assumptions": "fixed p0, true effect exactly target delta, 12 independent resolutions/week",
        }

    headline = (
        f"under the stated fixed-baseline/12-per-week assumptions, the planned sample sizes are reached "
        f"around {targets['+5pp']['assumption_bound_date']} (for +5pp) / "
        f"{targets['+10pp']['assumption_bound_date']} (for +10pp)"
    )

    return {
        "n_candidates_total": len(candidates),
        "n_resolved": n,
        "n_pending": len(pending),
        "wins": wins,
        "losses": losses,
        "wilson_95ci": ci,
        "method": (
            "Wilson score interval, two-sided z=1.96, 95% CI on wins/n. "
            "Power analysis: one-sample proportion test vs fixed baseline p0, "
            "unpooled-variance sample size formula, alpha=0.05 two-sided, power=0.80 "
            "(z_alpha/2=1.96, z_beta=0.8416); n=(z_a*sqrt(p0(1-p0))+z_b*sqrt(p1(1-p1)))^2/(p1-p0)^2; "
            "weeks = ceil(n_needed / 12_resolutions_per_week)."
        ),
        "power_analysis": targets,
        "headline_deliverable": headline,
        "note_on_stated_background": (
            "Task background cited ~36 resolved (15W/21L); as of 2026-07-10 the journal "
            "actually has fewer resolved research candidates than that (see n_resolved above) "
            "-- reported as measured, not reconciled to the stated approximation."
        ),
    }


# ---------------------------------------------------------------------------
# Task 2: take-all baseline (index, bullish)
# ---------------------------------------------------------------------------

def task2_take_all_baseline(bullish_df: pd.DataFrame) -> dict:
    overall = r_summary(bullish_df["r"].to_numpy())
    ts = bullish_df["ts"]
    span_days = int((ts.max() - ts.min()).days)
    n_unique_days = int(ts.dt.strftime("%Y-%m-%d").nunique())

    quarters = bullish_df["ts"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    per_quarter = {}
    for q, grp in bullish_df.groupby(quarters):
        per_quarter[q] = r_summary(grp["r"].to_numpy())

    wr_values = [v["win_rate"] for v in per_quarter.values() if v.get("n", 0) >= 20]
    wr_range = (min(wr_values), max(wr_values)) if wr_values else (None, None)

    return {
        "overall": overall,
        "date_range": {
            "min": str(ts.min()),
            "max": str(ts.max()),
            "calendar_span_days": span_days,
            "unique_trading_days": n_unique_days,
        },
        "per_quarter": per_quarter,
        "quarter_wr_range_n_ge_20": {"low": wr_range[0], "high": wr_range[1]},
        "stability_note": (
            "WR range across quarters with n>=20 spans "
            f"{wr_range[0]}-{wr_range[1]}" if wr_values else "insufficient per-quarter n to assess"
        ),
    }


# ---------------------------------------------------------------------------
# Task 3: top-K / bottom-K counterfactuals with day-block bootstrap
# ---------------------------------------------------------------------------

def _row_id(ticker: str, timestamp: str) -> str:
    return f"{ticker}|{timestamp}"


def _select_and_score(sorted_rows: list[tuple], k_frac: float, tail_r: float, from_top: bool) -> dict:
    """sorted_rows: list of (score, r, day) sorted DESC by score already."""
    n = len(sorted_rows)
    take = max(1, round(k_frac * n))
    subset = sorted_rows[:take] if from_top else sorted_rows[-take:]
    r_vals = np.array([row[1] for row in subset], dtype=float)
    wr = float(np.mean(r_vals > 0)) if len(r_vals) else 0.0
    avg_r = float(np.mean(r_vals)) if len(r_vals) else 0.0
    tail_total = sum(1 for row in sorted_rows if row[1] >= tail_r)
    tail_in_subset = sum(1 for row in subset if row[1] >= tail_r)
    tail_capture = (tail_in_subset / tail_total) if tail_total else None
    return {"n": len(subset), "win_rate": wr, "avg_r": avg_r, "tail_capture": tail_capture}


def _bootstrap_deltas(
    rows_by_day: dict[str, list[tuple]],
    days: list[str],
    k_frac: float,
    tail_r: float,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    top_wr_deltas, top_r_deltas = [], []
    bot_wr_deltas, bot_r_deltas = [], []
    for _ in range(n_boot):
        drawn = []
        while len(drawn) < len(days):
            start = int(rng.integers(0, len(days)))
            drawn.extend((start + offset) % len(days) for offset in range(BOOT_BLOCK_DAYS))
        drawn = drawn[: len(days)]
        pooled: list[tuple] = []
        for idx in drawn:
            pooled.extend(rows_by_day[days[int(idx)]])
        if len(pooled) < 9:
            continue
        r_all = np.array([row[1] for row in pooled], dtype=float)
        takeall_wr = float(np.mean(r_all > 0))
        takeall_avg = float(np.mean(r_all))
        ordered = sorted(pooled, key=lambda row: -row[0])
        top = _select_and_score(ordered, k_frac, tail_r, from_top=True)
        bot = _select_and_score(ordered, k_frac, tail_r, from_top=False)
        top_wr_deltas.append(top["win_rate"] - takeall_wr)
        top_r_deltas.append(top["avg_r"] - takeall_avg)
        bot_wr_deltas.append(bot["win_rate"] - takeall_wr)
        bot_r_deltas.append(bot["avg_r"] - takeall_avg)

    def pct(vals):
        if not vals:
            return {"low": None, "high": None}
        return {"low": round(float(np.percentile(vals, 2.5)), 4), "high": round(float(np.percentile(vals, 97.5)), 4)}

    return {
        "top_wr_delta_ci95": pct(top_wr_deltas),
        "top_avgR_delta_ci95": pct(top_r_deltas),
        "bottom_wr_delta_ci95": pct(bot_wr_deltas),
        "bottom_avgR_delta_ci95": pct(bot_r_deltas),
        "n_boot_used": len(top_wr_deltas),
        "bootstrap_method": "circular_moving_entry_day_blocks",
        "bootstrap_block_days": BOOT_BLOCK_DAYS,
    }


def task3_topk_counterfactuals(records) -> dict:
    out = {}
    for objective in ("p_win", "expected_r", "tail_prob"):
        result = walk_forward_calibration(records, direction="bullish", objective=objective)
        predictions = result.get("predictions") or {}
        n_evaluated = result.get("n_evaluated", 0)

        if not predictions:
            out[objective] = {
                "n_evaluated": n_evaluated,
                "insufficient": True,
                "acceptance": result.get("acceptance"),
                "note": "no OOF predictions returned; skipping top-K counterfactual for this objective",
            }
            continue

        rows = []
        for rec in records:
            if rec.direction != "bullish":
                continue
            rid = _row_id(rec.ticker, rec.timestamp)
            score = predictions.get(rid)
            if score is None:
                continue
            ts = pd.to_datetime(rec.timestamp, errors="coerce", utc=True)
            if pd.isna(ts) or not math.isfinite(rec.r_multiple):
                continue
            day = ts.strftime("%Y-%m-%d")
            rows.append((float(score), float(rec.r_multiple), day))

        n = len(rows)
        r_all = np.array([row[1] for row in rows], dtype=float)
        takeall = r_summary(r_all)

        rows_by_day: dict[str, list[tuple]] = {}
        for row in rows:
            rows_by_day.setdefault(row[2], []).append(row)
        days = sorted(rows_by_day)

        ordered_full = sorted(rows, key=lambda row: -row[0])

        k_results = {}
        for k in K_VALUES:
            k_frac = k / 100.0
            top = _select_and_score(ordered_full, k_frac, 2.0, from_top=True)
            bottom = _select_and_score(ordered_full, k_frac, 2.0, from_top=False)
            boot = _bootstrap_deltas(rows_by_day, days, k_frac, 2.0, N_BOOT, SEED)
            k_results[str(k)] = {
                "top": top,
                "bottom": bottom,
                "bootstrap_2000iter_seed42": boot,
                "label": "descriptive counterfactual -- selection uses OOF scores but K was not pre-registered before today",
            }

        out[objective] = {
            "n_evaluated_oof": n_evaluated,
            "n_matched_to_r_multiple": n,
            "n_distinct_days": len(days),
            "acceptance": result.get("acceptance"),
            "metrics": result.get("metrics"),
            "take_all_on_evaluable_subset": takeall,
            "top_bottom_k": k_results,
        }
    return out


# ---------------------------------------------------------------------------
# Task 4: bearish check
# ---------------------------------------------------------------------------

def task4_bearish_check(records) -> dict:
    out = {}
    for objective in ("p_win", "expected_r", "tail_prob"):
        result = walk_forward_calibration(records, direction="bearish", objective=objective)
        out[objective] = {
            "n_records": result.get("n_records"),
            "n_evaluated": result.get("n_evaluated"),
            "metrics": result.get("metrics"),
            "acceptance": result.get("acceptance"),
        }
    any_ranks = any(
        (v.get("acceptance") or {}).get("passed") for v in out.values()
    )
    return {
        "per_objective": out,
        "any_objective_ranks_bearish_oof": any_ranks,
        "context": (
            "scanner/brief.py hardcodes bearish as blocked on negative take-all expectancy "
            "('Bearish setups have negative expectancy in validation'; "
            "scanner/reports/daily_brief.md showed bearish n=563 avgR -0.23 BLOCKED as of last brief run). "
            "Even if an objective clears OOF rank-IC acceptance here, it ranks within a direction whose "
            "take-all expectancy is negative -- promotion stays blocked on that gate regardless of ranking skill."
        ),
    }


# ---------------------------------------------------------------------------
# Task 5: cost reality
# ---------------------------------------------------------------------------

def task5_cost_reality(bullish_df: pd.DataFrame) -> dict:
    out = {"classification": "fixed-return-subtraction sensitivity scenario; not an execution/slippage model", "assumptions": (
        "adjusted_return_pct = outcome_return_pct - 2*slippage_decimal*100 "
        "(entry leg + exit leg, both against the trade direction, on the underlying price move, "
        "in percentage points). adjusted_r = clip(adjusted_return_pct / risk_pct_used, -10, 10) using the "
        "SAME stored risk_pct_used (assumes slippage does not materially move the stop-distance-in-percent "
        "denominator). No bar-level re-walk of the triple barrier -- approximate floor-level sensitivity only."
    )}
    baseline = r_summary(bullish_df["r"].to_numpy())
    out["baseline_no_slippage"] = baseline

    risk = bullish_df["risk_pct_used"].to_numpy()
    ret = bullish_df["outcome_return_pct"].to_numpy()
    valid = np.isfinite(risk) & (risk > 0) & np.isfinite(ret)
    out["n_valid_risk_field"] = int(valid.sum())
    out["n_total"] = int(len(bullish_df))

    for bps in SLIPPAGE_BPS:
        slip_decimal = bps / 10000.0
        adj_ret = np.where(valid, ret - 2 * slip_decimal * 100.0, np.nan)
        adj_r = np.clip(adj_ret / np.where(valid, risk, np.nan), -10, 10)
        adj_r = adj_r[valid]
        out[f"slippage_{bps}bps_per_side"] = r_summary(adj_r)
        out[f"slippage_{bps}bps_per_side"]["delta_wr_vs_baseline"] = round(
            out[f"slippage_{bps}bps_per_side"]["win_rate"] - baseline["win_rate"], 4
        )
        out[f"slippage_{bps}bps_per_side"]["delta_avgR_vs_baseline"] = round(
            out[f"slippage_{bps}bps_per_side"]["avg_r"] - baseline["avg_r"], 4
        )
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    records = load_edge_index(EDGE_INDEX_PATH)
    bullish_records = [r for r in records if r.direction == "bullish"]

    bullish_rows = []
    for rec in bullish_records:
        ts = pd.to_datetime(rec.timestamp, errors="coerce", utc=True)
        if pd.isna(ts) or not math.isfinite(rec.r_multiple):
            continue
        bullish_rows.append(
            {
                "ticker": rec.ticker,
                "timestamp": rec.timestamp,
                "ts": ts,
                "r": rec.r_multiple,
                "outcome_return_pct": rec.outcome_return_pct,
                "risk_pct_used": rec.risk_pct_used,
            }
        )
    bullish_df = pd.DataFrame(bullish_rows)

    results = {
        "run_metadata": {
            "date": "2026-07-10",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_index_records_total": len(records),
            "n_bullish": len(bullish_records),
            "n_bearish": sum(1 for r in records if r.direction == "bearish"),
            "seed": SEED,
            "n_boot": N_BOOT,
            "bootstrap_block_days": BOOT_BLOCK_DAYS,
            "input_sha256": {
                "edge_index": sha256_file(Path(EDGE_INDEX_PATH)),
                "journal": sha256_file(JOURNAL_PATH),
            },
            "protocol_sha256": {
                "preregistration": sha256_file(OUT_DIR / "preregistration.json"),
                "runner": sha256_file(Path(__file__)),
                "calibration": sha256_file(REPO_ROOT / "scanner" / "edge" / "calibration.py"),
                "stats": sha256_file(REPO_ROOT / "scanner" / "edge" / "stats.py"),
            },
            "reproduction_note": "Exact reproduction requires every listed hash to match; record counts alone are insufficient.",
        },
        "task1_journal_honesty": task1_journal_honesty(),
        "task2_take_all_baseline": task2_take_all_baseline(bullish_df),
        "task3_topk_counterfactuals": task3_topk_counterfactuals(records),
        "task4_bearish_check": task4_bearish_check(records),
        "task5_cost_reality": task5_cost_reality(bullish_df),
    }

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(results["task1_journal_honesty"]["headline_deliverable"]))


if __name__ == "__main__":
    main()
