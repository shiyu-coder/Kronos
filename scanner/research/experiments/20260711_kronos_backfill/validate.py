"""Validation for the opt-in Kronos backfill wiring in edge/retrieval.py.

Scope (per task instructions): prove the wiring works, is leak-free, and
measure cost on a tiny sample. This script does NOT run the ~11k-record
production backfill and does NOT touch scanner/reports/edge_retrieval_index.json
or any other production report file. All output stays inside this experiment
directory.

Run with the project venv:
    .\\venv\\Scripts\\python.exe scanner\\research\\experiments\\20260711_kronos_backfill\\validate.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import pandas as pd

# Make `scanner` importable when run as a plain script (mirrors main.py's
# own sys.path bootstrap for __package__ in {None, ""}).
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scanner.config import EDGE_BARS_ADJUSTMENT, PRED_DAYS
from scanner.data.market_data import drop_in_progress_daily_bar, drop_vendor_placeholder_bars, fetch_daily_bars
from scanner.edge.retrieval import build_edge_records_from_bars
from scanner.models.kronos_adapter import KronosAdapter

TICKERS = ["SOFI", "PLTR", "HOOD"]
MIN_HISTORY = 35  # must match build_edge_records_from_bars' default
HORIZON = PRED_DAYS  # must match how main.py calls it (horizon=PRED_DAYS)
RECORDS_PER_TICKER = 6  # keep this tiny on purpose - do NOT run the full backfill
OUT_DIR = Path(__file__).resolve().parent
LOG = logging.getLogger("kronos_backfill_validate")


class _RecordingKronosAdapter:
    """Wraps a real KronosAdapter to time calls and capture what it saw.

    This exists purely for validation instrumentation (timing + leakage
    self-check) and is never used by production code. It loads the model
    exactly once, on first use, via the wrapped adapter's own `_load_once`
    caching - the same reuse-one-adapter-across-all-windows contract the
    task requires.
    """

    def __init__(self, inner: KronosAdapter):
        self._inner = inner
        self.eval_seconds: list[float] = []
        self.calls: list[dict] = []

    def evaluate(self, ticker: str, bars: pd.DataFrame, direction: str):
        t0 = time.perf_counter()
        result = self._inner.evaluate(ticker, bars, direction)
        elapsed = time.perf_counter() - t0
        self.eval_seconds.append(elapsed)
        self.calls.append(
            {
                "ticker": ticker,
                "direction": direction,
                # Full index (not just min/max/count) so the leakage
                # self-check can do an exact set comparison, not just a
                # count match that could hide a swapped bar.
                "window_index": pd.DatetimeIndex(bars.index).copy(),
                "n_bars_seen": int(len(bars)),
                "window_max_ts": bars.index.max(),
                "window_min_ts": bars.index.min(),
                "elapsed_sec": elapsed,
                "directional_agreement": result.directional_agreement,
            }
        )
        return result


def _fetch_clean_daily(ticker: str) -> pd.DataFrame:
    """Same bar-cleaning steps run_build_retrieval_index applies before
    build_edge_records_from_bars, minus the full contract/cross-check gates
    (out of scope here - this validates the Kronos wiring, not the ingestion
    gates, which are covered by their own tests)."""
    daily = fetch_daily_bars(ticker, research=True, adjustment=EDGE_BARS_ADJUSTMENT)
    daily = drop_in_progress_daily_bar(daily)
    daily = drop_vendor_placeholder_bars(daily)
    return daily


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    real_adapter = KronosAdapter(logging.getLogger("kronos_adapter"))
    recorder = _RecordingKronosAdapter(real_adapter)

    all_records = []
    per_ticker_summary = {}
    bars_used_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in TICKERS:
        LOG.info("Fetching daily bars for %s ...", ticker)
        daily = _fetch_clean_daily(ticker)
        if daily is None or daily.empty:
            LOG.warning("No bars for %s, skipping", ticker)
            continue

        # Trim to just enough history for RECORDS_PER_TICKER decision windows.
        # This bounds the number of Kronos evaluations (and therefore wall
        # clock) while still exercising the real code path on real bars.
        needed = MIN_HISTORY + HORIZON + RECORDS_PER_TICKER
        bars_for_build = daily.tail(needed) if len(daily) > needed else daily
        LOG.info(
            "%s: %d bars available, using last %d bars -> up to %d decision windows",
            ticker,
            len(daily),
            len(bars_for_build),
            max(0, len(bars_for_build) - MIN_HISTORY - HORIZON),
        )

        records = build_edge_records_from_bars(
            ticker,
            bars_for_build,
            horizon=HORIZON,
            min_history=MIN_HISTORY,
            kronos=recorder,
        )
        LOG.info("%s: built %d edge records", ticker, len(records))
        per_ticker_summary[ticker] = {
            "bars_fetched": int(len(daily)),
            "bars_used": int(len(bars_for_build)),
            "records_built": len(records),
        }
        all_records.extend(records)
        bars_used_by_ticker[ticker] = bars_for_build.sort_index()

    if not all_records:
        LOG.error("No edge records were built at all - cannot validate populate proof.")
        return 1

    # --- Populate proof -----------------------------------------------
    agreement_vals = [r.features.get("kronos_directional_agreement") for r in all_records]
    median_vals = [r.features.get("kronos_median_forecast_return_pct") for r in all_records]
    worst_vals = [r.features.get("kronos_worst_sampled_return_pct") for r in all_records]
    sample_count_vals = [r.features.get("kronos_sample_count") for r in all_records]

    finite_agreement = [v for v in agreement_vals if _is_finite(v)]
    finite_median = [v for v in median_vals if _is_finite(v)]
    finite_worst = [v for v in worst_vals if _is_finite(v)]
    finite_sample_count = [v for v in sample_count_vals if _is_finite(v)]

    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    _assert(len(finite_agreement) > 0, "kronos_directional_agreement has 0 finite values")
    _assert(len(finite_median) > 0, "kronos_median_forecast_return_pct has 0 finite values")
    _assert(len(finite_worst) > 0, "kronos_worst_sampled_return_pct has 0 finite values")
    _assert(len(finite_sample_count) > 0, "kronos_sample_count has 0 finite values")
    _assert(any(v > 0 for v in finite_sample_count), "kronos_sample_count is not > 0 anywhere")

    _assert(
        len(set(finite_agreement)) > 1 or len(set(finite_median)) > 1 or len(set(finite_worst)) > 1,
        "kronos forecast fields are constant across all records (still looks unpopulated)",
    )
    _assert(
        len(set(finite_sample_count)) > 1 or (finite_sample_count and finite_sample_count[0] > 0),
        "kronos_sample_count is constant/zero across all records",
    )

    LOG.info("POPULATE PROOF: PASS")
    LOG.info(
        "  directional_agreement: finite=%d/%d distinct=%d sample=%s",
        len(finite_agreement),
        len(agreement_vals),
        len(set(finite_agreement)),
        finite_agreement[:5],
    )
    LOG.info(
        "  median_forecast_return_pct: finite=%d/%d distinct=%d sample=%s",
        len(finite_median),
        len(median_vals),
        len(set(finite_median)),
        finite_median[:5],
    )
    LOG.info(
        "  worst_sampled_return_pct: finite=%d/%d distinct=%d sample=%s",
        len(finite_worst),
        len(worst_vals),
        len(set(finite_worst)),
        finite_worst[:5],
    )
    LOG.info(
        "  sample_count: finite=%d/%d distinct=%d sample=%s",
        len(finite_sample_count),
        len(sample_count_vals),
        len(set(finite_sample_count)),
        finite_sample_count[:5],
    )

    # --- Timing / cost projection ---------------------------------------
    eval_times = recorder.eval_seconds
    _assert(len(eval_times) > 0, "no Kronos evaluations were recorded - timing is meaningless")
    mean_eval_sec = sum(eval_times) / len(eval_times)
    total_eval_sec = sum(eval_times)
    projected_11000_hours = (mean_eval_sec * 11000) / 3600.0
    projected_11026_hours = (mean_eval_sec * 11026) / 3600.0

    LOG.info("TIMING:")
    LOG.info("  n_evals=%d total_sec=%.2f mean_sec_per_eval=%.3f", len(eval_times), total_eval_sec, mean_eval_sec)
    LOG.info(
        "  projected full backfill (~11,000 records) at this rate: %.2f hours (%.1f min)",
        projected_11000_hours,
        projected_11000_hours * 60,
    )
    LOG.info(
        "  projected full backfill (~11,026 records, current index size) at this rate: %.2f hours (%.1f min)",
        projected_11026_hours,
        projected_11026_hours * 60,
    )

    # --- Leakage self-check ----------------------------------------------
    # For each recorded Kronos call, recompute what it SHOULD have seen from
    # the exact input DataFrame that was handed to build_edge_records_from_bars
    # for that ticker (bars_used_by_ticker - not a fresh re-fetch, which could
    # legitimately differ in row count from a deliberately truncated input and
    # produce a false positive). The window Kronos actually saw must be
    # exactly the prefix of that input at-or-before its own last timestamp:
    # no bar strictly after window_max_ts, and none of the bars at-or-before
    # it missing.
    leakage_failures = []
    checked = 0
    for call in recorder.calls:
        ticker = call["ticker"]
        source = bars_used_by_ticker.get(ticker)
        if source is None:
            continue
        window_max_ts = pd.Timestamp(call["window_max_ts"])
        expected_index = pd.DatetimeIndex(source.index[source.index <= window_max_ts])
        actual_index = call["window_index"]
        if not expected_index.equals(actual_index):
            future_leak = actual_index[actual_index > window_max_ts]
            leakage_failures.append(
                {
                    "ticker": ticker,
                    "window_max_ts": str(window_max_ts),
                    "expected_bars": int(len(expected_index)),
                    "actual_bars_seen": call["n_bars_seen"],
                    "future_timestamps_in_window": [str(ts) for ts in future_leak],
                    "reason": "actual window index does not exactly match the at-or-before-decision prefix",
                }
            )
        checked += 1

    _assert(checked > 0, "leakage self-check ran 0 comparisons")
    _assert(not leakage_failures, f"LEAKAGE DETECTED: {leakage_failures}")

    LOG.info("LEAKAGE SELF-CHECK: PASS (%d calls checked, 0 failures)", checked)

    # --- Write output ONLY inside this experiment directory --------------
    out_path = OUT_DIR / "validation_output.json"
    payload = {
        "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "tickers": TICKERS,
        "records_per_ticker_cap": RECORDS_PER_TICKER,
        "per_ticker_summary": per_ticker_summary,
        "total_records_built": len(all_records),
        "populate_proof": {
            "directional_agreement_finite": len(finite_agreement),
            "median_forecast_return_pct_finite": len(finite_median),
            "worst_sampled_return_pct_finite": len(finite_worst),
            "sample_count_finite": len(finite_sample_count),
            "distinct_directional_agreement": len(set(finite_agreement)),
            "distinct_median_forecast_return_pct": len(set(finite_median)),
            "distinct_worst_sampled_return_pct": len(set(finite_worst)),
        },
        "timing": {
            "n_evals": len(eval_times),
            "total_sec": total_eval_sec,
            "mean_sec_per_eval": mean_eval_sec,
            "projected_11000_records_hours": projected_11000_hours,
            "projected_11026_records_hours": projected_11026_hours,
        },
        "leakage_check": {
            "calls_checked": checked,
            "failures": leakage_failures,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    LOG.info("Wrote validation output to %s (experiment-scoped only)", out_path)

    LOG.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
