"""Build a Kronos-populated edge-index SNAPSHOT for the 20260711 ranking test.

This rebuilds the edge index the SAME way scanner.main.run_build_retrieval_index
does - same universe, same bar-fetch + cleanup path - but threads a single
reused KronosAdapter through build_edge_records_from_bars so the three kronos_*
forecast fields are populated on (bullish, when --bullish-only-kronos) records.
The result differs from scanner/reports/edge_retrieval_index.json ONLY by those
now-populated fields.

HARD SAFETY INVARIANTS:
  * NEVER writes scanner/reports/edge_retrieval_index.json (or anything under
    scanner/reports/ or scanner/models/). The snapshot + checkpoint live only
    under this experiment directory (or an explicit --snapshot/--checkpoint
    path you pass, e.g. a temp path for a sanity run). A resolved output path
    that lands under scanner/reports or scanner/models aborts the run.
  * Checkpointed per-ticker to a resumable JSONL: each completed ticker appends
    one line (flushed + fsync'd). Re-running with the same --checkpoint skips
    tickers already completed OK and retries only never-attempted or
    previously-errored tickers, so a crash mid-run loses no completed work.

This script FETCHES BARS LIVE (a network read) and runs Kronos inference on CPU.
It is intentionally slow for the full universe. It does NOT touch scanner.main,
gates, thresholds, or acceptance code.

CLI:
  --universe {full,subsample}   full = WATCHLIST + EDGE_INDEX_EXTRA_UNIVERSE
                                (55 tickers, the production index universe);
                                subsample = a deterministic small slice
                                (--subsample-size, default 10) of that union.
  --tickers T1,T2,...           Explicit ticker list; overrides --universe.
  --subsample-size N            Size of the subsample slice (default 10).
  --max-records N               Stop starting new tickers once this many total
                                records have accumulated (cost/sanity cap).
  --bullish-only-kronos         Only evaluate Kronos on bullish windows
                                (kronos_direction_filter='bullish'); non-bullish
                                records are still built with kr=None. ~40% fewer
                                evals at zero cost to the bullish ranking test.
  --snapshot PATH               Final snapshot path (default:
                                <dir>/kronos_index_snapshot.json).
  --checkpoint PATH             Resumable per-ticker JSONL (default:
                                <dir>/kronos_backfill_checkpoint.jsonl).
  --no-kronos                   Build without Kronos (kr=None everywhere) - for
                                plumbing/sanity of the fetch+checkpoint path
                                without paying inference cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import (  # noqa: E402
    EDGE_BARS_ADJUSTMENT,
    EDGE_INDEX_EXTRA_UNIVERSE,
    EDGE_INDEX_PATH,
    PRED_DAYS,
    REPORT_DIR,
    ROOT_DIR,
)
from scanner.data.market_data import (  # noqa: E402
    drop_in_progress_daily_bar,
    drop_vendor_placeholder_bars,
    fetch_daily_bars,
)
from scanner.edge.retrieval import build_edge_records_from_bars  # noqa: E402
from scanner.tickers import WATCHLIST  # noqa: E402
from scanner.utils.atomic_io import atomic_write_text  # noqa: E402
from scanner.utils.logging_setup import setup_logging  # noqa: E402

DIR = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = DIR / "kronos_index_snapshot.json"
DEFAULT_CHECKPOINT = DIR / "kronos_backfill_checkpoint.jsonl"
MODELS_DIR = ROOT_DIR / "models"


class _CountingKronosAdapter:
    """Reuses ONE real KronosAdapter, counting evaluations for the ETA."""

    def __init__(self, inner):
        self._inner = inner
        self.n_evals = 0
        self.eval_seconds = 0.0

    def evaluate(self, ticker, bars, direction):
        t0 = time.perf_counter()
        result = self._inner.evaluate(ticker, bars, direction)
        self.eval_seconds += time.perf_counter() - t0
        self.n_evals += 1
        return result


def _resolve_universe(args) -> list[str]:
    if args.tickers:
        return list(dict.fromkeys([t.strip().upper() for t in args.tickers.split(",") if t.strip()]))
    union = list(dict.fromkeys([*WATCHLIST, *EDGE_INDEX_EXTRA_UNIVERSE]))
    if args.universe == "subsample":
        return union[: max(1, int(args.subsample_size))]
    return union


def _assert_safe_output(path: Path, label: str) -> None:
    resolved = path.resolve()
    forbidden = resolved == Path(EDGE_INDEX_PATH).resolve()
    try:
        under_reports = REPORT_DIR.resolve() in resolved.parents
        under_models = MODELS_DIR.resolve() in resolved.parents
    except Exception:
        under_reports = under_models = False
    if forbidden or under_reports or under_models:
        raise SystemExit(
            f"REFUSING to write {label} to {resolved}: it is the production index or lives under "
            f"scanner/reports or scanner/models. Choose a path under the experiment dir or a temp path."
        )


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Return the LAST line per ticker (a later successful retry supersedes an
    earlier errored attempt)."""
    last_by_ticker: dict[str, dict] = {}
    if not path.exists():
        return last_by_ticker
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard crash
            ticker = entry.get("ticker")
            if ticker:
                last_by_ticker[str(ticker)] = entry
    return last_by_ticker


def _append_checkpoint(path: Path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", choices=["full", "subsample"], default="full")
    parser.add_argument("--tickers", default=None, help="Explicit comma-separated list; overrides --universe.")
    parser.add_argument("--subsample-size", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--bullish-only-kronos", action="store_true", default=False)
    parser.add_argument("--no-kronos", action="store_true", default=False)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--log-dir", default=str(DIR / "logs"), help="Where run logs go (point at a temp dir for sanity runs).")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    checkpoint_path = Path(args.checkpoint)
    _assert_safe_output(snapshot_path, "snapshot")
    _assert_safe_output(checkpoint_path, "checkpoint")

    logger = setup_logging(Path(args.log_dir))
    universe = _resolve_universe(args)
    direction_filter = "bullish" if args.bullish_only_kronos else None

    kronos = None
    if not args.no_kronos:
        from scanner.models.kronos_adapter import KronosAdapter

        kronos = _CountingKronosAdapter(KronosAdapter(logger))

    done = _load_checkpoint(checkpoint_path)
    done_ok = {t for t, e in done.items() if e.get("status") == "ok"}
    todo = [t for t in universe if t not in done_ok]

    print(
        f"universe={args.universe} n_tickers={len(universe)} already_done_ok={len(done_ok)} "
        f"todo={len(todo)} bullish_only_kronos={args.bullish_only_kronos} no_kronos={args.no_kronos}"
    )
    print(f"snapshot -> {snapshot_path}")
    print(f"checkpoint -> {checkpoint_path}")

    # Records already durable from prior runs (last OK line per ticker).
    accumulated_records = 0
    for t in done_ok:
        accumulated_records += int(done[t].get("n_records", 0))

    run_start = time.perf_counter()
    completed_this_run = 0

    for i, ticker in enumerate(todo, start=1):
        if args.max_records is not None and accumulated_records >= args.max_records:
            print(f"max-records cap ({args.max_records}) reached at {accumulated_records}; stopping before {ticker}.")
            break

        t0 = time.perf_counter()
        try:
            daily = fetch_daily_bars(ticker, research=True, adjustment=EDGE_BARS_ADJUSTMENT)
            daily = drop_in_progress_daily_bar(daily)
            daily = drop_vendor_placeholder_bars(daily)
            records = build_edge_records_from_bars(
                ticker,
                daily,
                horizon=PRED_DAYS,
                kronos=kronos,
                kronos_direction_filter=direction_filter,
            )
            elapsed = time.perf_counter() - t0
            entry = {
                "ticker": ticker,
                "status": "ok",
                "n_records": len(records),
                "n_bullish": sum(1 for r in records if r.direction == "bullish"),
                "elapsed_sec": round(elapsed, 3),
                "records": [asdict(r) for r in records],
            }
            _append_checkpoint(checkpoint_path, entry)
            accumulated_records += len(records)
            completed_this_run += 1
        except Exception as exc:  # fail-soft per ticker: record + continue
            elapsed = time.perf_counter() - t0
            entry = {"ticker": ticker, "status": "error", "error": str(exc), "elapsed_sec": round(elapsed, 3)}
            _append_checkpoint(checkpoint_path, entry)
            logger.warning("KRONOS_BACKFILL_SKIP: %s %s", ticker, exc)
            print(f"[{i}/{len(todo)}] {ticker:6s} ERROR {exc}")
            continue

        # Running ETA from tickers completed THIS run (steady-state rate; prior
        # runs' tickers are already durable and excluded from the rate).
        run_elapsed = time.perf_counter() - run_start
        mean_per_ticker = run_elapsed / completed_this_run
        remaining = len(todo) - i
        eta = mean_per_ticker * remaining
        eval_note = ""
        if kronos is not None and kronos.n_evals:
            eval_note = f" kronos_evals={kronos.n_evals} mean_eval={kronos.eval_seconds / kronos.n_evals:.2f}s"
        print(
            f"[{i}/{len(todo)}] {ticker:6s} recs={entry['n_records']:4d} "
            f"(bull={entry['n_bullish']:4d}) {elapsed:6.1f}s | total_recs={accumulated_records} "
            f"| mean/ticker={mean_per_ticker:5.1f}s ETA={_fmt_hms(eta)}{eval_note}"
        )

    # Assemble the snapshot from the LAST OK line per ticker (idempotent:
    # re-reads the checkpoint so a resumed run includes prior + new tickers).
    final = _load_checkpoint(checkpoint_path)
    all_records: list[dict] = []
    ok_tickers = 0
    err_tickers = 0
    for t, e in final.items():
        if e.get("status") == "ok":
            all_records.extend(e.get("records", []))
            ok_tickers += 1
        else:
            err_tickers += 1

    _assert_safe_output(snapshot_path, "snapshot")  # re-check before write
    atomic_write_text(snapshot_path, json.dumps(all_records, indent=2, default=str))

    n_bullish = sum(1 for r in all_records if r.get("direction") == "bullish")
    kron_finite = sum(
        1
        for r in all_records
        if isinstance(r.get("features"), dict)
        and isinstance(r["features"].get("kronos_directional_agreement"), (int, float))
    )
    print("---")
    print(
        f"snapshot written: {snapshot_path} | tickers_ok={ok_tickers} tickers_err={err_tickers} "
        f"records={len(all_records)} bullish={n_bullish} rows_with_finite_kronos_agreement={kron_finite}"
    )
    print(f"total wall clock this run: {_fmt_hms(time.perf_counter() - run_start)}")
    if err_tickers:
        print(f"note: {err_tickers} ticker(s) errored; re-run the same command to retry ONLY those (OK tickers are skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
