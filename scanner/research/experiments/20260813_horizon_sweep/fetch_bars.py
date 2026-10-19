"""Fetch and cache the index universe's daily bars, identical to the index build path.

Mirrors scanner/main.py:1519-1524 (fetch -> drop in-progress -> drop placeholders).
Cached to a local pickle so the sweep is reproducible without refetching.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import EDGE_BARS_ADJUSTMENT, EDGE_INDEX_PATH
from scanner.data.market_data import (
    drop_in_progress_daily_bar,
    drop_vendor_placeholder_bars,
    fetch_daily_bars,
)

HERE = Path(__file__).parent
CACHE = HERE / "bars_cache.pkl"


def main() -> int:
    records = json.loads(Path(EDGE_INDEX_PATH).read_text())
    tickers = sorted({r["ticker"] for r in records})
    print(f"universe: {len(tickers)} tickers")

    bars: dict = {}
    failed: dict[str, str] = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            df = fetch_daily_bars(ticker, research=True, adjustment=EDGE_BARS_ADJUSTMENT)
            df = drop_in_progress_daily_bar(df)
            df = drop_vendor_placeholder_bars(df)
            if df is None or df.empty:
                raise RuntimeError("empty frame")
            bars[ticker] = df.sort_index()
            print(f"  [{i:>2}/{len(tickers)}] {ticker:<6} {len(df):>4} bars  "
                  f"{df.index.min().date()} -> {df.index.max().date()}")
        except Exception as exc:  # noqa: BLE001 - cache build reports, does not mask
            failed[ticker] = str(exc)
            print(f"  [{i:>2}/{len(tickers)}] {ticker:<6} FAILED: {exc}")

    with CACHE.open("wb") as fh:
        pickle.dump({"bars": bars, "failed": failed, "adjustment": EDGE_BARS_ADJUSTMENT}, fh)
    print(f"\ncached {len(bars)} tickers -> {CACHE}")
    if failed:
        print(f"FAILED {len(failed)}: {sorted(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
