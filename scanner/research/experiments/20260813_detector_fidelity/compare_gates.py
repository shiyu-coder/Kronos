"""Phase 1: implemented consolidation gates vs the handoff spec's stated formulas.

Source of the spec text: C:/Users/Jacob Higgins/Downloads/PotterBox_Scanner_Handoff.docx,
section "Step 4 - Potter Box detection (per ticker)":

  1. ATR compression   - "the current 14-period ATR must be less than 75% of the
                          average ATR over the prior 30 bars"
  2. Range compression - "the most recent candle's high-minus-low range must be less
                          than 65% of the median candle range over the consolidation window"
  3. No trend          - "the absolute difference between the first and last close in the
                          consolidation window must be less than 1.5 times the current ATR"

The config CONSTANTS (0.75, 0.65, CONSOLIDATION_BARS=15, ATR_PERIOD=14) match the spec.
The QUANTITIES they are compared against do not. This script measures how much that costs.

Ambiguity flagged, not resolved silently: "the most recent candle" in rule 2 could mean
the breakout candle or the last consolidation candle. Read as the breakout candle the rule
is self-defeating (a breakout candle is by definition an expansion), so the consolidation
reading is primary - but BOTH are measured below.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import (
    ATR_COMPRESSION,
    ATR_PERIOD,
    CONSOLIDATION_BARS,
    MIN_BOX_BOTTOM_TOUCHES,
    MIN_BOX_TOP_TOUCHES,
    BOX_TOUCH_TOLERANCE_PCT,
    NO_TREND_SLOPE_ABS_MAX,
    RANGE_COMPRESSION,
)

HERE = Path(__file__).parent
BARS = REPO_ROOT / "scanner/research/experiments/20260813_horizon_sweep/bars_cache.pkl"
SPEC_NO_TREND_ATR_MULT = 1.5  # from the spec sentence, not a tuned value


def atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def evaluate(df: pd.DataFrame, i: int, atr_full: pd.Series) -> dict | None:
    """Both gate sets at decision bar i, using only bars <= i."""
    need = CONSOLIDATION_BARS + ATR_PERIOD + 2
    if i + 1 < need + 30:
        return None
    win = df.iloc[: i + 1]
    breakout = win.iloc[-1]
    cons = win.iloc[-(CONSOLIDATION_BARS + 1): -1]
    pre = win.iloc[:-1]

    atr_now = atr_full.iloc[i - 1]  # ATR at last pre-breakout bar
    if not np.isfinite(atr_now) or atr_now <= 0:
        return None

    cons_rng = (cons["High"] - cons["Low"])
    avg_cons_range = float(cons_rng.mean())
    prior_window = pre.iloc[-(CONSOLIDATION_BARS + ATR_PERIOD): -CONSOLIDATION_BARS]
    prior_ranges = (prior_window["High"] - prior_window["Low"]).dropna()
    if prior_ranges.empty:
        return None
    prior_avg_range = float(prior_ranges.mean())
    if prior_avg_range <= 0:
        return None

    # ---- implemented (scanner/strategy/potter_box.py:102,106,131-133)
    impl_atr = atr_now <= ATR_COMPRESSION * prior_avg_range
    impl_range = (avg_cons_range / prior_avg_range) <= RANGE_COMPRESSION
    closes = cons["Close"].values
    slope = np.polyfit(np.arange(len(closes)), closes, 1)[0] / (np.mean(closes) + 1e-9)
    impl_notrend = abs(float(slope)) <= NO_TREND_SLOPE_ABS_MAX

    # ---- spec
    prior30_atr = atr_full.iloc[max(0, i - 30): i].dropna()
    if prior30_atr.empty:
        return None
    spec_atr = atr_now < ATR_COMPRESSION * float(prior30_atr.mean())
    med_cons_range = float(cons_rng.median())
    spec_range_cons = float(cons_rng.iloc[-1]) < RANGE_COMPRESSION * med_cons_range
    spec_range_brk = float(breakout["High"] - breakout["Low"]) < RANGE_COMPRESSION * med_cons_range
    spec_notrend = abs(float(closes[0] - closes[-1])) < SPEC_NO_TREND_ATR_MULT * float(atr_now)

    # ---- shared: touches + breakout, close-based control (doctrine doc rule 1-2)
    ctop, cbot = float(cons["Close"].max()), float(cons["Close"].min())
    cost_basis = (ctop + cbot) / 2.0
    bclose, pclose = float(breakout["Close"]), float(win.iloc[-2]["Close"])
    box_range = max(ctop - cbot, 1e-9)
    tol = max(box_range * 0.10, bclose * BOX_TOUCH_TOLERANCE_PCT)
    touches = ((cons["Close"] >= ctop - tol).sum() >= MIN_BOX_TOP_TOUCHES
               and (cons["Close"] <= cbot + tol).sum() >= MIN_BOX_BOTTOM_TOUCHES)
    bullish = bclose > ctop and pclose > cost_basis
    bearish = bclose < cbot and pclose < cost_basis
    # spec box uses high/low extremes rather than close-based control
    stop_, sbot_ = float(cons["High"].max()), float(cons["Low"].min())
    spec_bullish = bclose > stop_
    spec_bearish = bclose < sbot_

    return {
        "impl_atr": impl_atr, "impl_range": impl_range, "impl_notrend": impl_notrend,
        "spec_atr": spec_atr, "spec_range_cons": spec_range_cons,
        "spec_range_brk": spec_range_brk, "spec_notrend": spec_notrend,
        "touches": touches, "brk": bullish or bearish,
        "spec_brk": spec_bullish or spec_bearish,
        "impl_all": impl_atr and impl_range and impl_notrend and touches and (bullish or bearish),
        "spec_all": (spec_atr and spec_range_cons and spec_notrend and touches
                     and (bullish or bearish)),
        "spec_all_specbox": (spec_atr and spec_range_cons and spec_notrend
                             and (spec_bullish or spec_bearish)),
        # Doctrine-only: what potter_visual_doctrine.md (derived from the actual
        # videos + labeled charts) actually specifies - close-based control levels,
        # cost-basis midpoint, touch minimums, and "break requires close outside
        # control level plus prior-close bias vs cost basis". It names NO volatility
        # or trend-compression gate; those appear only in the AI-authored handoff doc.
        "doctrine_only": touches and (bullish or bearish),
        "doctrine_bullish": touches and bullish,
    }


def main() -> int:
    bars = pickle.loads(BARS.read_bytes())["bars"]
    rows = []
    for tk, df in bars.items():
        df = df.sort_index()
        a = atr_series(df, ATR_PERIOD)
        for i in range(len(df)):
            r = evaluate(df, i, a)
            if r:
                rows.append(r)
    n = len(rows)
    print(f"decision bars evaluated: {n} across {len(bars)} tickers\n")

    def pct(k):
        return 100.0 * sum(1 for r in rows if r[k]) / n

    print("GATE-BY-GATE marginal pass rate")
    print(f"{'gate':<34} {'implemented':>12} {'per spec':>10}")
    print(f"{'ATR compression':<34} {pct('impl_atr'):>11.1f}% {pct('spec_atr'):>9.1f}%")
    print(f"{'range compression':<34} {pct('impl_range'):>11.1f}% {pct('spec_range_cons'):>9.1f}%")
    print(f"{'  (breakout-candle reading)':<34} {'':>12} {pct('spec_range_brk'):>9.1f}%")
    print(f"{'no trend':<34} {pct('impl_notrend'):>11.1f}% {pct('spec_notrend'):>9.1f}%")
    print(f"{'touches >=2/>=2 (shared)':<34} {pct('touches'):>11.1f}% {'same':>10}")
    print(f"{'close outside control (shared)':<34} {pct('brk'):>11.1f}% {'same':>10}")
    print()
    print(f"{'ALL GATES (close-based box)':<34} {pct('impl_all'):>11.2f}% {pct('spec_all'):>9.2f}%")
    print(f"{'ALL GATES (spec high/low box)':<34} {'':>12} {pct('spec_all_specbox'):>9.2f}%")
    print()
    print(f"{'DOCTRINE ONLY (no compression)':<34} {'':>12} {pct('doctrine_only'):>9.2f}%")
    print()
    fires_impl = sum(1 for r in rows if r["impl_all"])
    fires_spec = sum(1 for r in rows if r["spec_all"])
    fires_specbox = sum(1 for r in rows if r["spec_all_specbox"])
    fires_doc = sum(1 for r in rows if r["doctrine_only"])
    fires_doc_b = sum(1 for r in rows if r["doctrine_bullish"])
    print(f"absolute firings over ~2y x 55 tickers:")
    print(f"  implemented          : {fires_impl}")
    print(f"  spec formulas        : {fires_spec}")
    print(f"  spec formulas + box  : {fires_specbox}")
    print(f"  DOCTRINE ONLY        : {fires_doc}   (bullish {fires_doc_b})")
    print(f"     -> ~{fires_doc/ (len(bars)*2):.1f} signals per ticker per year")

    out = {"n_decision_bars": n, "tickers": len(bars),
           "marginal": {k: pct(k) for k in
                        ("impl_atr", "impl_range", "impl_notrend", "spec_atr",
                         "spec_range_cons", "spec_range_brk", "spec_notrend",
                         "touches", "brk", "spec_brk")},
           "joint": {"implemented": pct("impl_all"), "spec": pct("spec_all"),
                     "spec_with_spec_box": pct("spec_all_specbox")},
           "firings": {"implemented": fires_impl, "spec": fires_spec,
                       "spec_with_spec_box": fires_specbox}}
    (HERE / "gate_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'gate_comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
