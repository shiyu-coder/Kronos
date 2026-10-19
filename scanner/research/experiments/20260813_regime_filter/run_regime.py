"""Long-only, market-regime-gated sweep. See preregistration.json.

The decisive control is the REGIME-MATCHED placebo: random entries drawn from
the same regime days. Against an all-days placebo a bull-regime filter looks
brilliant for the trivial reason that bull markets go up.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scanner/research/experiments/20260813_horizon_sweep"))

from run_sweep import walk_np  # validated harness, max |dR| = 0.000e+00

from scanner.config import EDGE_COST_BPS_PER_SIDE, EDGE_INDEX_PATH
from scanner.data.market_data import (
    drop_in_progress_daily_bar,
    drop_vendor_placeholder_bars,
    fetch_daily_bars,
)
from scanner.edge.stats import _hac_day_mean_summary

HERE = Path(__file__).parent
HORIZON_CACHE = REPO_ROOT / "scanner/research/experiments/20260813_horizon_sweep/bars_cache.pkl"
COST_PCT = 2.0 * EDGE_COST_BPS_PER_SIDE / 100.0
R_CAP = 10.0
HORIZONS = [5, 10, 20, 40]
TARGETS: list[float | None] = [None, 2.0, 3.0, 4.0]
T_SUGGESTIVE, T_BONFERRONI = 2.0, 3.3


def build_regimes() -> dict[str, dict[str, bool]]:
    spy = fetch_daily_bars("SPY", period="5y", research=True, adjustment="split")
    spy = drop_vendor_placeholder_bars(drop_in_progress_daily_bar(spy)).sort_index()
    c = spy["Close"]
    sma200 = c.rolling(200).mean()
    sma50 = c.rolling(50).mean()
    out: dict[str, dict[str, bool]] = {}
    for name, cond in (
        ("R1_spy_above_200sma", c > sma200),
        ("R2_spy_above_50sma", c > sma50),
        ("R3_golden_cross", sma50 > sma200),
    ):
        out[name] = {
            str(ts.date()): bool(v)
            for ts, v in cond.items()
            if not pd.isna(sma200.loc[ts]) and not pd.isna(sma50.loc[ts])
        }
    print(f"SPY regime days built: {[(k, sum(v.values()), len(v)) for k, v in out.items()]}")
    return out


def stats(rows: list[dict], lags: int) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "n_days": 0, "target_hit_pct": 0.0, "win_rate_pct": 0.0,
                "gross_r": 0.0, "net_r": 0.0, "t_day_clustered": 0.0}
    gross, net, days, wins = [], [], [], []
    for row in rows:
        g = row["ret"] / row["risk"]
        gross.append(float(np.clip(g, -R_CAP, R_CAP)))
        net.append(float(np.clip((row["ret"] - COST_PCT) / row["risk"], -R_CAP, R_CAP)))
        days.append(row["day"])
        wins.append(1.0 if (row["ret"] - COST_PCT) > 0 else 0.0)
    hac = _hac_day_mean_summary(net, days, hac_lags=lags)
    return {
        "n": n, "n_days": hac["n_days"],
        "target_hit_pct": 100.0 * sum(1 for r in rows if r["exit"] == "target") / n,
        "win_rate_pct": 100.0 * float(np.mean(wins)),
        "gross_r": float(np.mean(gross)), "net_r": float(np.mean(net)),
        "t_day_clustered": hac["t_stat"],
    }


def margin_t(a: list[dict], b: list[dict], lags: int) -> tuple[float, float, int]:
    def dm(rows):
        acc = defaultdict(list)
        for row in rows:
            acc[row["day"]].append(
                float(np.clip((row["ret"] - COST_PCT) / row["risk"], -R_CAP, R_CAP)))
        return {d: float(np.mean(v)) for d, v in acc.items()}

    x, y = dm(a), dm(b)
    shared = sorted(set(x) & set(y))
    if len(shared) < 2:
        return 0.0, 0.0, len(shared)
    hac = _hac_day_mean_summary([x[d] - y[d] for d in shared], shared, hac_lags=lags)
    return hac["mean_of_day_means"], hac["t_stat"], hac["n_days"]


def main() -> int:
    regimes = build_regimes()
    bars = pickle.loads(HORIZON_CACHE.read_bytes())["bars"]
    records = json.loads(Path(EDGE_INDEX_PATH).read_text())

    arr, pos_of, day_of = {}, {}, {}
    for tk, df in bars.items():
        arr[tk] = (df["Open"].to_numpy(float).tolist(), df["High"].to_numpy(float).tolist(),
                   df["Low"].to_numpy(float).tolist(), df["Close"].to_numpy(float).tolist())
        pos_of[tk] = {ts: i for i, ts in enumerate(df.index)}
        day_of[tk] = [str(ts.date()) for ts in df.index]

    max_h = max(HORIZONS)
    prep, dropped_no_spy = [], 0
    for r in records:
        if r["direction"] != "bullish":
            continue
        tk = r["ticker"]
        i = pos_of[tk].get(pd.Timestamp(r["timestamp"]))
        risk = float(r.get("risk_pct_used") or 0.0)
        if i is None or risk <= 0 or len(arr[tk][3]) - 1 - i < max_h:
            continue
        day = str(r["timestamp"])[:10]
        if day not in regimes["R1_spy_above_200sma"]:
            dropped_no_spy += 1  # never forward-fill a regime onto a record
            continue
        prep.append({"tk": tk, "i": i, "risk": risk, "day": day})
    print(f"bullish fixed-cohort records: {len(prep)} ({dropped_no_spy} dropped: no SPY bar)")

    med_risk = {}
    for tk in arr:
        rs = [p["risk"] for p in prep if p["tk"] == tk]
        if rs:
            med_risk[tk] = float(np.median(rs))

    results = {"cells": [], "meta": {"cost_bps_per_side": EDGE_COST_BPS_PER_SIDE,
                                     "horizons": HORIZONS, "t_bonferroni": T_BONFERRONI}}

    for rname, rmap in regimes.items():
        bull_days = sum(1 for p in prep if rmap.get(p["day"], False))
        print(f"\n===== {rname}  (setups in bull regime: {bull_days}/{len(prep)} = "
              f"{100*bull_days/len(prep):.1f}%) =====")
        print(f"{'h':>3} {'tgt':>5} | {'n':>5} {'hit%':>6} {'WR%':>6} {'netR':>7} {'t':>6} "
              f"| {'placeboR':>9} {'margin':>7} {'t_m':>6} | {'SKIPPED(bear)':>13} {'n_bear':>7}")
        for h in HORIZONS:
            for tgt in TARGETS:
                bull_rows, bear_rows, plac_rows = [], [], []
                for p in prep:
                    o, hi, lo, c = arr[p["tk"]]
                    w = walk_np(o, hi, lo, c, p["i"] + 1, p["i"] + 1 + h, "bullish",
                                c[p["i"]], p["risk"], tgt)
                    if not w:
                        continue
                    row = {"ret": w["return_pct"], "risk": p["risk"], "day": p["day"],
                           "exit": w["exit_reason"]}
                    (bull_rows if rmap.get(p["day"], False) else bear_rows).append(row)
                # REGIME-MATCHED placebo: every bar of every ticker, bull-regime bars only
                for tk in bars:
                    risk = med_risk.get(tk)
                    if risk is None:
                        continue
                    o, hi, lo, c = arr[tk]
                    days = day_of[tk]
                    for i in range(0, len(c) - h - 1):
                        if not rmap.get(days[i], False):
                            continue
                        w = walk_np(o, hi, lo, c, i + 1, i + 1 + h, "bullish", c[i], risk, tgt)
                        if w:
                            plac_rows.append({"ret": w["return_pct"], "risk": risk,
                                              "day": days[i], "exit": w["exit_reason"]})
                s = stats(bull_rows, h)
                sb = stats(bear_rows, h)
                sp = stats(plac_rows, h)
                m, mt, _ = margin_t(bull_rows, plac_rows, h)
                tag = "none" if tgt is None else f"{tgt:g}R"
                print(f"{h:>3} {tag:>5} | {s['n']:>5} {s['target_hit_pct']:>6.1f} "
                      f"{s['win_rate_pct']:>6.1f} {s['net_r']:>7.3f} {s['t_day_clustered']:>6.2f} "
                      f"| {sp['net_r']:>9.3f} {m:>7.3f} {mt:>6.2f} "
                      f"| {sb['net_r']:>13.3f} {sb['n']:>7}")
                results["cells"].append({
                    "regime": rname, "horizon": h, "target": tag,
                    "bull": s, "bear_skipped": sb, "placebo": sp,
                    "margin": m, "margin_t": mt,
                })

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {HERE / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
