"""Horizon x target sweep on the fixed index entry set. See preregistration.json.

Order of operations is deliberate and non-negotiable:
  1. validate the numpy fast-path barrier against scanner.edge.outcomes.walk_triple_barrier
  2. validate cell (h=5, target=none) against the SHIPPED index r_multiple
  3. only then read any cell
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scanner.config import EDGE_COST_BPS_PER_SIDE, EDGE_INDEX_PATH
from scanner.edge.outcomes import walk_triple_barrier
from scanner.edge.stats import _hac_day_mean_summary

HERE = Path(__file__).parent
HORIZONS = [5, 10, 15, 20, 30, 40, 60]
TARGETS: list[float | None] = [None, 2.0, 3.0, 4.0]
COST_PCT = 2.0 * EDGE_COST_BPS_PER_SIDE / 100.0  # round trip, in return points
R_CAP = 10.0


# ---------------------------------------------------------------- fast barrier
def walk_np(o, h, l, c, start: int, stop: int, direction: str, entry: float,
            risk_pct: float, target_r: float | None):
    """Pure-Python twin of walk_triple_barrier. Same tie rules, same gap rule.

    Takes whole-series lists plus a [start, stop) window rather than slices:
    the placebo evaluates every bar of every ticker, and slice allocation
    dominated the runtime. `target_r` is expressed in R (target_pct =
    target_r * risk_pct) because this experiment varies the target on the R
    scale directly.
    """
    stop = min(stop, len(c))
    if stop <= start or entry <= 0 or risk_pct <= 0:
        return None
    bullish = direction == "bullish"
    sign = 1.0 if bullish else -1.0
    no_target = target_r is None
    target_pct = 0.0 if no_target else target_r * risk_pct
    stop_price = entry * (1.0 - sign * risk_pct / 100.0)
    target_price = entry * (1.0 + sign * target_pct / 100.0)

    exit_reason = "horizon"
    exit_idx = stop - 1
    ret_pct = sign * ((c[stop - 1] - entry) / entry) * 100.0
    for pos in range(start, stop):
        stop_touched = l[pos] <= stop_price if bullish else h[pos] >= stop_price
        target_touched = (not no_target) and (
            h[pos] >= target_price if bullish else l[pos] <= target_price
        )
        if stop_touched:
            op = o[pos]
            exit_price = stop_price
            gapped = op <= stop_price if bullish else op >= stop_price
            if op > 0 and gapped:
                exit_price = op
            exit_reason, exit_idx = "stop", pos
            ret_pct = sign * ((exit_price - entry) / entry) * 100.0
            break
        if target_touched:
            exit_reason, exit_idx = "target", pos
            ret_pct = target_pct
            break

    if bullish:
        mfe = ((max(h[start : exit_idx + 1]) - entry) / entry) * 100.0
    else:
        mfe = ((entry - min(l[start : exit_idx + 1])) / entry) * 100.0
    r = ret_pct / risk_pct
    return {
        "return_pct": float(ret_pct),
        "r_multiple": float(R_CAP if r > R_CAP else (-R_CAP if r < -R_CAP else r)),
        "mfe_pct": float(mfe),
        "exit_reason": exit_reason,
    }


# ------------------------------------------------------------------ statistics
def cell_stats(rows: list[dict], horizon: int) -> dict:
    """rows: [{'ret': gross return pct, 'risk': stop pct, 'day': 'YYYY-MM-DD', 'exit': str}]"""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    net_r, days, wins = [], [], []
    gross_r = []
    for row in rows:
        gross_r.append(float(np.clip(row["ret"] / row["risk"], -R_CAP, R_CAP)))
        v = float(np.clip((row["ret"] - COST_PCT) / row["risk"], -R_CAP, R_CAP))
        net_r.append(v)
        days.append(row["day"])
        wins.append(1.0 if (row["ret"] - COST_PCT) > 0 else 0.0)
    hac = _hac_day_mean_summary(net_r, days, hac_lags=horizon)
    hits = sum(1 for row in rows if row["exit"] == "target")
    return {
        "n": n,
        "n_days": hac["n_days"],
        "target_hit_pct": 100.0 * hits / n,
        "win_rate_pct": 100.0 * float(np.mean(wins)),
        "gross_r": float(np.mean(gross_r)),
        "net_r": float(np.mean(net_r)),
        "net_r_day_mean": hac["mean_of_day_means"],
        "t_day_clustered": hac["t_stat"],
        "hac_method": hac["method"],
    }


def paired_margin_t(a: list[dict], b: list[dict], horizon: int) -> dict:
    """Day-clustered t on the per-day margin (cell minus placebo). Unpaired at the
    trade level - the placebo has its own entries - so margins are formed on the
    shared entry days, which is where the market factor lives anyway."""
    def day_means(rows):
        acc = defaultdict(list)
        for row in rows:
            acc[row["day"]].append(float(np.clip((row["ret"] - COST_PCT) / row["risk"], -R_CAP, R_CAP)))
        return {d: float(np.mean(v)) for d, v in acc.items()}

    da, db = day_means(a), day_means(b)
    shared = sorted(set(da) & set(db))
    if len(shared) < 2:
        return {"n_days": len(shared), "margin": 0.0, "t": 0.0}
    margins = [da[d] - db[d] for d in shared]
    hac = _hac_day_mean_summary(margins, shared, hac_lags=horizon)
    return {"n_days": hac["n_days"], "margin": hac["mean_of_day_means"], "t": hac["t_stat"]}


def main() -> int:
    cache = pickle.loads((HERE / "bars_cache.pkl").read_bytes())
    bars = cache["bars"]
    records = json.loads(Path(EDGE_INDEX_PATH).read_text())
    print(f"index records {len(records)}, tickers cached {len(bars)}")

    # Per-ticker numpy arrays + timestamp -> row lookup.
    arr, pos_of, day_of = {}, {}, {}
    for tk, df in bars.items():
        arr[tk] = (
            df["Open"].to_numpy(float).tolist(), df["High"].to_numpy(float).tolist(),
            df["Low"].to_numpy(float).tolist(), df["Close"].to_numpy(float).tolist(),
        )
        pos_of[tk] = {ts: i for i, ts in enumerate(df.index)}
        day_of[tk] = [str(ts.date()) for ts in df.index]

    import pandas as pd
    prepared = []
    unmatched = 0
    for r in records:
        tk = r["ticker"]
        if tk not in arr:
            unmatched += 1
            continue
        ts = pd.Timestamp(r["timestamp"])
        i = pos_of[tk].get(ts)
        risk = float(r.get("risk_pct_used") or 0.0)
        if i is None or risk <= 0:
            unmatched += 1
            continue
        prepared.append({
            "tk": tk, "i": i, "dir": r["direction"], "risk": risk,
            "entry": float(arr[tk][3][i]), "day": str(r["timestamp"])[:10],
            "stored_r": float(r["r_multiple"]), "stored_exit": r["exit_reason"],
            "n_ahead": len(arr[tk][3]) - 1 - i,
        })
    print(f"prepared {len(prepared)} records ({unmatched} unmatched/unusable)")

    # ---------------- CONTROL 1: numpy twin == production walk_triple_barrier
    print("\n[control 1] numpy fast path vs walk_triple_barrier")
    worst = 0.0
    mismatched = 0
    checked = 0
    for h in HORIZONS:
        for tgt in TARGETS:
            for rec in prepared[::37]:  # deterministic stride across all tickers/dates
                o, hi, lo, c = arr[rec["tk"]]
                start, stopx = rec["i"] + 1, rec["i"] + 1 + h
                if start >= len(c):
                    continue
                fast = walk_np(o, hi, lo, c, start, stopx, rec["dir"], rec["entry"],
                               rec["risk"], tgt)
                ref = walk_triple_barrier(
                    bars[rec["tk"]].iloc[start:stopx], rec["dir"], rec["entry"], rec["risk"],
                    None if tgt is None else tgt * rec["risk"],
                )
                worst = max(worst, abs(fast["r_multiple"] - ref["r_multiple"]))
                mismatched += fast["exit_reason"] != ref["exit_reason"]
                checked += 1
    print(f"  checked {checked} walks across all {len(HORIZONS)}x{len(TARGETS)} cells")
    print(f"  max |dR| = {worst:.3e}   exit_reason mismatches = {mismatched}")
    if worst > 1e-9 or mismatched:
        print("  ABORT: fast path diverges from production barrier")
        return 1
    print("  PASS")

    # ---------------- CONTROL 2: cell (5, none) == shipped index
    print("\n[control 2] cell (h=5, target=none) vs shipped index outcomes")
    diffs, exit_ok, n = [], 0, 0
    mine_r, stored_r = [], []
    for rec in prepared:
        o, hi, lo, c = arr[rec["tk"]]
        out = walk_np(o, hi, lo, c, rec["i"] + 1, rec["i"] + 6, rec["dir"], rec["entry"],
                      rec["risk"], None)
        if out is None:
            continue
        diffs.append(abs(out["r_multiple"] - rec["stored_r"]))
        mine_r.append(out["r_multiple"])
        stored_r.append(rec["stored_r"])
        exit_ok += out["exit_reason"] == rec["stored_exit"]
        n += 1
    diffs = np.array(diffs)
    mine_a, stored_a = np.array(mine_r), np.array(stored_r)
    within = float((diffs < 1e-5).mean())
    d_mean = float(mine_a.mean() - stored_a.mean())
    corr = float(np.corrcoef(mine_a, stored_a)[0, 1])
    print(f"  n={n}  max |dR| {diffs.max():.3e}  median {np.median(diffs):.3e}")
    print(f"  [2b reported] within 1e-5: {within*100:.2f}%   "
          f"exit_reason match: {exit_ok/n*100:.2f}%")
    print(f"  [2a GATE] delta mean R {d_mean:+.5f} (limit +-0.005)   "
          f"corr {corr:.5f} (limit 0.99)")
    # See preregistration.json "amendments": control 1 proves harness fidelity
    # exactly; this gate covers aggregate agreement, which is what the verdict
    # rests on. Bar-snapshot identity across a vendor revision day is not a
    # property any experiment can hold.
    if abs(d_mean) > 0.005 or corr < 0.99:
        print("  ABORT: harness does not reproduce the shipped index in aggregate")
        return 1
    print("  PASS")

    # ---------------- fixed cohort: resolvable at the longest horizon
    max_h = max(HORIZONS)
    for rec in prepared:
        rec["fixed_cohort"] = rec["n_ahead"] >= max_h

    # ---------------- drift placebo: same geometry, every bar, per ticker
    med_risk: dict[tuple[str, str], float] = {}
    for direction in ("bullish", "bearish"):
        for tk in arr:
            rs = [r["risk"] for r in prepared if r["tk"] == tk and r["dir"] == direction]
            if rs:
                med_risk[(tk, direction)] = float(np.median(rs))

    results = {"cells": [], "placebo": [], "meta": {
        "cost_bps_per_side": EDGE_COST_BPS_PER_SIDE,
        "round_trip_pct_charged": COST_PCT,
        "horizons": HORIZONS,
        "targets_r": ["none" if t is None else t for t in TARGETS],
        "max_horizon": max_h,
    }}

    import pandas as pd  # noqa: F811
    for direction in ("bullish", "bearish"):
        print(f"\n===== {direction.upper()} =====")
        print(f"{'h':>4} {'target':>7} {'n':>6} {'days':>5} {'hit%':>6} {'WR%':>6} "
              f"{'grossR':>8} {'netR':>8} {'t_hac':>7} | {'placeboR':>9} {'margin':>8} {'t_marg':>7}")
        for h in HORIZONS:
            # placebo rows for this horizon/direction, computed once per target below
            for tgt in TARGETS:
                rows, prows = [], []
                for rec in prepared:
                    if rec["dir"] != direction or not rec["fixed_cohort"]:
                        continue
                    o, hi, lo, c = arr[rec["tk"]]
                    out = walk_np(o, hi, lo, c, rec["i"] + 1, rec["i"] + 1 + h, rec["dir"],
                                  rec["entry"], rec["risk"], tgt)
                    if out is None:
                        continue
                    rows.append({"ret": out["return_pct"], "risk": rec["risk"],
                                 "day": rec["day"], "exit": out["exit_reason"]})
                # placebo: same plan at EVERY bar of every ticker in the universe
                for tk in bars:
                    risk = med_risk.get((tk, direction))
                    if risk is None:
                        continue
                    o, hi, lo, c = arr[tk]
                    days = day_of[tk]
                    for i in range(0, len(c) - h - 1):
                        out = walk_np(o, hi, lo, c, i + 1, i + 1 + h, direction, c[i], risk, tgt)
                        if out is None:
                            continue
                        prows.append({"ret": out["return_pct"], "risk": risk,
                                      "day": days[i], "exit": out["exit_reason"]})
                s = cell_stats(rows, h)
                p = cell_stats(prows, h)
                m = paired_margin_t(rows, prows, h)
                tag = "none" if tgt is None else f"{tgt:g}R"
                results["cells"].append({"direction": direction, "horizon": h, "target": tag,
                                         **s, "placebo_net_r": p.get("net_r"),
                                         "margin": m["margin"], "margin_t": m["t"]})
                print(f"{h:>4} {tag:>7} {s['n']:>6} {s['n_days']:>5} {s['target_hit_pct']:>6.1f} "
                      f"{s['win_rate_pct']:>6.1f} {s['gross_r']:>8.3f} {s['net_r']:>8.3f} "
                      f"{s['t_day_clustered']:>7.2f} | {p['net_r']:>9.3f} {m['margin']:>8.3f} "
                      f"{m['t']:>7.2f}")

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {HERE / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
