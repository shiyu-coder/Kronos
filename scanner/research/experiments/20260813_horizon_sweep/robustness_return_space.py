"""Robustness: redo the cell-vs-placebo margin in RETURN-PERCENT space.

The headline margin is on the R scale, where the cell divides by each record's
own risk_pct_used and the placebo divides by that ticker's median record risk.
If setup bars systematically carry wider stops, the cell's R is deflated
against the placebo by construction and the negative margin would be an
artifact of the denominator rather than of selection.

Return percent has no denominator. If the margin stays negative there, the
selection really is worse than an arbitrary entry.
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
sys.path.insert(0, str(Path(__file__).parent))

from run_sweep import COST_PCT, HORIZONS, walk_np

from scanner.config import EDGE_INDEX_PATH
from scanner.edge.stats import _hac_day_mean_summary

HERE = Path(__file__).parent


def day_margin(cell, placebo, lags):
    def dm(rows):
        acc = defaultdict(list)
        for day, val in rows:
            acc[day].append(val)
        return {d: float(np.mean(v)) for d, v in acc.items()}

    a, b = dm(cell), dm(placebo)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return 0.0, 0.0, 0
    margins = [a[d] - b[d] for d in shared]
    hac = _hac_day_mean_summary(margins, shared, hac_lags=lags)
    return hac["mean_of_day_means"], hac["t_stat"], hac["n_days"]


def main() -> int:
    bars = pickle.loads((HERE / "bars_cache.pkl").read_bytes())["bars"]
    records = json.loads(Path(EDGE_INDEX_PATH).read_text())
    arr, pos_of, day_of = {}, {}, {}
    for tk, df in bars.items():
        arr[tk] = (df["Open"].to_numpy(float).tolist(), df["High"].to_numpy(float).tolist(),
                   df["Low"].to_numpy(float).tolist(), df["Close"].to_numpy(float).tolist())
        pos_of[tk] = {ts: i for i, ts in enumerate(df.index)}
        day_of[tk] = [str(ts.date()) for ts in df.index]

    max_h = max(HORIZONS)
    prep = []
    for r in records:
        if r["direction"] != "bullish":
            continue
        tk = r["ticker"]
        i = pos_of[tk].get(pd.Timestamp(r["timestamp"]))
        risk = float(r.get("risk_pct_used") or 0.0)
        if i is None or risk <= 0 or len(arr[tk][3]) - 1 - i < max_h:
            continue
        prep.append({"tk": tk, "i": i, "risk": risk, "day": str(r["timestamp"])[:10]})

    med_risk = {}
    for tk in arr:
        rs = [p["risk"] for p in prep if p["tk"] == tk]
        if rs:
            med_risk[tk] = float(np.median(rs))

    print("risk (stop distance) sanity check")
    cell_risk = np.array([p["risk"] for p in prep])
    plac_risk = np.array([med_risk[p["tk"]] for p in prep])
    print(f"  cell    mean {cell_risk.mean():.3f}%  median {np.median(cell_risk):.3f}%")
    print(f"  placebo mean {plac_risk.mean():.3f}%  median {np.median(plac_risk):.3f}%")
    print(f"  ratio of means {cell_risk.mean()/plac_risk.mean():.4f}\n")

    print("BULLISH cell vs placebo, NET, in RETURN-PERCENT space (no R denominator)")
    print(f"{'h':>4} {'target':>7} {'cell%':>8} {'placebo%':>9} {'margin%':>9} {'t_marg':>8} {'days':>6}")
    out = []
    for h in HORIZONS:
        for tgt in (None, 3.0):
            cell, plac = [], []
            for p in prep:
                o, hi, lo, c = arr[p["tk"]]
                w = walk_np(o, hi, lo, c, p["i"] + 1, p["i"] + 1 + h, "bullish",
                            c[p["i"]], p["risk"], tgt)
                if w:
                    cell.append((p["day"], w["return_pct"] - COST_PCT))
            for tk in bars:
                risk = med_risk.get(tk)
                if risk is None:
                    continue
                o, hi, lo, c = arr[tk]
                days = day_of[tk]
                for i in range(0, len(c) - h - 1):
                    w = walk_np(o, hi, lo, c, i + 1, i + 1 + h, "bullish", c[i], risk, tgt)
                    if w:
                        plac.append((days[i], w["return_pct"] - COST_PCT))
            cm = float(np.mean([v for _, v in cell]))
            pm = float(np.mean([v for _, v in plac]))
            m, t, nd = day_margin(cell, plac, h)
            tag = "none" if tgt is None else f"{tgt:g}R"
            print(f"{h:>4} {tag:>7} {cm:>8.3f} {pm:>9.3f} {m:>9.3f} {t:>8.2f} {nd:>6}")
            out.append({"horizon": h, "target": tag, "cell_net_pct": cm,
                        "placebo_net_pct": pm, "margin_pct": m, "margin_t": t, "n_days": nd})

    (HERE / "robustness_return_space.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'robustness_return_space.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
