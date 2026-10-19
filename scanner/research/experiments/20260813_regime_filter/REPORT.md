# Regime filter — long-only, gated on market trend, at 1:3

**Date:** 2026-08-13 · **Branch:** `codex/finish-kronos-cleanup`
**Pre-registration:** [preregistration.json](preregistration.json) (written before any cell was read, including the power limitation)
**Verdict: FALSIFIED — 0 of 48 cells pass. The rule as stated is backwards on this data.**

## The rule under test

Take **longs only**, and only while the market is in an uptrend. Take **no shorts**. **Skip longs** while
the market is in a downtrend. Then re-run at a 1:3 bracket.

## Three findings, in order of importance

### 1. The primary cell is significantly negative

R1 (SPY > 200-SMA), 3R target, shipped 5-bar horizon: **net −0.045R, t −2.10**, n=5,263.
Adding the regime gate made the 1:3 bracket *worse* than the ungated version (−0.023R), not better.

### 2. The gate is pointed the wrong way — it declines the profitable trades

The `SKIPPED` column is the trades the rule refuses. Under the two 200-SMA-based definitions, they are
**better than the trades it keeps**, at every horizon and every target:

| regime | cell | kept (bull) net R | **declined (bear) net R** |
|---|---|--:|--:|
| R1 SPY>200SMA | h=5, 3R | −0.045 | **+0.216** |
| R1 SPY>200SMA | h=40, none | +0.178 | **+1.037** |
| R3 golden cross | h=5, 3R | −0.054 | **+0.147** |
| R3 golden cross | h=40, none | +0.116 | **+0.994** |
| R2 SPY>50SMA | h=5, 3R | −0.004 | −0.158 |

**Do not trade on this.** The bear bucket is 513 records across just **51 distinct entry days**,
concentrated in two episodes — Mar–May 2025 (79% of it; 34% in April 2025 alone) and Mar–Apr 2026. Its
day-clustered **t-stats are ~0.1** (0.10, 0.05, 0.06, 0.53 across cells) despite the large point
estimates. This is one drawdown-and-recovery episode, not a finding. The honest read: **the data
gives no support for a downtrend filter, and what little signal there is points the other way** —
consistent with buying dips, not with avoiding them.

R2 (SPY > 50-SMA) is the one definition that points the way the rule assumes, and even it produces no
passing cell (best: h=40 4R, net +0.239, **t 1.47**, below the 2.0 gate).

### 3. The regime gate does not fix negative selection value — **0 of 48 cells beat the regime-matched placebo**

This is the control that decides the question. The placebo takes random entries **on the same regime
days**. Against an all-days placebo a bull filter looks brilliant for the trivial reason that bull
markets go up; matched properly, the margin is negative in **48 of 48 cells**, best margin −0.017.

The negative selection value found in the horizon sweep (28/28 cells) survives every regime gate tested.
Filtering *when* you trade does not repair *what* you are selecting.

## Full grid — R1, SPY > 200-SMA (91.1% of setups in-regime)

| h | target | n | 3R hit % | WR % | net R | t | placebo R | margin | t | declined |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | none | 5263 | — | 44.5 | −0.029 | −1.49 | −0.014 | −0.057 | −1.81 | +0.246 |
| 5 | 2R | 5263 | 10.9 | 45.3 | −0.064 | −2.62 | −0.022 | −0.085 | −2.99 | +0.180 |
| **5** | **3R** | 5263 | 5.2 | 44.8 | **−0.045** | **−2.10** | −0.016 | −0.077 | −2.66 | +0.216 |
| 20 | 3R | 5263 | 18.4 | 40.2 | +0.076 | −0.04 | 0.121 | −0.108 | −1.82 | +0.440 |
| 40 | 3R | 5263 | 25.7 | 36.1 | +0.101 | 0.08 | 0.211 | −0.144 | −1.85 | +0.721 |
| 40 | none | 5263 | — | 30.5 | +0.178 | 0.50 | 0.321 | −0.141 | −1.50 | +1.037 |

R2 and R3 grids in [results.json](results.json). All 48 cells logged to `scanner/reports/trial_registry.jsonl`
as `regime_filter_trial`.

## Pre-stated limitation, and it bit

The pre-registration warned this window has **low power** for a downtrend filter: SPY closed above its
200-day SMA on 88.3% of index days. That held — 91.1% of setups sit in-regime under R1, and the declined
bucket is 51 days. So the correct reading of the null is **"not demonstrable in this window"**, which is
weaker than the horizon sweep's "demonstrably absent". A longer history containing a real bear market
would be needed to test a downtrend filter properly.

What is *not* limited by power: the placebo result (48/48, n≈5,263 per cell) and the negative primary
cell (t −2.10). Those are solid.

## Controls

- **Harness:** reuses `walk_np`, validated at max |ΔR| = 0.000e+00 vs `scanner.edge.outcomes.walk_triple_barrier`.
- **No lookahead:** every SMA is a trailing window ending at the decision bar; regime is read from SPY's bar
  on the record's own date and never forward-filled (0 records dropped for a missing SPY bar).
- **Regime-matched placebo** — the decisive control, described above.
- **Fixed cohort:** same 5,776 bullish entries across all cells; HAC lags = cell horizon.
- **Shorts:** bearish records discarded entirely, per the operator rule — not reported, not scored.

## Bottom line

Adding "longs only, in an uptrend" to the 1:3 bracket does not produce a tradeable edge. It makes the
primary cell significantly negative, it fails the regime-matched placebo in every cell, and the trades
it declines were — in the only two episodes available to test it — the better ones.

The open lever remains the one the horizon sweep surfaced: **entry selection has negative value against
random entry**, and no filter applied *after* selection has fixed it.

## Reproduce

```bash
venv\Scripts\python.exe scanner\research\experiments\20260813_regime_filter\run_regime.py
```

No production code modified. Full suite green (349 passed).
