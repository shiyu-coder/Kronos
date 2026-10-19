# Horizon sweep — does a 1:3 bracket work with a longer hold?

**Date:** 2026-08-13 · **Branch:** `codex/finish-kronos-cleanup`
**Pre-registration:** [preregistration.json](preregistration.json) (written before any cell was read; one amendment recorded there, see Controls)
**Verdict: FALSIFIED — 0 of 56 cells pass. The horizon lever is CLOSED.**

## The question

The [2026-08-13 R-target probe](#context) showed every fixed profit target underperforms the shipped
no-target geometry, and that only 5.2% of bullish setups ever touch +3R. Median stop distance is 7.0%
of price, so 3R is a ~21% move inside the 5-bar horizon. That left one thing untested: `PRED_DAYS=5`
was inherited from the Kronos forecast length, never chosen from outcome evidence. **Was the horizon,
not the target multiple, the binding constraint?**

## Answer, in one line

**Yes on reachability, no on money.** Extending the hold does exactly what the mechanism predicts —
the 3R hit rate climbs from 5.2% to 30.4%, and net expectancy goes from −0.003R to +0.280R. And it is
**entirely market exposure**: the same trade plan entered on an *arbitrary* day in the same tickers
pays **more**. Every one of the 28 bullish cells underperforms its drift placebo.

## Bullish results (fixed cohort, n=5,534, 406 entry days, net of 25 bps/side)

| horizon | target | 3R hit % | WR % | gross R | **net R** | t (HAC) | placebo R | **margin** | t |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | none | — | 45.7 | 0.103 | −0.003 | −1.32 | 0.003 | −0.061 | −2.01 |
| 5 | 3R | 5.2 | 46.0 | 0.086 | −0.022 | −1.89 | 0.002 | −0.082 | −2.87 |
| 10 | none | — | 44.0 | 0.171 | +0.065 | −0.52 | 0.072 | −0.115 | −2.32 |
| 20 | none | — | 39.3 | 0.292 | +0.186 | 0.27 | 0.202 | −0.159 | −2.39 |
| 20 | 3R | 18.6 | 41.8 | 0.219 | +0.110 | −0.12 | 0.163 | −0.160 | −2.35 |
| 40 | none | — | 32.9 | 0.385 | **+0.280** | 0.60 | **0.404** | −0.224 | −2.29 |
| 40 | 3R | 26.4 | 38.2 | 0.275 | +0.167 | 0.26 | 0.269 | −0.192 | −2.41 |
| 60 | none | — | 28.0 | 0.359 | +0.256 | 0.43 | **0.508** | −0.265 | −3.30 |
| 60 | 3R | 30.4 | 36.1 | 0.295 | +0.186 | 0.29 | 0.298 | −0.202 | −2.70 |

Full 56-cell grid in [results.json](results.json); all 56 appended to `scanner/reports/trial_registry.jsonl`
as `horizon_geometry_trial`.

### Three things the grid says

1. **The horizon really was the constraint on target reachability.** 3R hit rate: 5.2% (h=5) → 18.6%
   (h=20) → 30.4% (h=60). The earlier "3R is unreachable" finding was a statement about the 5-day
   window, not about 3R.
2. **Targets still truncate, at every horizon.** Within every single horizon, `none` beats 4R beats
   3R beats 2R on both gross and net R. At h=40: 0.280 / 0.232 / 0.167 / 0.073. The 2026-07-02 exit-geometry
   conclusion is horizon-invariant — extending the hold does not rescue profit targets, it makes their
   truncation cost *larger*.
3. **Nothing is statistically distinguishable from zero.** Max t across all 28 bullish cells is **+0.60**,
   against a pre-registered gate of 2.0 (3.2 Bonferroni). Rising point estimates with flat t is the
   signature of longer windows buying variance, not signal.

## The finding that actually matters: negative selection value

The drift placebo runs the identical plan at **every bar of every ticker** in the universe. Margins are
negative in **28 of 28** bullish cells, and significantly so (|t| ≥ 2) in most.

At h=60 no-target: setups **+0.256R** vs arbitrary entries **+0.508R**. The Potter Box breakout selection
gives up half the available drift.

**Why the raw comparison hides this.** In return-percent terms at h=5, the cell averages +0.276% and the
placebo +0.267% — the setups look marginally better. Day-matched, the margin is −0.496% (t −1.99). The
gap is *when* these setups fire: breakouts cluster on days the whole universe is already ripping. Compared
against those same days' baseline, they underperform. Unconditional averages credit the setup for market
timing luck it did not have.

## Controls

| Control | Result |
|---|---|
| **1. Harness fidelity** — numpy twin vs `scanner.edge.outcomes.walk_triple_barrier`, 8,372 walks across all 28 cells | **max \|ΔR\| = 0.000e+00**, 0 exit mismatches |
| **2a. Aggregate agreement with shipped index** (gate) | Δ mean R **+0.00025**, corr **0.99972** — PASS with ~20× margin |
| **2b. Per-record agreement** (reported) | 98.81% within 1e-5; 5 exit flips in 11,047 (0.045%), all knife-edge stop→horizon on sub-$1 names |
| **3. Fixed cohort** | Same 5,534 entries in every cell — no survivorship confound |
| **4. Dependence** | HAC lags = cell horizon, not the shipped lag-5 |
| **5. Denominator (adversarial)** | Cell risk 7.777% vs placebo 7.809% (ratio 0.996) — the negative margin is not a denominator artifact; it survives in return-percent space in 14/14 cells |

**One amendment**, fully recorded in `preregistration.json`: control 2 as written required ≥99% per-record
reproduction and hit 98.81%. It conflated harness fidelity (proven *exactly* by control 1) with bar-snapshot
identity across a vendor-revision day, which no experiment can hold. It was split into a stricter aggregate
gate (specified before any cell was read) plus a reported drift characterization. The drift is sign-favorable
to *rejecting* the null (+0.0003R), so it cannot manufacture this negative verdict.

## What this closes and what it doesn't

**Closed — do not re-run:**
- Horizon extension as a lever for this universe and geometry. 5 → 60 bars, four target modes, 56 cells, nothing passes.
- "A 1:3 bracket works if you hold longer." It reaches 3R far more often and still loses to holding nothing in particular.

**Newly open, and more important than what was tested:** the placebo result is a *finding*, not just a control.
The scanner's setup selection appears to have **negative** value against a same-ticker random-entry baseline.
That is a sharper indictment than the cost-model result from 2026-08-08 — it says the problem may not be
that the edge is too small to clear 50bps, but that the entry criteria are selecting the wrong bars.
Worth a dedicated pre-registered test before any further work on ranking, features, or costs.

**Untouched by this experiment:** stop placement. Every cell used the shipped empty-space risk. A 7%
median stop on a 5-day hold is the other half of the geometry and has never been swept.

## Reproduce

```bash
venv\Scripts\python.exe scanner\research\experiments\20260813_horizon_sweep\fetch_bars.py
venv\Scripts\python.exe scanner\research\experiments\20260813_horizon_sweep\run_sweep.py
venv\Scripts\python.exe scanner\research\experiments\20260813_horizon_sweep\robustness_return_space.py
```

No production code was modified. Full suite green (349 passed) after this work.

<a name="context"></a>*Context: this experiment answers the open question left by the 2026-08-13 R-target
probe, which itself independently replicated the 2026-07-02 six-variant exit-geometry sweep.*
