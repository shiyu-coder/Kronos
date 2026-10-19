# Post-breakout retest entry experiment

**Date:** 2026-08-18
**Verdict:** FALSIFIED — no production entry change.

## Question

The current Potter entry is taken at the breakout decision-bar close. The
source doctrine emphasizes punchbacks and reclaiming control. Does waiting up
to three completed daily bars for a price-improving control-level retest and a
close back in the breakout direction repair the mature-negative entry edge?

The rule, controls, hashes, gates, and failure condition were fixed in
`preregistration.json` before output was read. A metadata amendment records and
corrects a hand-entered future-time typo using the preregistration file's
filesystem creation timestamp; no rule or gate changed.

## Primary bullish result

The rule found 179 confirmations across 138 entry days from 632 independent
source-valid triggers. Median entry improved by 0.95% versus the breakout
close.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Coverage | >=100 entries / >=40 days | 179 / 138 | PASS |
| Absolute net edge | mean R > 0, HAC t >= 2 | +0.0386R, t=0.35 | FAIL |
| Same-setup improvement | margin > 0, HAC t >= 2 | +0.1859R, t=3.67 | PASS |
| Day-matched drift | margin > 0, HAC t >= 2 | +0.0403R, t=0.45 | FAIL |

Waiting for the retest gives a materially better fill than entering those exact
setups immediately: immediate entries lost -0.2323R/day while retest entries
were +0.0386R/day. That is not enough. The retest estimate is statistically
indistinguishable from zero and from ordinary same-direction exposure on the
same entry days. It improves a bad entry without demonstrating an edge.

## Locked bearish diagnostic

Bearish retests also improved the exact immediate setups (+0.1348R, t=2.19),
but still lost -0.1984R/day (t=-2.43) and did not beat drift (-0.0062R,
t=-0.09). Bearish remains blocked.

## Decision

- Do not add a retest entry, score bonus, shadow promotion, or alert path.
- Close this exact three-daily-bar reclaim rule. Do not rescue it with a delay
  bucket, threshold, direction exception, or the favorable same-setup margin.
- The result does not test intraday punchbacks, stop placement, or a persistent
  multi-timeframe box state machine.

## Reproduce

```powershell
.\venv\Scripts\python.exe scanner\research\experiments\20260818_retest_entry\run_experiment.py
```

The runner prints JSON and intentionally does not mutate production reports,
tuning, alerts, or evidence.
