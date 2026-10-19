# Close-based cost-basis invalidation experiment

**Date:** 2026-08-19

**Verdict:** FALSIFIED -- no production stop change.

## Question

The shipped five-session plan stops when an intraday wick crosses the Potter
cost-basis midpoint. The source doctrine gives completed candle closes priority
over wick extremes. Would exiting only after a completed daily close beyond the
same cost-basis line remove recoverable wick stop-outs and repair the mature-
negative entry edge?

The rule, input hashes, controls, and gates were fixed in
`preregistration.json` before output was read. The treatment retained the exact
same entry, risk denominator, cost-basis price, five-bar horizon, no-target
plan, and 25 bps/side cost floor. A close beyond cost basis filled at the actual
close, so delayed exits were allowed to lose more than 1R.

## Harness control

The cached-bar production re-walk reproduced the pinned index across 1,373
non-overlapping source-valid rows:

- Pearson correlation: 0.99999657 (required >=0.99).
- Mean gross-R difference: +0.00007286R (required absolute <=0.005R).

The harness passed before the treatment result was evaluated.

## Primary bullish result

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Coverage | >=100 rows / >=40 days | 751 / 314 | PASS |
| Absolute net edge | mean R >0, HAC t >=2 | -0.0189R/day, t=-0.27 | FAIL |
| Same-setup improvement | margin >0, HAC t >=2 | -0.0029R/day, t=-0.23 | FAIL |
| Day-matched drift | margin >0, HAC t >=2 | -0.0022R/day, t=-0.04 | FAIL |

The close rule converted 115 production stops into later close invalidations
and let 59 wick-stopped rows reach the horizon. Its raw row mean rose from
+0.0064R to +0.0273R, but those gains clustered on already favorable entry
days. The pre-registered independent-day estimate became slightly worse than
the shipped rule (-0.0189R versus -0.0160R/day) and did not beat arbitrary
same-day exposure (-0.0167R/day).

## Locked bearish diagnostic

Bearish treatment remained negative at -0.1151R/day (t=-1.55), versus
-0.1104R/day for the shipped stop. Its +0.0266R/day margin over bearish drift
was weak (t=0.54). Bearish remains blocked.

## Decision

- Do not change the production stop, score, alerts, promotion, or readiness.
- Close this exact daily-close invalidation at the existing cost-basis line.
- Do not rescue it with a direction carve-out, threshold, close-delay bucket,
  ATR buffer, or favorable unclustered row mean.
- The result does not test a stateful intraday/multi-timeframe invalidation
  model; that remains a distinct future hypothesis, not a variant of this one.

The preregistration timestamp initially contained a hand-entered future-time
typo. Its amendment records the correction from the filesystem creation time:
the file was created ten seconds before experiment output, and no rule or gate
changed.

## Reproduce

```powershell
.\venv\Scripts\python.exe scanner\research\experiments\20260819_close_invalidation_stop\run_experiment.py
```

The runner prints JSON and does not mutate production reports, tuning, alerts,
or evidence.
