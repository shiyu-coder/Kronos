# Kronos scanner retirement decision - 2026-08-20

## Decision

The Potter Box scanner trading strategy and its daily go-live program are
retired. Do not allocate capital, paper-promote candidates, send live alerts,
or continue daily parameter/feature searches on this design.

This is a terminal strategy verdict, not another temporary `NO-GO`. The
engineering work produced a substantially safer and more honest evidence lab,
but it did not produce a profitable trading system. Continuing to search the
same evidence for a favorable threshold, subgroup, or geometry would be
post-selection overfitting, not autonomous improvement.

The upstream Kronos model and the separate desktop forecasting app remain in
the repository. Retirement applies to the Potter Box options scanner and the
automation whose objective was to make it live-ready.

## Evidence is sufficient to stop

Project work spans the May design specifications through this August decision,
with 42 daily audit notes from 2026-06-24 through 2026-08-19. The append-only
trial registry contains 256 declared trial rows:

| Trial family | Rows | Accepted | Shipped |
|---|---:|---:|---:|
| Calibration / ranking | 134 | 0 | 0 |
| Horizon geometry | 56 | 0 | 0 |
| Regime filters | 48 | 0 | 0 |
| Exit geometry | 7 | 0 | 0 |
| Adaptive policy | 11 | 0 | 0 |
| **Total** | **256** | **0** | **0** |

Three adaptive-policy rows applied research-threshold or doctrine-baseline
changes. None authorized a production strategy, passed the production
acceptance gates, or shipped.

The final corrected-semantic, source-fingerprinted market-hours audit used
1,500 recent purged walk-forward candidates, charged 25 bps per side, and
blocked future analogs:

- Score 55: 32 raw signals / 27 independent entry days, **-0.2434R/day**,
  HAC t=-2.1654, precision lower bound 0.2155 versus the required 0.55.
- Top decile: 150 raw signals / 55 independent entry days,
  **-0.2102R/day**, HAC t=-2.1188, precision lower bound 0.2932 versus 0.45.
- Rank IC: +0.0362 with clustered p=0.139661 versus the required
  IC>=+0.07 and p<=0.05.
- Bullish and bearish direction cohorts were both net-negative and blocked.
- The research journal had 76 resolved candidates: 34 wins / 42 losses raw;
  the embargoed independent cohort was 19 wins / 23 losses. Adaptive policy's
  terminal recommendation was `hold_no_edge`.

This is mature negative evidence. It is not a request for more samples.

## Major strategy routes tested and closed

The following were evaluated with point-in-time or purged walk-forward
controls and did not justify production:

- Raw-R, win-probability, tail-probability, Huber, quantile, winsorized, and
  rank objectives on the historical feature set.
- Classic chart-pattern transformations and feature ablations; all 13
  preregistered chart-pattern cells ranked backward.
- Real Kronos forecast features on 5,792 bullish out-of-fold samples; 0 of 6
  cells passed and all rank ICs were negative.
- Forty-eight regime-filter trials; none passed and the proposed regime rule
  was backward on the data.
- Seven target/exit variants and 56 horizon/target cells; none passed, and
  longer-horizon returns did not beat matched market drift.
- Detector-gate correction, which restored source-faithful coverage but did
  not restore economic edge.
- Three-day post-breakout retest entry: better fills on the same bad setups,
  but no significant absolute edge and no significant drift margin.
- Daily-close cost-basis invalidation: adequate coverage, but negative
  absolute edge and no improvement over the shipped stop or matched drift.
- Bearish direction, which stayed consistently negative.

The project also fixed multiple issues that could otherwise have manufactured
false optimism: future leakage, insufficient purge windows, missing
transaction costs, a cost-floor fail-open, inconsistent live/historical score
scales, incomplete cost denominators, stale evidence reuse, invalid runtime
provenance, detector/source mismatch, and false punchback-reclaim labels. Each
correction made the final negative verdict more credible.

## Retirement controls applied

- Windows Task Scheduler task `Kronos Daily Research Ops` disabled before its
  2026-08-20 launch.
- Codex automation `daily-kronos-adjustment` paused.
- `scanner/RETIRED.json` added as the machine-readable retirement marker.
- Scanner live preflight now rejects `--mode live` whenever that marker exists,
  before credentials, environment flags, or readiness are considered.
- The scheduled batch launcher exits successfully without running research
  while the marker exists, protecting against accidental task re-enablement.
- Daily brief Telegram delivery defaults off.

These controls are reversible, but not automatically. Code, reports, journals,
experiment artifacts, and tests are preserved for audit.

## Revival criteria

This strategy must not be revived by removing the marker after one favorable
backtest. Every condition below is required:

1. A genuinely new strategy mechanism, not a threshold, direction, target,
   horizon, regime, retest, stop, chart-pattern, loss-function, or favorable
   subgroup variation of the retired design.
2. A preregistration committed before outcomes are read, with immutable input
   hashes, explicit failure criteria, and a multiplicity ledger.
3. Purged walk-forward evaluation net of at least 25 bps per side, with full
   cost-denominator coverage and a same-day ticker-universe drift control.
4. All existing ranking, top-decile, precision, direction, provenance,
   data-quality, and execution-quality gates passing without exceptions.
5. Confirmation on at least 20 new independent market days collected after
   retirement and not used to invent the new strategy.
6. Explicit human authorization to remove `scanner/RETIRED.json`, re-enable
   Windows scheduling, unpause the Codex automation, and begin paper-only
   operation. Live capital would still require a later separate decision.

Until all six are satisfied, the correct project state is **RETIRED**.
