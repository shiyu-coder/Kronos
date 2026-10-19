# Potter Box Scanner V1

> [!CAUTION]
> **RETIRED 2026-08-20.** The trading strategy and daily go-live program failed
> their mature out-of-sample evidence gates and are no longer active. Live mode
> is hard-blocked by `scanner/RETIRED.json`, scheduled research is disabled, and
> daily Telegram briefs default off. The code and evidence remain available for
> audit only. See `docs/RETIREMENT-2026-08-20.md`; revival requires a genuinely
> new preregistered strategy and new post-retirement confirmation data.

Local Windows Python scanner that identifies Potter Box-style options setups and sends Telegram alerts only when every gate passes.

## Safety Defaults
- Retired state is checked before every live preflight; environment flags and a passing audit cannot bypass it.
- Dry-run is default.
- Fail closed on missing/uncertain data.
- Live alerting requires both `--mode live` and `LIVE_MODE_ENABLED=true`.
- Live alerting also requires a current Edge Readiness Audit with `readiness=paper_trade_only`; run `--mode run_edge_lab` first. Preflight verifies the scan and validation timestamps inside the audit are both under 24 hours old, carry the same evidence-lab run ID, and were produced by the exact current production-source fingerprint. Rewriting `audit_edge` from stale artifacts or changing scoring/execution code after a passing lab run therefore cannot refresh or reuse the live gate. Immediately before any Telegram send, the exact current ticker/direction is rescored and must itself earn an Edge `promote` recommendation in a validation-supported direction; a different promoted ticker cannot authorize the alert.
- `paper_trade_only` additionally requires current equity-delay provenance (at most one minute), a consolidated-capable feed, and execution-grade Tradier OPRA evidence. The free 16-minute historical SIP route remains research-only even though its consolidated coverage has high confidence.
- Secrets are loaded from `.env` only (do not hardcode keys/tokens in code).

## Potter v3 research path (moved)
The source-faithful re-encoding of the method (`scanner/potter_v3/`, failed its
2026-09-05 pre-registered test) moved on 2026-10-19 to its own project,
`C:\Users\Jacob Higgins\projects\Potter Box Strategy`, together with its
research record, bar cache, and the @potter.stocks source material. This
scanner keeps only the retired V1 detector.

## Setup
```bat
cd /d C:\Users\Jacob Higgins\projects\kronos-predictor
copy scanner\.env.example scanner\.env
scanner\setup_dependencies.bat
```

## Alpaca Market Data (Recommended)
Set these in `.env` for free IEX-based bars:
```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
MARKET_DATA_PROVIDER=auto
ALPACA_FEED=iex
ALPACA_OPTIONS_FEED=indicative
MINIMAX_ENABLED=false
MINIMAX_API_KEY=your_key
MINIMAX_BASE_URL=https://api.minimax.io/v1
MINIMAX_MODEL=MiniMax-M2.7-highspeed
```

Provider behavior:
- `MARKET_DATA_PROVIDER=auto`: Alpaca first, then yfinance fallback.
- `MARKET_DATA_PROVIDER=alpaca`: Alpaca only (fails if unavailable).
- `MARKET_DATA_PROVIDER=yfinance`: yfinance only.
- Current scans use `ALPACA_FEED` (`iex` on the Basic plan).
- Historical index building and `research_scan` request SIP data with a 16-minute delay, which fits Alpaca Basic's delayed consolidated-data access.
- All Alpaca bars (daily and 30m intraday) are fetched with split adjustment (price AND volume), so a split inside a lookback window no longer reads as a giant gap or a fake volume spike. The yfinance daily fallback runs with `repair=True` (fixes Yahoo's occasional unapplied splits and 100x glitches); yfinance intraday is as-traded and relies on the bar contract to flag split gaps.
- Outcome review (`review_outcomes`) resolves pending decisions on the 16-minute-delayed SIP feed - resolution happens days after the decision, so the delay is free and consolidated High/Low (not the ~3% IEX slice) decides whether stops/targets were really touched. Source-aware records are anchored to the exact synthetic session that produced the signal, and the five-session target must reach its closing anchor plus feed delay before it can resolve; this keeps market-hours runs from training on a shifted or partial terminal session. Records whose entry price no longer matches the refetched history's scale (a split between decision and review) are quarantined as `not_applicable` instead of resolving to fiction.
- Options selection uses Tradier's OPRA-consolidated chain first. Infrastructure failures may fall back to the Alpaca/yfinance indicative join, which remains research-only.
- Kronos forecasts are collected and shown as advisory evidence, but do not hard-gate alerts or change the Edge Score by default. A preregistered 5,792-sample bullish backfill found 0/6 Kronos ranking cells passed the production gate, so `KRONOS_LIVE_GATE_ENABLED=true` is an explicit experimental opt-in until new independent evidence supports restoring the gate.

Data-accuracy gates on the evidence index (2026-07-03):
- Fail-closed OHLCV bar contract per ticker: NaN/non-positive prices, High/Low body violations, negative volume, duplicate/unsorted timestamps, zero-volume bars with price range, forward-filled flat bars, stale-feed runs (5+ identical closes / 4+ identical bars), plus warnings for split-ratio-shaped overnight gaps (catches the 3:2 splits the 45% move rule misses).
- NYSE session-completeness check via `pandas_market_calendars`: missing expected sessions warn at <=5 and fail the ticker beyond that (halted/partially-delivered histories cannot silently become evidence).
- Halted/non-traded sessions (2026-07-10): Alpaca and Yahoo forward-fill a session with no prints as a zero-volume flat placeholder bar (e.g. CLSK's full-day Nasdaq halt on 2024-11-08); Tradier omits it. Placeholder bars are dropped before the contract check — the halt then counts as a missing session under the session-completeness rule above instead of failing the ticker for the whole two-year lookback. Zero-volume bars *with* price range are genuine corruption and still hard-fail. The same normalization runs before outcome review so a halt inside a resolution window cannot block a record until expiry.
- Cross-source verification: when `TRADIER_API_TOKEN` is set, each ticker's daily returns are compared against Tradier's independent history feed; a split-scale divergence (>20pp on one session) or a systemic (>5%) disagreement rate fails the ticker for the run. No token or transport failure = skipped, never a block.
- The current session's still-forming daily bar is dropped at index build, so a mid-session scheduled run cannot ingest ~60%-of-a-day bars as evidence.

## Modes
```bat
scanner\run_scanner.bat --mode dry_run
scanner\run_scanner.bat --mode live
scanner\run_scanner.bat --mode backtest_intraday_60d
scanner\run_scanner.bat --mode backtest_daily_proxy_2y
scanner\run_scanner.bat --mode calibration --ticker PLTR --tradingview_csv C:\path\to\tradingview_export.csv
scanner\run_scanner.bat --mode calibration --ticker PLTR --tradingview_csv C:\path\to\tradingview_export.csv --sweep_anchors
scanner\run_scanner.bat --mode calibration --calibration_csv_glob "C:\Users\Jacob Higgins\Downloads\BATS_*, 1D.csv" --sweep_anchors
scanner\run_scanner.bat --mode test_telegram --test_message "Potter scanner connectivity test"
scanner\run_scanner.bat --mode test_minimax --test_message "MiniMax connectivity test"
scanner\run_scanner.bat --mode review_outcomes
scanner\run_scanner.bat --mode autotune
scanner\run_scanner.bat --mode autotune --apply_tuning
scanner\run_scanner.bat --mode adaptive_policy
scanner\run_scanner.bat --mode adaptive_policy --apply_tuning
scanner\run_scanner.bat --mode replay_eval --replay_dataset "C:\Users\Jacob Higgins\projects\kronos-predictor\scanner\replay\sample_replay_dataset.json"
scanner\run_scanner.bat --mode research_scan
scanner\run_scanner.bat --mode diagnose_zero_results
scanner\run_scanner.bat --mode build_retrieval_index
scanner\run_scanner.bat --mode validate_edge
scanner\run_scanner.bat --mode edge_scan
scanner\run_scanner.bat --mode diagnose_edge
scanner\run_scanner.bat --mode audit_edge
scanner\run_scanner.bat --mode run_edge_lab
scanner\run_scanner.bat --mode research_ops
scanner\run_scanner.bat --mode brief
scanner\run_scanner.bat --mode doctor
```

Equivalent Python commands:
```bat
.\venv\Scripts\python.exe -m scanner.main --mode dry_run
.\venv\Scripts\python.exe -m scanner.main --mode backtest_intraday_60d
.\venv\Scripts\python.exe -m scanner.main --mode backtest_daily_proxy_2y
.\venv\Scripts\python.exe -m scanner.main --mode calibration --ticker PLTR --tradingview_csv C:\path\to\tradingview_export.csv
.\venv\Scripts\python.exe -m scanner.main --mode calibration --calibration_csv_glob "C:\Users\Jacob Higgins\Downloads\BATS_*, 1D.csv" --sweep_anchors
.\venv\Scripts\python.exe -m scanner.main --mode run_edge_lab
.\venv\Scripts\python.exe -m scanner.main --mode doctor
```

## Automation (hands-off daily loop)
A Windows scheduled task named `Kronos Daily Research Ops` runs `scanner\run_research_ops_scheduled.bat` every weekday at 13:30 Central (14:30 ET, mid-session so Tradier quotes are execution-grade). It wakes the PC if asleep, catches up if the start was missed, runs on battery power, and uses both Task Scheduler failure retries and guarded recovery launches at 14:00 and 14:30 Central. The recovery launcher checks for a complete `research_ops_report.json` from the current New York market date, so a successful first run makes later launches no-ops while a hard interruption such as `0xC000013A` is retried inside the market-hours evidence window. Before launching, it verifies the actual NYSE session window with the exchange calendar; holidays, early-close afternoons, and delayed after-hours catch-ups are skipped instead of replacing valid evidence with closed-market quotes. Calendar failures block the run and return a nonzero exit code. Output appends to `scanner\logs\scheduled_research_ops.log`; the latest launcher state is recorded in `scanner\reports\scheduled_research_ops_status.json`, and recovery no-ops retain the verified market-open and market-close timestamps instead of erasing session provenance.

Each run finishes by writing `scanner\reports\daily_brief.md` and sending the condensed brief to Telegram (`BRIEF_TELEGRAM_ENABLED`, uses `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` from `.env`). The Telegram message is a status report only; live trade alerting remains behind the evidence-gated live-mode checks.

To restore the hardened schedule after moving the checkout, run `powershell -ExecutionPolicy Bypass -File scanner\register_research_ops_task.ps1`. To inspect or change it manually, open Task Scheduler > "Kronos Daily Research Ops".

## Daily Brief
`brief` reads the latest report artifacts (no network, no model loads) and renders a verdict-first operator summary: evidence-gate progress, today's scan, accepted counterfactual research samples, learning-loop state including Kronos lift, every blocker in plain English with its fix, and the single next action. Both the markdown and phone summary distinguish an under-sampled gate from an adequately sampled, negative design; mature-negative evidence is never described as merely "not yet." Edge scan recommendations are never labelled as live trades: blocked/research-only recommendations render under `EDGE RESEARCH`, while an audit-authorized candidate can render only as a `PAPER CANDIDATE`. `research_ops` runs it automatically as its final stage.

Gate progress distinguishes `COLLECTING` from `NEGATIVE`. Once both the score-55 and top-decile routes have enough independent entry days but non-positive dependence-aware expectancy, the brief calls for a pre-registered entry-selection redesign instead of implying that more samples alone will repair the tested strategy.

```bat
.\venv\Scripts\python.exe -m scanner.main --mode brief
```

Output: printed to the console and saved at `scanner\reports\daily_brief.md`.

## Project Doctor
`doctor` is a no-secrets health report for local review. It verifies the Python version, core runtime imports, and whether secret/runtime artifact paths are ignored by Git. Use it before demos, handoffs, or deeper evidence-lab runs:

```bat
.\venv\Scripts\python.exe -m scanner.main --mode doctor
```

## What Is Logged
- PASS/SKIP/ERROR per ticker with reason.
- Rotating log file at `scanner\logs\scanner.log`.
- Alpaca request tracing at `scanner\logs\request_ids.log` (request id when available).
- Backtest and calibration reports in `scanner\reports\`.
- Decision journal for self-tuning at `scanner\reports\scan_decisions.jsonl`.
- Outcome review report at `scanner\reports\outcome_review_summary.json`.
- Replay report at `scanner\reports\replay_eval_report.json`.
- Calibration batch summary at `scanner\reports\calibration_summary.json`.
- Zero-result diagnostic at `scanner\reports\zero_result_diagnostic.json`.
- Edge Evidence Lab runs at `scanner\reports\evidence\<run_id>\manifest.json`.
- Edge readiness audit at `scanner\reports\edge_audit_report.json`.
- Adaptive policy report at `scanner\reports\adaptive_policy_report.json`.
- Daily operator brief at `scanner\reports\daily_brief.md`.
- Trial registry (every tuning change evaluated/applied) at `scanner\reports\trial_registry.jsonl`.

## Edge Evidence Lab
`run_edge_lab` executes the full research loop in one local run:
1. Build the retrieval index.
2. Validate scored historical candidates.
3. Scan the current watchlist.
4. Write diagnostics.

Each lab run writes a manifest plus JSONL row artifacts for index records, validation candidates, scan candidates, metrics, and diagnostics. If a Parquet engine such as `pyarrow` is installed, matching `.parquet` sidecars are written automatically. If not, JSONL remains the canonical fallback. To enable Parquet sidecars, install the optional package extra with `.\venv\Scripts\python.exe -m pip install -e ".[evidence]"`.

Edge validation uses purged walk-forward analogs: historical candidates are scored only against records available before that candidate timestamp. Current scans may use the full saved index, but validation reports mark `validation_method=purged_walk_forward` and `future_analogs_allowed=false`.

How the evidence is measured (2026-07 revision):
- The retrieval index is built from the watchlist plus `EDGE_INDEX_EXTRA_UNIVERSE` (index/validation only), and analogs match on a curated set of scale-free setup features with direction matching and a cross-ticker embargo during validation.
- Historical outcomes are triple-barrier labeled (stop at -risk, target at +target, time exit at the horizon, evaluated against the High/Low path). A stopped-out trade is a loss even if price later recovers.
- Stop placement remains the Potter cost-basis midpoint with intraday High/Low invalidation. A pre-registered 2026-08-19 test of completed-daily-close invalidation at the exact same level failed absolute edge, same-setup improvement, and day-matched drift gates; do not substitute that rule or cite its favorable unclustered row mean. The test did not cover a genuinely stateful intraday/multi-timeframe invalidation model.
- The Potter setup pass condition follows the source-derived visual doctrine: close-based control levels, at least two touches of each control, a close outside control, and prior-close bias on the matching side of cost basis. ATR compression, range compression, and normalized slope remain recorded diagnostics and research/ranking features; they are not setup gates. A 2026-08-13 detector-fidelity audit traced those three hard gates only to an engineering handoff (not the underlying strategy), found their implementations did not match that handoff's formulas, and measured that their conjunction starved the detector. This correction does not bypass Edge promotion, validation, cost, feed, liquidity, or live-preflight gates; the corrected setup corpus must earn readiness from scratch.
- The encoded plan's TARGET side is chosen by `EDGE_EXIT_TARGET_MODE` (`none` | `nearest_empty_space` | `next_empty_space` | `atr_multiple`, plus an `EDGE_EXIT_TARGET_R_FLOOR`), overridable per process via `KRONOS_EXIT_TARGET_*` env vars for lab sweeps. The shipped default is `none` (stop/horizon exits only): the 2026-07-02 six-variant sweep (logged as `exit_geometry_trial` in the trial registry) showed every tested profit target truncated more bullish upside than it locked in (bullish avg R: nearest -0.01, 1.5R floor +0.13, 2R floor +0.15, 2xATR +0.16, no target +0.19 at t=5.2), while bearish stayed negative under all six geometries. Each index record carries `target_mode`/`target_pct_used` provenance and the validation report stamps `exit_geometry_config`.
- Validation samples the most recent `EDGE_VALIDATION_MAX_RECORDS` records by timestamp across all tickers, and reports Spearman rank IC over all samples, top-5/10/20% percentile blocks, decile spread, per-direction expectancy, precision bounds, and R-multiple t-stats. Dependence-aware blocks aggregate trades by entry day, use five Newey-West lags for overlapping outcome windows, and bound daily precision by the more conservative of the HAC normal bound and a Wilson bound whose effective sample size is entry days.
- **Every gate and candidate score is net of transaction costs (2026-08-08/11/12).** `walk_triple_barrier` stores outcomes gross - it charges nothing - so before this change rank IC, average R, precision and the HAC t-stats all described an edge no execution could capture, and the only cost-aware number in the report (`average_return_pct_after_slippage`, a flat 5 bps) was cosmetic and read by no gate. `compute_edge_validation_report` charges `EDGE_COST_BPS_PER_SIDE` (default 25 bps/side = 0.50 return pct pts round trip, the pre-registered basis from the 20260710 sprint's `e4_decision`) against every row BEFORE any metric is computed, re-divides by the stored `risk_pct_used` to keep net R on the R scale, and recomputes the win/loss label from the net return so precision is net too. Candidate scoring applies the same conversion to retrieved analogs before analog expectancy affects the Edge score or promotion; missing analog risk denominators block promotion, and each score exposes both gross/net summaries plus cost provenance. Gross values stay visible as `gross_*` / `average_gross_return_pct` for diagnosis, and the report stamps a `cost_model` block. The audit's `costs_charged` check fails closed unless the block meets the immutable 25 bps/side production floor, has the exact implied round-trip charge and `net_of_costs` basis, covers returns/R/labels, and proves complete `risk_pct_used` denominator coverage matching the validation sample count. Lowering `KRONOS_COST_BPS_PER_SIDE` may still support an explicitly labeled research scenario, but it cannot authorize paper/live readiness; a token-positive, decorative, partial, or internally inconsistent cost stamp cannot open a gate. Charging validation costs moved bullish from "unproven, +0.05 avg daily R" to blocked at -0.06, and the top decile from -0.02 to -0.13 avg R.
- **The numeric Edge score contains only historically reconstructable strategy evidence (2026-08-15).** Point-in-time equity-feed confidence and option-chain quality are unavailable in the historical retrieval index, so they cannot add or subtract score points without putting live candidates on a different scale from walk-forward validation. Data freshness/feed confidence, contract availability, spread, and options provenance remain mandatory promotion/execution gates with explicit blocking reasons, but their scorecard components are fixed at zero. Scan reports carry both gross/net analog summaries and the analog cost model so the score's cost basis can be audited directly.
- The audit accepts either evidence route: the legacy absolute-threshold gate, or the ranking gate (rank IC >= 0.07 with entry-day-clustered p <= 0.05, plus a profitable top decile with >= 20 evidence days, HAC t >= 2, dependence-aware precision lower bound >= 0.45). A direction becomes promotion-eligible only with >= 15 evidence days, positive average R, and entry-day/HAC t >= 2; negative directions are blocked and all other directions remain explicitly unproven. Raw trade counts remain visible for coverage, but they are never treated as independent evidence.

Research recommendations are gated by live setup quality. Strong historical analogs cannot promote or research-label a candidate when both Potter Box and Empty Space gates fail.

### Potter Doctrine v2 Features
The edge feature vector includes a research-only Potter Doctrine v2 score. It records punchback/retest reclaim state, failed reentry risk, cost-basis hold/reclaim/loss state, and overlap-box stack alignment. A reclaim requires an earlier observable close outside the control followed by a later control-level retest and continuation-side close; a pre-breakout consolidation touch is not a reclaim. Doctrine evidence is versioned, and adaptive policy excludes legacy unversioned scores after semantic corrections instead of mixing incompatible cohorts. These fields are written into edge features and decision records so analog retrieval, scorecards, zero-result diagnostics, and future adaptive policy work can learn from failed or near-miss setups.

Doctrine v2 can improve research ranking, but it does not bypass live safety gates. Promotion still requires setup gates, enough analog samples, positive expectancy, usable feed confidence, and execution-grade options data quality.

`adaptive_policy --apply_tuning` may safely tighten `DOCTRINE_V2_SCORE_BASELINE` when resolved doctrine-scored research candidates are loss-heavy. It only tightens the baseline; it does not automatically lower doctrine standards.

## Research Operations
`research_ops` is the repeatable evidence-maintenance cycle. It backs up and deduplicates the decision journal, resolves aged outcomes, collects a fresh delayed-SIP research scan, runs diagnostics, runs bounded autotuning, applies only safe adaptive-policy overrides, refreshes the complete edge lab, and writes `scanner\reports\research_ops_report.json`.

```bat
.\venv\Scripts\python.exe -m scanner.main --mode research_ops
```

The decision journal rejects repeat observations of the same ticker/setup/day so repeated scans cannot inflate sample counts or bias autotuning.

## Edge Readiness Audit
`audit_edge` summarizes whether the current evidence is usable, research-only, or blocked. It checks:
- Purged walk-forward validation is enabled and future analogs are blocked.
- Threshold 55 has enough out-of-sample entry days, dependence-aware precision, and statistically positive average R.
- Current candidates have usable feed confidence and non-missing options liquidity fields.

The audit is intentionally conservative. Free IEX-only market data and indicative or missing options data are useful for research, but they should not be treated as capital-ready evidence without better coverage and liquidity checks.

The free-data ensemble records stock provider/feed/delay plus option provider/feed/quote age. Indicative options receive a quality penalty and cannot produce a `promote` recommendation; real-time OPRA-quality options data is required for promotion.

## Self-Tuning Loop
1. Run scanner (`dry_run` or `live`) so decisions are logged.
2. Run `--mode review_outcomes` to resolve pending decisions to win/loss after horizon.
3. Run `--mode autotune` to generate a bounded threshold proposal using resolved outcomes.
4. Run `--mode autotune --apply_tuning` to persist safe overrides into `scanner\tuning\overrides.json`.
5. Restart scanner process so new thresholds are loaded from overrides.

## Adaptive Policy Loop
`adaptive_policy` evaluates resolved research candidates by score threshold and outcome quality. It first removes overlapping same-ticker outcome windows, then bases automatic changes on entry-day evidence: same-day cross-ticker rows are one market cluster, adjacent five-session outcomes use HAC-adjusted uncertainty, and raw row counts remain coverage only. It uses the dependence-adjusted win-rate lower bound and average entry-day return before recommending any improvement threshold. When the resolved research cohort is loss-heavy across enough evidence days, it may safely tighten `RESEARCH_CANDIDATE_MIN_SCORE`; it does not loosen live gates or promote signals without evidence.

The policy is two-sided with asymmetric safeguards. `RESEARCH_CANDIDATE_MIN_SCORE` is a data-collection throttle (research candidates are paper counterfactuals, never alerts), so a noise-driven tightening must not starve the journal forever. Loosening is recommended only when a lower threshold's cohort dominates the current one on conservative, dependence-adjusted bounds (at least 12 evidence days, positive average daily return, win-rate LB >= 0.30, and LB/return margins over the current cohort), capped at 10 points per change. Tightening applies immediately; loosening additionally requires a 7-day cooldown since the last automatic change plus a confirmation on a later calendar day. Every evaluated or applied change is appended to `scanner\reports\trial_registry.jsonl` so the multiple-testing trial count stays honest.

The report also includes a `kronos_lift` section: resolved research candidates split by Kronos directional agreement (research candidates are evaluated by Kronos at scan time). This is the evidence that decides whether the Kronos confirmation stage earns its gate.

The same report includes a `doctrine_v2` section with threshold cohorts, punchback-state outcomes, cost-basis-state outcomes, and risk flag counts. When Doctrine v2 evidence is loss-heavy at the current baseline, the safe auto-action is to raise `DOCTRINE_V2_SCORE_BASELINE`.

```bat
.\venv\Scripts\python.exe -m scanner.main --mode adaptive_policy
.\venv\Scripts\python.exe -m scanner.main --mode adaptive_policy --apply_tuning
```

`research_ops` runs this stage with safe application enabled, so repeated maintenance cycles can automatically reduce bad research candidates while keeping live alerts fail-closed.

How learning records behave:
- Passed alerts are recorded as live decisions.
- Skipped setups can be recorded as counterfactual decisions (when direction/entry can be inferred) so missed calls can be evaluated later.
- Validation-only failures (for example, price below min threshold) are marked `not_applicable` and excluded from tuning.

Current gating for tuning:
- Outcomes must age past `OUTCOME_MIN_AGE_DAYS` before resolution.
- Autotune requires at least `AUTOTUNE_MIN_SAMPLES` resolved outcomes before proposing changes.

Suggested automation cadence:
- Every hour/day: `research_scan` while the strict alert path is still producing few or zero signals.
- Daily after close: `review_outcomes`.
- Daily after review: `diagnose_zero_results`.
- Daily after review: `autotune --apply_tuning`.
- Daily after review: `adaptive_policy --apply_tuning`, or run `research_ops` to include it automatically.

Research mode:
- `research_scan` logs graded near-miss Potter candidates as pending counterfactuals.
- These candidates do not send Telegram alerts and do not weaken live alert safety.
- Use this mode to build enough labeled data for autotune before enabling or relaxing live alerts.

## Quick Validation Runbook
```bat
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m scanner.main --mode dry_run
.\venv\Scripts\python.exe -m scanner.main --mode backtest_intraday_60d
.\venv\Scripts\python.exe -m scanner.main --mode backtest_daily_proxy_2y
.\venv\Scripts\python.exe -m scanner.main --mode replay_eval --replay_dataset "C:\Users\Jacob Higgins\projects\kronos-predictor\scanner\replay\sample_replay_dataset.json"
```

## Tradier Options Data (wired 2026-07-02)
Set `TRADIER_API_TOKEN` in `scanner\.env` (a PRODUCTION brokerage token - sandbox tokens are 15-min delayed and stay research-grade). Option selection then uses Tradier's real-time OPRA-consolidated chains first: NBBO bid/ask with sizes, native volume and open interest, ORATS greeks, true quote timestamps.

Behavior:
- Liquidity-gate failures on Tradier data are authoritative (no fallback to lower-grade data to force a pass). Only infrastructure failures (auth/transport) fall back to the legacy Alpaca-indicative + yfinance pipeline, with a warning.
- `options_data_quality` is 0.9 for fresh quotes (execution-grade; clears the audit's 0.75 bar during market hours) and degrades past 30 minutes, so after-hours scans are honestly research-grade. Run scans during market hours for execution-grade evidence.
- Keep at least $2k in the account or make 2 trades/yr to avoid Tradier's $50 inactivity fee; unfunded accounts lose API access after 60 days.

## Remaining Data Upgrade Paths
- `low_feed_confidence` (equities): free Alpaca IEX remains the bar source. Full-SIP options: Alpaca Algo Trader Plus $99/mo (also real-time OPRA options on the existing SDK: `feed="sip"` / `feed="opra"`), or Polygon/Massive Stocks Starter $29/mo for delayed full-market bars.

## Limitations
- Free Alpaca IEX feed is not full SIP coverage; live/dry-run scans see IEX-only bars (~3% of consolidated volume, missing bars on thin names). Evidence paths (index build, research scans, outcome review) use delayed SIP.
- Earnings dates come from Yahoo and are estimates, not confirmations (the gate adds a +1 day cushion and fails closed on unknown/stale dates, but a source error larger than one day can still slip through; a second earnings source would need a paid/keyed API).
- yfinance data quality and intraday history depth are limited; it is a fallback, not the primary.
- Synthetic 24h sessions include extended-hours prints; a barrier touch in thin ETH trade may not have been executable for an options position (options trade ~9:30-16:15 ET). The lab's daily-bar index does not have this exposure.
- Daily 2y backtest is proxy only and does not validate true 24h ETH parity.
- TradingView parity is unverified until calibration CSV comparison is run.
- Options liquidity can change intraday; pass/fail is point-in-time.
- Kronos output is advisory by default; an unknown format is recorded as an evaluation error and blocks alerts only when the experimental `KRONOS_LIVE_GATE_ENABLED=true` gate is enabled.
- Early-stage self-tuning will show `insufficient_samples` until enough resolved outcomes accumulate.
- Tradier greeks/IV are hourly ORATS snapshots (research-grade); chain bid/ask/volume/OI are realtime on the production token.

## Before Enabling Live Alerts
1. Rotate Telegram bot token and update `.env`.
2. Run `--mode run_edge_lab` and verify `scanner\reports\edge_audit_report.json` reports `readiness=paper_trade_only`.
3. Run `--mode dry_run` and verify ticker-by-ticker logs.
4. Run both backtests and inspect `reports\*.json`.
5. Run calibration with TradingView CSV and review mismatch report.
6. Set `LIVE_MODE_ENABLED=true` only after verification.
