# Codebase Concerns

**Analysis Date:** 2026-07-02

## Tech Debt

**`scanner/main.py` god module:**
- Issue: 1,681-line entrypoint mixing CLI parsing, env loading, per-ticker scan pipeline, calibration math, edge-lab orchestration, research-ops sequencing, zero-result diagnostics, and report writing
- Files: `scanner/main.py`
- Impact: Every pipeline change touches the same file; the 2026-07-01 stage-order bug (scan ran before tightened threshold applied) lived here. Hard to unit-test orchestration in isolation — most tests exercise helpers, not `main()` mode dispatch
- Fix approach: Extract `research_ops`/`edge_lab` orchestration into `scanner/ops/`, calibration into `scanner/calibration.py`, diagnostics into `scanner/learning/diagnostics.py`. Keep `main.py` as thin dispatch
- Severity: Medium

**Split-brain config override system (highest-leverage debt):**
- Issue: Tunable values live in three layers — `scanner/config.py` module globals, `scanner/tuning/overrides.json`, and import-time bindings in consumers. Two import styles coexist: `from ..config import MIN_RR` (frozen at import) vs `scanner_config.MIN_RR` (live). `config.reload_overrides()` (`scanner/config.py:145`) mutates module globals, so a mid-process reload only reaches consumers using the live style
- Files: `scanner/config.py:102-150`, `scanner/strategy/empty_space.py:6` (`MIN_EMPTY_SPACE_SCORE`, `MIN_RR` frozen), `scanner/data/options_data.py:9` (`MAX_ATM_BID_ASK_SPREAD_PCT`, `MIN_ATM_OPEN_INTEREST` frozen), `scanner/models/kronos_adapter.py:11` (`MIN_KRONOS_AGREEMENT` frozen), `scanner/strategy/potter_box.py:7-19` (ATR/range/slope thresholds frozen), vs live reads at `scanner/strategy/potter_box.py:280` and `scanner/edge/scoring.py:82`
- Impact: Only 2 of the 10 tunable keys (`RESEARCH_CANDIDATE_MIN_SCORE`, `DOCTRINE_V2_SCORE_BASELINE`) actually take effect after an in-process `reload_overrides()`. The other 8 require a process restart (documented in `scanner/README.md` step 5, but nothing enforces it). This exact class of bug already fired once — the 2026-07-01 ordering fix added the live-read path for research score. Any new tunable added with the frozen import style silently reintroduces the bug
- Fix approach: Standardize all tunable reads on `scanner_config.<KEY>` (or a `get_config()` accessor); add a test asserting no module imports tunable names directly
- Severity: High

**Autotuner clobbers adaptive-policy overrides:**
- Issue: `apply_overrides()` in `scanner/learning/autotuner.py:139-146` writes its 9-key override dict with `OVERRIDES_PATH.write_text(json.dumps(overrides))` — a full replace, no merge. `apply_adaptive_overrides()` in `scanner/learning/adaptive_policy.py:308-328` merges correctly and calls `reload_overrides()`; the autotuner does neither
- Impact: A `DOCTRINE_V2_SCORE_BASELINE` override set by `adaptive_policy --apply_tuning` is silently deleted the next time `autotune --apply_tuning` runs (the autotuner's dict never includes that key). The README-recommended daily cadence runs both. Additionally, the autotuner seeds proposals from import-time constants (`scanner/learning/autotuner.py:62-70`), so within a `research_ops` process the autotune proposal ignores thresholds adaptive policy just tightened
- Fix approach: Merge with existing `overrides.json` in `apply_overrides()`, call `scanner_config.reload_overrides()` after write, and seed baselines from `scanner_config.*` live values
- Severity: High (latent — currently only `RESEARCH_CANDIDATE_MIN_SCORE=72` is set in `scanner/tuning/overrides.json`)

**Override file pins all keys forever after one autotune apply:**
- Issue: `propose_overrides()` returns `status=ok` with ALL 9 keys (even unchanged ones) whenever the hold conditions don't trigger (`scanner/learning/autotuner.py:112-136`); applying writes every key into `overrides.json`
- Files: `scanner/learning/autotuner.py`, `scanner/config.py:114-142`
- Impact: After one apply, future edits to `scanner/config.py` defaults for those keys silently do nothing — the override file wins and nothing reports the shadowing
- Fix approach: Only write keys that differ from current config; have `doctor` (`scanner/doctor.py`) report active overrides vs defaults
- Severity: Medium

**Identity split across three READMEs plus a stale agent-memory file:**
- Issue: `README.md` describes the upstream Kronos research model; `README_JAKE.md` describes a desktop Streamlit forecasting app (claims "Kronos-base, RTX 5060 Ti, CUDA 13.2, PyTorch 2.11+cu128" — repo pins torch 2.7.0); `scanner/README.md` describes the fail-closed trading evidence lab that is the actual center of gravity. `LLM_PROJECT_MEMORY.md` says "Last updated: 2026-05-24" and contains zero mentions of `research_ops`, `adaptive_policy`, or Potter Doctrine v2 — the three most active subsystems — despite declaring itself "the first document future LLM agents should read"
- Files: `README.md`, `README_JAKE.md`, `LLM_PROJECT_MEMORY.md`, `scanner/README.md`, `docs/daily-notes/`
- Impact: Agents and humans bootstrapping from the wrong doc build the wrong mental model; README_JAKE's stale toolchain claims feed the dependency drift below
- Fix approach: Update `LLM_PROJECT_MEMORY.md` to cover doctrine v2 / adaptive policy / research_ops and daily-notes workflow; add a one-paragraph "three products, one repo" map at the top of `README.md`; correct or delete the stale hardware/toolchain claims in `README_JAKE.md`
- Severity: Medium

**Dead code and upstream dead weight:**
- Issue: `HEARTBEAT_ENABLED` is parsed into env (`scanner/main.py:202`, `scanner/.env.example`) and never used anywhere. Upstream fork baggage: `examples/` (3.7MB), `figures/` (2.7MB), `finetune_csv/` (8MB), `finetune/` with placeholder TODO configs (`finetune/config.py:12,40,81,99`), and 29 stale `webui/prediction_results/prediction_2025*.json` files still git-tracked from before the directory was gitignored
- Files: `scanner/main.py:202`, `finetune/config.py`, `webui/prediction_results/`, `examples/`, `finetune_csv/`
- Impact: ~16MB repo bloat; tracked-but-ignored prediction JSONs contradict the `doctor` hygiene check's spirit (it only verifies `git check-ignore` on a sample path, `scanner/doctor.py:18-25`, which passes while 29 real files remain tracked)
- Fix approach: `git rm --cached webui/prediction_results/*.json`; remove `HEARTBEAT_ENABLED` or implement heartbeats; consider moving upstream research assets behind a separate branch or deleting them from the fork
- Severity: Low

**Decision journal append is O(n) per write:**
- Issue: `append_decision()` calls `load_decisions()` (full-file JSON parse) for every single append to dedupe/enrich, and rewrites the whole file on enrichment (`scanner/learning/outcome_store.py:81-97`)
- Impact: A 30-ticker scan performs 30 full journal reads. At the current 443 rows this is invisible; at tens of thousands it becomes quadratic scan overhead
- Fix approach: Load once per scan run and pass a journal handle through `run_watchlist_scan()` (`scanner/main.py:1427`), or maintain an in-memory fingerprint set
- Severity: Low (now), Medium at scale

## Known Bugs

**Pending outcomes older than the intraday window are unresolvable zombies (learning-loop starvation vector):**
- Symptoms: Decision-journal rows stay `outcome_status=pending` forever; `review_outcomes` reports them as reviewed but never resolves or expires them
- Files: `scanner/learning/outcome_reviewer.py:66-76` (`start_pos < 0 → continue`), `scanner/config.py:70` (`OUTCOME_REVIEW_MAX_RECORDS = 500`), `scanner/config.py:44` (`INTRADAY_LOOKBACK = "60d"`)
- Trigger: Any pending decision whose timestamp falls before the oldest bar in the rolling 60-day intraday fetch — `searchsorted(...) - 1` returns `-1` and the record is skipped with no state change. The 2026-07-02 anchoring fix solved after-hours anchoring but not window-expiry
- Impact: These zombies permanently occupy the front of the `pending[:OUTCOME_REVIEW_MAX_RECORDS]` slice (journal is oldest-first), so once >500 zombies accumulate, fresh pending records are never reviewed and sample counts for autotune/adaptive policy silently stop growing — exactly the "learning loop starves" failure mode
- Workaround: None automatic. Manually mark stale rows `not_applicable`
- Fix approach: When a pending record's timestamp predates the fetched window, resolve it via `fetch_daily_bars` or mark it `expired_unresolvable`; add an `unresolvable` counter to `outcome_review_summary.json`
- Severity: High (for the learning loop; currently only 4-6 pendings so not yet biting)

**Journal dedupe is weaker than documented:**
- Symptoms: `scanner/README.md:129` claims "the decision journal rejects repeat observations of the same ticker/setup/day", but the fingerprint includes `entry_price` rounded to 4 decimals (`scanner/learning/outcome_store.py:14-29`)
- Trigger: Re-scanning the same ticker later the same day after price moves produces a different `breakout_close`, hence a new fingerprint and a duplicate same-day record
- Impact: Repeated intraday scans can still inflate sample counts and bias autotune/adaptive cohorts, the exact thing the dedupe exists to prevent (historical "duplicate journal enrichment" bug was fixed for identical rows only)
- Fix approach: Drop `entry_price` from the fingerprint for research/counterfactual rows, or bucket by session date + stage only
- Severity: Medium

**Stale past earnings dates permanently block tickers:**
- Symptoms: A ticker whose yfinance calendar returns an old (past) earnings date is blocked forever: `days_to` is negative and `days_to <= EARNINGS_BLOCK_DAYS` is always true (`scanner/data/events.py:70-79`)
- Files: `scanner/data/events.py`
- Trigger: yfinance `calendar` not updated after an earnings event (common)
- Workaround: None; fails closed (safe direction, wrong reason)
- Fix approach: Treat `days_to < 0` as "unknown/stale" and route through the `BLOCK_ON_UNKNOWN_EARNINGS` policy with a distinct skip reason
- Severity: Low-Medium (suppresses signal collection, invisible in diagnostics as a distinct cause)

**Market-holiday false staleness penalty:**
- Symptoms: `_equity_market_elapsed_minutes()` counts all non-weekend days as trading days (`scanner/main.py:971-989`), so full-day holidays (e.g., July 4) accrue ~390 stale minutes and shave `quality_score` in `_edge_data_quality()` (`scanner/main.py:1038-1041`)
- Trigger: Any scan on/after a market holiday. The 2026-06-28 fix handled weekends only
- Impact: Undeserved `data_quality` scorecard penalty (`scanner/edge/scoring.py:99`); not blocking at one holiday (score ~0.76 vs 0.5 blocker floor) but compounds with missing bars
- Fix approach: Add a static NYSE holiday list to `_equity_market_elapsed_minutes`
- Severity: Low

**R-multiple can explode when risk is degenerate:**
- Symptoms: `r_multiple = ret_pct / max(abs(risk_pct), 0.01)` (`scanner/edge/retrieval.py:116`) — a record where `risk_pct` collapses to ~0 (entry equals invalidation, e.g. `pb.cost_basis or entry` fallback at `scanner/edge/retrieval.py:144` when cost basis is falsy) yields up to 100x-inflated r_multiples
- Impact: `average_r_multiple` in analog summaries (`scanner/edge/scoring.py:54`) and validation blocks (`scanner/edge/validation.py:44`) is an uncapped mean, so a single degenerate record can flip `non_positive_analog_expectancy` or the audit's `average_r > 0` check
- Fix approach: Winsorize/clamp r_multiples at record-build time, or use median in gating math
- Severity: Low-Medium (probabilistic; audit currently blocked for other reasons)

## Structural Evidence-Pipeline Concern

**Historical validation records carry a built-in ~18-point score handicap, which helps explain why threshold-55 evidence never accumulates:**
- Problem: Index records built by `build_edge_records_from_bars()` (`scanner/edge/retrieval.py:120-165`) never run Kronos or fetch options, but `extract_edge_features()` fills those features with concrete zeros rather than omitting them (`scanner/edge/features.py:186-190`). Downstream, `score_edge_candidate()` reads `kronos_directional_agreement=0.0` (key present, so the neutral 0.5 default at `scanner/edge/scoring.py:70` never applies) → kronos component −10; `options_data_quality=0.0` → −8 (`scanner/edge/scoring.py:101`). Every historical validation candidate therefore starts ~18 points below an equivalent live candidate
- Evidence: Daily notes are monotonic on this — threshold 55 has 0 signals in every run since 2026-06-24; threshold 45 has exactly 1 signal out of 600; the audit blocker `validation_threshold_55_unsupported` (`scanner/edge/audit.py:90-91`, needs 20 signals) has never moved. The gate may be measuring the feature-default artifact as much as edge quality
- Files: `scanner/edge/features.py`, `scanner/edge/scoring.py`, `scanner/edge/retrieval.py`, `scanner/edge/audit.py:35-43`, `scanner/edge/validation.py`
- Secondary effect: the analog distance metric compares live feature vectors (real options/kronos/data-quality values) against historical vectors of zeros for the same keys (`scanner/edge/retrieval.py:243-265`), adding systematic noise to nearest-neighbor retrieval
- Fix approach: Omit never-measured features from historical records (the index/distance code already handles missing keys via NaN masking), or make scoring treat "absent" differently from "measured zero"; then re-baseline the 55-threshold audit target against achievable historical score ranges
- Severity: High — if uncorrected, the readiness audit can stay `blocked` indefinitely regardless of how much data accumulates, while appearing to be a pure data-volume problem

## Security Considerations

**Secrets handling (currently sound, one legacy caveat):**
- Risk: `scanner/.env` holds Telegram bot token, Alpaca keys, MiniMax key. `scanner/README.md:199` instructs rotating the Telegram token before live use, implying past exposure
- Files: `scanner/.env` (present, gitignored — existence only, contents not read), `scanner/main.py:185-209`, `scanner/doctor.py:18-25`
- Current mitigation: `.gitignore` covers `.env`; `doctor` verifies ignore status; no secrets found hardcoded in code
- Recommendations: Complete the token rotation called out in the README before any live mode; consider adding `request_ids.log`/`scanner.log` scrub checks since Alpaca URLs are logged (headers/keys are not logged — verified in `scanner/data/market_data.py:69-94`)

**Live-mode gate trusts an editable report file:**
- Risk: `_preflight_checks()` enables live mode if `scanner/reports/edge_audit_report.json` says `readiness=paper_trade_only` (`scanner/main.py:405-423`). The file is plain JSON; a hand-edited or stale file passes the gate
- Current mitigation: Also requires `LIVE_MODE_ENABLED=true` + Telegram creds; single-user local machine threat model
- Recommendations: Embed a freshness timestamp + git commit in the audit payload and reject audits older than N hours in preflight
- Severity: Low (local-only), worth closing before real capital

**Web UI:**
- Risk: Flask app with file-load endpoint
- Files: `webui/app.py:17-63`, `tests/test_webui_security.py`
- Current mitigation: Defaults to `127.0.0.1:7070`, debug off, CORS restricted, `DATA_DIR` path-traversal rejection with tests
- Recommendations: None urgent; keep host default local

## Performance Bottlenecks

**`research_ops` takes 4-6 minutes, dominated by full edge-index rebuild:**
- Problem: Every `run_edge_lab` rebuilds all ~5,686 index records from scratch — 2y of daily bars x 30 tickers, running `detect_potter_box`/`score_empty_space`/`score_potter_doctrine_v2` per bar per ticker (`scanner/edge/retrieval.py:120-165`, called from `scanner/main.py:1198-1232`)
- Files: `scanner/main.py:1389-1424`, `scanner/edge/retrieval.py`
- Cause: No incremental indexing; the whole 2-year window is recomputed even though only ~1 new session per ticker per day exists
- Improvement path: Cache records keyed by (ticker, session date, feature_version); append new sessions only; rebuild fully only when `FEATURE_VERSION` (`scanner/edge/features.py:11`) changes
- Severity: Medium (runs daily; 250-345s per `docs/daily-notes/2026-07-02.md`)

**Serial per-ticker yfinance calls in the scan path:**
- Problem: `validate_ticker` hits `yf.Ticker(t).info` + options list per ticker (`scanner/data/market_data.py:196-206`), `assess_event_risk` hits `.info` + `.calendar` again (`scanner/data/events.py:45-47`), `select_options_contract` walks option chains (`scanner/data/options_data.py:97-120`) — all sequential, no shared session or cache within a run
- Cause: Yahoo endpoints are slow and rate-limited; duplicate `.info` fetches per ticker per stage
- Improvement path: Share one `yf.Ticker` object per ticker per run; cache `.info` for the scan duration
- Severity: Medium

**Outcome review re-fetches bars per pending record:**
- Problem: `review_pending_outcomes` calls `fetch_intraday_bars(rec["ticker"])` inside the loop (`scanner/learning/outcome_reviewer.py:66`) — N pending records for one ticker = N identical fetches
- Improvement path: Memoize bars per (ticker, anchor) within the review call
- Severity: Low-Medium (grows with pending count)

**Unbounded `request_ids.log`:**
- Problem: `_persist_request_id` appends forever with no rotation (`scanner/data/market_data.py:69-79`); already 691KB while `scanner.log` rotates at 1.5MB x3 (`scanner/utils/logging_setup.py`)
- Improvement path: Route through the rotating logger or truncate on startup
- Severity: Low

## Fragile Areas

**Windows launcher/bootstrap chain (circular and drifting):**
- Files: `launch_kronos.bat`, `install_deps.bat`, `scanner/setup_dependencies.bat`, `scanner/run_scanner.bat`
- Why fragile: No script creates the venv — `launch_kronos.bat` says "run install_deps.bat first" but `install_deps.bat` requires `venv\Scripts\pip.exe` to already exist; `scanner/setup_dependencies.bat` likewise errors if venv is missing. `launch_kronos.bat` hardcodes `C:\Users\Jacob Higgins\projects\kronos-predictor` (breaks on any move/rename); `install_deps.bat` silently pip-installs unpinned packages mid-launch if streamlit is missing
- Torch venv ping-pong: the SAME venv serves the GPU desktop app and the scanner. `requirements.txt` pins `torch==2.7.0+cpu` (CPU wheel index) — running `scanner/setup_dependencies.bat` replaces CUDA torch with CPU torch and breaks GPU inference; `install_deps.bat` then installs unpinned latest CUDA torch, violating the `pyproject.toml` pin `torch==2.7.0`; `webui/requirements.txt` says `torch>=2.1.0`; `README_JAKE.md` claims 2.11+cu128. Four conflicting torch specs, one venv
- Safe modification: Pick one torch spec as truth; either split venvs (scanner has no torch need beyond Kronos adapter) or gate the CPU pin out of the shared file; add a `python -m venv venv` bootstrap step to one script
- Test coverage: None (batch files untestable as-is)
- Severity: Medium-High for environment drift (this is the documented historical "environment/worktree drift" pain)

**Silent exception swallowing in config/data plumbing:**
- Files and behaviors:
  - `scanner/config.py:114-121` — corrupt/unreadable `overrides.json` is silently ignored; the process runs on defaults while the operator believes tuning is active
  - `scanner/data/market_data.py:271-273, 310-312` — in `auto` provider mode, ANY Alpaca failure (auth, 403 on SIP, network) falls through to yfinance with no log line; scans silently degrade to lower-confidence data (provenance attrs record it, but nothing warns)
  - `scanner/main.py:117-135` — `_resolve_calibrated_anchor` swallows all errors and silently returns the default 20:00 anchor; a corrupt `calibration_summary.json` de-calibrates every scan invisibly
  - `scanner/learning/outcome_store.py:104-113` — corrupt journal lines are skipped without counting; partial file corruption shrinks the training set silently
  - `scanner/main.py:1346-1359, 1370-1378` — unreadable validation/scan reports become `{}`, and the audit then reports `blocked` for a data-shaped reason rather than an I/O reason
- Why fragile: All of these fail toward "keep running with quietly different behavior," which is the hardest failure class to notice in a system whose whole job is evidence integrity
- Safe modification: Add one WARN log per swallow site with the file path and exception; add counters (`corrupt_lines`, `alpaca_fallbacks`) to run summaries
- Severity: Medium

**yfinance as a load-bearing dependency for gates:**
- Files: `scanner/data/market_data.py` (validation, fallback bars), `scanner/data/events.py` (earnings/dividends), `scanner/data/options_data.py` (chains + open interest — the ONLY OI source, `open_interest_source="yfinance"` hardcoded at `scanner/data/options_data.py:179`)
- Why fragile: Yahoo breaks its unofficial API regularly; `.info`/`.calendar` schema shifts have historically nuked fields. Because events and options gates fail closed, a yfinance outage manifests as "all tickers skip" rather than an explicit provider-down error
- Safe modification: Pin `yfinance` to a tested version per release; add a provider-health preflight that distinguishes "data provider down" from "no setups"
- Test coverage: `scanner/tests/test_options.py` covers selection logic with fixtures; no tests simulate provider failure modes in `validate_ticker`/`assess_event_risk`
- Severity: Medium

**Kronos gate degrades to fleet-wide skip when the model can't load:**
- Files: `scanner/models/kronos_adapter.py:21-63`, `scanner/config.py:53-56`
- Why fragile: First use downloads `NeoQuasar/Kronos-small` from HuggingFace; if offline or HF is down, `evaluate()` fails closed per ticker and every candidate dies at the kronos stage. Correct direction, but a full-scan zero can look like "no setups" instead of "model unavailable"
- Safe modification: Surface a distinct scan-summary flag when >N kronos-stage failures share the same load error
- Severity: Low-Medium

**Branch/worktree drift:**
- Files: repo root; branch `codex/finish-kronos-cleanup` is 1 commit (`0b22b79` — adaptive policy, doctrine v2, anchoring fix) ahead of `master`
- Why fragile: The entire current learning stack exists only on the working branch; `master` still lacks doctrine v2/adaptive policy. Historical notes cite environment/worktree drift as a recurring failure source (`.gitignore` even has `worktrees/` entries from past cleanup)
- Safe modification: Merge the checkpoint to `master` after review; keep daily-note commits on the same branch as the code they describe
- Severity: Low-Medium (process risk)

## Scaling Limits

**Decision journal (`scanner/reports/scan_decisions.jsonl`):**
- Current capacity: 443 rows; full read per append (`scanner/learning/outcome_store.py:86`), full rewrite per enrichment/review
- Limit: O(n^2) behavior noticeable around ~10k rows; `OUTCOME_REVIEW_MAX_RECORDS=500` interacts badly with unresolvable zombies (see Known Bugs) well before raw I/O hurts
- Scaling path: SQLite or per-day JSONL sharding; add pending-expiry first

**Edge retrieval index (`scanner/reports/edge_retrieval_index.json`):**
- Current capacity: 5,686 records, single pretty-printed JSON file rewritten every lab run (`scanner/edge/retrieval.py:316-320`); loaded fully into memory and vectorized per scan (`EdgeAnalogIndex`, `scanner/edge/retrieval.py:206-313`)
- Limit: Fine to ~50k records; JSON parse + full matrix build per run grows linearly; `EdgeRecord(**row)` strict constructor (`scanner/edge/retrieval.py:330`) breaks the whole load on any schema drift in the saved file
- Scaling path: Parquet sidecar as primary (machinery exists in `scanner/evidence/store.py:121-127`), tolerant record parsing with per-row skip counting

**Watchlist:**
- Current capacity: 30 hardcoded tickers (`scanner/config.py:92-99`, re-exported by `scanner/tickers.py`)
- Limit: Serial scans mean runtime scales linearly (~8-12s/ticker in research_ops); no CLI/env way to change the watchlist without editing code
- Scaling path: Watchlist file + `--watchlist` flag; per-ticker concurrency with provider rate-limit budget

## Dependencies at Risk

**`yfinance` (unpinned floor `>=0.2.54` in `scanner/requirements-scanner.txt` and `pyproject.toml`):**
- Risk: Frequent breaking changes against unofficial Yahoo endpoints; supplies price fallback, earnings calendar, option chains, and the only open-interest source
- Impact: Scans fail closed fleet-wide; events gate blocks everything on `.calendar` schema changes
- Migration plan: Pin per-release; long-term replace OI/chains with a real options data subscription (also the top readiness blocker)

**Alpaca free tier (IEX bars + `indicative` options feed):**
- Risk: Not a code bug but a structural cap — `_options_quality()` caps indicative at 0.6 (`scanner/data/options_data.py:79-85`) vs the 0.75 promote floor (`scanner/edge/scoring.py:115`), so `promote` is unreachable by design until OPRA-grade data exists. Audit warnings `low_feed_confidence`/`options_liquidity_missing`/`options_data_not_execution_grade` are permanent fixtures of the current data tier
- Migration plan: Budget for Alpaca options subscription or another OPRA source before expecting `readiness=paper_trade_only`

**Torch (three conflicting specs):**
- Risk: `requirements.txt` (`2.7.0+cpu`), `pyproject.toml` (`2.7.0`), `install_deps.bat` (unpinned CUDA), `webui/requirements.txt` (`>=2.1.0`) — see Fragile Areas
- Migration plan: Single source of truth; separate GPU-app extras from scanner deps

**MiniMax API (`MiniMax-M2.7-highspeed` hardcoded default, `scanner/config.py:59`):**
- Risk: Model-name churn breaks `test_minimax`/scoring; adapter fails soft (alert still sends with error insight, `scanner/main.py:657-658`), so degradation is quiet
- Migration plan: None urgent; disabled by default

**`flask==2.3.3` / unpinned `streamlit`:**
- Risk: Older Flask line (EOL trajectory); streamlit floats freely in a venv that other scripts mutate
- Migration plan: Bump/pin during next webui touch; low priority (local-only)

## Missing Critical Features

**Execution-grade options truth data:**
- Problem: Everything downstream of the audit is gated on options data quality that the current free stack cannot provide; `docs/daily-notes/2026-07-02.md` names this the highest-value blocker
- Blocks: `promote` recommendations, `readiness=paper_trade_only`, live alerts

**Pending-outcome expiry / unresolvable accounting:**
- Problem: No mechanism marks pendings that can never resolve (see Known Bugs); no metric separates "waiting" from "stuck"
- Blocks: Trustworthy sample-count growth for autotune (`AUTOTUNE_MIN_SAMPLES=20`) and adaptive policy (min 8 resolved)

**Scheduler/automation for the documented cadence:**
- Problem: `scanner/README.md:169-174` prescribes an hourly/daily cadence, but nothing schedules it — the loop only learns when someone remembers to run `research_ops`
- Blocks: Steady evidence accumulation; a missed week silently stalls the whole learning flywheel

**Market holiday calendar:**
- Problem: Staleness math and outcome horizons treat every weekday as a session
- Blocks: Accurate data-quality scoring around holidays

## Test Coverage Gaps

**Override merge/clobber semantics:**
- What's not tested: `autotuner.apply_overrides` preserving keys written by `apply_adaptive_overrides`; `reload_overrides()` propagation to import-bound consumers
- Files: `scanner/learning/autotuner.py:139-146`, `scanner/config.py:102-150` (tests exist for adaptive policy logic in `scanner/tests/test_adaptive_policy.py` but not cross-writer file semantics)
- Risk: The documented daily cadence silently reverts adaptive tightening
- Priority: High

**Unresolvable-pending expiry:**
- What's not tested: Behavior when a pending decision predates the intraday window (the `start_pos < 0` path in `scanner/learning/outcome_reviewer.py:71-73`); `scanner/tests/test_outcome_reviewer.py` covers after-hours anchoring only
- Risk: Zombie accumulation goes unnoticed until the review cap starves fresh records
- Priority: High

**Provider failure modes:**
- What's not tested: Alpaca-down silent fallback to yfinance (`scanner/data/market_data.py:271-273`), yfinance `.info`/`.calendar` schema breakage in `validate_ticker`/`assess_event_risk`, corrupt `overrides.json`/`calibration_summary.json` swallow paths
- Files: `scanner/data/market_data.py`, `scanner/data/events.py`, `scanner/config.py`, `scanner/main.py:117-135`
- Risk: Silent data-tier degradation in the exact system that certifies data quality
- Priority: Medium

**Historical-vs-live feature parity in edge scoring:**
- What's not tested: That validation records without kronos/options measurements are not structurally penalized versus live candidates (the ~18-point handicap above); no test pins the maximum achievable historical validation score against the 55-threshold audit gate
- Files: `scanner/edge/features.py`, `scanner/edge/scoring.py`, `scanner/edge/validation.py`, `scanner/edge/audit.py`
- Risk: The audit stays `blocked` forever for artifact reasons while everyone waits for "more data"
- Priority: High

**`main.py` mode dispatch and preflight:**
- What's not tested: `_preflight_checks` live-gating (audit-file trust, readiness parsing), `research_ops` stage ordering end-to-end (the 07-01 regression class); `scanner/tests/test_research_ops.py` and `scanner/tests/test_edge_cli_units.py` cover pieces, not the dispatch/order contract
- Files: `scanner/main.py:390-427, 1502-1559`
- Risk: Orchestration regressions (already happened once) reach the journal before detection
- Priority: Medium

---

*Concerns audit: 2026-07-02*
