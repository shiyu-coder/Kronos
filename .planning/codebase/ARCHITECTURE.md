<!-- refreshed: 2026-07-02 -->
# Architecture

**Analysis Date:** 2026-07-02

## System Overview

This repo is a fork of the open-source Kronos financial foundation model with a locally-grown options scanner as the center of gravity. Two loosely-coupled layers:

1. **Kronos model layer** (`model/`, `webui/`, `finetune/`, `kronos_app.py`) — upstream transformer forecaster, kept mostly as-is.
2. **Potter Box scanner** (`scanner/`) — the active product: a fail-closed staged gate pipeline plus a self-tuning learning loop and an edge evidence lab. `scanner/main.py` is the single CLI orchestrator.

```text
┌───────────────────────────────────────────────────────────────────────┐
│                       CLI Entry: python -m scanner.main               │
│              `scanner/main.py:1596-1677` (mode dispatch)              │
├────────────────────┬─────────────────────┬────────────────────────────┤
│  Scan Pipeline     │  Edge Evidence Lab  │  Learning / Self-Tuning    │
│  `_run_single_     │  `run_edge_lab`     │  `scanner/learning/`       │
│   ticker`          │  `main.py:1389`     │  outcome_store/reviewer,   │
│  `main.py:462-712` │  index→validate→    │  adaptive_policy, autotuner│
│                    │  scan→diagnose→audit│                            │
└─────────┬──────────┴──────────┬──────────┴─────────────┬──────────────┘
          │                     │                        │
          ▼                     ▼                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Domain layer: `scanner/strategy/` (potter_box, empty_space,          │
│  potter_doctrine), `scanner/edge/` (features, retrieval, scoring,     │
│  validation, audit), `scanner/models/kronos_adapter.py`               │
└─────────┬─────────────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Data layer: `scanner/data/` (market_data via Alpaca/yfinance,        │
│  synthetic_sessions, options_data, events)                            │
└─────────┬─────────────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Persistence (files, no DB): `scanner/reports/*.json` reports,        │
│  `scanner/reports/scan_decisions.jsonl` journal,                      │
│  `scanner/reports/evidence/<run_id>/` evidence runs,                  │
│  `scanner/tuning/overrides.json` learned thresholds                   │
└───────────────────────────────────────────────────────────────────────┘
```

The Kronos model itself participates in the scanner only as one gate: `scanner/models/kronos_adapter.py` lazy-loads `model/kronos.py` (`KronosPredictor`) and converts N sampled forecast paths into a directional-agreement score.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI orchestrator | Arg parsing, env load, preflight, mode dispatch, staged pipeline | `scanner/main.py` |
| Runtime config + learned overrides | All thresholds; merges `tuning/overrides.json` at import | `scanner/config.py` |
| Potter Box detector | Consolidation box + breakout/breakdown detection; research-grade near-miss scoring | `scanner/strategy/potter_box.py` |
| Potter Doctrine v2 | Research-only scoring of punchback/cost-basis/box-stack mechanics | `scanner/strategy/potter_doctrine.py` |
| Empty Space gate | Nearest target, R/R ratio, empty-space score | `scanner/strategy/empty_space.py` |
| R/R math | `compute_rr(entry, target, invalidation, direction)` | `scanner/strategy/risk_reward.py` |
| Market data | Alpaca-first (yfinance fallback) intraday/daily bars, ticker validation, provenance attrs | `scanner/data/market_data.py` |
| Synthetic sessions | Groups 30m bars into anchor-hour "synthetic daily" sessions | `scanner/data/synthetic_sessions.py` |
| Options selection | ATM contract pick; Alpaca snapshots joined with yfinance OI; data-quality grading | `scanner/data/options_data.py:88` |
| Event risk | Earnings/ex-dividend proximity gate | `scanner/data/events.py:43` |
| Kronos gate | Lazy-loads Kronos, samples N paths, directional agreement | `scanner/models/kronos_adapter.py` |
| MiniMax AI advisory | Optional non-gating LLM second opinion on final candidates | `scanner/ai/minimax_adapter.py` |
| Telegram alerts | Message rendering + send with retry | `scanner/alerts/telegram.py` |
| Edge features | Stable JSON-safe feature vector (FEATURE_VERSION=2) | `scanner/edge/features.py` |
| Edge retrieval | `EdgeRecord` history + vectorized k-NN analog search with embargo | `scanner/edge/retrieval.py` |
| Edge scoring | Transparent scorecard → promote/research/reject | `scanner/edge/scoring.py` |
| Edge validation | Purged walk-forward precision/recall per threshold | `scanner/edge/validation.py` |
| Edge audit | Readiness verdict: blocked / watch_only / research_only / paper_trade_only | `scanner/edge/audit.py` |
| Decision journal | Append/dedupe/enrich JSONL decision records | `scanner/learning/outcome_store.py` |
| Outcome reviewer | Resolves pending decisions to win/loss after 3+ days | `scanner/learning/outcome_reviewer.py` |
| Adaptive policy | Wilson-lower-bound threshold search; safe auto-apply | `scanner/learning/adaptive_policy.py` |
| Autotuner | Bounded step proposals for live gate thresholds | `scanner/learning/autotuner.py` |
| Replay eval | Confusion matrix over labeled replay datasets | `scanner/learning/replay_runner.py` |
| Backtests | Intraday-60d and daily-proxy-2y simulations | `scanner/backtest/backtest_runner.py` |
| Evidence store | Immutable run directories: JSONL + parquet + manifest | `scanner/evidence/store.py` |
| Env doctor | Dependency/artifact health checks | `scanner/doctor.py` |
| Shared result dataclasses | `PotterBoxResult`, `EmptySpaceResult`, `KronosResult`, etc. | `scanner/utils/validation.py` |
| Kronos model | Tokenizer, transformer, `KronosPredictor.predict` | `model/kronos.py`, `model/module.py` |
| Desktop forecaster | Streamlit one-click forecast app | `kronos_app.py` |
| Legacy web UI | Flask forecast UI (upstream) | `webui/app.py` |

## Pattern Overview

**Overall:** Mode-dispatched CLI monolith with a staged gate pipeline (chain of responsibility) and a file-based evidence/learning feedback loop.

**Key Characteristics:**
- **Fail-closed gates:** every stage returns a result object with `passed` + `skip_reason`; first failure short-circuits the ticker and journals the decision. Weak evidence is rejected by default.
- **Counterfactual journaling:** rejected-but-interesting candidates are still journaled with `counterfactual: True` and a pending outcome, so the learning loop can later measure what the gates cost (`scanner/main.py:532-561`).
- **Everything is a JSON report:** every mode writes a report file under `scanner/reports/`; downstream modes consume prior reports rather than in-memory state (e.g. `audit_edge` reads `edge_validation_report.json` + `edge_scan_report.json`).
- **Bounded self-tuning:** learned overrides live in `scanner/tuning/overrides.json`, are clamped to `*_BOUNDS` constants in `scanner/config.py:81-90`, and only "tighten-or-hold" changes auto-apply.

## Layers

**Orchestration (`scanner/main.py`):**
- Purpose: CLI parsing, env/preflight, timed stage running, mode dispatch, the per-ticker gate pipeline, and all edge-lab mode runners.
- Depends on: every scanner subpackage.
- Used by: `scanner/run_scanner.bat`, `python -m scanner.main`.

**Strategy (`scanner/strategy/`):**
- Purpose: pure-ish setup detection and scoring over bar DataFrames.
- Contains: `potter_box.py` (gate + research scoring), `empty_space.py`, `potter_doctrine.py` (dict-returning research scorer), `risk_reward.py`.
- Depends on: `scanner/config.py`, `scanner/utils/validation.py`.
- Used by: main pipeline, edge retrieval/backtests/replay (re-run detection over historical windows).

**Edge (`scanner/edge/`):**
- Purpose: evidence engine — turn history into `EdgeRecord`s, find analogs, score candidates, validate walk-forward, audit readiness.
- Depends on: strategy layer (rebuilds setups per historical window), `scanner/edge/features.py`.
- Used by: `edge_scan`, `validate_edge`, `run_edge_lab`, `research_ops` modes.

**Learning (`scanner/learning/`):**
- Purpose: the closed loop — journal decisions, resolve outcomes, propose/apply threshold changes.
- Depends on: `scanner/data/` (re-fetch bars to resolve outcomes), `scanner/config.py` (bounds + `reload_overrides()`).
- Used by: `review_outcomes`, `autotune`, `adaptive_policy`, `research_ops` modes; `append_decision` is called from the pipeline itself.

**Data (`scanner/data/`):**
- Purpose: all external market data I/O. Provider choice via `MARKET_DATA_PROVIDER` env (auto → Alpaca if credentialed, else yfinance). Provenance travels on `DataFrame.attrs` (`data_provider`, `data_feed`, `data_delay_minutes`).
- Used by: everything above it.

**Model adapters (`scanner/models/`, `scanner/ai/`):**
- Purpose: wrap heavyweight/remote models behind result objects. `KronosAdapter._load_once` (`scanner/models/kronos_adapter.py:21-41`) is a lazy per-process singleton; `MiniMaxAdapter` degrades to a `skipped` payload when disabled.

**Persistence (files only):**
- `scanner/reports/` (JSON reports + `scan_decisions.jsonl` journal), `scanner/reports/evidence/<run_id>/` (immutable evidence runs), `scanner/tuning/overrides.json` (learned thresholds), `scanner/logs/` (rotating logs). No database anywhere.

## Data Flow

### Full Mode List (dispatch in `scanner/main.py:1596-1677`, choices at `scanner/main.py:141-164`)

| Mode | Handler | Notes |
|------|---------|-------|
| `dry_run` (default) | `run_watchlist_scan` (`main.py:1674`, fn at `main.py:1427`) | Full gate pipeline, alert preview only |
| `live` | `run_watchlist_scan` (`main.py:1674`) | Requires preflight: Telegram creds + `LIVE_MODE_ENABLED=true` + edge audit `readiness=paper_trade_only` (`main.py:399-423`) |
| `research_scan` | `run_watchlist_scan` (`main.py:1674`) | Branches inside `_run_single_ticker` at `main.py:506-530`; grades near-miss candidates instead of gating |
| `backtest_intraday_60d` | `run_intraday_60d_backtest` (`main.py:1603` → `scanner/backtest/backtest_runner.py:74`) | |
| `backtest_daily_proxy_2y` | `run_daily_proxy_2y_backtest` (`main.py:1607` → `backtest_runner.py:102`) | |
| `calibration` | `run_calibration` / `run_batch_calibration` (`main.py:1611-1616`, fns at `main.py:843`, `main.py:912`) | TradingView CSV mismatch check; `--sweep_anchors` tests anchors 16:00–22:00 |
| `test_telegram` | `run_telegram_test` (`main.py:1618`, fn at `main.py:430`) | |
| `test_minimax` | `run_minimax_test` (`main.py:1620`, fn at `main.py:445`) | |
| `review_outcomes` | `review_pending_outcomes` (`main.py:1622-1626` → `scanner/learning/outcome_reviewer.py:44`) | |
| `autotune` | `propose_overrides` (+`apply_overrides` with `--apply_tuning`) (`main.py:1627-1634` → `scanner/learning/autotuner.py:37,139`) | |
| `adaptive_policy` | `run_adaptive_policy` (`main.py:1635`, fn at `main.py:1470` → `scanner/learning/adaptive_policy.py:179,308`) | |
| `replay_eval` | `run_replay_eval` (`main.py:1638` → `scanner/learning/replay_runner.py:13`) | Requires `--replay_dataset` |
| `diagnose_zero_results` | `_write_zero_result_diagnostic` (`main.py:1644`, fn at `main.py:254`) | Journal bottleneck analysis |
| `build_retrieval_index` | `run_build_retrieval_index` (`main.py:1647`, fn at `main.py:1198`) | Daily bars → `EdgeRecord`s → `edge_retrieval_index.json` |
| `validate_edge` | `run_validate_edge` (`main.py:1650`, fn at `main.py:1235`) | Purged walk-forward, `allow_future=False` |
| `edge_scan` | `run_edge_scan` (`main.py:1653`, fn at `main.py:1285`) | Live watchlist vs analog index |
| `diagnose_edge` | `run_diagnose_edge` (`main.py:1656`, fn at `main.py:1346`) | Reads validation + scan reports |
| `audit_edge` | `run_audit_edge` (`main.py:1659`, fn at `main.py:1370` → `scanner/edge/audit.py:35`) | Produces the readiness verdict live mode depends on |
| `run_edge_lab` | `run_edge_lab` (`main.py:1662`, fn at `main.py:1389`) | Composite: index → validate → scan → diagnose → audit under one `EvidenceRun` |
| `research_ops` | `run_research_ops` (`main.py:1665`, fn at `main.py:1502`) | Master daily loop (see below) |
| `doctor` | `run_doctor` (`main.py:1668` → `scanner/doctor.py:56`) | Exit code reflects health |

### Primary Request Path: staged gate pipeline (`_run_single_ticker`, `scanner/main.py:462-712`)

Each stage appends a decision record (`append_decision`, `scanner/learning/outcome_store.py:81`) whether it passes or fails; failures record `stage_failed` and often `counterfactual: True` with a pending outcome.

1. **Validate ticker** — `validate_ticker` (`main.py:469` → `scanner/data/market_data.py:166`): active, price ≥ $5, has listed options. Fail → `stage_failed="validation"`.
2. **Fetch bars + synthetic sessions** — calibrated anchor from `calibration_summary.json` via `_resolve_calibrated_anchor` (`main.py:483`, fn at `main.py:117`); `fetch_intraday_bars` (`main.py:484` → `market_data.py:243`); `build_synthetic_sessions` (`main.py:485` → `scanner/data/synthetic_sessions.py:16`). Fail → `stage_failed="market_data"`.
3. **Potter Box detect** — `detect_potter_box` (`main.py:505` → `scanner/strategy/potter_box.py:42`). In `research_scan` mode, branch to `score_potter_research_candidate` + doctrine v2 and return (`main.py:506-530`). On gate failure, still compute research score + doctrine and journal as counterfactual (`main.py:532-561`), `stage_failed="potter_box"`.
4. **Empty Space score** — `score_empty_space` (`main.py:563` → `scanner/strategy/empty_space.py:11`); doctrine v2 scored alongside (`main.py:564` → `scanner/strategy/potter_doctrine.py:98`, recorded on every downstream record via `_doctrine_record_fields`, `main.py:225`). Fail → `stage_failed="empty_space"`.
5. **Event risk** — `assess_event_risk` (`main.py:582` → `scanner/data/events.py:43`): earnings within 10 days or unknown blocks. Fail → `stage_failed="event_risk"`.
6. **Options contract** — `select_options_contract` (`main.py:600` → `scanner/data/options_data.py:88`): ATM contract, spread/OI liquidity gates. Fail → `stage_failed="options"`.
7. **Kronos confirm** — `kronos.evaluate` (`main.py:618` → `scanner/models/kronos_adapter.py:50`): 10 sampled 5-day paths; directional agreement ≥ `MIN_KRONOS_AGREEMENT` (0.65). Fail → `stage_failed="kronos"`.
8. **MiniMax advisory (non-gating)** — `minimax.score_setup` (`main.py:636-656`): errors only warn.
9. **Alert/decision** — build `AlertCandidate` (`main.py:660`), journal `final_pass: True` (`main.py:672`), render message (`main.py:688` → `scanner/alerts/telegram.py:26`). `dry_run` logs preview (`main.py:690-698`); `live` re-checks `LIVE_MODE_ENABLED` + Telegram creds and sends (`main.py:700-712` → `telegram.py:80`).

### Learning Loop (self-tuning)

1. Pipeline writes decision records → `scanner/reports/scan_decisions.jsonl` (`scanner/learning/outcome_store.py:11`). Dedup by fingerprint of ticker/mode/direction/entry/stage/day (`outcome_store.py:14-29`); re-appends merge-enrich instead of duplicating (`outcome_store.py:81-97`).
2. `review_outcomes` resolves records older than `OUTCOME_MIN_AGE_DAYS` (3): re-fetches bars, rebuilds synthetic sessions with the record's anchor, computes 5-bar forward return, labels `win`/`loss` (`scanner/learning/outcome_reviewer.py:44-103`).
3. `adaptive_policy` grids `RESEARCH_CANDIDATE_MIN_SCORE` and `DOCTRINE_V2_SCORE_BASELINE` over resolved research candidates using Wilson lower-bound win rates; only tighten-or-equal proposals are `auto_apply_safe` (`scanner/learning/adaptive_policy.py:179-305`); apply writes `scanner/tuning/overrides.json` and hot-reloads config (`adaptive_policy.py:308-328` → `scanner/config.py:145`).
4. `autotune` proposes bounded step changes to live gate thresholds from missed-winner/false-positive stage counts (`scanner/learning/autotuner.py:37-136`); returns `hold_no_edge` when evidence doesn't justify loosening. Apply is manual (`--apply_tuning`).
5. `scanner/config.py:102-150` applies `tuning/overrides.json` at import time, so every subsequent run uses learned thresholds.

### Edge Evidence Lab (`run_edge_lab`, `scanner/main.py:1389-1424`)

Runs five stages under a single `EvidenceRun` (`scanner/evidence/store.py:36`, started at `main.py:1390` with git commit tag from `_git_commit`, `main.py:1164`):

1. `run_build_retrieval_index` (`main.py:1198`): for each watchlist ticker, fetch 2y daily bars and slide a window calling `detect_potter_box`/`score_empty_space`/`score_potter_doctrine_v2`/`extract_edge_features` per bar, labeling forward 5-bar outcomes → `EdgeRecord` list (`scanner/edge/retrieval.py:120-165`) → `scanner/reports/edge_retrieval_index.json`.
2. `run_validate_edge` (`main.py:1235`): rescore the last 600 index records against past-only analogs (`allow_future=False`, 5-day same-ticker embargo) and compute precision/recall/R-multiple per edge-score threshold (45/55/65) → `edge_validation_report.json` (`scanner/edge/validation.py:50`).
3. `run_edge_scan` (`main.py:1285`): score today's watchlist via `_score_edge_for_bars` (`main.py:1054`) — features + `find_analogs` (`scanner/edge/retrieval.py:168`, vectorized by `EdgeAnalogIndex`, `retrieval.py:206`) + `score_edge_candidate` (`scanner/edge/scoring.py:61`; promote ≥65 with positive analog expectancy and execution-grade options data, research ≥45, else reject) → `edge_scan_report.json`.
4. `run_diagnose_edge` (`main.py:1346`): summarizes index/validation/scan into a diagnosis string → `edge_diagnostic_report.json`.
5. `run_audit_edge` (`main.py:1370` → `scanner/edge/audit.py:35`): checks purged walk-forward, no future analogs, threshold-55 evidence (≥20 signals, precision ≥0.55, avg R > 0) → readiness `blocked`/`watch_only`/`research_only`/`paper_trade_only` → `edge_audit_report.json`. **This file is the live-mode preflight gate** (`main.py:405-423`).

Finally `evidence_run.flush()` writes JSONL+parquet rows, copied report artifacts, and `manifest.json` to `scanner/reports/evidence/<run_id>/`.

### Research Ops (master loop, `run_research_ops`, `scanner/main.py:1502-1559`)

Timed stages via `_run_timed_stage` (`main.py:98`):
1. `journal_integrity` — dedupe journal, back up if duplicates removed (`main.py:1512-1522`).
2. `outcome_review` — resolve pending outcomes (`main.py:1524-1529`).
3. `adaptive_policy` — with `apply_tuning=True` (auto-applies safe overrides) (`main.py:1530`).
4. `research_scan` — full watchlist in research mode (`main.py:1531`).
5. `diagnostic` — zero-result bottleneck analysis (`main.py:1532`).
6. `autotune` — propose only, never auto-applied here (`main.py:1533`).
7. `edge_lab` — full five-stage lab (`main.py:1534`).
8. `next_actions` — derived from audit readiness + autotune status + adaptive recommendation (`_research_next_actions`, `main.py:1482`) → `research_ops_report.json`.

**State Management:**
- No long-lived process; every run is a fresh process. All cross-run state is files: the decision journal, report JSONs, the edge index, tuning overrides, and evidence run dirs. Model weights cache in `~/.cache/huggingface/hub/`.

## Key Abstractions

**Stage result dataclasses** (`scanner/utils/validation.py`):
- Purpose: uniform gate contract — `passed: bool`, `skip_reason: str | None`, plus stage-specific metrics and a `diagnostics` dict.
- Examples: `PotterBoxResult`, `EmptySpaceResult`, `EventRiskResult`, `OptionsContractResult`, `KronosResult`, `AlertCandidate`.
- Pattern: each gate function returns its result type; the orchestrator inspects `.passed`.

**Decision record** (dict, journaled to `scanner/reports/scan_decisions.jsonl`):
- Purpose: the unit of learning. Fields include `ticker`, `mode`, `decision_ts`, `final_pass`, `stage_failed`, `skip_reason`, `direction`, `entry_price`, `anchor_hour/minute`, `counterfactual`, `outcome_status/label/ret_5bar_pct`, `research_score`, `doctrine_v2_*`.
- Pattern: append-once with fingerprint dedup + enrichment merge (`scanner/learning/outcome_store.py`).

**`EdgeRecord`** (`scanner/edge/retrieval.py:18`):
- Purpose: one historical setup with its feature vector and realized outcome (return, win/loss label, R-multiple, MAE/MFE).
- Pattern: k-NN retrieval over normalized numeric features with per-ticker time embargo; `EdgeAnalogIndex` is the vectorized in-memory index.

**Feature vector** (`extract_edge_features`, `scanner/edge/features.py:85`):
- Purpose: stable, JSON-safe, versioned (`FEATURE_VERSION = 2`, `features.py:11`) representation used for both index building and live scoring. Add new features here, never ad hoc.

**`EvidenceRun`** (`scanner/evidence/store.py:36`):
- Purpose: immutable audit trail per lab run — `record_rows`, `record_metrics`, `log_artifact`, `flush()` → `manifest.json`.

**Timed stage wrapper** (`_run_timed_stage`, `scanner/main.py:98`):
- Purpose: STAGE_START/STAGE_DONE/STAGE_FAILED logging with durations; used by `research_ops`.

## Entry Points

**`python -m scanner.main` / `scanner/run_scanner.bat`:**
- Location: `scanner/main.py:1596` (`main()`); bat wrapper resolves repo venv and forwards args.
- Triggers: manual or scheduled (daily `research_ops` is the intended cadence).
- Responsibilities: everything scanner-side. Note the script-execution fallback at `scanner/main.py:21-23` inserts the repo root into `sys.path` so the file also runs as a plain script.

**`kronos_app.py` (+ `launch_kronos.bat`):**
- Streamlit desktop forecaster over `model/` — independent of the scanner.

**`webui/app.py` (+ `webui/run.py`, `webui/start.sh`):**
- Upstream Flask forecast UI writing to `webui/prediction_results/`.

**`finetune/train_tokenizer.py`, `finetune/train_predictor.py`:**
- Upstream qlib-based fine-tuning scripts; not wired to the scanner.

## Architectural Constraints

- **Threading:** single-threaded, sequential per-ticker loops. `yf.download(..., threads=False)` is deliberate. No async.
- **Global state:** `scanner/config.py` module globals are mutated at import (`_apply_overrides`, `config.py:150`) and at runtime (`reload_overrides`, `config.py:145`). Modules that must see live values import the module (`from .. import config as scanner_config`) rather than the names — copy that pattern for any tunable threshold (see `scanner/strategy/potter_box.py:280`, `scanner/edge/scoring.py:82`).
- **sys.path bootstrapping:** three places manipulate `sys.path` to reach `model/` or the repo root: `scanner/main.py:21-23`, `scanner/models/kronos_adapter.py:25-28`, `scanner/tests/conftest.py`. `webui/app.py` and `kronos_app.py` do their own.
- **Report-file coupling:** modes communicate through JSON files with paths fixed in `scanner/config.py:6-16`. Renaming a report path breaks consumers (`audit_edge`, live preflight, `_resolve_calibrated_anchor`).
- **Fail-closed live gating:** `live` mode cannot run without a fresh `edge_audit_report.json` with `readiness=paper_trade_only` (`scanner/main.py:405-423`). Do not weaken this chain.
- **Windows-first:** bat launchers, `venv/` at repo root (`.venv/` also present), paths assume `C:\Users\Jacob Higgins\projects\kronos-predictor`.

## Anti-Patterns

### God module: `scanner/main.py` (1681 lines)

**What happens:** CLI, env handling, the gate pipeline, calibration, zero-result diagnostics, and all seven edge-mode runners live in one file.
**Why it's wrong:** every feature touches `main.py`; merge conflicts and accidental coupling (e.g. `_edge_data_quality` at `main.py:992` is domain logic stranded in the CLI layer).
**Do this instead:** when adding a new mode, implement the logic in the owning subpackage (like `scanner/edge/audit.py` does) and keep only the thin `run_*` dispatch wrapper in `main.py`.

### O(n) journal rewrite on every append

**What happens:** `append_decision` calls `load_decisions()` (full JSONL read) per append, and may rewrite the whole file for enrichment merges (`scanner/learning/outcome_store.py:81-97`). A 30-ticker scan does this 30 times.
**Why it's wrong:** quadratic growth as the journal grows; already thousands of records.
**Do this instead:** for batch operations follow `run_research_ops`'s pattern — load once, mutate the list, `save_decisions` once (`scanner/main.py:1512-1527`). Don't add new per-record `append_decision` loops.

### Mixed result conventions (dataclass vs dict)

**What happens:** first-generation gates return dataclasses (`PotterBoxResult`), while research/doctrine/edge scorers return plain dicts (`score_potter_research_candidate`, `score_potter_doctrine_v2`, `score_edge_candidate`). Bridging requires `_as_dict` shims (`scanner/edge/features.py:14`, `scanner/strategy/potter_doctrine.py:9`).
**Why it's wrong:** no type safety on the dict half; key typos fail silently through `_finite_float(default)` coercion.
**Do this instead:** new gate stages should use dataclasses in `scanner/utils/validation.py`; new research/report payloads may stay dicts but must define keys in one constructor function, not scattered literals.

### Silent exception swallowing for control flow

**What happens:** broad `except Exception: return default` in `_resolve_calibrated_anchor` (`scanner/main.py:133`), config override parsing (`scanner/config.py:118`), report reads (`main.py:1350-1359`).
**Why it's wrong:** a corrupt calibration or overrides file degrades behavior silently (falls back to defaults with no signal).
**Do this instead:** keep the fail-closed fallback but log a warning with the file path; never bare-swallow in new code.

## Error Handling

**Strategy:** fail closed, journal the failure, keep scanning.

**Patterns:**
- Gate functions catch their own exceptions and return `passed=False` result objects with `skip_reason` (e.g. `validate_ticker`, `market_data.py:231-240`; `KronosAdapter.evaluate`, `kronos_adapter.py:138-150`).
- The per-ticker loop wraps each ticker so one blow-up can't kill the scan (`run_watchlist_scan`, `main.py:1438-1442`; `run_edge_scan`, `main.py:1301-1303`).
- HTTP retries with backoff for Alpaca on 429/5xx, request IDs persisted to `scanner/logs/request_ids.log` (`market_data.py:69-94`).
- Preflight hard-fails the process with exit code 1 before any work (`main.py:390-427`, `main.py:1600`).

## Cross-Cutting Concerns

**Logging:** single `"scanner"` logger via `setup_logging` (`scanner/utils/logging_setup.py`) — console + rotating `scanner/logs/scanner.log` (1.5MB x3). Convention: `UPPER_SNAKE` event tags with JSON payloads (`SCAN_SUMMARY:`, `EDGE_AUDIT_REPORT:`, `STAGE_DONE:`), greppable as a poor-man's event stream.
**Validation:** result dataclasses + `skip_reason` strings; `_finite_float`/`_clamp` coercion helpers duplicated per module (features, scoring, retrieval, audit, validation, adaptive_policy).
**Configuration:** `.env` loaded from repo root and `scanner/.env` without clobbering existing env (`main.py:185-194`); non-secret thresholds in `scanner/config.py`; learned overrides in `scanner/tuning/overrides.json`.
**Timezones:** everything normalized to `America/New_York` (`TIMEZONE`, `config.py:51`; `_to_ny_index`, `market_data.py:25`); synthetic sessions anchor at 20:00 ET by default, per-ticker anchors from calibration.
**Authentication:** API keys only via env (`ALPACA_API_KEY/SECRET_KEY`, `TELEGRAM_BOT_TOKEN/CHAT_ID`, `MINIMAX_API_KEY`); never in code or reports.

---

*Architecture analysis: 2026-07-02*
