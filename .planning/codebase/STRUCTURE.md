# Codebase Structure

**Analysis Date:** 2026-07-02

## Directory Layout

```
kronos-predictor/
├── scanner/                    # THE PRODUCT: Potter Box options scanner (active development)
│   ├── main.py                 # CLI orchestrator: 21 modes, staged gate pipeline (1681 lines)
│   ├── config.py               # All thresholds, report paths, bounds; applies tuning/overrides.json
│   ├── tickers.py              # WATCHLIST (re-exports DEFAULT_WATCHLIST from config)
│   ├── doctor.py               # Environment/dependency health check (--mode doctor)
│   ├── README.md               # Scanner runbook: setup, modes, safety defaults
│   ├── requirements-scanner.txt# Scanner-specific deps
│   ├── run_scanner.bat         # Windows launcher → venv python -m scanner.main
│   ├── setup_dependencies.bat  # Dependency installer
│   ├── strategy/               # Setup detection & scoring (Potter Box, Empty Space, Doctrine v2)
│   ├── data/                   # Market data providers, synthetic sessions, options, events
│   ├── edge/                   # Edge evidence engine: features, retrieval, scoring, validation, audit
│   ├── learning/               # Self-tuning loop: journal, outcome review, adaptive policy, autotune
│   ├── models/                 # KronosAdapter (bridges to root model/)
│   ├── ai/                     # MiniMaxAdapter (optional LLM advisory)
│   ├── alerts/                 # Telegram rendering + send
│   ├── backtest/               # Historical simulations + metrics
│   ├── evidence/               # EvidenceRun store (immutable run dirs)
│   ├── replay/                 # Replay datasets (sample_replay_dataset.json)
│   ├── utils/                  # Shared dataclasses (validation.py), logging setup
│   ├── tests/                  # Scanner unit tests (26 test files, pytest)
│   ├── reports/                # GENERATED: JSON reports, scan_decisions.jsonl, evidence/ (gitignored)
│   ├── tuning/                 # GENERATED: overrides.json learned thresholds (gitignored)
│   └── logs/                   # GENERATED: scanner.log, request_ids.log (gitignored)
├── model/                      # Upstream Kronos model (fork base)
│   ├── kronos.py               # KronosTokenizer, Kronos, KronosPredictor
│   ├── module.py               # Transformer building blocks (BSQ, RoPE attention, etc.)
│   └── __init__.py             # Exports + model registry
├── tests/                      # Root-level Kronos model tests (regression, sampling safety, webui security)
│   └── data/                   # Regression fixtures (regression_input.csv, expected outputs)
├── webui/                      # Upstream Flask forecast UI (legacy, independent of scanner)
│   ├── app.py                  # Flask app
│   ├── templates/              # HTML templates
│   └── prediction_results/     # GENERATED (gitignored)
├── finetune/                   # Upstream qlib-based fine-tuning scripts
├── finetune_csv/               # CSV-based fine-tuning variant (configs/, data/, examples/)
├── examples/                   # Upstream prediction examples (+ yuce/)
├── figures/                    # Upstream README images
├── docs/                       # Local planning docs
│   ├── superpowers/plans/      # Dated implementation plans (e.g. 2026-06-21-potter-doctrine-v2.md)
│   ├── superpowers/specs/      # Dated design specs (e.g. 2026-05-14-kronos-edge-engine-design.md)
│   └── daily-notes/            # Dated session notes (YYYY-MM-DD.md)
├── .planning/codebase/         # GSD codebase analysis documents (this directory)
├── kronos_app.py               # Streamlit desktop forecaster (Jake's one-click app)
├── launch_kronos.bat           # Desktop launcher for kronos_app.py
├── install_deps.bat            # Root dependency helper
├── pyproject.toml              # Packaging: includes model*, scanner*; excludes tests
├── requirements.txt            # Root deps
├── pytest.ini                  # testpaths = tests, scanner/tests
├── LLM_PROJECT_MEMORY.md       # First-read orientation doc for agents (guardrails, layout)
├── README.md                   # Upstream Kronos README
├── README_JAKE.md              # Local desktop usage notes
├── venv/                       # Runtime venv used by bat launchers (gitignored)
└── .venv/                      # Secondary venv (gitignored)
```

## Directory Purposes

**`scanner/` (center of gravity):**
- Purpose: the local options scanner product; everything new happens here.
- Key files: `scanner/main.py` (orchestrator + mode dispatch), `scanner/config.py` (single source of thresholds/paths).

**`scanner/strategy/`:**
- Purpose: setup detection and scoring over synthetic-session DataFrames.
- Contains: `potter_box.py` (`detect_potter_box`, `score_potter_research_candidate`), `empty_space.py` (`score_empty_space`), `potter_doctrine.py` (`score_potter_doctrine_v2`, dict-returning research scorer), `risk_reward.py` (`compute_rr`).

**`scanner/data/`:**
- Purpose: all external market-data I/O.
- Contains: `market_data.py` (Alpaca-first fetch + `validate_ticker`; provenance in `DataFrame.attrs`), `synthetic_sessions.py` (`build_synthetic_sessions` anchor-hour aggregation), `options_data.py` (`select_options_contract`), `events.py` (`assess_event_risk`).

**`scanner/edge/`:**
- Purpose: evidence engine for the edge lab.
- Contains: `features.py` (`extract_edge_features`, `FEATURE_VERSION`), `retrieval.py` (`EdgeRecord`, `EdgeAnalogIndex`, `find_analogs`, `build_edge_records_from_bars`, index save/load), `scoring.py` (`score_edge_candidate` → promote/research/reject), `validation.py` (`compute_edge_validation_report`), `audit.py` (`compute_edge_audit_report` → readiness verdict).

**`scanner/learning/`:**
- Purpose: closed-loop self-tuning.
- Contains: `outcome_store.py` (JSONL journal, `DECISIONS_PATH`, fingerprint dedup), `outcome_reviewer.py` (`review_pending_outcomes`), `adaptive_policy.py` (Wilson-bound threshold search + safe apply), `autotuner.py` (`propose_overrides`/`apply_overrides`), `replay_runner.py` (`run_replay_eval`).

**`scanner/models/` and `scanner/ai/`:**
- Purpose: adapters around models. `models/kronos_adapter.py` lazy-loads the root `model/` package; `ai/minimax_adapter.py` wraps the MiniMax HTTP API and degrades gracefully when disabled.

**`scanner/evidence/`:**
- Purpose: immutable research-run artifact store. `store.py` exposes `EvidenceRun` / `start_evidence_run` (the only subpackage with a populated `__init__.py` re-export).

**`scanner/utils/`:**
- Purpose: shared plumbing. `validation.py` holds ALL pipeline result dataclasses (`PotterBoxResult`, `EmptySpaceResult`, `EventRiskResult`, `OptionsContractResult`, `KronosResult`, `AlertCandidate`, `TickerValidationResult`); `logging_setup.py` configures the rotating `"scanner"` logger.

**`scanner/reports/` (generated, gitignored):**
- Purpose: every mode's JSON output plus the decision journal.
- Key files: `scan_decisions.jsonl` (learning journal), `edge_retrieval_index.json`, `edge_scan_report.json`, `edge_validation_report.json`, `edge_diagnostic_report.json`, `edge_audit_report.json` (live-mode gate), `calibration_summary.json`, `calibration_<TICKER>.json`, `adaptive_policy_report.json`, `research_ops_report.json`, `zero_result_diagnostic.json`, `outcome_review_summary.json`, `evidence/<UTC-timestamp>-<hex8>/` run dirs (`manifest.json`, `*.jsonl`, `*.parquet`, `artifacts/`).

**`model/` (fork base):**
- Purpose: upstream Kronos implementation. Touch only for model-level fixes (e.g. NaN-safe sampling); the scanner consumes it exclusively through `scanner/models/kronos_adapter.py`.

**`docs/superpowers/`:**
- Purpose: dated plans and specs that document how each subsystem was designed (edge engine, evidence lab, doctrine v2). Read these before large changes.

## Key File Locations

**Entry Points:**
- `scanner/main.py`: scanner CLI (`python -m scanner.main --mode <mode>`); dispatch at `scanner/main.py:1596-1677`
- `scanner/run_scanner.bat`: Windows wrapper (uses root `venv/`)
- `kronos_app.py`: Streamlit forecaster
- `webui/app.py`: Flask forecast UI
- `finetune/train_predictor.py`, `finetune/train_tokenizer.py`: training scripts

**Configuration:**
- `scanner/config.py`: thresholds, bounds, report paths, watchlist
- `scanner/tuning/overrides.json`: learned threshold overrides (generated)
- `.env` (repo root) and `scanner/.env`: secrets — existence only, never committed
- `pyproject.toml`, `pytest.ini`, `requirements.txt`, `scanner/requirements-scanner.txt`

**Core Logic:**
- Gate pipeline: `scanner/main.py:462-712` (`_run_single_ticker`)
- Edge lab composite: `scanner/main.py:1389-1424` (`run_edge_lab`)
- Master daily loop: `scanner/main.py:1502-1559` (`run_research_ops`)
- Shared dataclasses: `scanner/utils/validation.py`

**Testing:**
- `scanner/tests/`: scanner unit tests, `conftest.py` adds repo root to `sys.path`
- `tests/`: Kronos model regression/safety tests + `tests/data/` fixtures
- `pytest.ini`: collects both roots

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (`outcome_store.py`, `potter_doctrine.py`)
- Tests: `test_<module>.py` mirroring module names (`test_edge_scoring.py` for `edge/scoring.py`); CLI-level units tested in `test_edge_cli_units.py`
- Reports: `<subject>_report.json` or `<subject>_summary.json`; per-ticker calibration `calibration_<TICKER>.json`
- Docs: date-prefixed `YYYY-MM-DD-<slug>.md` in `docs/superpowers/{plans,specs}/`; daily notes `YYYY-MM-DD.md`
- Windows launchers: `*.bat` at repo root or `scanner/`

**Directories:**
- Scanner subpackages are single-word domain nouns: `strategy/`, `data/`, `edge/`, `learning/`, `alerts/`, `evidence/`
- Evidence run dirs: `<UTC yyyymmddTHHMMSSZ>-<uuid4 hex8>` (e.g. `20260528T233618Z-2e5b3c6e`)

**Code:**
- Functions: `snake_case`; module-private helpers prefixed `_` (`_finite_float`, `_run_timed_stage`)
- Mode runner functions: `run_<mode_name>` in `scanner/main.py`
- Result dataclasses: `<Thing>Result` in `scanner/utils/validation.py`
- Config constants: `UPPER_SNAKE`; tunables have matching `<NAME>_BOUNDS` tuples (`scanner/config.py:81-90`)
- Log event tags: `UPPER_SNAKE:` prefix with JSON payload (`EDGE_SCAN_REPORT:`, `STAGE_DONE:`)
- Decision journal field names: `snake_case` with stage names matching gate order (`validation`, `market_data`, `potter_box`, `potter_box_research`, `empty_space`, `event_risk`, `options`, `kronos`)

## Where to Add New Code

**New pipeline gate (e.g. a new filter stage):**
- Detection/scoring logic: new module in `scanner/strategy/` or `scanner/data/` returning a dataclass defined in `scanner/utils/validation.py`
- Wire into `_run_single_ticker` in `scanner/main.py` following the existing pattern: check `.passed`, `append_decision` with a new `stage_failed` value, `_log_skip`, return skip dict
- Threshold constants (+ `_BOUNDS` if tunable): `scanner/config.py`
- Tests: `scanner/tests/test_<module>.py`

**New CLI mode:**
- Add to argparse choices at `scanner/main.py:141-164`
- Implement the real logic in the owning subpackage; add a thin `run_<mode>` wrapper in `scanner/main.py` and a dispatch branch in `main()` (`scanner/main.py:1596-1677`)
- Write a JSON report to `REPORT_DIR` with a path constant in `scanner/config.py`; document the mode in `scanner/README.md`
- Tests: CLI-level behavior in `scanner/tests/test_edge_cli_units.py` style (import functions from `scanner.main`)

**New edge feature:**
- Add the key in `extract_edge_features` (`scanner/edge/features.py:85`) and bump `FEATURE_VERSION` (`features.py:11`); rebuild the index with `--mode build_retrieval_index`

**New learned/tunable threshold:**
- Constant + bounds in `scanner/config.py`, override plumbing in `_apply_overrides` (`config.py:102`), proposal logic in `scanner/learning/autotuner.py` or `scanner/learning/adaptive_policy.py`
- Read it via `scanner_config.<NAME>` (module attribute access) so `reload_overrides()` takes effect

**New data provider:**
- `scanner/data/market_data.py`; preserve the `DataFrame.attrs` provenance contract (`data_provider`, `data_feed`, `data_delay_minutes`)

**Utilities:**
- Shared helpers: `scanner/utils/` (note: `_finite_float`/`_clamp` are currently duplicated per module — prefer consolidating into `scanner/utils/` if touched)

**Kronos model changes:**
- `model/kronos.py` / `model/module.py`; add regression coverage in root `tests/`

## Special Directories

**`scanner/reports/`:**
- Purpose: generated evidence (reports, journal, evidence runs)
- Generated: Yes
- Committed: No (`.gitignore`: `scanner/reports/*.json`, `*.jsonl`, `evidence/`) — treat contents as data, never as source

**`scanner/tuning/`:**
- Purpose: learned overrides consumed by `scanner/config.py` at import
- Generated: Yes (by `autotune --apply_tuning` / `adaptive_policy` auto-apply)
- Committed: No (`scanner/tuning/*.json` ignored)

**`scanner/logs/`:**
- Purpose: rotating runtime logs (`scanner.log`, `request_ids.log`)
- Generated: Yes; Committed: No

**`venv/` and `.venv/`:**
- Purpose: `venv/` is the runtime environment the bat launchers expect (`scanner/run_scanner.bat` hard-requires `venv\Scripts\python.exe`); `.venv/` is a secondary environment
- Generated: Yes; Committed: No

**`webui/prediction_results/`:**
- Purpose: Flask UI outputs; Generated: Yes; Committed: No

**`__pycache__/`, `.pytest_cache/`:**
- Generated: Yes; Committed: No

---

*Structure analysis: 2026-07-02*
