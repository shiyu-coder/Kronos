# Coding Conventions

**Analysis Date:** 2026-07-02

Scope: conventions below describe the actively developed `scanner/` package (Potter Box options scanner). Upstream Kronos model code (`model/`, `webui/`, `kronos_app.py`, `finetune/`) is inherited open-source code and follows looser, older patterns — match `scanner/` conventions for all new work.

## Naming Patterns

**Files:**
- snake_case module names, one domain concept per file: `scanner/strategy/potter_box.py`, `scanner/edge/scoring.py`, `scanner/learning/outcome_store.py`
- Packages grouped by domain, not by layer type: `scanner/strategy/`, `scanner/edge/`, `scanner/learning/`, `scanner/data/`, `scanner/alerts/`, `scanner/models/`, `scanner/ai/`, `scanner/evidence/`, `scanner/utils/`, `scanner/backtest/`
- Adapters for external systems end in `_adapter.py`: `scanner/models/kronos_adapter.py`, `scanner/ai/minimax_adapter.py`

**Functions:**
- snake_case with verb prefixes that signal intent: `detect_potter_box`, `score_edge_candidate`, `score_potter_doctrine_v2`, `fetch_intraday_bars`, `build_adaptive_policy_report`, `assess_event_risk`, `render_alert_message`
- Module-private helpers ALWAYS get a leading underscore: `_atr`, `_count_touches`, `_finite_float`, `_clamp`, `_analog_summary`, `_preflight_checks`, `_run_single_ticker` (used pervasively; public surface per module is deliberately small)
- CLI mode entry points in `scanner/main.py` use `run_*`: `run_watchlist_scan`, `run_edge_lab`, `run_research_ops`, `run_telegram_test`

**Variables:**
- snake_case; short domain abbreviations are accepted for the stage results inside a function: `pb` (potter box), `es` (empty space), `ev` (event risk), `op` (options), `kr` (kronos) — see `scanner/alerts/telegram.py` `render_alert_message`
- Config constants are UPPER_SNAKE in `scanner/config.py`; tunable-bound pairs use the `<NAME>_BOUNDS` suffix (e.g., `MIN_RR_BOUNDS`, `DOCTRINE_V2_SCORE_BASELINE_BOUNDS`)

**Types:**
- PascalCase dataclasses; stage-gate result contracts end in `Result`: `TickerValidationResult`, `PotterBoxResult`, `EmptySpaceResult`, `EventRiskResult`, `OptionsContractResult`, `KronosResult` in `scanner/utils/validation.py`
- Classes only for adapters and stateful stores: `KronosAdapter`, `MiniMaxAdapter`, `EvidenceRun` (`scanner/evidence/store.py`). Everything else is module-level functions.

## Code Style

**Formatting:**
- No formatter configured (no black/ruff/prettier config anywhere in the repo). De facto style: 4-space indent, double quotes, f-strings, trailing commas in multi-line literals, ~120–130 char practical line limit (`scanner/main.py` max observed line is 133 chars). Do not reflow existing code to a stricter width.
- Multi-line call/collection style: one arg per line with trailing comma when the call spans lines (see `PotterBoxResult(...)` constructions in `scanner/strategy/potter_box.py`).
- Quirk: most `scanner/` files carry a UTF-8 BOM (e.g., `scanner/config.py`, `scanner/main.py`, `scanner/strategy/potter_box.py`). Python tolerates it. Preserve file encoding as-is when editing; write new files as plain UTF-8 (no BOM) — both coexist today.

**Linting:**
- Not configured. No `.flake8`, `ruff.toml`, `mypy.ini`, `.pre-commit-config.yaml`, or CI workflows exist. Verification = the pytest suite (`.\venv\Scripts\python.exe -m pytest -q`).

**Typing:**
- `requires-python = ">=3.10"` (`pyproject.toml`); use modern builtin generics and unions: `float | None`, `dict[str, Any]`, `list[str]`, `tuple[int, int]` — never `Optional[...]`/`Dict[...]`
- Nearly every scanner module starts with `from __future__ import annotations`; keep doing this
- Type hints on public function signatures and dataclass fields; internal locals mostly untyped. `logger` params are often untyped (`def _preflight_checks(mode: str, env: dict, logger) -> bool`) — either style is fine.

## Import Organization

**Order (blank-line separated groups):**
1. `from __future__ import annotations`
2. Standard library (`import json`, `from pathlib import Path`, `from dataclasses import dataclass, field`)
3. Third-party (`import numpy as np`, `import pandas as pd`, `import requests`, `import yfinance as yf`)
4. Local relative imports (`from ..config import ...`, `from ..utils.validation import PotterBoxResult`)

**Path Aliases / relative imports:**
- Inside `scanner/`, always relative imports: `from ..config import MIN_RR`, `from .outcome_store import deduplicate_decisions`
- Tests import absolutely: `from scanner.strategy.potter_box import detect_potter_box`
- Multi-name from-imports use parenthesized, alphabetized, one-per-line style (see the config import block at `scanner/main.py:28-54`)
- `scanner/main.py` has a self-bootstrapping header so `python scanner/main.py` works without install: `if __package__ in {None, ""}: sys.path.insert(0, ...)` (`scanner/main.py:21-23`). Do not replicate this outside entry points.

**Config access — two deliberate styles (important):**
- Static thresholds: import directly — `from ..config import ATR_PERIOD`
- Runtime-tunable thresholds (anything with a `*_BOUNDS` pair or touched by overrides): access via module attribute so tuning/monkeypatching takes effect — `from .. import config as scanner_config` then `scanner_config.RESEARCH_CANDIDATE_MIN_SCORE` (see `scanner/strategy/potter_box.py:280`, `scanner/edge/scoring.py:82`). Direct-import a tunable and you freeze its value at import time — this is a real bug class here.

## Configuration Pattern

- `scanner/config.py` holds non-secret constants only (module docstring says so). Secrets NEVER go there.
- Tuning overrides live in `scanner/tuning/overrides.json`, applied at import via `_apply_overrides()` and refreshable via `config.reload_overrides()` (`scanner/config.py:145-150`). Each override is explicitly whitelisted and type-coerced (`float(...)`/`int(...)`) — add new tunables to both the whitelist and a `*_BOUNDS` constant.
- Secrets/env: loaded in `scanner/main.py` `_load_env()` via `dotenv_values` from `ENV_PATHS` (repo-root `.env`, then `scanner/.env`); existing `os.environ` values win. Parsed into a plain lowercase-key dict (`telegram_token`, `alpaca_key`, ...). Boolean env vars parse as `os.getenv("X", "false").lower() == "true"`.
- Adaptive tuning writes merged overrides back with `OVERRIDES_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")` then calls `scanner_config.reload_overrides()` (`scanner/learning/adaptive_policy.py:308-328`).

## Error Handling

**Core pattern — fail closed, never crash the scan loop:**
- Every pipeline stage returns a dataclass with `passed: bool` and `skip_reason: str | None` instead of raising (`scanner/utils/validation.py`). On any uncertainty or error, return `passed=False` with a reason string. Canonical examples:
  - `scanner/data/events.py` `assess_event_risk`: unknown earnings date → `skip_reason="earnings data unavailable (fail-closed)"`; any exception → `passed=False, status="blocked"`
  - `scanner/models/kronos_adapter.py` `evaluate`: load failure, unknown output format, or inference exception all return `KronosResult(passed=False, output_mode="error"/"unknown", ...)`
- `scanner/main.py` `_run_single_ticker` wraps data fetching in `try/except Exception`, journals the failure as a decision record (`stage_failed`, `skip_reason`), logs via `_log_skip`, and returns a `{"status": "skip"}` dict — one bad ticker never kills the watchlist run.
- Live mode is gated by `_preflight_checks` (`scanner/main.py:390-427`): returns `False` (blocks the run) if credentials are missing, `LIVE_MODE_ENABLED` is not true, or the edge audit report is missing / readiness != `"paper_trade_only"`. Follow this pattern for any new externally-visible mode.

**Coercion instead of exceptions for dirty data:**
- `_finite_float(value, default)` — try/except `(TypeError, ValueError)` + `math.isfinite` check — is the standard scalar-coercion helper. It is intentionally re-declared per module (`scanner/edge/scoring.py:12`, `scanner/learning/adaptive_policy.py:18`, `scanner/strategy/potter_doctrine.py:19`); copy it locally rather than importing across domains.
- Guard denominators with `max(x, 1e-9)`; clamp scores with a local `_clamp(value, low, high)` (`scanner/edge/scoring.py:20`).

**Broad excepts are deliberate at I/O boundaries:**
- Reading JSON reports/overrides: `try: json.loads(...) except Exception: return <safe default>` (`scanner/config.py:116-119`, `scanner/main.py:117-135`). Silent fallback to defaults is the accepted style for optional files.
- Network calls retry with linear backoff on 429/5xx: `time.sleep(0.8 * (attempt + 1))`, 3 attempts, then return `False`/last response — see `send_telegram_message` (`scanner/alerts/telegram.py:80-105`) and `_alpaca_get` (`scanner/data/market_data.py:82-94`). Log the `X-Request-ID` response header when present.

## Logging

**Framework:** stdlib `logging`, single named logger `"scanner"` with console + rotating file handler (1.5MB x3 backups) built by `setup_logging(log_dir)` in `scanner/utils/logging_setup.py`. Format: `%(asctime)s | %(levelname)s | %(message)s`.

**Patterns:**
- The logger is created once in `main()` and passed as a parameter down the call stack (`detect_potter_box` excepted — pure functions take no logger). Do not call `logging.getLogger(__name__)` per module.
- Use %-style lazy formatting: `logger.info("SKIP: %s %s", ticker, reason)` — not f-strings inside log calls.
- Machine-greppable UPPERCASE event tags prefix structured lines: `STAGE_START:`, `STAGE_DONE:`, `STAGE_FAILED:`, `SKIP:`, `RESEARCH_CANDIDATE:`, `ZERO_RESULT_DIAGNOSTIC:`, `ADAPTIVE_POLICY_OVERRIDES_APPLIED:`, `MINIMAX_TEST_RESULT:` (see `scanner/main.py:98-114`). Payloads are `key=value` pairs or a `json.dumps(...)` blob.
- `logger.exception(...)` inside stage wrappers before re-raising (`_run_timed_stage`, `scanner/main.py:104-106`).
- External request IDs are also persisted to `scanner/logs/request_ids.log` via `_persist_request_id` (`scanner/data/market_data.py:69-79`).

## JSONL Journals & Reports

- Decision journal: append-only JSONL at `scanner/reports/scan_decisions.jsonl` managed exclusively through `scanner/learning/outcome_store.py`. Never write to it directly — use `append_decision` (fingerprint-dedupes per ticker/mode/direction/entry/day and enriches existing rows instead of duplicating), `load_decisions` (skips blank/corrupt lines silently), `save_decisions`, `deduplicate_decisions`.
- JSON reports: `path.write_text(json.dumps(payload, indent=2), encoding="utf-8")` with `REPORT_DIR.mkdir(parents=True, exist_ok=True)` first. All report paths are constants in `scanner/config.py` (`EDGE_AUDIT_REPORT_PATH`, etc.) so tests can monkeypatch them.
- Evidence runs: `EvidenceRun` (`scanner/evidence/store.py`) writes per-run JSONL (+ best-effort parquet) plus a `manifest.json`; `_json_safe` converts Timestamps/NaN before serialization. Reuse it for any new research artifact instead of ad-hoc files.
- Always pass `encoding="utf-8"` on every `open`/`read_text`/`write_text`. No exceptions.
- Timestamps: journal/report timestamps are ISO strings — `pd.Timestamp.utcnow().isoformat()` or `datetime.now(timezone.utc).isoformat()`; market-data indexes are tz-aware `America/New_York` (`TIMEZONE` in `scanner/config.py`). Data provenance rides on `DataFrame.attrs` (`data_provider`, `data_feed`, `data_delay_minutes`).

## Comments

**When to Comment:**
- Sparingly — only where intent is non-obvious: `# Consolidation window excludes the latest breakout candle.` (`scanner/strategy/potter_box.py:66`), inline option notes on config constants (`ALPACA_FEED = "iex"  # iex (free) or sip (subscription)`). No section-banner comments.

**Docstrings:**
- One-line docstrings on modules and on functions whose contract needs a caveat: `"""Grade near-miss setups for research logging without relaxing live alerts."""` (`scanner/strategy/potter_box.py:183`), `"""Scanner runtime configuration (non-secret values only)."""` (`scanner/config.py:1`). No param/return docstring blocks; the type hints are the documentation.

## Function Design

**Size:** small pure helpers composed into one large per-domain public function (e.g., `detect_potter_box` ~140 lines but linear). `scanner/main.py` is the intentional exception (1681-line orchestrator).

**Parameters:**
- Keyword-only (`*,`) for optional behavior knobs: `_fetch_alpaca_bars(..., *, feed=None, delay_minutes=0)` (`scanner/data/market_data.py:104`), `build_adaptive_policy_report(records, *, current_research_score=None, ...)` (`scanner/learning/adaptive_policy.py:179`)
- Injectable dependencies default to production behavior for testability: `options_selector=...` param, `now=None` param on data-quality/market functions.

**Return Values:**
- Gate stages return dataclasses (`*Result`); analysis/reporting functions return plain `dict` payloads with a `"status"`/`"recommendation"`/`"reason"` key and rounded floats (`round(..., 4)`). Scoring functions return a transparent `"scorecard"` dict of named components plus `blocking_reasons` lists (`scanner/edge/scoring.py:145-152`).

## Module Design

**Exports:** direct imports from the defining module; no barrel re-exports except `scanner/evidence/__init__.py` (exports `EvidenceRun`, `start_evidence_run` with `__all__`). All other `__init__.py` are empty or docstring-only — keep them that way.

**Dataclasses:** shared cross-stage contracts live in `scanner/utils/validation.py`; use `field(default_factory=dict)` for `diagnostics`/`metadata`, and put `skip_reason: str | None = None` last among required-context fields. Accept dataclass-or-dict at boundaries with an `_as_dict` helper (`scanner/strategy/potter_doctrine.py:9-16`).

**Packaging:** setuptools finds `model*` and `scanner*`, excludes tests (`pyproject.toml:33-35`). Entry: `python -m scanner.main --mode <mode>` or `scanner\run_scanner.bat`.

---

*Convention analysis: 2026-07-02*
