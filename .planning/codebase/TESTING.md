# Testing Patterns

**Analysis Date:** 2026-07-02

## Test Framework

**Runner:**
- pytest >= 8.2.0 (declared as the `test` optional dependency in `pyproject.toml`)
- Config: `pytest.ini` — `testpaths = tests, scanner/tests`, `python_files = test_*.py`
- No plugins configured (no pytest-cov, no pytest-mock, no markers, no addopts)

**Assertion Library:**
- Plain `assert` statements; `np.testing.assert_allclose(..., rtol=...)` for numeric model regression (`tests/test_kronos_regression.py:100-109`)

**Run Commands:**
```bash
.\venv\Scripts\python.exe -m pytest -q                       # Run all tests (101 pass)
.\venv\Scripts\python.exe -m pytest -q scanner/tests         # Scanner suite only (fast, no network)
.\venv\Scripts\python.exe -m pytest -q tests/test_webui_security.py   # WebUI security tests
.\venv\Scripts\python.exe -m pytest -q scanner/tests/test_potter_box.py -k breakout   # Single file/pattern
```
- Note: both `venv/` and `.venv/` exist at repo root; the documented runner is `.\venv\Scripts\python.exe`.
- `tests/test_kronos_regression.py` downloads pinned HuggingFace model weights (revisions pinned at `tests/test_kronos_regression.py:35-36`) — needs network/HF cache and CPU minutes. Everything in `scanner/tests/` runs offline.
- No CI (no `.github/workflows/`). The suite is the only verification gate — run it locally before claiming done.

## Test File Organization

**Location:**
- Separate test directories, not co-located:
  - `scanner/tests/` — 25 test files, 86 test functions, covers the scanner product
  - `tests/` — 3 files: `test_kronos_regression.py`, `test_kronos_sampling_safety.py` (upstream model), `test_webui_security.py` (Flask webui hardening)
- Fixture data files only for model regression: `tests/data/regression_input.csv`, `tests/data/regression_output_{256,512}.csv` (+ `tests/data/generate_regression_output.py` regenerator)

**Naming:**
- `test_<module_under_test>.py` mirrors the source module: `scanner/edge/scoring.py` → `scanner/tests/test_edge_scoring.py`; behavior-suite exceptions: `test_hardening.py` (nullable-field/fallback safety), `test_edge_cli_units.py` (unit tests for `scanner/main.py` helpers), `test_package_entrypoint.py` (CLI arg/env/preflight), `test_zero_result_diagnostic.py`, `test_edge_evidence_lab.py`
- Test functions read as behavior sentences: `test_append_decision_skips_duplicate_setup_on_same_day`, `test_live_preflight_blocks_research_only_audit`, `test_bullish_requires_prior_close_above_cost_basis`

**Structure:**
```
scanner/tests/
├── conftest.py                  # sys.path bootstrap only (adds repo root); no fixtures
├── test_<module>.py             # one file per source module
tests/
├── data/                        # CSV fixtures for model regression
└── test_*.py
```

## Test Structure

**Suite Organization:**
```python
# Standard shape (scanner/tests/test_outcome_store.py)
from scanner.learning import outcome_store

def _record(**overrides):                      # module-private builder with override kwargs
    payload = {"ticker": "TEST", "mode": "research_scan", ...}
    payload.update(overrides)
    return payload

def test_append_decision_skips_duplicate_setup_on_same_day(monkeypatch, tmp_path):
    path = tmp_path / "decisions.jsonl"                     # arrange
    monkeypatch.setattr(outcome_store, "DECISIONS_PATH", path)
    monkeypatch.setattr(outcome_store, "REPORT_DIR", tmp_path)

    first = outcome_store.append_decision(_record())        # act
    second = outcome_store.append_decision(_record(decision_ts="..."))

    assert first is True                                    # assert
    assert second is False
```

**Patterns:**
- Function-based tests only — no test classes, no unittest.TestCase anywhere
- Arrange/act/assert separated by blank lines; no comments labeling the sections
- No custom pytest fixtures — only built-ins: `monkeypatch` (32 tests) and `tmp_path` (21 tests). `scanner/tests/conftest.py` contains only a sys.path insert
- `@pytest.mark.parametrize` used only in `tests/test_kronos_regression.py` (context lengths); scanner tests write explicit separate functions instead
- Setup/teardown: none needed — `tmp_path` + monkeypatched path constants isolate all filesystem writes

## Mocking

**Framework:** `monkeypatch` exclusively. `unittest.mock`/`MagicMock` is NOT used anywhere — do not introduce it.

**Patterns:**
```python
# 1. Patch where used (module attribute on the consumer), lambda stubs
monkeypatch.setattr("scanner.main.select_options_contract", _valid_options_contract)
monkeypatch.setattr("scanner.main.fetch_intraday_bars", lambda ticker, research=False: _bars())

# 2. Redirect path constants into tmp_path (JSONL journals / reports)
monkeypatch.setattr(outcome_store, "DECISIONS_PATH", tmp_path / "decisions.jsonl")
monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", tmp_path / "edge_audit_report.json")

# 3. Mutate runtime-tunable config values directly on the config module
monkeypatch.setattr(config, "RESEARCH_CANDIDATE_MIN_SCORE", score + 1)   # scanner/tests/test_potter_box.py:62

# 4. Minimal duck-typed doubles as local classes
class DummyLogger:
    def info(self, *args, **kwargs): return None
    def error(self, *args, **kwargs): return None

class DummyPredictor:                                   # scanner/tests/test_kronos_adapter.py:31
    def predict(self, **kwargs): return {"unexpected": True}
monkeypatch.setattr(adapter, "_load_once", lambda: DummyPredictor())

# 5. Deterministic clocks via exhausting iterators
clock = iter([100.0, 101.0, 103.5, 107.0, 108.0])
monkeypatch.setattr("scanner.main._monotonic_seconds", lambda: next(clock))   # test_edge_cli_units.py:299-303

# 6. Env control
monkeypatch.setenv("ALPACA_FEED", "iex"); monkeypatch.delenv("ALPACA_API_KEY", raising=False)

# 7. HTTP stub by faking requests.post response object
class Resp: status_code = 500; text = "fail"
monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())              # scanner/tests/test_telegram.py
```

**What to Mock:**
- ALL network I/O: yfinance/Alpaca fetchers (`_fetch_alpaca_bars`, `validate_ticker`, `fetch_intraday_bars`), `requests.post`, Kronos model loading (`_load_once`)
- Filesystem path constants (`DECISIONS_PATH`, `REPORT_DIR`, `EDGE_*_PATH`, `ENV_PATHS`) — always redirect to `tmp_path`
- Time sources (`_monotonic_seconds`, `_utc_now_iso`) when asserting durations/timestamps
- Neighboring pipeline stages when unit-testing an orchestrator (`test_research_ops.py` stubs every stage of `run_research_ops` and asserts stage ordering + report shape)

**What NOT to Mock:**
- Pure computation: `detect_potter_box`, `score_edge_candidate`, `score_potter_doctrine_v2`, `deduplicate_decisions`, `build_synthetic_sessions` run against real hand-built DataFrames
- The filesystem itself — tests write/read real files under `tmp_path` and assert on JSON/JSONL contents (`json.loads(path.read_text(...))`)
- One real subprocess smoke test exists: `test_scanner_help_runs_from_repo_root` runs `python -m scanner.main --help` with `timeout=15` (`scanner/tests/test_package_entrypoint.py:9-21`)

## Fixtures and Factories

**Test Data:**
```python
# Deterministic OHLCV frames built inline per test file — no shared fixtures
def _bars():
    rows = []
    for i in range(45):
        if i < 29:   rows.append([100, 104, 96, 100 + (0.2 if i % 2 == 0 else -0.2), 1000])
        elif i < 44: rows.append([100, 101, 99, 100 + (0.05 if i % 2 == 0 else -0.05), 1200])
        else:        rows.append([101, 104, 100.5, 103, 2600])   # breakout bar
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])

# Record builders take **overrides
def _record(**overrides): ...          # test_outcome_store.py
def _edge_record(ticker="TEST", timestamp="..."): ...   # test_edge_evidence_lab.py
```

**Location:**
- Builders are module-private (`_` prefixed) at the top of each test file and intentionally duplicated per file rather than shared — follow that; do not create a shared factories module
- DataFrame indexes are always tz-aware `America/New_York`; columns always `["Open", "High", "Low", "Close", "Volume"]`

## Coverage

**Requirements:** None enforced. No coverage tooling configured (no pytest-cov, no threshold, no CI).

**View Coverage:**
```bash
# Not configured; would require: pip install pytest-cov
.\venv\Scripts\python.exe -m pytest --cov=scanner -q   # only if pytest-cov is added
```

### Subsystems WITH tests (scanner)

| Subsystem | Source | Tests |
|-----------|--------|-------|
| Potter Box detection + research scoring | `scanner/strategy/potter_box.py` | `scanner/tests/test_potter_box.py` |
| Potter Doctrine v2 scoring | `scanner/strategy/potter_doctrine.py` | `scanner/tests/test_potter_doctrine.py` |
| Empty space scoring | `scanner/strategy/empty_space.py` | `scanner/tests/test_empty_space.py` |
| Edge features / retrieval / scoring / validation / audit | `scanner/edge/*.py` | `scanner/tests/test_edge_features.py`, `test_edge_retrieval.py`, `test_edge_scoring.py`, `test_edge_validation.py`, `test_edge_audit.py` |
| Outcome journal (JSONL dedup/enrich) | `scanner/learning/outcome_store.py` | `scanner/tests/test_outcome_store.py` |
| Outcome reviewer / autotuner / adaptive policy / replay | `scanner/learning/*.py` | `scanner/tests/test_outcome_reviewer.py`, `test_autotuner.py`, `test_adaptive_policy.py`, `test_replay_runner.py` |
| Evidence store (JSONL/parquet runs) | `scanner/evidence/store.py` | `scanner/tests/test_evidence_store.py`, `test_edge_evidence_lab.py` |
| Market data provider routing (Alpaca feeds/delays) | `scanner/data/market_data.py` | `scanner/tests/test_market_data.py` |
| Options contract selection | `scanner/data/options_data.py` | `scanner/tests/test_options.py` |
| Kronos adapter fail-safe behavior | `scanner/models/kronos_adapter.py` | `scanner/tests/test_kronos_adapter.py` |
| Telegram send failure path + alert rendering | `scanner/alerts/telegram.py` | `scanner/tests/test_telegram.py`, `test_hardening.py` |
| MiniMax fallback parsing | `scanner/ai/minimax_adapter.py` | `scanner/tests/test_hardening.py` (fallback regex only) |
| CLI helpers, preflight gates, env loading, entrypoint | `scanner/main.py` | `scanner/tests/test_edge_cli_units.py`, `test_package_entrypoint.py`, `test_zero_result_diagnostic.py`, `test_research_ops.py`, `test_edge_evidence_lab.py` |
| Doctor diagnostics | `scanner/doctor.py` | `scanner/tests/test_doctor.py` |

### Subsystems WITH tests (root)

| Subsystem | Source | Tests |
|-----------|--------|-------|
| Kronos model deterministic regression + MSE health | `model/kronos.py`, `model/module.py` | `tests/test_kronos_regression.py` (pinned HF revisions, heavy) |
| Logit sampling NaN/Inf safety | `model/kronos.py` `sample_from_logits` | `tests/test_kronos_sampling_safety.py` |
| WebUI path traversal / CORS / server defaults / result saving | `webui/app.py` | `tests/test_webui_security.py` |

### Subsystems WITHOUT tests (gaps)

- `scanner/backtest/backtest_runner.py` and `scanner/backtest/metrics.py` — backtest modes (`backtest_intraday_60d`, `backtest_daily_proxy_2y`) are entirely untested
- `scanner/data/events.py` — earnings/ex-dividend fail-closed gate has no direct tests (its fail-closed behavior is asserted only implicitly via config)
- `scanner/data/synthetic_sessions.py` — no dedicated test file; only exercised as a helper inside `scanner/tests/test_edge_cli_units.py`
- `scanner/strategy/risk_reward.py` — no dedicated tests
- `scanner/ai/minimax_adapter.py` — HTTP call path, timeout, and response parsing untested (only the regex fallback is covered)
- `scanner/utils/logging_setup.py`, `scanner/tickers.py` — untested (low risk)
- `scanner/main.py` mode dispatch in `main()` and the full `dry_run`/`live` alert path (Telegram send on pass, MiniMax insight enrichment) — only helpers and preflight are unit-tested
- `kronos_app.py` (404-line Flask desktop app) — untested
- `webui/app.py` — only security-relevant functions tested; prediction endpoints untested
- `finetune/`, `finetune_csv/`, `examples/` — untested upstream code

## Test Types

**Unit Tests:**
- The dominant type. Pure-logic tests on hand-built DataFrames/dicts; orchestrator helpers tested with every collaborator monkeypatched (`scanner/tests/test_edge_cli_units.py`)

**Integration Tests:**
- Light, file-level: evidence-lab tests run real `EvidenceRun` flush → assert JSONL rows and manifest on disk (`scanner/tests/test_edge_evidence_lab.py`); journal tests write/read real JSONL under `tmp_path`
- One subprocess CLI smoke test (`scanner/tests/test_package_entrypoint.py:9`)

**E2E Tests:**
- Not used. No live-network or live-broker tests; live mode is instead guarded by runtime preflight gates which are themselves unit-tested (`test_live_preflight_*` in `scanner/tests/test_package_entrypoint.py:54-133`)

## Common Patterns

**Fail-closed testing (the house specialty):**
```python
# Assert the gate BLOCKS on missing/ambiguous inputs, not just that it passes on good ones
def test_live_preflight_requires_edge_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", tmp_path / "missing_audit.json")
    env = {..., "live_mode_enabled": True, ...}
    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False

def test_unknown_output_format_fails_safely(monkeypatch):    # test_kronos_adapter.py
    ...
    assert result.passed is False
    assert result.output_mode == "unknown"
```
Any new gate/stage needs both a blocking-path and passing-path test.

**Journal assertion pattern:**
```python
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 1
assert rows[0]["doctrine_v2_score"] == 74
```

**Determinism for stochastic model code:**
```python
# tests/test_kronos_regression.py — pin everything
torch.use_deterministic_algorithms(True, warn_only=True); torch.set_num_threads(1)
set_seed(SEED)   # random + numpy + torch (+ cudnn deterministic)
KronosTokenizer.from_pretrained(..., revision=TOKENIZER_REVISION)   # pinned HF revision
```

**Async Testing:** Not applicable — the codebase is fully synchronous.

**Error Testing:**
```python
# Errors are values here: stub the failure, assert the returned result object / bool
monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())   # Resp.status_code = 500
ok = send_telegram_message("token", "chat", "msg", DummyLogger())
assert ok is False
```
`pytest.raises` is not used anywhere — scanner code returns fail-closed results instead of raising, and tests assert on those.

---

*Testing analysis: 2026-07-02*
