# Kronos Edge Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local edge engine that ranks scanner candidates with deterministic features, historical analog retrieval, uncertainty penalties, and validation reports while preserving fail-closed live alerts.

**Architecture:** Add a focused `scanner/edge` package for feature extraction, retrieval indexing, scoring, and validation. Wire new CLI modes into `scanner/main.py` without changing existing dry-run/live behavior. Harden Kronos sampling so model failures degrade safely instead of crashing.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, existing Kronos/Potter scanner modules.

---

## File Structure

- Create `scanner/edge/__init__.py`: package marker and public exports.
- Create `scanner/edge/features.py`: deterministic feature vector extraction from Potter, Empty Space, bars, data quality, options, event, and Kronos results.
- Create `scanner/edge/retrieval.py`: local historical analog index, distance scoring, outcome calculations, JSON persistence.
- Create `scanner/edge/scoring.py`: transparent `edge_score` and scorecard generation.
- Create `scanner/edge/validation.py`: threshold/top-K validation metrics for ranked candidates.
- Create `scanner/tests/test_edge_features.py`: feature extraction tests.
- Create `scanner/tests/test_edge_retrieval.py`: analog retrieval and leakage guard tests.
- Create `scanner/tests/test_edge_scoring.py`: edge score tests.
- Create `scanner/tests/test_edge_validation.py`: validation metric tests.
- Modify `scanner/config.py`: edge thresholds and report/index paths.
- Modify `scanner/main.py`: add `build_retrieval_index`, `edge_scan`, `validate_edge`, and `diagnose_edge` modes.
- Modify `scanner/learning/replay_runner.py`: make replay evaluation able to use valid longer datasets and report edge details.
- Modify `scanner/replay/sample_replay_dataset.json`: replace too-short replay sample with a valid full-window sample.
- Modify `model/kronos.py`: NaN/Inf-safe logits handling.
- Create `tests/test_kronos_sampling_safety.py`: root-level safety tests for `sample_from_logits`.

---

### Task 1: Kronos Sampling Safety

**Files:**
- Modify: `model/kronos.py`
- Create: `tests/test_kronos_sampling_safety.py`

- [ ] **Step 1: Write failing tests for invalid logits**

Create `tests/test_kronos_sampling_safety.py`:

```python
import torch

from model.kronos import sample_from_logits


def test_sample_from_logits_handles_nan_and_inf_values():
    logits = torch.tensor([[float("nan"), float("-inf"), 1.0, float("inf")]])

    sample = sample_from_logits(logits, temperature=1.0, top_k=1, top_p=1.0, sample_logits=True)

    assert sample.shape == (1, 1)
    assert torch.isfinite(sample.float()).all()
    assert 0 <= int(sample.item()) < logits.shape[-1]


def test_sample_from_logits_falls_back_when_all_logits_invalid():
    logits = torch.tensor([[float("nan"), float("-inf"), float("-inf")]])

    sample = sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True)

    assert sample.shape == (1, 1)
    assert 0 <= int(sample.item()) < logits.shape[-1]


def test_sample_from_logits_argmax_path_uses_torch_topk():
    logits = torch.tensor([[0.1, 0.2, 9.0]])

    sample = sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=False)

    assert int(sample.item()) == 2
```

- [ ] **Step 2: Run safety tests to verify failure**

Run: `.\venv\Scripts\python.exe -m pytest tests\test_kronos_sampling_safety.py -q`

Expected: at least one failure from invalid probability tensors or the `top_k` shadowing bug.

- [ ] **Step 3: Implement safe logits handling**

In `model/kronos.py`, update `sample_from_logits` so it clones logits, replaces NaN/Inf values with finite fallbacks, handles rows filtered to all `-inf`, validates probabilities before sampling, and calls `torch.topk` in the argmax path.

- [ ] **Step 4: Run safety tests to verify pass**

Run: `.\venv\Scripts\python.exe -m pytest tests\test_kronos_sampling_safety.py -q`

Expected: `3 passed`.

---

### Task 2: Edge Feature Engine

**Files:**
- Create: `scanner/edge/__init__.py`
- Create: `scanner/edge/features.py`
- Create: `scanner/tests/test_edge_features.py`

- [ ] **Step 1: Write failing feature tests**

Create tests that build a synthetic bullish Potter setup, call `extract_edge_features`, and assert deterministic keys:

```python
import pandas as pd

from edge.features import extract_edge_features
from strategy.empty_space import score_empty_space
from strategy.potter_box import detect_potter_box


def _bars():
    rows = []
    for i in range(40):
        if i < 24:
            rows.append([100, 104, 96, 100 + (0.2 if i % 2 == 0 else -0.2), 1000])
        elif i < 39:
            rows.append([100, 101, 99, 100 + (0.05 if i % 2 == 0 else -0.05), 1200])
        else:
            rows.append([101, 104, 100.5, 103, 2500])
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def test_extract_edge_features_has_stable_numeric_fields():
    bars = _bars()
    pb = detect_potter_box("TEST", bars)
    es = score_empty_space(bars, "bullish", pb.breakout_close, pb.cost_basis)

    features = extract_edge_features("TEST", bars, pb, es)

    assert features["ticker"] == "TEST"
    assert features["direction"] == "bullish"
    assert features["potter_passed"] == 1.0
    assert features["breakout_distance_pct"] > 0
    assert features["volume_expansion"] > 1
    assert "feature_version" in features
```

- [ ] **Step 2: Run feature tests to verify failure**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_features.py -q`

Expected: import failure because `edge.features` does not exist.

- [ ] **Step 3: Implement feature extraction**

Implement `extract_edge_features(ticker, bars, potter_box, empty_space=None, event_risk=None, options_contract=None, kronos=None, data_quality=None)` returning JSON-safe scalars with missing values defaulting conservatively.

- [ ] **Step 4: Run feature tests to verify pass**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_features.py -q`

Expected: `1 passed`.

---

### Task 3: Retrieval Index

**Files:**
- Create: `scanner/edge/retrieval.py`
- Create: `scanner/tests/test_edge_retrieval.py`

- [ ] **Step 1: Write failing retrieval tests**

Create tests for building an index, querying nearest analogs, and excluding records whose decision timestamp is too close to the query timestamp.

- [ ] **Step 2: Run retrieval tests to verify failure**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_retrieval.py -q`

Expected: import failure because `edge.retrieval` does not exist.

- [ ] **Step 3: Implement retrieval**

Implement:

- `EdgeRecord` dataclass.
- `build_edge_records_from_bars(ticker, bars, horizon=5)`.
- `find_analogs(query_features, records, k=7, embargo_days=5)`.
- `save_edge_index(records, path)` and `load_edge_index(path)`.

Use normalized Euclidean distance over numeric feature keys shared by query and candidates. Exclude same-ticker records inside the embargo window.

- [ ] **Step 4: Run retrieval tests to verify pass**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_retrieval.py -q`

Expected: retrieval tests pass.

---

### Task 4: Edge Scoring

**Files:**
- Create: `scanner/edge/scoring.py`
- Create: `scanner/tests/test_edge_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

Create tests that verify:

- Positive analog expectancy and Kronos agreement increase score.
- Negative expectancy reduces score.
- Weak data quality and low analog count apply penalties.
- Output includes a transparent `scorecard`.

- [ ] **Step 2: Run scoring tests to verify failure**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_scoring.py -q`

Expected: import failure because `edge.scoring` does not exist.

- [ ] **Step 3: Implement scoring**

Implement `score_edge_candidate(features, analogs, min_analogs=5)` returning:

```python
{
    "edge_score": float,
    "recommendation": "promote" | "research" | "reject",
    "scorecard": {...},
    "analog_summary": {...},
}
```

- [ ] **Step 4: Run scoring tests to verify pass**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_scoring.py -q`

Expected: scoring tests pass.

---

### Task 5: Validation Metrics

**Files:**
- Create: `scanner/edge/validation.py`
- Create: `scanner/tests/test_edge_validation.py`

- [ ] **Step 1: Write failing validation tests**

Create tests that pass ranked candidates with known outcomes and assert signal count, precision, recall, top-K precision, average return, and average R-multiple.

- [ ] **Step 2: Run validation tests to verify failure**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_validation.py -q`

Expected: import failure because `edge.validation` does not exist.

- [ ] **Step 3: Implement validation metrics**

Implement `compute_edge_validation_report(candidates, thresholds=(45, 55, 65), top_k=5, slippage_pct=0.0)`.

- [ ] **Step 4: Run validation tests to verify pass**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_validation.py -q`

Expected: validation tests pass.

---

### Task 6: CLI Edge Modes

**Files:**
- Modify: `scanner/config.py`
- Modify: `scanner/main.py`
- Create: `scanner/tests/test_edge_cli_units.py`

- [ ] **Step 1: Write failing CLI unit tests**

Write tests for pure helper functions that build edge scan payloads without network calls.

- [ ] **Step 2: Run CLI tests to verify failure**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_cli_units.py -q`

Expected: missing helper/mode failures.

- [ ] **Step 3: Add config and mode helpers**

Add config constants for edge index/report paths and thresholds. Add helpers in `main.py`:

- `_score_edge_for_bars(ticker, synthetic, index_records, logger)`
- `_write_edge_report(filename, payload)`
- `_write_edge_diagnostic(logger)`

Add modes:

- `build_retrieval_index`
- `edge_scan`
- `validate_edge`
- `diagnose_edge`

- [ ] **Step 4: Run CLI tests to verify pass**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests\test_edge_cli_units.py -q`

Expected: CLI unit tests pass.

---

### Task 7: Replay Dataset and Replay Edge Details

**Files:**
- Modify: `scanner/replay/sample_replay_dataset.json`
- Modify: `scanner/learning/replay_runner.py`
- Modify: `scanner/tests/test_kronos_adapter.py` only if existing expectations need import-safe changes.

- [ ] **Step 1: Write or update failing replay test**

Add a replay test that confirms the sample replay dataset has enough bars to run Potter detection and produces detail records with skip reasons.

- [ ] **Step 2: Run replay test to verify failure**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest tests -q`

Expected: replay-related failure before dataset update.

- [ ] **Step 3: Update replay sample and details**

Replace the too-short replay sample with at least 45 bars. Ensure replay details include `stage`, `reason`, `potter_passed`, and `edge_score` when available.

- [ ] **Step 4: Run scanner tests**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest -q`

Expected: scanner tests pass.

---

### Task 8: Environment and End-to-End Verification

**Files:**
- Modify only if needed: `requirements.txt`, `scanner/requirements-scanner.txt`, docs.

- [ ] **Step 1: Install root requirements if local venv is missing packages**

Run: `.\venv\Scripts\python.exe -m pip install -r requirements.txt`

Expected: required root test dependencies, including `matplotlib`, are present.

- [ ] **Step 2: Run root safety tests**

Run: `.\venv\Scripts\python.exe -m pytest tests\test_kronos_sampling_safety.py -q`

Expected: sampling safety tests pass.

- [ ] **Step 3: Run scanner tests**

Run: `cd scanner; ..\venv\Scripts\python.exe -m pytest -q`

Expected: scanner tests pass.

- [ ] **Step 4: Build retrieval index**

Run: `cd scanner; ..\venv\Scripts\python.exe main.py --mode build_retrieval_index`

Expected: report and index JSON files are written under `scanner/reports`.

- [ ] **Step 5: Run edge validation**

Run: `cd scanner; ..\venv\Scripts\python.exe main.py --mode validate_edge`

Expected: validation report JSON is written under `scanner/reports`.

- [ ] **Step 6: Run edge scan**

Run: `cd scanner; ..\venv\Scripts\python.exe main.py --mode edge_scan`

Expected: ranked candidate report JSON is written under `scanner/reports`; live alerts remain off unless existing live safeguards allow them.

- [ ] **Step 7: Run edge diagnostic**

Run: `cd scanner; ..\venv\Scripts\python.exe main.py --mode diagnose_edge`

Expected: diagnostic report explains candidate count, edge distribution, index availability, and current blockers.
