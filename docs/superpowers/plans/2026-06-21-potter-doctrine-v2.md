# Potter Doctrine V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a research-only Potter Doctrine v2 layer that captures multi-stage Potter-style setup mechanics without weakening live gates.

**Architecture:** Keep the existing strict Potter Box detector intact. Add a separate `scanner.strategy.potter_doctrine` module that scores box stack, punchback/retest, cost-basis lifecycle, and empty-space target quality from existing bar data and Potter/Empty Space outputs. Feed these features into edge retrieval/scoring and research logs so the system can learn which v2 mechanics actually improve outcomes before any promotion.

**Tech Stack:** Python 3.12, pandas, pytest, existing scanner edge/research pipeline.

---

### Task 1: Doctrine Detector

**Files:**
- Create: `scanner/strategy/potter_doctrine.py`
- Test: `scanner/tests/test_potter_doctrine.py`

- [ ] **Step 1: Write failing tests**

```python
def test_doctrine_scores_punchback_reclaim_after_breakout():
    bars = make_box_break_retest_reclaim()
    pb = detect_potter_box("TEST", bars)
    doctrine = score_potter_doctrine_v2("TEST", bars, pb, None)
    assert doctrine["punchback_state"] == "reclaim"
    assert doctrine["cost_basis_state"] == "held"
    assert doctrine["score"] >= 70
```

```python
def test_doctrine_rejects_failed_punchback_back_inside_box():
    bars = make_failed_retest()
    pb = detect_potter_box("TEST", bars)
    doctrine = score_potter_doctrine_v2("TEST", bars, pb, None)
    assert doctrine["punchback_state"] == "failed_reentry"
    assert doctrine["passed"] is False
```

- [ ] **Step 2: Verify tests fail**

Run: `.\venv\Scripts\python.exe -m pytest scanner\tests\test_potter_doctrine.py -q`
Expected: import/function missing failure.

- [ ] **Step 3: Implement minimal detector**

Implement:
- `score_potter_doctrine_v2(ticker, bars, potter_box, empty_space=None) -> dict`
- recent breakout direction inference from Potter Box or close vs control levels
- punchback state from last 3 bars: `fresh_breakout`, `reclaim`, `failed_reentry`, `inside`
- cost basis state: `held`, `lost`, `reclaimed`, `unknown`
- box stack proxy from rolling 15/30/60 bar ranges
- score and reasons, with `passed` for research only.

- [ ] **Step 4: Verify tests pass**

Run: `.\venv\Scripts\python.exe -m pytest scanner\tests\test_potter_doctrine.py -q`
Expected: all tests pass.

### Task 2: Edge Feature Integration

**Files:**
- Modify: `scanner/edge/features.py`
- Modify: `scanner/edge/retrieval.py`
- Modify: `scanner/main.py`
- Test: `scanner/tests/test_edge_features.py`
- Test: `scanner/tests/test_edge_retrieval.py`

- [ ] **Step 1: Write failing tests**

Assert extracted features include:
- `doctrine_v2_score`
- `doctrine_v2_passed`
- `punchback_state`
- `cost_basis_state`
- `box_stack_score`

- [ ] **Step 2: Verify tests fail**

Run: `.\venv\Scripts\python.exe -m pytest scanner\tests\test_edge_features.py scanner\tests\test_edge_retrieval.py -q`
Expected: missing keys/assertions fail.

- [ ] **Step 3: Wire doctrine into feature extraction**

Add optional `doctrine_v2` argument to `extract_edge_features`. Compute doctrine in edge retrieval and current edge scan before extracting features. Preserve compatibility when omitted.

- [ ] **Step 4: Verify tests pass**

Run: `.\venv\Scripts\python.exe -m pytest scanner\tests\test_edge_features.py scanner\tests\test_edge_retrieval.py -q`
Expected: all tests pass.

### Task 3: Research Scan Logging

**Files:**
- Modify: `scanner/main.py`
- Test: `scanner/tests/test_package_entrypoint.py` or new focused test if needed

- [ ] **Step 1: Add research payload checks**

Ensure dry-run/research decisions include `doctrine_v2_score`, `doctrine_v2_diagnostics`, and `doctrine_v2_passed` when Potter Box fails.

- [ ] **Step 2: Implement logging fields**

Include doctrine fields in counterfactual decision records. Do not send Telegram alerts from v2.

- [ ] **Step 3: Verify**

Run focused tests plus `.\venv\Scripts\python.exe -m scanner.main --mode diagnose_zero_results`.

### Task 4: Scoring and Guardrails

**Files:**
- Modify: `scanner/edge/scoring.py`
- Test: `scanner/tests/test_edge_scoring.py`

- [ ] **Step 1: Write tests**

Assert v2 score can raise research rank only when setup gate/retest evidence exists, and cannot promote when options data quality is below execution grade.

- [ ] **Step 2: Implement guarded scorecard addition**

Add `doctrine_v2` scorecard component capped to research influence. Promotion remains blocked by existing options/data quality gates.

- [ ] **Step 3: Verify**

Run `.\venv\Scripts\python.exe -m pytest scanner\tests\test_edge_scoring.py -q`.

### Task 5: Reports and Verification

**Files:**
- Modify: `scanner/README.md`

- [ ] **Step 1: Document v2**

Add a short section explaining that v2 is research-only, logs punchback/cost-basis/multi-timeframe proxy features, and must pass edge validation before live use.

- [ ] **Step 2: Run verification**

Run:
- `.\venv\Scripts\python.exe -m pytest -q`
- `.\venv\Scripts\python.exe -m compileall scanner model tests`
- `.\venv\Scripts\python.exe -m pip check`
- `.\venv\Scripts\python.exe -m scanner.main --mode doctor`
- `.\venv\Scripts\python.exe -m scanner.main --mode diagnose_zero_results`
- `.\venv\Scripts\python.exe -m scanner.main --mode run_edge_lab`

Expected: tests/build/doctor pass. Edge lab may remain blocked; if so, report blockers honestly.
