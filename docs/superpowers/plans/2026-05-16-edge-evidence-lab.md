# Edge Evidence Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible local evidence lab that records edge index, scan, validation, and diagnostic outputs as durable experiment artifacts.

**Architecture:** Keep the existing scanner modes and JSON reports intact, then add a small `scanner.evidence` package that writes typed row artifacts under `scanner/reports/evidence`. Use Parquet when an engine is installed and JSONL as a deterministic fallback so the feature works in the current Windows venv. Add `run_edge_lab` to execute index, validation, scan, and diagnosis in sequence with one shared experiment id.

**Tech Stack:** Python 3.12, pandas, pytest, optional pyarrow/fastparquet Parquet support, existing scanner CLI.

---

### Task 1: Evidence Store

**Files:**
- Create: `scanner/evidence/__init__.py`
- Create: `scanner/evidence/store.py`
- Create: `scanner/tests/test_evidence_store.py`
- Modify: `scanner/config.py`

- [x] Write tests that create an `EvidenceRun`, record candidate rows and metric rows, flush them, and assert `manifest.json`, `candidates.jsonl`, and `metrics.jsonl` exist with the expected run metadata.
- [x] Run `.\venv\Scripts\python.exe -m pytest scanner\tests\test_evidence_store.py -q` and confirm it fails with `ModuleNotFoundError: No module named 'scanner.evidence'`.
- [x] Implement `EvidenceRun` with `record_rows`, `record_metrics`, `log_artifact`, `flush`, and `start_evidence_run`.
- [x] Run the evidence store tests and confirm they pass.

### Task 2: Edge Mode Evidence Hooks

**Files:**
- Modify: `scanner/main.py`
- Create: `scanner/tests/test_edge_evidence_lab.py`

- [x] Write tests that call `run_build_retrieval_index`, `run_validate_edge`, `run_edge_scan`, and `run_diagnose_edge` with a temporary evidence directory and monkeypatched data fetchers.
- [x] Run the new tests and confirm they fail because the modes do not accept or write evidence runs yet.
- [x] Add optional `evidence_run` parameters to the four edge mode functions and write index rows, validation candidates, scan candidates, diagnostics, metrics, and JSON artifacts into the evidence run.
- [x] Add `run_edge_lab` to execute the four edge steps in order and flush one manifest.
- [x] Run the new tests and confirm they pass.

### Task 3: CLI And Docs

**Files:**
- Modify: `scanner/main.py`
- Modify: `scanner/README.md`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [x] Add `run_edge_lab` to CLI choices and dispatch.
- [x] Document the mode, the evidence directory, and the JSONL/Parquet fallback behavior.
- [x] Add optional Parquet engine dependencies without making the lab unusable when they are unavailable.
- [x] Run `.\venv\Scripts\python.exe -m pytest scanner\tests\test_evidence_store.py scanner\tests\test_edge_evidence_lab.py scanner\tests\test_edge_cli_units.py scanner\tests\test_package_entrypoint.py -q`.

### Task 4: Verification

**Files:**
- No new files expected.

- [x] Run `.\venv\Scripts\python.exe -m pytest -q`.
- [x] If the known Kronos regression test still fails, run the scanner/evidence test subset and report the exact remaining failure.
- [x] Run `git diff --stat` and inspect the changed files before final response.
