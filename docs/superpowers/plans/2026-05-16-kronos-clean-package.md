# Kronos Clean Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current upgraded but messy worktree into a clean, packaged, committed project state.

**Architecture:** Keep source code and durable docs in Git, while treating scanner reports, logs, tuning overrides, caches, and local credentials as generated runtime state. Make the scanner executable as a Python package from the repo root so future runs do not depend on changing directories or fragile import paths.

**Tech Stack:** Python 3.12, pytest, setuptools via `pyproject.toml`, Windows batch launchers, Streamlit, scanner CLI.

---

### Task 1: Package Boundary And Imports

**Files:**
- Create: `scanner/__init__.py`
- Create: `scanner/tests/test_package_entrypoint.py`
- Modify: `scanner/main.py`
- Modify: scanner modules importing sibling packages
- Modify: scanner tests importing top-level scanner modules

- [x] Add a subprocess regression test proving `python -m scanner.main --help` works from the repo root.
- [x] Run the new test and confirm it fails with `ModuleNotFoundError`.
- [x] Convert scanner imports to package-relative imports.
- [x] Keep direct script compatibility by setting `__package__` in `scanner/main.py` when run as `python scanner/main.py`.
- [x] Run the new test and scanner suite.

### Task 2: Generated Artifact Boundaries

**Files:**
- Modify: `.gitignore`
- Modify: `scanner/README.md`

- [x] Ignore scanner logs, generated reports, local tuning overrides, and local credentials.
- [x] Keep `scanner/.env.example` and source tests tracked.
- [x] Update scanner docs to prefer root package execution.

### Task 3: Project Packaging Metadata

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`

- [x] Add minimal package metadata.
- [x] Include scanner/runtime dependencies already used by source.
- [x] Preserve current `pytest.ini` behavior.

### Task 4: Signal Calibration Verification

**Files:**
- No source changes expected unless a verified issue appears.

- [x] Run `build_retrieval_index`, `validate_edge`, `edge_scan`, `diagnose_edge`, and `dry_run`.
- [x] Treat generated reports as local evidence, not committed source.
- [x] Do not loosen thresholds solely to produce alerts.

### Task 5: Final Commit

**Files:**
- Stage only intentional source, tests, docs, and packaging files.

- [x] Run `python -m pytest`.
- [x] Run `git diff --cached --stat` and inspect staged scope.
- [x] Commit with a message describing scanner packaging and safety cleanup.
- [x] Confirm final status is clean except ignored runtime artifacts.
