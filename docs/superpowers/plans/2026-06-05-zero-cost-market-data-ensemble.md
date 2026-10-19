# Zero-Cost Market Data Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve research data with free delayed SIP equities and an Alpaca-indicative/yfinance options ensemble while preserving fail-closed live behavior.

**Architecture:** Add explicit research-vs-current data intent to the existing market-data adapter, then enrich option selection by joining Alpaca snapshots to yfinance chains. Thread provenance and quality metadata into existing edge features and audits without changing promotion thresholds.

**Tech Stack:** Python 3.12, pandas, requests, yfinance, pytest, Alpaca Market Data REST API

---

### Task 1: Provider-Aware Research Bars

**Files:**
- Modify: `scanner/data/market_data.py`
- Modify: `scanner/main.py`
- Test: `scanner/tests/test_market_data.py`
- Test: `scanner/tests/test_edge_evidence_lab.py`

- [ ] Write failing tests proving research/history requests use delayed SIP and current scans use configured IEX.
- [ ] Run focused tests and confirm failures are caused by missing data-intent support.
- [ ] Add a small bar-result metadata mechanism and delayed-SIP request path.
- [ ] Thread research intent into retrieval-index and research-scan workflows.
- [ ] Run focused tests until green.

### Task 2: Hybrid Options Ensemble

**Files:**
- Modify: `scanner/data/options_data.py`
- Modify: `scanner/utils/validation.py`
- Test: `scanner/tests/test_options.py`

- [ ] Write failing tests for Alpaca indicative quote enrichment, yfinance open-interest joining, pagination, and safe fallback.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement Alpaca option snapshot retrieval and OCC-symbol joining.
- [ ] Extend option results with provider/feed/quote-age metadata.
- [ ] Run focused tests until green.

### Task 3: Edge Provenance and Conservative Quality

**Files:**
- Modify: `scanner/edge/features.py`
- Modify: `scanner/edge/scoring.py`
- Modify: `scanner/edge/audit.py`
- Test: `scanner/tests/test_edge_features.py`
- Test: `scanner/tests/test_edge_scoring.py`
- Test: `scanner/tests/test_edge_audit.py`

- [ ] Write failing tests proving provenance is recorded and indicative/stale options cannot improve live readiness.
- [ ] Run focused tests and confirm expected failures.
- [ ] Add stable provenance/quality features and conservative scoring penalties.
- [ ] Preserve existing audit thresholds and live fail-closed behavior.
- [ ] Run focused tests until green.

### Task 4: Documentation and Evidence Refresh

**Files:**
- Modify: `scanner/README.md`
- Modify: `scanner/.env.example`

- [ ] Document delayed SIP research behavior and option-ensemble limitations.
- [ ] Run the full test suite, compile check, dependency check, and doctor.
- [ ] Run `run_edge_lab` and inspect validation, scan, diagnostic, and audit reports.
- [ ] Run `research_scan`, then report the measured data-quality improvement and remaining blockers.
