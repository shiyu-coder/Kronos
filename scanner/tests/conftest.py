import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_mutable_runtime_state(monkeypatch, tmp_path):
    # Tests must never mutate operator evidence, tuning state, or append-only
    # journals. These artifacts influence readiness and adaptive decisions;
    # a green verification run must not change the evidence it is verifying.
    report_dir = tmp_path / "reports"
    tuning_dir = tmp_path / "tuning"
    retirement_marker_path = tmp_path / "RETIRED.json"
    report_paths = {
        "EDGE_INDEX_PATH": report_dir / "edge_retrieval_index.json",
        "EDGE_SCAN_REPORT_PATH": report_dir / "edge_scan_report.json",
        "EDGE_VALIDATION_REPORT_PATH": report_dir / "edge_validation_report.json",
        "EDGE_DIAGNOSTIC_REPORT_PATH": report_dir / "edge_diagnostic_report.json",
        "EDGE_AUDIT_REPORT_PATH": report_dir / "edge_audit_report.json",
        "META_MODEL_PATH": report_dir / "meta_model.json",
    }
    monkeypatch.setattr("scanner.config.REPORT_DIR", report_dir)
    monkeypatch.setattr("scanner.config.EVIDENCE_DIR", report_dir / "evidence")
    monkeypatch.setattr("scanner.config.TUNING_DIR", tuning_dir)
    monkeypatch.setattr("scanner.config.RETIREMENT_MARKER_PATH", retirement_marker_path)
    monkeypatch.setattr("scanner.config.OVERRIDES_PATH", tuning_dir / "overrides.json")
    for name, path in report_paths.items():
        monkeypatch.setattr(f"scanner.config.{name}", path)

    monkeypatch.setattr("scanner.main.REPORT_DIR", report_dir)
    monkeypatch.setattr("scanner.main.EVIDENCE_DIR", report_dir / "evidence")
    monkeypatch.setattr("scanner.main.RETIREMENT_MARKER_PATH", retirement_marker_path)
    for name, path in report_paths.items():
        monkeypatch.setattr(f"scanner.main.{name}", path, raising=False)

    monkeypatch.setattr("scanner.learning.outcome_store.REPORT_DIR", report_dir)
    monkeypatch.setattr(
        "scanner.learning.outcome_store.DECISIONS_PATH",
        report_dir / "scan_decisions.jsonl",
    )
    monkeypatch.setattr(
        "scanner.learning.outcome_store.QUARANTINE_PATH",
        report_dir / "scan_decisions.quarantine.jsonl",
    )
    monkeypatch.setattr("scanner.learning.outcome_reviewer.REPORT_DIR", report_dir)
    monkeypatch.setattr("scanner.learning.replay_runner.REPORT_DIR", report_dir)
    monkeypatch.setattr("scanner.learning.trial_registry.REPORT_DIR", report_dir)
    monkeypatch.setattr(
        "scanner.learning.trial_registry.TRIAL_REGISTRY_PATH",
        report_dir / "trial_registry.jsonl",
    )
    monkeypatch.setattr("scanner.learning.adaptive_policy.TUNING_DIR", tuning_dir)
    monkeypatch.setattr("scanner.learning.adaptive_policy.OVERRIDES_PATH", tuning_dir / "overrides.json")
    monkeypatch.setattr("scanner.learning.autotuner.TUNING_DIR", tuning_dir)
    monkeypatch.setattr("scanner.learning.autotuner.OVERRIDES_PATH", tuning_dir / "overrides.json")
    monkeypatch.setattr("scanner.brief.REPORT_DIR", report_dir)
    monkeypatch.setattr("scanner.backtest.backtest_runner.REPORT_DIR", report_dir)

    # Live credentials in the shell would let tests reach real provider APIs
    # (e.g. the Tradier cross-source check). Tests that need creds set their
    # own fakes via monkeypatch.setenv.
    monkeypatch.delenv("TRADIER_API_TOKEN", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
