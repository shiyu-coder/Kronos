import subprocess
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scanner.main as scanner_main


def test_scanner_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-m", "scanner.main", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "Potter Box Scanner V1" in result.stdout


def test_parse_args_accepts_research_ops(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scanner.main", "--mode", "research_ops"])

    args = scanner_main.parse_args()

    assert args.mode == "research_ops"


def test_load_env_reads_scanner_env_file(monkeypatch, tmp_path):
    root_env = tmp_path / ".env"
    scanner_env = tmp_path / "scanner.env"
    root_env.write_text("ALPACA_API_KEY=\nMARKET_DATA_PROVIDER=\n", encoding="utf-8")
    scanner_env.write_text(
        "ALPACA_API_KEY=test-key\n"
        "ALPACA_SECRET_KEY=test-secret\n"
        "MARKET_DATA_PROVIDER=alpaca\n"
        "ALPACA_FEED=iex\n",
        encoding="utf-8",
    )
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "MARKET_DATA_PROVIDER", "ALPACA_FEED"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(scanner_main, "ENV_PATHS", (root_env, scanner_env))

    env = scanner_main._load_env()

    assert env["alpaca_key"] == "test-key"
    assert env["alpaca_secret"] == "test-secret"
    assert env["market_data_provider"] == "alpaca"


def test_live_preflight_blocks_when_edge_audit_is_blocked(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "blocked",
                "blockers": ["validation_threshold_55_unsupported"],
                "warnings": ["no_current_actionable_candidates"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_requires_edge_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", tmp_path / "missing_audit.json")
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_research_only_audit(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    audit_path.write_text(
        json.dumps({"readiness": "research_only", "blockers": [], "warnings": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_stale_audit(monkeypatch, tmp_path):
    import os
    import time

    audit_path = tmp_path / "edge_audit_report.json"
    audit_path.write_text(
        json.dumps({"readiness": "paper_trade_only", "blockers": [], "warnings": []}),
        encoding="utf-8",
    )
    # Age the file two days: a stale verdict must not authorize live mode.
    two_days_ago = time.time() - 48 * 3600
    os.utime(audit_path, (two_days_ago, two_days_ago))
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_retired_scanner_before_other_live_gates(monkeypatch, tmp_path):
    retirement_path = tmp_path / "RETIRED.json"
    retirement_path.write_text(
        json.dumps({"status": "retired", "retired_at": "2026-08-20T11:00:00-05:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "RETIREMENT_MARKER_PATH", retirement_path, raising=False)
    audit_path = tmp_path / "edge_audit_report.json"
    completed_at = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "paper_trade_only",
                "blockers": [],
                "warnings": [],
                "evidence_provenance": {
                    "scan_completed_at": completed_at,
                    "validation_completed_at": completed_at,
                    "scan_run_id": "current-run",
                    "validation_run_id": "current-run",
                    "scan_runtime_fingerprint": "sha256:current-runtime",
                    "validation_runtime_fingerprint": "sha256:current-runtime",
                    "runtime_fingerprint": "sha256:current-runtime",
                },
                "summary": {
                    "promotable_directions": ["bullish"],
                    "execution_ready_promoted_candidates": ["TEST"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    monkeypatch.setattr(scanner_main, "_edge_runtime_fingerprint", lambda: "sha256:current-runtime")
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_fresh_file_rewritten_from_stale_evidence(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    stale_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "paper_trade_only",
                "blockers": [],
                "warnings": [],
                "evidence_provenance": {
                    "scan_completed_at": stale_at,
                    "validation_completed_at": stale_at,
                    "scan_run_id": "stale-run",
                    "validation_run_id": "stale-run",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_paper_audit_without_intrinsic_provenance(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    audit_path.write_text(
        json.dumps({"readiness": "paper_trade_only", "blockers": [], "warnings": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_mixed_evidence_lab_runs(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    completed_at = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "paper_trade_only",
                "blockers": [],
                "warnings": [],
                "evidence_provenance": {
                    "scan_completed_at": completed_at,
                    "validation_completed_at": completed_at,
                    "scan_run_id": "scan-run",
                    "validation_run_id": "validation-run",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_allows_paper_trade_only_audit(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    completed_at = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(scanner_main, "_edge_runtime_fingerprint", lambda: "sha256:current-runtime")
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "paper_trade_only",
                "blockers": [],
                "warnings": [],
                "evidence_provenance": {
                    "scan_completed_at": completed_at,
                    "validation_completed_at": completed_at,
                    "scan_run_id": "current-run",
                    "validation_run_id": "current-run",
                    "scan_runtime_fingerprint": "sha256:current-runtime",
                    "validation_runtime_fingerprint": "sha256:current-runtime",
                    "runtime_fingerprint": "sha256:current-runtime",
                },
                "summary": {
                    "promotable_directions": ["bullish"],
                    "execution_ready_promoted_candidates": ["TEST"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is True
    assert env["_live_promotable_directions"] == ("bullish",)


def test_live_preflight_blocks_audit_from_different_runtime_source(monkeypatch, tmp_path):
    """Fresh evidence from old scoring code must not authorize current code."""
    audit_path = tmp_path / "edge_audit_report.json"
    completed_at = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "paper_trade_only",
                "blockers": [],
                "warnings": [],
                "evidence_provenance": {
                    "scan_completed_at": completed_at,
                    "validation_completed_at": completed_at,
                    "scan_run_id": "current-run",
                    "validation_run_id": "current-run",
                    "scan_runtime_fingerprint": "sha256:previous-runtime",
                    "validation_runtime_fingerprint": "sha256:previous-runtime",
                    "runtime_fingerprint": "sha256:previous-runtime",
                },
                "summary": {
                    "promotable_directions": ["bullish"],
                    "execution_ready_promoted_candidates": ["TEST"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    monkeypatch.setattr(scanner_main, "_edge_runtime_fingerprint", lambda: "sha256:current-runtime")
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_preflight_blocks_paper_audit_without_candidate_authorization(monkeypatch, tmp_path):
    audit_path = tmp_path / "edge_audit_report.json"
    completed_at = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(
            {
                "readiness": "paper_trade_only",
                "blockers": [],
                "warnings": [],
                "evidence_provenance": {
                    "scan_completed_at": completed_at,
                    "validation_completed_at": completed_at,
                    "scan_run_id": "current-run",
                    "validation_run_id": "current-run",
                },
                "summary": {
                    "promotable_directions": [],
                    "execution_ready_promoted_candidates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner_main, "EDGE_AUDIT_REPORT_PATH", audit_path)
    env = {
        "market_data_provider": "auto",
        "alpaca_key": "key",
        "alpaca_secret": "secret",
        "telegram_token": "token",
        "telegram_chat_id": "chat",
        "live_mode_enabled": True,
        "minimax_api_key": "",
    }

    assert scanner_main._preflight_checks("live", env, scanner_main.setup_logging(tmp_path)) is False


def test_live_candidate_authorization_rescores_exact_setup(monkeypatch):
    monkeypatch.setattr(scanner_main, "load_edge_index", lambda path: [object()])
    scored = {
        "direction": "bullish",
        "recommendation": "reject",
        "edge_score": 44.0,
        "blocking_reasons": ["edge_score_below_promotion_threshold"],
        "rejection_reasons": ["edge_score_below_promotion_threshold"],
    }
    monkeypatch.setattr(scanner_main, "_score_edge_for_bars", lambda *args, **kwargs: scored)
    execution_ready = {"value": False}
    monkeypatch.setattr(
        scanner_main,
        "candidate_execution_ready",
        lambda candidate: execution_ready["value"],
    )

    result = scanner_main._authorize_live_candidate(
        "TEST",
        "bullish",
        object(),
        object(),
        {"_live_promotable_directions": ("bullish",)},
        logger=None,
    )

    assert result["authorized"] is False
    assert result["edge_recommendation"] == "reject"

    scored["recommendation"] = "promote"
    scored["edge_score"] = 72.0
    scored["blocking_reasons"] = []
    scored["rejection_reasons"] = []
    result = scanner_main._authorize_live_candidate(
        "TEST",
        "bullish",
        object(),
        object(),
        {"_live_promotable_directions": ("bullish",)},
        logger=None,
    )

    assert result["authorized"] is False
    assert result["reason"] == "edge_execution_quality_not_ready"

    execution_ready["value"] = True
    result = scanner_main._authorize_live_candidate(
        "TEST",
        "bullish",
        object(),
        object(),
        {"_live_promotable_directions": ("bullish",)},
        logger=None,
    )

    assert result["authorized"] is True
    assert result["edge_recommendation"] == "promote"
