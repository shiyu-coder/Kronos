"""Potter Box Scanner V1 entrypoint."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "scanner"

from . import config as scanner_config
from .alerts.telegram import render_alert_message, send_telegram_message
from .ai.minimax_adapter import MiniMaxAdapter
from .backtest.backtest_runner import run_daily_proxy_2y_backtest, run_intraday_60d_backtest
from .brief import run_brief
from .config import (
    CALIBRATION_MIN_MATCHED_ROWS,
    CALIBRATION_PASS_AVG_ABS_MAX,
    CALIBRATION_WARN_AVG_ABS_MAX,
    ALPACA_FEED,
    DRY_RUN_DEFAULT,
    EDGE_ANALOG_DIRECTION_MATCH,
    EDGE_ANALOG_K,
    EDGE_AUDIT_REPORT_PATH,
    EDGE_BARS_ADJUSTMENT,
    EDGE_COST_BPS_PER_SIDE,
    EDGE_CROSS_TICKER_EMBARGO_DAYS,
    EDGE_DIAGNOSTIC_REPORT_PATH,
    EDGE_EMBARGO_DAYS,
    EDGE_INDEX_EXTRA_UNIVERSE,
    EVIDENCE_DIR,
    EDGE_INDEX_PATH,
    EDGE_MIN_ANALOGS,
    EDGE_SCAN_REPORT_PATH,
    EDGE_VALIDATION_MAX_RECORDS,
    EDGE_VALIDATION_REPORT_PATH,
    EDGE_VALIDATION_THRESHOLDS,
    EDGE_VALIDATION_TOP_K,
    LOG_DIR,
    MARKET_DATA_PROVIDER_DEFAULT,
    META_MODEL_PATH,
    PRED_DAYS,
    REPORT_DIR,
    RETIREMENT_MARKER_PATH,
    SYNTHETIC_SESSION_ANCHOR_HOUR,
    SYNTHETIC_SESSION_ANCHOR_MINUTE,
    TIMEZONE,
)
from .data.bar_contract import check_ohlcv_contract, check_session_completeness
from .data.cross_check import cross_check_daily_bars
from .data.events import assess_event_risk
from .data.market_data import (
    drop_in_progress_daily_bar,
    drop_vendor_placeholder_bars,
    fetch_daily_bars,
    fetch_intraday_bars,
    validate_ticker,
)
from .data.options_data import select_options_contract
from .doctor import run_doctor
from .data.synthetic_sessions import build_synthetic_sessions
from .edge.audit import candidate_execution_ready, compute_edge_audit_report
from .edge.calibration import (
    META_MODEL_VERSION as _META_MODEL_VERSION_CURRENT,
    predict_expected_r,
    predict_score,
    select_shippable_objective,
    walk_forward_calibration_suite,
)
from .edge.features import extract_edge_features
from .edge.retrieval import (
    EdgeAnalogIndex,
    EdgeRecord,
    build_edge_records_from_bars,
    find_analogs,
    load_edge_index,
    save_edge_index,
    select_recent_records,
)
from .edge.scoring import score_edge_candidate
from .edge.validation import compute_edge_validation_report
from .evidence.provenance import edge_runtime_fingerprint as _edge_runtime_fingerprint
from .evidence.store import EvidenceRun, start_evidence_run
from .learning.adaptive_policy import apply_adaptive_overrides, build_adaptive_policy_report
from .learning.autotuner import apply_overrides, propose_overrides
from .learning.outcome_reviewer import review_pending_outcomes
from .learning.outcome_store import DECISIONS_PATH, append_decision, deduplicate_decisions, load_decisions, save_decisions
from .learning.replay_runner import run_replay_eval
from .learning.trial_registry import load_trials, record_trial
from .models.kronos_adapter import KronosAdapter
from .strategy.empty_space import score_empty_space
from .strategy.potter_box import detect_potter_box, score_potter_research_candidate
from .strategy.potter_doctrine import score_potter_doctrine_v2
from .tickers import WATCHLIST
from .utils.atomic_io import atomic_write_json
from .utils.logging_setup import setup_logging
from .utils.validation import AlertCandidate

ENV_PATHS = (
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parent / ".env",
)


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _monotonic_seconds() -> float:
    return time.monotonic()


def _elapsed_seconds(start: float, end: float | None = None) -> float:
    current = _monotonic_seconds() if end is None else end
    return round(current - start, 3)


def _run_timed_stage(name: str, logger, func):
    started_at = _utc_now_iso()
    start = _monotonic_seconds()
    logger.info("STAGE_START: %s", name)
    try:
        result = func()
    except Exception:
        logger.exception("STAGE_FAILED: %s duration_seconds=%.3f", name, _elapsed_seconds(start))
        raise
    completed_at = _utc_now_iso()
    duration = _elapsed_seconds(start)
    logger.info("STAGE_DONE: %s duration_seconds=%.3f", name, duration)
    return result, {
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration,
    }


def _resolve_calibrated_anchor(ticker: str) -> tuple[int, int]:
    summary_path = REPORT_DIR / "calibration_summary.json"
    if not summary_path.exists():
        return SYNTHETIC_SESSION_ANCHOR_HOUR, SYNTHETIC_SESSION_ANCHOR_MINUTE
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("ticker", "").upper() != ticker.upper():
                continue
            if row.get("quality_status") == "fail":
                break
            anchor = str(row.get("best_anchor", "")).strip()
            if ":" not in anchor:
                break
            hour_str, minute_str = anchor.split(":", 1)
            return int(hour_str), int(minute_str)
    except Exception:
        return SYNTHETIC_SESSION_ANCHOR_HOUR, SYNTHETIC_SESSION_ANCHOR_MINUTE
    return SYNTHETIC_SESSION_ANCHOR_HOUR, SYNTHETIC_SESSION_ANCHOR_MINUTE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Potter Box Scanner V1")
    parser.add_argument(
        "--mode",
        choices=[
            "dry_run",
            "live",
            "backtest_intraday_60d",
            "backtest_daily_proxy_2y",
            "calibration",
            "test_telegram",
            "test_minimax",
            "review_outcomes",
            "autotune",
            "adaptive_policy",
            "replay_eval",
            "research_scan",
            "diagnose_zero_results",
            "build_retrieval_index",
            "edge_scan",
            "validate_edge",
            "diagnose_edge",
            "audit_edge",
            "run_edge_lab",
            "research_ops",
            "brief",
            "doctor",
        ],
        default="dry_run" if DRY_RUN_DEFAULT else "live",
    )
    parser.add_argument("--tradingview_csv", default=None, help="TradingView CSV export path for calibration mode")
    parser.add_argument("--ticker", default="PLTR", help="Ticker for calibration mode")
    parser.add_argument("--test_message", default=None, help="Optional custom message for test_telegram mode")
    parser.add_argument("--replay_dataset", default=None, help="Path to replay dataset JSON for replay_eval mode")
    parser.add_argument("--apply_tuning", action="store_true", help="Apply proposed overrides in autotune mode")
    parser.add_argument(
        "--calibration_csv_glob",
        default=None,
        help="Glob pattern for batch calibration CSVs (e.g., C:\\Users\\...\\BATS_*.csv)",
    )
    parser.add_argument(
        "--sweep_anchors",
        action="store_true",
        help="When calibrating, test multiple session anchors and keep the best mismatch result.",
    )
    return parser.parse_args()


def _load_project_env_files() -> None:
    for path in ENV_PATHS:
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value is None or not str(value).strip():
                continue
            if os.getenv(key, "").strip():
                continue
            os.environ[key] = str(value).strip()


def _load_env() -> dict:
    _load_project_env_files()
    return {
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        "heartbeat_enabled": os.getenv("HEARTBEAT_ENABLED", "false").lower() == "true",
        "live_mode_enabled": os.getenv("LIVE_MODE_ENABLED", "false").lower() == "true",
        "kronos_live_gate_enabled": os.getenv("KRONOS_LIVE_GATE_ENABLED", "false").lower() == "true",
        "alpaca_key": os.getenv("ALPACA_API_KEY", "").strip(),
        "alpaca_secret": os.getenv("ALPACA_SECRET_KEY", "").strip(),
        "market_data_provider": os.getenv("MARKET_DATA_PROVIDER", "auto").strip().lower(),
        "minimax_enabled": os.getenv("MINIMAX_ENABLED", "false").strip().lower() == "true",
        "minimax_api_key": os.getenv("MINIMAX_API_KEY", "").strip(),
    }


def _log_skip(logger, ticker: str, reason: str):
    logger.info("SKIP: %s %s", ticker, reason)


def _data_provenance(bars: pd.DataFrame | None) -> dict:
    attrs = getattr(bars, "attrs", {}) if bars is not None else {}
    latest_source_ts = attrs.get("latest_source_timestamp")
    if not latest_source_ts and bars is not None and not bars.empty:
        latest_source_ts = pd.Timestamp(bars.index[-1]).isoformat()
    payload = {
        "data_provider": attrs.get("data_provider"),
        "data_feed": attrs.get("data_feed"),
        "data_delay_minutes": int(attrs.get("data_delay_minutes", 0) or 0),
    }
    if latest_source_ts:
        payload["latest_source_timestamp"] = str(latest_source_ts)
        try:
            payload["source_session_date"] = pd.Timestamp(latest_source_ts).date().isoformat()
        except Exception:
            pass
    return payload


def _doctrine_record_fields(doctrine: dict | None) -> dict:
    doctrine = doctrine if isinstance(doctrine, dict) else {}
    return {
        "doctrine_v2_version": doctrine.get("version"),
        "doctrine_v2_score": doctrine.get("score"),
        "doctrine_v2_passed": bool(doctrine.get("passed")),
        "doctrine_v2_punchback_state": doctrine.get("punchback_state"),
        "doctrine_v2_cost_basis_state": doctrine.get("cost_basis_state"),
        "doctrine_v2_risk_flags": doctrine.get("risk_flags", []),
        "doctrine_v2_diagnostics": doctrine,
    }


def _kronos_result_fields(result) -> dict:
    """Keep usable Kronos evidence without turning inference failures into disagreement."""
    if result.directional_agreement is None:
        return {"kronos_eval_error": str(result.skip_reason or "Kronos forecast unavailable")}
    return {
        "kronos_directional_agreement": result.directional_agreement,
        "kronos_median_forecast_return_pct": result.median_forecast_return_pct,
        "kronos_worst_sampled_return_pct": result.worst_sampled_return_pct,
        "kronos_passed": bool(result.passed),
    }


def _kronos_research_fields(kronos: KronosAdapter, ticker: str, synthetic: pd.DataFrame, direction, logger) -> dict:
    """Kronos evaluation for a research candidate, best-effort.

    Research candidates are the only decisions that resolve regularly, so
    they are where Kronos agreement can earn (or lose) evidence-based trust.
    """
    if not scanner_config.KRONOS_RESEARCH_ENABLED or direction not in {"bullish", "bearish"}:
        return {}
    try:
        kr = kronos.evaluate(ticker, synthetic, direction)
    except Exception as exc:
        logger.warning("KRONOS_RESEARCH_EVAL_FAILED: %s %s", ticker, exc)
        return {"kronos_eval_error": str(exc)}
    if kr.directional_agreement is None:
        # Model load/inference failure, not a real disagreement - journaling
        # kronos_passed=False here would poison the lift measurement. The
        # error string keeps a persistent model failure visible in the
        # journal instead of looking like normal early accumulation.
        logger.warning("KRONOS_RESEARCH_EVAL_FAILED: %s %s", ticker, kr.skip_reason)
    return _kronos_result_fields(kr)


def _infer_direction_for_counterfactual(pb) -> str | None:
    if pb.direction in {"bullish", "bearish"}:
        return pb.direction
    if pb.breakout_close is None:
        return None
    if pb.cost_basis is not None:
        return "bullish" if float(pb.breakout_close) >= float(pb.cost_basis) else "bearish"
    if pb.box_top is not None and pb.box_bottom is not None:
        mid = (float(pb.box_top) + float(pb.box_bottom)) / 2.0
        return "bullish" if float(pb.breakout_close) >= mid else "bearish"
    return None


def _outcome_status(direction: str | None, entry_price: float | None) -> str:
    return "pending" if direction in {"bullish", "bearish"} and entry_price is not None else "not_applicable"


def _write_zero_result_diagnostic(logger) -> dict:
    from collections import Counter

    rows = load_decisions()
    strict_rows = [r for r in rows if r.get("mode") in {"dry_run", "live"}]
    status_counts = Counter(r.get("outcome_status", "unknown") for r in rows)
    stage_counts = Counter(str(r.get("stage_failed") or "none") for r in rows)
    final_pass_counts = Counter("pass" if r.get("final_pass") else "fail" for r in rows)
    strict_stage_counts = Counter(str(r.get("stage_failed") or "none") for r in strict_rows)
    resolved = [r for r in rows if r.get("outcome_status") == "resolved"]
    labels = Counter(r.get("outcome_label", "unknown") for r in resolved)
    stage_labels = Counter(f"{r.get('stage_failed') or 'none'}:{r.get('outcome_label')}" for r in resolved)
    missed_winners = [
        r for r in resolved if not r.get("final_pass") and r.get("outcome_label") == "win"
    ]
    correct_skips = [
        r for r in resolved if not r.get("final_pass") and r.get("outcome_label") == "loss"
    ]

    def research_diagnostics(row: dict) -> dict:
        diagnostics = row.get("research_diagnostics")
        return diagnostics if isinstance(diagnostics, dict) else {}

    def is_research_candidate(row: dict) -> bool:
        diagnostics = research_diagnostics(row)
        return bool(diagnostics.get("passed")) or row.get("skip_reason") == "research_candidate"

    def score_bucket(score) -> str:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return "missing"
        if value < 45:
            return "below_45"
        if value < 62:
            return "45_to_61"
        if value < 70:
            return "62_to_69"
        return "70_plus"

    research_candidates = [r for r in rows if is_research_candidate(r)]
    resolved_research_candidates = [r for r in research_candidates if r.get("outcome_status") == "resolved"]
    research_candidate_labels = Counter(r.get("outcome_label", "unknown") for r in resolved_research_candidates)
    research_candidate_returns = []
    for row in resolved_research_candidates:
        try:
            research_candidate_returns.append(float(row.get("outcome_ret_5bar_pct")))
        except (TypeError, ValueError):
            continue
    research_candidate_resolved_count = sum(research_candidate_labels.values())
    research_candidate_win_rate = (
        research_candidate_labels.get("win", 0) / research_candidate_resolved_count
        if research_candidate_resolved_count
        else None
    )
    research_award_counts = Counter()
    for row in research_candidates:
        reasons = research_diagnostics(row).get("reasons", [])
        if not isinstance(reasons, list):
            continue
        research_award_counts.update(str(reason) for reason in reasons)
    potter_research_reason_counts = Counter(
        str(research_diagnostics(row).get("reason") or "missing")
        for row in strict_rows
        if row.get("stage_failed") == "potter_box"
    )
    score_buckets = Counter(score_bucket(row.get("research_score")) for row in rows)

    bottleneck_counts = strict_stage_counts if strict_stage_counts else stage_counts
    primary_bottleneck_stage = bottleneck_counts.most_common(1)[0][0] if bottleneck_counts else "none"
    if not resolved_research_candidates:
        research_edge_status = "insufficient_resolved"
    elif research_candidate_labels.get("loss", 0) > research_candidate_labels.get("win", 0):
        research_edge_status = "loss_heavy"
    elif research_candidate_labels.get("win", 0) > research_candidate_labels.get("loss", 0):
        research_edge_status = "positive"
    else:
        research_edge_status = "mixed"
    if research_edge_status == "positive":
        live_gate_action = "review_edge_validation_before_live"
    elif research_candidates:
        live_gate_action = "do_not_loosen_without_validated_edge"
    else:
        live_gate_action = "continue_research_scan"

    payload = {
        "mode": "diagnose_zero_results",
        "total_records": len(rows),
        "outcome_status_counts": dict(status_counts),
        "final_pass_counts": dict(final_pass_counts),
        "resolved_label_counts": dict(labels),
        "stage_counts": dict(stage_counts.most_common()),
        "resolved_stage_label_counts": dict(stage_labels.most_common()),
        "missed_winners": len(missed_winners),
        "correct_skips": len(correct_skips),
        "strict_path": {
            "modes": ["dry_run", "live"],
            "records": len(strict_rows),
            "final_pass_counts": dict(Counter("pass" if r.get("final_pass") else "fail" for r in strict_rows)),
            "stage_counts": dict(strict_stage_counts.most_common()),
        },
        "research_candidates": {
            "records": len(research_candidates),
            "resolved": research_candidate_resolved_count,
            "pending": sum(1 for r in research_candidates if r.get("outcome_status") == "pending"),
            "not_applicable": sum(1 for r in research_candidates if r.get("outcome_status") == "not_applicable"),
            "resolved_outcomes": dict(research_candidate_labels),
            "resolved_win_rate": research_candidate_win_rate,
            "average_outcome_ret_5bar_pct": (
                round(sum(research_candidate_returns) / len(research_candidate_returns), 4)
                if research_candidate_returns
                else None
            ),
        },
        "research_score_buckets": dict(score_buckets),
        "potter_research_reason_counts": dict(potter_research_reason_counts.most_common()),
        "research_award_counts": dict(research_award_counts.most_common()),
        "diagnostic_summary": {
            "primary_bottleneck_stage": primary_bottleneck_stage,
            "research_edge_status": research_edge_status,
            "recommended_live_gate_action": live_gate_action,
        },
        "diagnosis": (
            "potter_box gate is the primary bottleneck; use research_scan to collect graded candidates"
            if stage_counts.get("potter_box", 0) or stage_counts.get("potter_box_research", 0)
            else "insufficient bottleneck evidence"
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "zero_result_diagnostic.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("ZERO_RESULT_DIAGNOSTIC: %s", json.dumps(payload))
    logger.info("Zero-result diagnostic saved: %s", str(path.resolve()))
    return payload


def _preflight_checks(mode: str, env: dict, logger) -> bool:
    if mode == "live" and RETIREMENT_MARKER_PATH.is_file():
        logger.error(
            "Preflight failed: the Potter Box scanner is retired and live mode is permanently "
            "disabled by %s. See docs/RETIREMENT-2026-08-20.md for the evidence and explicit "
            "revival criteria.",
            RETIREMENT_MARKER_PATH,
        )
        return False
    provider = env["market_data_provider"]
    if provider == "alpaca" and (not env["alpaca_key"] or not env["alpaca_secret"]):
        logger.error(
            "Preflight failed: MARKET_DATA_PROVIDER=alpaca requires both ALPACA_API_KEY "
            "and ALPACA_SECRET_KEY. The dashboard Key/Key ID alone is not enough; "
            "regenerate the paper API key and copy the Secret Key into scanner/.env."
        )
        return False
    if mode in {"live", "test_telegram"} and (not env["telegram_token"] or not env["telegram_chat_id"]):
        logger.error("Preflight failed: Telegram token/chat id missing for mode=%s.", mode)
        return False
    if mode == "live" and not env["live_mode_enabled"]:
        logger.error("Preflight failed: LIVE_MODE_ENABLED must be true for live mode.")
        return False
    if mode == "live":
        try:
            audit = json.loads(EDGE_AUDIT_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.error(
                "Preflight failed: live mode requires a current edge audit. "
                "Run --mode run_edge_lab and inspect %s first.",
                EDGE_AUDIT_REPORT_PATH,
            )
            return False
        # A stale audit must not authorize live mode: readiness reflects the
        # evidence as of the last lab run, and the edge can degrade between
        # runs. 24h covers the daily research_ops cadence with slack. The file
        # age is only a first check; audit_edge can legitimately rewrite a
        # report from older source artifacts, so their intrinsic timestamps
        # are checked below as well.
        audit_age_hours = (
            pd.Timestamp.now(tz="UTC")
            - pd.Timestamp(EDGE_AUDIT_REPORT_PATH.stat().st_mtime, unit="s", tz="UTC")
        ).total_seconds() / 3600.0
        if audit_age_hours < -(5.0 / 60.0) or audit_age_hours > 24.0:
            logger.error(
                "Preflight failed: edge audit is %.1f hours old (max 24). "
                "Run --mode run_edge_lab for a current readiness verdict.",
                audit_age_hours,
            )
            return False
        readiness = str(audit.get("readiness", "")).strip().lower()
        if readiness != "paper_trade_only":
            logger.error(
                "Preflight failed: edge audit readiness=%s is not live-eligible. blockers=%s warnings=%s",
                readiness or "missing",
                audit.get("blockers", []),
                audit.get("warnings", []),
            )
            return False
        provenance = audit.get("evidence_provenance")
        if not isinstance(provenance, dict):
            logger.error(
                "Preflight failed: edge audit lacks intrinsic evidence provenance. "
                "Run --mode run_edge_lab; rewriting audit_edge alone is not sufficient."
            )
            return False
        scan_run_id = str(provenance.get("scan_run_id") or "").strip()
        validation_run_id = str(provenance.get("validation_run_id") or "").strip()
        if not scan_run_id or scan_run_id != validation_run_id:
            logger.error(
                "Preflight failed: scan and validation do not come from the same evidence-lab run. "
                "scan_run_id=%s validation_run_id=%s",
                scan_run_id or "missing",
                validation_run_id or "missing",
            )
            return False
        audit_runtime_fingerprint = str(provenance.get("runtime_fingerprint") or "").strip()
        scan_runtime_fingerprint = str(provenance.get("scan_runtime_fingerprint") or "").strip()
        validation_runtime_fingerprint = str(
            provenance.get("validation_runtime_fingerprint") or ""
        ).strip()
        try:
            current_runtime_fingerprint = _edge_runtime_fingerprint()
        except Exception as exc:
            logger.error("Preflight failed: could not fingerprint the current runtime: %s", exc)
            return False
        if (
            not audit_runtime_fingerprint
            or audit_runtime_fingerprint != scan_runtime_fingerprint
            or audit_runtime_fingerprint != validation_runtime_fingerprint
            or audit_runtime_fingerprint != current_runtime_fingerprint
        ):
            logger.error(
                "Preflight failed: Edge evidence was not produced by the current runtime source. "
                "Run --mode run_edge_lab after every production code change."
            )
            return False
        now_utc = pd.Timestamp.now(tz="UTC")
        for source_name in ("scan", "validation"):
            field = f"{source_name}_completed_at"
            try:
                completed_at = pd.Timestamp(provenance.get(field))
                if pd.isna(completed_at):
                    raise ValueError("timestamp is NaT")
                if completed_at.tzinfo is None:
                    completed_at = completed_at.tz_localize("UTC")
                else:
                    completed_at = completed_at.tz_convert("UTC")
            except Exception:
                logger.error(
                    "Preflight failed: edge audit has no valid %s evidence timestamp. "
                    "Run --mode run_edge_lab.",
                    source_name,
                )
                return False
            evidence_age_hours = (now_utc - completed_at).total_seconds() / 3600.0
            if evidence_age_hours < -(5.0 / 60.0) or evidence_age_hours > 24.0:
                logger.error(
                    "Preflight failed: %s evidence is %.1f hours old (max 24). "
                    "Run --mode run_edge_lab for current evidence.",
                    source_name,
                    evidence_age_hours,
                )
                return False
        summary = audit.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        direction_values = summary.get("promotable_directions", [])
        direction_values = direction_values if isinstance(direction_values, (list, tuple, set)) else []
        promotable_directions = tuple(
            direction
            for direction in direction_values
            if direction in {"bullish", "bearish"}
        )
        candidate_values = summary.get("execution_ready_promoted_candidates", [])
        candidate_values = candidate_values if isinstance(candidate_values, (list, tuple, set)) else []
        execution_ready_candidates = [
            str(ticker).strip().upper()
            for ticker in candidate_values
            if str(ticker).strip()
        ]
        if not promotable_directions or not execution_ready_candidates:
            logger.error(
                "Preflight failed: paper-trade audit lacks a promotable direction or an "
                "execution-ready promoted candidate. Run --mode run_edge_lab."
            )
            return False
        # The live candidate is rescored from its fresh bars/options immediately
        # before alerting. Carry only the validation-supported directions out of
        # preflight; the stale audit candidate itself is not blindly reused.
        env["_live_promotable_directions"] = promotable_directions
    if mode == "test_minimax" and not env["minimax_api_key"]:
        logger.error("Preflight failed: MINIMAX_API_KEY missing for test_minimax mode.")
        return False
    return True


def run_telegram_test(env: dict, logger, custom_message: str | None = None) -> bool:
    message = custom_message or (
        "Potter Scanner Test Signal\n\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Provider: {env['market_data_provider']}\n"
        "Status: Telegram integration is active."
    )
    ok = send_telegram_message(env["telegram_token"], env["telegram_chat_id"], message, logger)
    if ok:
        logger.info("PASS: test Telegram message sent successfully.")
    else:
        logger.error("FAIL: test Telegram message failed.")
    return ok


def run_minimax_test(env: dict, logger, custom_message: str | None = None) -> bool:
    adapter = MiniMaxAdapter(logger)
    adapter.enabled = True
    payload = {
        "ticker": "TEST",
        "direction": "bullish",
        "breakout_strength_pct": 1.2,
        "empty_space_score": 2,
        "rr_ratio": 1.8,
        "kronos_directional_agreement": 0.72,
        "notes": custom_message or "MiniMax API connectivity test from Potter scanner runtime.",
    }
    result = adapter.score_setup(payload)
    logger.info("MINIMAX_TEST_RESULT: %s", json.dumps(result))
    return result.get("status", "").startswith("ok")


def _authorize_live_candidate(
    ticker: str,
    direction: str,
    synthetic: pd.DataFrame,
    options_contract,
    env: dict,
    logger,
) -> dict:
    """Fail closed unless this exact, freshly scanned setup earns promotion.

    The readiness audit proves the global validation route and records which
    directions are statistically promotable. It cannot authorize a different
    live alert merely because some candidate in the earlier Edge scan passed.
    Re-score the current bars with the current options contract, then require
    Edge promotion, direction support, and capital-grade execution evidence.
    """
    direction_values = env.get("_live_promotable_directions", ())
    direction_values = direction_values if isinstance(direction_values, (list, tuple, set)) else ()
    promotable_directions = {
        str(value).strip().lower()
        for value in direction_values
        if str(value).strip()
    }
    if direction not in promotable_directions:
        return {
            "authorized": False,
            "reason": "edge_direction_not_promotable",
            "edge_score": None,
            "edge_recommendation": None,
            "blocking_reasons": ["direction_evidence_unsupported"],
        }

    try:
        records = load_edge_index(EDGE_INDEX_PATH)
        if not records:
            raise ValueError("edge retrieval index is empty")
        scored = _score_edge_for_bars(
            ticker,
            synthetic,
            records,
            logger,
            options_selector=lambda *_args, **_kwargs: options_contract,
        )
    except Exception as exc:
        logger.error("LIVE_EDGE_AUTHORIZATION_ERROR: %s %s", ticker, exc)
        return {
            "authorized": False,
            "reason": "edge_authorization_unavailable",
            "edge_score": None,
            "edge_recommendation": None,
            "blocking_reasons": ["edge_authorization_unavailable"],
        }

    edge_direction = str(scored.get("direction") or "").strip().lower()
    recommendation = str(scored.get("recommendation") or "").strip().lower()
    blocking_reasons = list(scored.get("blocking_reasons") or [])
    result = {
        "authorized": False,
        "reason": "edge_recommendation_not_promote",
        "edge_score": scored.get("edge_score"),
        "edge_recommendation": recommendation or None,
        "blocking_reasons": blocking_reasons,
        "rejection_reasons": list(scored.get("rejection_reasons") or []),
    }
    if edge_direction != direction:
        result["reason"] = "edge_direction_mismatch"
        result["blocking_reasons"] = [*blocking_reasons, "edge_direction_mismatch"]
        return result
    if recommendation != "promote":
        return result
    if not candidate_execution_ready(scored):
        result["reason"] = "edge_execution_quality_not_ready"
        result["blocking_reasons"] = [*blocking_reasons, "execution_quality_not_ready"]
        return result

    result["authorized"] = True
    result["reason"] = "edge_candidate_authorized"
    return result


def _run_single_ticker(ticker: str, mode: str, env: dict, kronos: KronosAdapter, minimax: MiniMaxAdapter, logger) -> dict:
    base_record = {
        "ticker": ticker,
        "mode": mode,
        "decision_ts": pd.Timestamp.now(tz="UTC").tz_convert("America/New_York").isoformat(),
        "final_pass": False,
    }
    validation = validate_ticker(ticker, logger)
    if validation.skip_reason:
        rec = {
            **base_record,
            "outcome_status": "not_applicable",
            "stage_failed": "validation",
            "skip_reason": validation.skip_reason,
            "counterfactual": False,
        }
        append_decision(rec)
        _log_skip(logger, ticker, validation.skip_reason)
        return {"ticker": ticker, "status": "skip", "reason": validation.skip_reason}

    try:
        anchor_hour, anchor_minute = _resolve_calibrated_anchor(ticker)
        intraday = fetch_intraday_bars(ticker, research=mode == "research_scan")
        synthetic, synth_diag = build_synthetic_sessions(
            intraday_df=intraday,
            session_anchor_hour=anchor_hour,
            session_anchor_minute=anchor_minute,
            source_interval="30m",
            prepost_enabled=True,
        )
        base_record.update(_data_provenance(intraday))
    except Exception as exc:
        rec = {
            **base_record,
            "outcome_status": "not_applicable",
            "stage_failed": "market_data",
            "skip_reason": str(exc),
            "counterfactual": False,
        }
        append_decision(rec)
        _log_skip(logger, ticker, f"market/synthetic data failure: {exc}")
        return {"ticker": ticker, "status": "skip", "reason": str(exc)}

    pb = detect_potter_box(ticker, synthetic)
    if mode == "research_scan":
        research = score_potter_research_candidate(pb, synthetic)
        doctrine = score_potter_doctrine_v2(ticker, synthetic, pb, None)
        kronos_fields = (
            _kronos_research_fields(kronos, ticker, synthetic, research.get("direction"), logger)
            if research.get("passed")
            else {}
        )
        rec = {
            **base_record,
            **_doctrine_record_fields(doctrine),
            **kronos_fields,
            "direction": research.get("direction"),
            "entry_price": research.get("entry_price"),
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": _outcome_status(research.get("direction"), research.get("entry_price"))
            if research.get("passed")
            else "not_applicable",
            "stage_failed": "potter_box_research",
            "skip_reason": research.get("reason"),
            "counterfactual": bool(research.get("passed")),
            "research_score": research.get("score"),
            "research_diagnostics": research,
        }
        append_decision(rec)
        if research.get("passed"):
            logger.info("RESEARCH_CANDIDATE: %s score=%s direction=%s", ticker, research.get("score"), research.get("direction"))
            return {"ticker": ticker, "status": "pass", "reason": "research_candidate"}
        _log_skip(logger, ticker, f"research score {research.get('score')} below threshold")
        return {"ticker": ticker, "status": "skip", "reason": "research_score_below_threshold"}

    if not pb.passed or pb.direction is None:
        research = score_potter_research_candidate(pb, synthetic)
        doctrine = score_potter_doctrine_v2(ticker, synthetic, pb, None)
        if research.get("passed"):
            counter_direction = research.get("direction")
            counter_entry = research.get("entry_price")
            outcome_status = _outcome_status(counter_direction, counter_entry)
            counterfactual = True
        else:
            counter_direction = None
            counter_entry = None
            outcome_status = "not_applicable"
            counterfactual = False
        kronos_fields = (
            _kronos_research_fields(kronos, ticker, synthetic, counter_direction, logger) if counterfactual else {}
        )
        rec = {
            **base_record,
            **_doctrine_record_fields(doctrine),
            **kronos_fields,
            "direction": counter_direction,
            "entry_price": counter_entry,
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": outcome_status,
            "stage_failed": "potter_box",
            "skip_reason": pb.skip_reason or "potter_box_failed",
            "counterfactual": counterfactual,
            "research_score": research.get("score"),
            "research_diagnostics": research,
        }
        append_decision(rec)
        _log_skip(logger, ticker, pb.skip_reason or "no Potter Box breakout")
        return {"ticker": ticker, "status": "skip", "reason": pb.skip_reason or "potter_box_failed"}

    es = score_empty_space(synthetic, pb.direction, pb.breakout_close, pb.cost_basis)
    doctrine = score_potter_doctrine_v2(ticker, synthetic, pb, es)
    if not es.passed:
        rec = {
            **base_record,
            **_doctrine_record_fields(doctrine),
            "stage_failed": "empty_space",
            "direction": pb.direction,
            "entry_price": pb.breakout_close,
            "skip_reason": es.skip_reason or "empty_space_failed",
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": _outcome_status(pb.direction, pb.breakout_close),
            "counterfactual": True,
        }
        append_decision(rec)
        _log_skip(logger, ticker, es.skip_reason or "empty space failed")
        return {"ticker": ticker, "status": "skip", "reason": es.skip_reason or "empty_space_failed"}

    ev = assess_event_risk(ticker, logger)
    if not ev.passed:
        rec = {
            **base_record,
            **_doctrine_record_fields(doctrine),
            "stage_failed": "event_risk",
            "direction": pb.direction,
            "entry_price": pb.breakout_close,
            "skip_reason": ev.skip_reason or "event_risk_failed",
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": _outcome_status(pb.direction, pb.breakout_close),
            "counterfactual": True,
        }
        append_decision(rec)
        _log_skip(logger, ticker, ev.skip_reason or "event risk")
        return {"ticker": ticker, "status": "skip", "reason": ev.skip_reason or "event_risk_failed"}

    opt = select_options_contract(ticker, pb.direction, pb.breakout_close, logger)
    if not opt.passed:
        rec = {
            **base_record,
            **_doctrine_record_fields(doctrine),
            "stage_failed": "options",
            "direction": pb.direction,
            "entry_price": pb.breakout_close,
            "skip_reason": opt.skip_reason or "options_failed",
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": _outcome_status(pb.direction, pb.breakout_close),
            "counterfactual": True,
        }
        append_decision(rec)
        _log_skip(logger, ticker, opt.skip_reason or "options liquidity failed")
        return {"ticker": ticker, "status": "skip", "reason": opt.skip_reason or "options_failed"}

    live_edge_authorization = {}
    if mode == "live":
        live_edge_authorization = _authorize_live_candidate(
            ticker,
            pb.direction,
            synthetic,
            opt,
            env,
            logger,
        )
        if not live_edge_authorization.get("authorized"):
            reason = str(live_edge_authorization.get("reason") or "edge_candidate_not_authorized")
            append_decision(
                {
                    **base_record,
                    **_doctrine_record_fields(doctrine),
                    "stage_failed": "edge_promotion",
                    "direction": pb.direction,
                    "entry_price": pb.breakout_close,
                    "anchor_hour": anchor_hour,
                    "anchor_minute": anchor_minute,
                    "outcome_status": _outcome_status(pb.direction, pb.breakout_close),
                    "counterfactual": True,
                    "skip_reason": reason,
                    "edge_score": live_edge_authorization.get("edge_score"),
                    "edge_recommendation": live_edge_authorization.get("edge_recommendation"),
                    "edge_blocking_reasons": live_edge_authorization.get("blocking_reasons", []),
                    "edge_rejection_reasons": live_edge_authorization.get("rejection_reasons", []),
                }
            )
            _log_skip(logger, ticker, reason)
            return {"ticker": ticker, "status": "skip", "reason": reason}

    kr = kronos.evaluate(ticker, synthetic, pb.direction)
    kronos_fields = _kronos_result_fields(kr)
    if env.get("kronos_live_gate_enabled", False) and not kr.passed:
        rec = {
            **base_record,
            **_doctrine_record_fields(doctrine),
            **kronos_fields,
            "stage_failed": "kronos",
            "direction": pb.direction,
            "entry_price": pb.breakout_close,
            "skip_reason": kr.skip_reason or "kronos_failed",
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": _outcome_status(pb.direction, pb.breakout_close),
            "counterfactual": True,
        }
        append_decision(rec)
        _log_skip(logger, ticker, kr.skip_reason or "kronos confirmation failed")
        return {"ticker": ticker, "status": "skip", "reason": kr.skip_reason or "kronos_failed"}
    if not kr.passed:
        logger.info(
            "KRONOS_ADVISORY_ONLY: %s agreement=%s reason=%s",
            ticker,
            kr.directional_agreement,
            kr.skip_reason,
        )

    ai_insight = minimax.score_setup(
        {
            "ticker": ticker,
            "direction": pb.direction,
            "box_top": pb.box_top,
            "box_bottom": pb.box_bottom,
            "cost_basis": pb.cost_basis,
            "breakout_close": pb.breakout_close,
            "breakout_strength_pct": pb.breakout_strength_pct,
            "empty_space_score": es.score,
            "rr_ratio": es.rr_ratio,
            "nearest_target": es.nearest_target,
            "risk_pct": es.risk_pct,
            "kronos_directional_agreement": kr.directional_agreement,
            "kronos_median_forecast_return_pct": kr.median_forecast_return_pct,
            "kronos_worst_sampled_return_pct": kr.worst_sampled_return_pct,
            "event_status": ev.status,
            "options_spread_pct": opt.spread_pct,
            "options_open_interest": opt.open_interest,
        }
    )
    if ai_insight.get("status") == "error":
        logger.warning("MiniMax scoring error for %s: %s", ticker, ai_insight.get("rationale"))

    candidate = AlertCandidate(
        ticker=ticker,
        direction=pb.direction,
        potter_box=pb,
        empty_space=es,
        event_risk=ev,
        options_contract=opt,
        kronos=kr,
        final_decision="pass",
        timestamp=datetime.now().isoformat(),
        ai_insight=ai_insight,
    )
    append_decision(
        {
            **base_record,
            **_doctrine_record_fields(doctrine),
            **kronos_fields,
            "final_pass": True,
            "direction": pb.direction,
            "entry_price": pb.breakout_close,
            "anchor_hour": anchor_hour,
            "anchor_minute": anchor_minute,
            "outcome_status": _outcome_status(pb.direction, pb.breakout_close),
            "stage_failed": None,
            "skip_reason": None,
            "counterfactual": False,
            "edge_score": live_edge_authorization.get("edge_score"),
            "edge_recommendation": live_edge_authorization.get("edge_recommendation"),
            "edge_blocking_reasons": live_edge_authorization.get("blocking_reasons", []),
        }
    )

    message = render_alert_message(candidate)

    if mode == "dry_run":
        logger.info("DRY_RUN_ALERT_PREVIEW:\n%s", message)
        logger.info(
            "PASS: %s breakout detected, Empty Space score %s, options passed, Kronos %.1f%% agreement",
            ticker,
            es.score,
            (kr.directional_agreement or 0.0) * 100.0,
        )
        return {"ticker": ticker, "status": "pass", "reason": "dry_run_preview"}

    if not env["live_mode_enabled"]:
        _log_skip(logger, ticker, "live mode blocked; set LIVE_MODE_ENABLED=true")
        return {"ticker": ticker, "status": "skip", "reason": "live_mode_not_enabled"}

    if not env["telegram_token"] or not env["telegram_chat_id"]:
        _log_skip(logger, ticker, "missing Telegram token/chat id")
        return {"ticker": ticker, "status": "skip", "reason": "missing_telegram_credentials"}

    sent = send_telegram_message(env["telegram_token"], env["telegram_chat_id"], message, logger)
    if sent:
        logger.info("PASS: %s live alert sent", ticker)
        return {"ticker": ticker, "status": "pass", "reason": "live_alert_sent"}
    return {"ticker": ticker, "status": "skip", "reason": "telegram_send_failed"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    remap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"timestamp", "time", "datetime", "date"}:
            remap[c] = "timestamp"
        elif cl in {"open", "o"}:
            remap[c] = "Open"
        elif cl in {"high", "h"}:
            remap[c] = "High"
        elif cl in {"low", "l"}:
            remap[c] = "Low"
        elif cl in {"close", "c"}:
            remap[c] = "Close"
        elif cl in {"volume", "v"}:
            remap[c] = "Volume"
    out = df.rename(columns=remap)
    if "timestamp" in out.columns:
        ts_raw = out["timestamp"]
        if pd.api.types.is_numeric_dtype(ts_raw):
            out["timestamp"] = pd.to_datetime(ts_raw, errors="coerce", unit="s", utc=True)
        else:
            out["timestamp"] = pd.to_datetime(ts_raw, errors="coerce", utc=True)
        out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        out.index = out.index.tz_convert("America/New_York")
    return out


def _infer_ticker_from_filename(csv_path: str) -> str:
    name = Path(csv_path).name
    match = re.search(r"BATS_([^,]+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    stem = Path(csv_path).stem
    cleaned = re.sub(r"[^A-Za-z]", "", stem).upper()
    return cleaned[-5:] if cleaned else "UNKNOWN"


def _calc_mismatch_and_merge(synthetic: pd.DataFrame, tv: pd.DataFrame) -> tuple[dict, int]:
    syn_ohlc = synthetic[["Open", "High", "Low", "Close"]].copy()
    tv_ohlc = tv[["Open", "High", "Low", "Close"]].copy()

    # Align by calendar trading date (daily exports commonly use different timestamp anchors).
    syn_ohlc["trade_date"] = syn_ohlc.index.tz_convert("America/New_York").date
    tv_ohlc["trade_date"] = tv_ohlc.index.tz_convert("America/New_York").date
    merged = syn_ohlc.merge(
        tv_ohlc,
        on="trade_date",
        how="inner",
        suffixes=("_synthetic", "_tv"),
    ).dropna()

    mismatches = {}
    for col in ("Open", "High", "Low", "Close"):
        delta = (merged[f"{col}_synthetic"] - merged[f"{col}_tv"]).abs()
        mismatches[col.lower()] = {
            "avg_abs_mismatch": float(delta.mean()) if not delta.empty else None,
            "max_abs_mismatch": float(delta.max()) if not delta.empty else None,
        }
    return mismatches, int(len(merged))


def _classify_calibration(mismatch: dict, matched_rows: int) -> dict:
    avg_values = [
        mismatch.get(field, {}).get("avg_abs_mismatch")
        for field in ("open", "high", "low", "close")
        if mismatch.get(field, {}).get("avg_abs_mismatch") is not None
    ]
    max_values = [
        mismatch.get(field, {}).get("max_abs_mismatch")
        for field in ("open", "high", "low", "close")
        if mismatch.get(field, {}).get("max_abs_mismatch") is not None
    ]
    overall_avg = float(sum(avg_values) / len(avg_values)) if avg_values else None
    overall_max = float(max(max_values)) if max_values else None

    if matched_rows < CALIBRATION_MIN_MATCHED_ROWS:
        status = "fail"
        reason = f"matched_rows {matched_rows} < {CALIBRATION_MIN_MATCHED_ROWS}"
    elif overall_avg is None:
        status = "fail"
        reason = "no overlap to compare"
    elif overall_avg <= CALIBRATION_PASS_AVG_ABS_MAX:
        status = "pass"
        reason = f"overall_avg_abs_mismatch {overall_avg:.4f} <= {CALIBRATION_PASS_AVG_ABS_MAX:.4f}"
    elif overall_avg <= CALIBRATION_WARN_AVG_ABS_MAX:
        status = "warn"
        reason = (
            f"overall_avg_abs_mismatch {overall_avg:.4f} between "
            f"{CALIBRATION_PASS_AVG_ABS_MAX:.4f} and {CALIBRATION_WARN_AVG_ABS_MAX:.4f}"
        )
    else:
        status = "fail"
        reason = f"overall_avg_abs_mismatch {overall_avg:.4f} > {CALIBRATION_WARN_AVG_ABS_MAX:.4f}"

    return {
        "status": status,
        "reason": reason,
        "overall_avg_abs_mismatch": overall_avg,
        "overall_max_abs_mismatch": overall_max,
    }


def _run_single_anchor_calibration(
    ticker: str,
    tv: pd.DataFrame,
    intraday: pd.DataFrame,
    anchor_hour: int,
    anchor_minute: int,
) -> dict:
    synthetic, diagnostics = build_synthetic_sessions(
        intraday,
        session_anchor_hour=anchor_hour,
        session_anchor_minute=anchor_minute,
        source_interval="30m",
        prepost_enabled=True,
    )
    mismatch, matched_rows = _calc_mismatch_and_merge(synthetic, tv)
    quality = _classify_calibration(mismatch, matched_rows)
    return {
        "anchor": f"{anchor_hour:02d}:{anchor_minute:02d}",
        "synthetic_diagnostics": diagnostics,
        "mismatch": mismatch,
        "matched_rows": matched_rows,
        "quality_gate": quality,
    }


def run_calibration(
    ticker: str,
    tradingview_csv: str | None,
    logger,
    sweep_anchors: bool = False,
) -> dict:
    intraday = fetch_intraday_bars(ticker)
    payload = {
        "mode": "calibration",
        "ticker": ticker,
        "status": "requires_tradingview_csv" if not tradingview_csv else "calculated",
    }

    if not tradingview_csv:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORT_DIR / f"calibration_{ticker}.json"
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("CALIBRATION_REPORT: %s", json.dumps(payload))
        logger.info("Calibration report saved: %s", str(report.resolve()))
        return payload

    tv = _normalize_columns(pd.read_csv(tradingview_csv))
    if sweep_anchors:
        candidates = []
        for hour in range(16, 23):
            candidates.append(_run_single_anchor_calibration(ticker, tv, intraday, hour, 0))

        best = min(
            candidates,
            key=lambda x: (
                x["quality_gate"]["overall_avg_abs_mismatch"]
                if x["quality_gate"]["overall_avg_abs_mismatch"] is not None
                else 1e9
            ),
        )
        payload.update(
            {
                "status": "calculated_sweep",
                "best_anchor": best["anchor"],
                "best_result": best,
                "all_anchor_results": candidates,
            }
        )
    else:
        result = _run_single_anchor_calibration(
            ticker=ticker,
            tv=tv,
            intraday=intraday,
            anchor_hour=SYNTHETIC_SESSION_ANCHOR_HOUR,
            anchor_minute=SYNTHETIC_SESSION_ANCHOR_MINUTE,
        )
        payload.update(
            {
                "anchor": result["anchor"],
                "synthetic_diagnostics": result["synthetic_diagnostics"],
                "mismatch": result["mismatch"],
                "matched_rows": result["matched_rows"],
                "quality_gate": result["quality_gate"],
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / f"calibration_{ticker}.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("CALIBRATION_REPORT: %s", json.dumps(payload))
    logger.info("Calibration report saved: %s", str(report.resolve()))
    return payload


def run_batch_calibration(csv_glob: str, logger, sweep_anchors: bool = False) -> dict:
    csv_files = sorted(glob.glob(csv_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched: {csv_glob}")

    per_ticker = []
    for csv_path in csv_files:
        ticker = _infer_ticker_from_filename(csv_path)
        result = run_calibration(ticker=ticker, tradingview_csv=csv_path, logger=logger, sweep_anchors=sweep_anchors)
        gate = result.get("quality_gate", {})
        if sweep_anchors:
            gate = result.get("best_result", {}).get("quality_gate", {})
        per_ticker.append(
            {
                "ticker": ticker,
                "csv": csv_path,
                "quality_status": gate.get("status", "unknown"),
                "quality_reason": gate.get("reason"),
                "overall_avg_abs_mismatch": gate.get("overall_avg_abs_mismatch"),
                "best_anchor": result.get("best_anchor", result.get("anchor")),
                "report_file": str((REPORT_DIR / f"calibration_{ticker}.json").resolve()),
            }
        )

    summary = {
        "mode": "calibration_batch",
        "sweep_anchors": sweep_anchors,
        "csv_glob": csv_glob,
        "total": len(per_ticker),
        "pass": sum(1 for r in per_ticker if r["quality_status"] == "pass"),
        "warn": sum(1 for r in per_ticker if r["quality_status"] == "warn"),
        "fail": sum(1 for r in per_ticker if r["quality_status"] == "fail"),
        "results": per_ticker,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("CALIBRATION_BATCH_SUMMARY: %s", json.dumps(summary))
    logger.info("Calibration batch summary saved: %s", str(summary_path.resolve()))
    return summary



def _write_edge_report(path: Path, payload: dict, logger=None) -> dict:
    # Atomic: the audit report written here authorizes live mode; a torn
    # write must never leave a half-report behind.
    atomic_write_json(path, payload)
    if logger is not None:
        logger.info("EDGE_REPORT_SAVED: %s", str(path.resolve()))
    return payload


def _scanner_tz_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(TIMEZONE)
    return ts.tz_convert(TIMEZONE)


def _equity_market_elapsed_minutes(start, end) -> int:
    start_ts = _scanner_tz_timestamp(start)
    end_ts = _scanner_tz_timestamp(end)
    if end_ts <= start_ts:
        return 0

    total_minutes = 0
    day = start_ts.normalize()
    end_day = end_ts.normalize()
    while day <= end_day:
        if day.dayofweek < 5:
            market_open = day + pd.Timedelta(hours=9, minutes=30)
            market_close = day + pd.Timedelta(hours=16)
            window_start = max(start_ts, market_open)
            window_end = min(end_ts, market_close)
            if window_end > window_start:
                total_minutes += int((window_end - window_start).total_seconds() // 60)
        day += pd.Timedelta(days=1)
    return total_minutes


def _edge_data_quality(
    bars: pd.DataFrame,
    *,
    provider: str | None = None,
    alpaca_feed: str | None = None,
    alpaca_credentials_available: bool | None = None,
    now: pd.Timestamp | None = None,
) -> dict:
    attrs = getattr(bars, "attrs", {}) if bars is not None else {}
    provider_choice = (
        provider or attrs.get("data_provider") or os.getenv("MARKET_DATA_PROVIDER", MARKET_DATA_PROVIDER_DEFAULT)
    ).strip().lower()
    feed = (alpaca_feed or attrs.get("data_feed") or os.getenv("ALPACA_FEED", ALPACA_FEED)).strip().lower() or "iex"
    has_alpaca_credentials = (
        bool(os.getenv("ALPACA_API_KEY", "").strip() and os.getenv("ALPACA_SECRET_KEY", "").strip())
        if alpaca_credentials_available is None
        else bool(alpaca_credentials_available)
    )

    effective_provider = provider_choice
    if provider_choice == "auto":
        effective_provider = "alpaca" if has_alpaca_credentials else "yfinance"

    if effective_provider == "alpaca" and has_alpaca_credentials:
        feed_confidence = 0.9 if feed == "sip" else 0.7
    elif effective_provider == "alpaca":
        feed_confidence = 0.35
    else:
        feed_confidence = 0.5

    missing_bars = 0
    stale_minutes = 0
    quality_score = 1.0
    if bars is None or bars.empty:
        missing_bars = 1
        quality_score = 0.0
    else:
        required_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in bars.columns]
        if required_cols:
            missing_bars = int(bars[required_cols].isna().any(axis=1).sum())
        latest_source_ts = attrs.get("latest_source_timestamp")
        latest_ts = pd.Timestamp(latest_source_ts) if latest_source_ts else pd.Timestamp(bars.index[-1])
        now_ts = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
        stale_minutes = _equity_market_elapsed_minutes(latest_ts, now_ts)
        if missing_bars:
            quality_score = max(0.0, 1.0 - min(missing_bars / max(len(bars), 1), 1.0))
        stale_grace_minutes = 30 + int(attrs.get("data_delay_minutes", 0) or 0)
        if stale_minutes > stale_grace_minutes:
            stale_penalty = min((stale_minutes - stale_grace_minutes) / (24 * 60), 1.0)
            quality_score = min(quality_score, max(0.0, 1.0 - stale_penalty))

    return {
        "quality_score": round(quality_score, 4),
        "feed_confidence": feed_confidence,
        "missing_bars": missing_bars,
        "stale_minutes": stale_minutes,
        "provider": effective_provider,
        "feed": feed if effective_provider == "alpaca" else None,
        "delay_minutes": int(attrs.get("data_delay_minutes", 0) or 0),
    }


def _score_edge_for_bars(
    ticker: str,
    synthetic: pd.DataFrame,
    index_records: list[EdgeRecord] | EdgeAnalogIndex,
    logger,
    options_selector=None,
) -> dict:
    if options_selector is None:
        options_selector = select_options_contract
    pb = detect_potter_box(ticker, synthetic)
    research = score_potter_research_candidate(pb, synthetic)
    direction = pb.direction if pb.direction in {"bullish", "bearish"} else research.get("direction")
    entry = pb.breakout_close
    if direction not in {"bullish", "bearish"} or entry is None:
        return {
            "ticker": ticker,
            "status": "skip",
            "reason": research.get("reason") or pb.skip_reason or "no edge candidate direction",
            "potter_passed": bool(pb.passed),
            "research_score": research.get("score", 0),
        }

    es = score_empty_space(synthetic, direction, float(entry), pb.cost_basis or float(entry))
    doctrine = score_potter_doctrine_v2(ticker, synthetic, pb, es)
    options_contract = options_selector(ticker, direction, float(entry), logger)
    features = extract_edge_features(
        ticker=ticker,
        bars=synthetic,
        potter_box=pb,
        empty_space=es,
        doctrine_v2=doctrine,
        options_contract=options_contract,
        data_quality=_edge_data_quality(synthetic),
    )
    features["direction"] = direction
    features["research_score"] = research.get("score", 0)
    features["research_passed"] = 1.0 if research.get("passed") else 0.0
    analogs = find_analogs(
        features,
        index_records,
        k=EDGE_ANALOG_K,
        embargo_days=EDGE_EMBARGO_DAYS,
        direction_match=EDGE_ANALOG_DIRECTION_MATCH,
    )
    scoring = score_edge_candidate(features, analogs, min_analogs=EDGE_MIN_ANALOGS)
    return {
        "ticker": ticker,
        "status": "candidate",
        "direction": direction,
        "entry_price": float(entry),
        "edge_score": scoring["edge_score"],
        "recommendation": scoring["recommendation"],
        "scorecard": scoring["scorecard"],
        "analog_summary": scoring["analog_summary"],
        "gross_analog_summary": scoring["gross_analog_summary"],
        "analog_cost_model": scoring["analog_cost_model"],
        "blocking_reasons": scoring.get("blocking_reasons", []),
        "rejection_reasons": scoring.get("rejection_reasons", []),
        "analog_count": len(analogs),
        "top_analogs": analogs[:5],
        "features": features,
        "doctrine_v2": doctrine,
        "potter_passed": bool(pb.passed),
        "empty_space_passed": bool(es.passed),
        "research_score": research.get("score", 0),
        "skip_reason": None
        if scoring["recommendation"] != "reject"
        else ", ".join(scoring.get("rejection_reasons", [])) or "edge score below promotion/research threshold",
    }


def _build_edge_diagnostic_payload(
    index_records: list[EdgeRecord],
    validation_report: dict | None,
    scan_report: dict | None,
) -> dict:
    from collections import Counter

    candidates = (scan_report or {}).get("candidates", [])
    recommendations = Counter(row.get("recommendation", row.get("status", "unknown")) for row in candidates)
    rejection_reasons = Counter(
        str(reason)
        for row in candidates
        if row.get("recommendation") == "reject"
        for reason in row.get("rejection_reasons", [])
    )
    blocking_reasons = Counter(
        str(reason)
        for row in candidates
        for reason in row.get("blocking_reasons", [])
    )
    scores = [float(row.get("edge_score", 0.0)) for row in candidates if row.get("edge_score") is not None]
    return {
        "mode": "diagnose_edge",
        "index_records": len(index_records),
        "validation_samples": int((validation_report or {}).get("samples", 0)),
        "candidate_count": len(candidates),
        "recommendation_counts": dict(recommendations),
        "rejection_reason_counts": dict(rejection_reasons.most_common()),
        "blocking_reason_counts": dict(blocking_reasons.most_common()),
        "max_edge_score": max(scores) if scores else 0.0,
        "avg_edge_score": sum(scores) / len(scores) if scores else 0.0,
        "index_available": bool(index_records),
        "latest_validation_thresholds": (validation_report or {}).get("thresholds", {}),
        "diagnosis": (
            "edge index missing; run build_retrieval_index"
            if not index_records
            else "edge scan missing; run edge_scan"
            if not candidates
            else "promoted candidates available for review"
            if recommendations.get("promote", 0)
            else "research candidates available; promotion blocked by score, uncertainty, or evidence"
            if recommendations.get("research", 0)
            else "no edge candidates passed current research scoring"
        ),
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _numeric_metrics(payload: dict, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            metrics[metric_key] = 1.0 if value else 0.0
        elif isinstance(value, int | float):
            metrics[metric_key] = float(value)
        elif isinstance(value, dict):
            metrics.update(_numeric_metrics(value, metric_key))
    return metrics


def _record_report_artifact(evidence_run: EvidenceRun | None, path: Path) -> None:
    if evidence_run is not None:
        evidence_run.log_artifact(path)


def run_build_retrieval_index(watchlist: list[str], logger, evidence_run: EvidenceRun | None = None) -> dict:
    # The index universe is wider than the live watchlist: more cross-
    # sectional history means more honest walk-forward samples per lab run.
    index_universe = list(dict.fromkeys([*watchlist, *EDGE_INDEX_EXTRA_UNIVERSE]))
    records: list[EdgeRecord] = []
    errors: dict[str, str] = {}
    contract_warnings: dict[str, list[str]] = {}
    partial_bars_dropped: dict[str, str] = {}
    placeholder_bars_dropped: dict[str, list[str]] = {}
    cross_checks: dict[str, dict] = {}
    for ticker in index_universe:
        try:
            daily = fetch_daily_bars(ticker, research=True, adjustment=EDGE_BARS_ADJUSTMENT)
            daily = drop_in_progress_daily_bar(daily)
            dropped_ts = daily.attrs.get("dropped_in_progress_bar")
            if dropped_ts:
                partial_bars_dropped[ticker] = str(dropped_ts)
            daily = drop_vendor_placeholder_bars(daily)
            dropped_placeholders = daily.attrs.get("dropped_placeholder_bars")
            if dropped_placeholders:
                placeholder_bars_dropped[ticker] = list(dropped_placeholders)
                logger.warning(
                    "EDGE_INDEX_PLACEHOLDER_BARS_DROPPED: %s %s (halted/non-traded sessions)",
                    ticker,
                    dropped_placeholders,
                )
            violations, warnings = check_ohlcv_contract(daily)
            session_violations, session_warnings, _session_stats = check_session_completeness(daily)
            violations.extend(session_violations)
            warnings.extend(session_warnings)
            cross = cross_check_daily_bars(ticker, daily, logger)
            cross_checks[ticker] = cross
            if cross.get("status") == "disagreement":
                violations.append(f"cross-source disagreement: {cross.get('reason')}")
            if violations:
                raise RuntimeError(f"bar contract violation: {'; '.join(violations)}")
            if warnings:
                contract_warnings[ticker] = warnings
                for warning in warnings:
                    logger.warning("EDGE_INDEX_BAR_WARNING: %s %s", ticker, warning)
            records.extend(build_edge_records_from_bars(ticker, daily, horizon=PRED_DAYS))
        except Exception as exc:
            errors[ticker] = str(exc)
            logger.warning("EDGE_INDEX_SKIP: %s %s", ticker, exc)

    save_edge_index(records, EDGE_INDEX_PATH)
    if evidence_run is not None:
        evidence_run.record_rows("edge_index_records", [asdict(record) for record in records])
    payload = {
        "mode": "build_retrieval_index",
        "records": len(records),
        "tickers": len(index_universe),
        "watchlist_tickers": len(watchlist),
        "extra_universe_tickers": len(index_universe) - len(watchlist),
        "errors": errors,
        "bars_adjustment": EDGE_BARS_ADJUSTMENT,
        "contract_warnings": contract_warnings,
        "partial_bars_dropped": partial_bars_dropped,
        "placeholder_bars_dropped": placeholder_bars_dropped,
        "cross_source_checks": cross_checks,
        "path": str(EDGE_INDEX_PATH.resolve()),
    }
    if evidence_run is not None:
        evidence_run.record_metrics(
            "build_retrieval_index",
            {
                "index_records": len(records),
                "watchlist_count": len(watchlist),
                "error_count": len(errors),
            },
        )
    logger.info("EDGE_INDEX_REPORT: %s", json.dumps(payload))
    report_path = REPORT_DIR / "edge_index_report.json"
    result = _write_edge_report(report_path, payload, logger)
    _record_report_artifact(evidence_run, report_path)
    return result


def run_validate_edge(logger, evidence_run: EvidenceRun | None = None) -> dict:
    started_at = _utc_now_iso()
    records = load_edge_index(EDGE_INDEX_PATH)
    analog_index = EdgeAnalogIndex(records)
    validation_records = select_recent_records(records, EDGE_VALIDATION_MAX_RECORDS)
    candidates = []
    for record in validation_records:
        analogs = find_analogs(
            record.features,
            analog_index,
            k=EDGE_ANALOG_K,
            embargo_days=EDGE_EMBARGO_DAYS,
            allow_future=False,
            direction_match=EDGE_ANALOG_DIRECTION_MATCH,
            cross_ticker_embargo_days=EDGE_CROSS_TICKER_EMBARGO_DAYS,
        )
        scoring = score_edge_candidate(record.features, analogs, min_analogs=EDGE_MIN_ANALOGS)
        candidates.append(
            {
                "ticker": record.ticker,
                "timestamp": record.timestamp,
                "direction": record.direction,
                "edge_score": scoring["edge_score"],
                "recommendation": scoring["recommendation"],
                "outcome_label": record.outcome_label,
                "outcome_return_pct": record.outcome_return_pct,
                "r_multiple": record.r_multiple,
                # Stop distance is the denominator the cost model needs to
                # restate R net of the round-trip charge.
                "risk_pct_used": record.risk_pct_used,
                "mae_pct": record.mae_pct,
                "mfe_pct": record.mfe_pct,
                "exit_reason": record.exit_reason,
                "outcome_method": record.outcome_method,
            }
        )
    # Within-direction calibration frontier, self-run: every pre-registered
    # objective is evaluated out-of-fold over the FULL index history each
    # lab run, registered in the trial registry, and shipped as the live
    # advisory ONLY on the two-touch rule (pass this run AND the previous
    # registered run). Strictly advisory - it ranks inside a direction, it
    # never gates. Nothing qualifying means take-all stays the policy.
    suite = walk_forward_calibration_suite(records)
    previous_passes = _previous_calibration_passes()
    shipped_objective = select_shippable_objective(suite, previous_passes)

    if shipped_objective is not None:
        shipped_predictions = suite[shipped_objective].get("predictions", {})
        for candidate in candidates:
            if str(candidate.get("direction")) != suite[shipped_objective].get("direction"):
                continue
            key = f"{candidate.get('ticker', '')}|{candidate.get('timestamp', '')}"
            score = shipped_predictions.get(key)
            if score is not None:
                candidate["meta_score_oof"] = round(float(score), 4)
                candidate["meta_objective"] = shipped_objective

    if evidence_run is not None:
        evidence_run.record_rows("validation_candidates", candidates)
    report = compute_edge_validation_report(
        candidates,
        thresholds=EDGE_VALIDATION_THRESHOLDS,
        top_k=EDGE_VALIDATION_TOP_K,
        cost_bps_per_side=EDGE_COST_BPS_PER_SIDE,
    )
    report["meta_model"] = {
        "ship_rule": "two_touch_then_highest_oof_ic",
        "shipped_objective": shipped_objective,
        "previous_run_passes": previous_passes,
        "objectives": {
            objective: {key: value for key, value in result.items() if key not in {"predictions", "final_model"}}
            for objective, result in suite.items()
        },
    }
    # Ship or clear the live advisory artifact. A model that has not passed
    # its acceptance twice in a row must never annotate live candidates -
    # P(win) already proved a ranker can be actively anti-informative
    # against this right-tail edge.
    if shipped_objective is not None:
        atomic_write_json(META_MODEL_PATH, suite[shipped_objective]["final_model"])
        report["meta_model"]["final_model_path"] = str(META_MODEL_PATH.resolve())
    else:
        META_MODEL_PATH.unlink(missing_ok=True)
    for objective, result in suite.items():
        record_trial(
            "calibration_trial",
            {
                "direction": result.get("direction"),
                "objective": objective,
                "model_class": result.get("model_class"),
                "model_version": result.get("model_version"),
                "config": result.get("config"),
                "n_evaluated": result.get("n_evaluated"),
                "metrics": result.get("metrics"),
                "acceptance": result.get("acceptance"),
                "previous_run_passed": bool(previous_passes.get(objective)),
                "shipped": objective == shipped_objective,
            },
        )
    report["mode"] = "validate_edge"
    report["started_at"] = started_at
    report["completed_at"] = _utc_now_iso()
    report["runtime_fingerprint"] = _edge_runtime_fingerprint()
    if evidence_run is not None:
        report["evidence_run_id"] = evidence_run.run_id
    report["validation_method"] = "purged_walk_forward"
    report["future_analogs_allowed"] = False
    report["purge_config"] = {
        "embargo_days": EDGE_EMBARGO_DAYS,
        "cross_ticker_embargo_days": EDGE_CROSS_TICKER_EMBARGO_DAYS,
        "outcome_horizon_bars": PRED_DAYS,
        "min_required_days": 11,
        "outcome_window_covered": min(EDGE_EMBARGO_DAYS, EDGE_CROSS_TICKER_EMBARGO_DAYS) >= 11,
    }
    report["candidate_count"] = len(candidates)
    report["index_records"] = len(records)
    report["validation_record_limit"] = EDGE_VALIDATION_MAX_RECORDS
    if evidence_run is not None:
        evidence_run.record_metrics("validate_edge", _numeric_metrics(report))
    logger.info("EDGE_VALIDATION_REPORT: %s", json.dumps(report))
    result = _write_edge_report(EDGE_VALIDATION_REPORT_PATH, report, logger)
    _record_report_artifact(evidence_run, EDGE_VALIDATION_REPORT_PATH)
    return result


def run_edge_scan(watchlist: list[str], logger, evidence_run: EvidenceRun | None = None) -> dict:
    started_at = _utc_now_iso()
    scan_start = _monotonic_seconds()
    scan_end = scan_start
    records = load_edge_index(EDGE_INDEX_PATH)
    analog_index = EdgeAnalogIndex(records)
    candidates = []
    ticker_timings = []
    for ticker in watchlist:
        ticker_start = _monotonic_seconds()
        logger.info("EDGE_SCAN_TICKER_START: %s", ticker)
        try:
            anchor_hour, anchor_minute = _resolve_calibrated_anchor(ticker)
            intraday = fetch_intraday_bars(ticker)
            synthetic, _ = build_synthetic_sessions(intraday, anchor_hour, anchor_minute, "30m", True)
            result = _score_edge_for_bars(ticker, synthetic, analog_index, logger)
        except Exception as exc:
            logger.warning("EDGE_SCAN_ERROR: %s %s", ticker, exc)
            result = {"ticker": ticker, "status": "error", "reason": str(exc)}
        scan_end = _monotonic_seconds()
        candidates.append(result)
        duration = _elapsed_seconds(ticker_start, scan_end)
        ticker_timings.append(
            {
                "ticker": ticker,
                "status": str(result.get("recommendation") or result.get("status", "unknown")),
                "duration_seconds": duration,
            }
        )
        logger.info("EDGE_SCAN_TICKER_DONE: %s status=%s duration_seconds=%.3f", ticker, ticker_timings[-1]["status"], duration)
    ranked = sorted(candidates, key=lambda row: float(row.get("edge_score", 0.0)), reverse=True)
    _attach_meta_advisory(ranked, logger)
    completed_at = _utc_now_iso()
    payload = {
        "mode": "edge_scan",
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_fingerprint": _edge_runtime_fingerprint(),
        "duration_seconds": _elapsed_seconds(scan_start, scan_end),
        "index_records": len(records),
        "total": len(ranked),
        "ticker_timings": ticker_timings,
        "candidates": ranked,
    }
    if evidence_run is not None:
        payload["evidence_run_id"] = evidence_run.run_id
    if evidence_run is not None:
        evidence_run.record_rows("scan_candidates", ranked)
        evidence_run.record_metrics(
            "edge_scan",
            {
                "scan_candidates": len(ranked),
                "index_records": len(records),
                "promote_count": sum(1 for row in ranked if row.get("recommendation") == "promote"),
                "research_count": sum(1 for row in ranked if row.get("recommendation") == "research"),
                "skip_count": sum(1 for row in ranked if row.get("status") == "skip"),
                "error_count": sum(1 for row in ranked if row.get("status") == "error"),
            },
        )
    logger.info("EDGE_SCAN_REPORT: %s", json.dumps({"mode": "edge_scan", "total": len(ranked), "index_records": len(records)}))
    result = _write_edge_report(EDGE_SCAN_REPORT_PATH, payload, logger)
    _record_report_artifact(evidence_run, EDGE_SCAN_REPORT_PATH)
    return result


def _previous_calibration_passes() -> dict[str, bool]:
    """Last registered acceptance verdict per objective (same model version).

    This is the memory behind the two-touch ship rule: the registry is the
    append-only ledger, so 'did this objective also pass on the previous lab
    run' is read from recorded facts, not from a mutable state file.
    """
    passes: dict[str, bool] = {}
    for row in load_trials("calibration_trial"):
        objective = row.get("objective")
        if not objective:
            continue
        if row.get("model_version") != _META_MODEL_VERSION_CURRENT:
            continue
        acceptance = row.get("acceptance")
        passes[str(objective)] = bool(isinstance(acceptance, dict) and acceptance.get("passed"))
    return passes


def _attach_meta_advisory(candidates: list[dict], logger) -> None:
    """Attach advisory meta_score / expected_r_meta to live scan candidates.

    Ranking and recommendations are untouched: the shipped model is a
    within-direction advisory layer that must never gate or promote. It only
    annotates candidates in its trained direction (bullish), and only a
    model that passed its acceptance two runs straight is ever on disk.
    """
    try:
        if not META_MODEL_PATH.exists():
            return
        model = json.loads(META_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("META_MODEL_LOAD_FAILED: %s", exc)
        return
    if not isinstance(model, dict):
        return
    for row in candidates:
        if str(row.get("direction")) != "bullish":
            continue
        features = row.get("features")
        if not isinstance(features, dict):
            continue
        score = predict_score(model, features)
        if score is None or not math.isfinite(score):
            continue
        row["meta_objective"] = model.get("objective")
        row["meta_score"] = round(float(score), 4)
        expected_r = predict_expected_r(model, features)
        if expected_r is not None and math.isfinite(expected_r):
            row["expected_r_meta"] = round(float(expected_r), 4)


def run_diagnose_edge(logger, evidence_run: EvidenceRun | None = None) -> dict:
    records = load_edge_index(EDGE_INDEX_PATH)
    validation_report = None
    scan_report = None
    try:
        if EDGE_VALIDATION_REPORT_PATH.exists():
            validation_report = json.loads(EDGE_VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        validation_report = None
    try:
        if EDGE_SCAN_REPORT_PATH.exists():
            scan_report = json.loads(EDGE_SCAN_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        scan_report = None
    payload = _build_edge_diagnostic_payload(records, validation_report, scan_report)
    if evidence_run is not None:
        evidence_run.record_rows("diagnostics", [payload])
        evidence_run.record_metrics("diagnose_edge", _numeric_metrics(payload))
    logger.info("EDGE_DIAGNOSTIC_REPORT: %s", json.dumps(payload))
    result = _write_edge_report(EDGE_DIAGNOSTIC_REPORT_PATH, payload, logger)
    _record_report_artifact(evidence_run, EDGE_DIAGNOSTIC_REPORT_PATH)
    return result


def run_audit_edge(logger, evidence_run: EvidenceRun | None = None) -> dict:
    try:
        validation_report = json.loads(EDGE_VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        validation_report = {}
    try:
        scan_report = json.loads(EDGE_SCAN_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        scan_report = {}
    runtime_fingerprint = _edge_runtime_fingerprint()
    payload = compute_edge_audit_report(
        validation_report,
        scan_report,
        expected_runtime_fingerprint=runtime_fingerprint,
    )
    payload["generated_at"] = _utc_now_iso()
    if evidence_run is not None:
        evidence_run.record_rows("audits", [payload])
        evidence_run.record_metrics("audit_edge", _numeric_metrics(payload))
    logger.info("EDGE_AUDIT_REPORT: %s", json.dumps(payload))
    result = _write_edge_report(EDGE_AUDIT_REPORT_PATH, payload, logger)
    _record_report_artifact(evidence_run, EDGE_AUDIT_REPORT_PATH)
    return result


def run_edge_lab(watchlist: list[str], logger) -> dict:
    runtime_fingerprint = _edge_runtime_fingerprint()
    evidence_run = start_evidence_run(
        mode="run_edge_lab",
        root_dir=EVIDENCE_DIR,
        params={
            "watchlist_count": len(watchlist),
            "analog_k": EDGE_ANALOG_K,
            "embargo_days": EDGE_EMBARGO_DAYS,
            "min_analogs": EDGE_MIN_ANALOGS,
            "validation_thresholds": list(EDGE_VALIDATION_THRESHOLDS),
            "validation_record_limit": EDGE_VALIDATION_MAX_RECORDS,
        },
        tags={"git_commit": _git_commit(), "runtime_fingerprint": runtime_fingerprint},
    )
    index_report = run_build_retrieval_index(watchlist, logger, evidence_run=evidence_run)
    validation_report = run_validate_edge(logger, evidence_run=evidence_run)
    scan_report = run_edge_scan(watchlist, logger, evidence_run=evidence_run)
    diagnostic_report = run_diagnose_edge(logger, evidence_run=evidence_run)
    audit_report = run_audit_edge(logger, evidence_run=evidence_run)
    manifest_path = evidence_run.flush()
    payload = {
        "mode": "run_edge_lab",
        "run_id": evidence_run.run_id,
        "manifest_path": str(manifest_path.resolve()),
        "index": index_report,
        "validation": validation_report,
        "scan": {
            "mode": scan_report.get("mode"),
            "total": scan_report.get("total", 0),
            "index_records": scan_report.get("index_records", 0),
        },
        "diagnostic": diagnostic_report,
        "audit": audit_report,
    }
    logger.info("EDGE_LAB_REPORT: %s", json.dumps(payload))
    return payload


def run_watchlist_scan(watchlist: list[str], mode: str, env: dict, logger) -> dict:
    started_at = _utc_now_iso()
    scan_start = _monotonic_seconds()
    scan_end = scan_start
    kronos = KronosAdapter(logger)
    minimax = MiniMaxAdapter(logger)
    results = []
    ticker_timings = []
    for ticker in watchlist:
        ticker_start = _monotonic_seconds()
        logger.info("SCAN_TICKER_START: %s mode=%s", ticker, mode)
        try:
            result = _run_single_ticker(ticker, mode, env, kronos, minimax, logger)
        except Exception as exc:
            logger.error("ERROR: %s unhandled ticker exception: %s", ticker, exc)
            result = {"ticker": ticker, "status": "error", "reason": str(exc)}
        scan_end = _monotonic_seconds()
        results.append(result)
        duration = _elapsed_seconds(ticker_start, scan_end)
        ticker_timings.append(
            {
                "ticker": ticker,
                "status": result["status"],
                "duration_seconds": duration,
            }
        )
        logger.info("SCAN_TICKER_DONE: %s status=%s duration_seconds=%.3f", ticker, result["status"], duration)
    completed_at = _utc_now_iso()
    summary = {
        "mode": mode,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": _elapsed_seconds(scan_start, scan_end),
        "total": len(results),
        "pass": sum(1 for row in results if row["status"] == "pass"),
        "skip": sum(1 for row in results if row["status"] == "skip"),
        "error": sum(1 for row in results if row["status"] == "error"),
        "ticker_timings": ticker_timings,
    }
    logger.info("SCAN_SUMMARY: %s", json.dumps(summary))
    return summary


def run_adaptive_policy(logger, apply_tuning: bool = False) -> dict:
    report = build_adaptive_policy_report(load_decisions())
    apply_result = apply_adaptive_overrides(report, logger) if apply_tuning else {"status": "not_requested"}
    payload = {**report, "apply_result": apply_result}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "adaptive_policy_report.json"
    atomic_write_json(report_path, payload)
    logger.info("ADAPTIVE_POLICY_REPORT: %s", json.dumps(payload))
    logger.info("Adaptive policy report saved: %s", str(report_path.resolve()))
    return payload


def _research_next_actions(audit: dict, autotune: dict, adaptive_policy: dict | None = None) -> list[str]:
    actions = []
    if audit.get("readiness") != "paper_trade_only":
        actions.append("keep_live_disabled")
    warnings = set(audit.get("warnings", []))
    if "no_current_actionable_candidates" in warnings:
        actions.append("continue_daily_research_scan")
    # options_no_liquid_contract is deliberately absent: a chain with no
    # tradeable strike is the market's answer, not data we can go collect.
    if warnings & {
        "options_liquidity_missing",
        "options_data_not_execution_grade",
        "options_provider_unavailable",
    }:
        actions.append("collect_better_options_truth_data")
    if autotune.get("status") == "hold_no_edge":
        actions.append("do_not_loosen_thresholds")
    if adaptive_policy:
        recommendation = adaptive_policy.get("recommendation", {})
        if recommendation.get("status") == "tighten_research_threshold":
            actions.append("tighten_loss_heavy_research_threshold")
        if recommendation.get("status") == "improve_research_threshold":
            actions.append("use_supported_research_score_threshold")
    return actions


def run_research_ops(watchlist: list[str], env: dict, logger) -> dict:
    started_at = _utc_now_iso()
    run_start = _monotonic_seconds()
    stages = {}

    def timed(name: str, func):
        result, meta = _run_timed_stage(name, logger, func)
        stages[name] = meta
        return result

    def prepare_journal():
        rows = load_decisions()
        clean_rows, dedupe_report = deduplicate_decisions(rows)
        if dedupe_report["duplicates_removed"] and DECISIONS_PATH.exists():
            backup = REPORT_DIR / f"scan_decisions_backup_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
            shutil.copy2(DECISIONS_PATH, backup)
            dedupe_report["backup_path"] = str(backup.resolve())
        save_decisions(clean_rows)
        return clean_rows, dedupe_report

    clean_rows, dedupe_report = timed("journal_integrity", prepare_journal)

    def review_outcomes():
        reviewed_rows, review_summary = review_pending_outcomes(clean_rows, logger)
        save_decisions(reviewed_rows)
        return review_summary

    review_summary = timed("outcome_review", review_outcomes)
    adaptive_policy = timed("adaptive_policy", lambda: run_adaptive_policy(logger, apply_tuning=True))
    research_summary = timed("research_scan", lambda: run_watchlist_scan(watchlist, "research_scan", env, logger))
    diagnostic = timed("diagnostic", lambda: _write_zero_result_diagnostic(logger))
    autotune = timed("autotune", lambda: propose_overrides(load_decisions()))
    edge_lab = timed("edge_lab", lambda: run_edge_lab(watchlist, logger))
    daily_brief = timed("daily_brief", lambda: run_brief(logger, telegram_env=env))
    audit = edge_lab.get("audit", {})
    completed_at = _utc_now_iso()
    payload = {
        "mode": "research_ops",
        "generated_at": completed_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": _elapsed_seconds(run_start),
        "stages": stages,
        "journal_integrity": dedupe_report,
        "outcome_review": review_summary,
        "research_scan": research_summary,
        "diagnostic": diagnostic,
        "autotune": autotune,
        "adaptive_policy": adaptive_policy,
        "edge_run_id": edge_lab.get("run_id"),
        "edge_readiness": audit,
        "daily_brief": daily_brief,
        "next_actions": _research_next_actions(audit, autotune, adaptive_policy),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "research_ops_report.json"
    atomic_write_json(report_path, payload)
    logger.info("RESEARCH_OPS_REPORT: %s", json.dumps(payload))
    logger.info("Research operations report saved: %s", str(report_path.resolve()))
    return payload


def _sample_alert_template() -> str:
    return (
        "$TICKER - POTTER BOX TRADE CANDIDATE\n\n"
        "Direction:\n"
        "Box Top:\n"
        "Box Bottom:\n"
        "Cost Basis:\n"
        "Breakout Close:\n"
        "Breakout Strength:\n\n"
        "Empty Space:\n"
        "Score:\n"
        "Nearest Target:\n"
        "Distance to Target:\n"
        "Invalidation:\n"
        "R/R:\n\n"
        "Event Risk:\n"
        "Earnings:\n"
        "Ex-Dividend:\n\n"
        "Options:\n"
        "Expiration:\n"
        "Strike:\n"
        "Bid:\n"
        "Ask:\n"
        "Spread:\n"
        "Open Interest:\n"
        "Volume:\n"
        "IV:\n\n"
        "Kronos:\n"
        "Directional Agreement:\n"
        "Median 5-Day Forecast:\n"
        "Worst Sampled Forecast:\n\n"
        "Rule:\n"
        "Setup invalid if synthetic 24h close returns inside box or closes back through cost basis."
    )
def main() -> int:
    args = parse_args()
    logger = setup_logging(LOG_DIR)
    env = _load_env()
    if not _preflight_checks(args.mode, env, logger):
        return 1

    if args.mode == "backtest_intraday_60d":
        run_intraday_60d_backtest(WATCHLIST, logger)
        return 0

    if args.mode == "backtest_daily_proxy_2y":
        run_daily_proxy_2y_backtest(WATCHLIST, logger)
        return 0

    if args.mode == "calibration":
        if args.calibration_csv_glob:
            run_batch_calibration(args.calibration_csv_glob, logger, sweep_anchors=args.sweep_anchors)
        else:
            run_calibration(args.ticker, args.tradingview_csv, logger, sweep_anchors=args.sweep_anchors)
        return 0

    if args.mode == "test_telegram":
        return 0 if run_telegram_test(env, logger, custom_message=args.test_message) else 1
    if args.mode == "test_minimax":
        return 0 if run_minimax_test(env, logger, custom_message=args.test_message) else 1
    if args.mode == "review_outcomes":
        rows = load_decisions()
        rows, summary = review_pending_outcomes(rows, logger)
        save_decisions(rows)
        return 0
    if args.mode == "autotune":
        rows = load_decisions()
        proposal = propose_overrides(rows)
        logger.info("AUTOTUNE_PROPOSAL: %s", json.dumps(proposal))
        if args.apply_tuning:
            applied = apply_overrides(proposal, logger)
            logger.info("AUTOTUNE_APPLY_RESULT: %s", json.dumps(applied))
        return 0
    if args.mode == "adaptive_policy":
        run_adaptive_policy(logger, apply_tuning=args.apply_tuning)
        return 0
    if args.mode == "replay_eval":
        if not args.replay_dataset:
            logger.error("replay_eval requires --replay_dataset path")
            return 1
        run_replay_eval(args.replay_dataset, logger)
        return 0
    if args.mode == "diagnose_zero_results":
        _write_zero_result_diagnostic(logger)
        return 0
    if args.mode == "build_retrieval_index":
        run_build_retrieval_index(WATCHLIST, logger)
        return 0
    if args.mode == "validate_edge":
        run_validate_edge(logger)
        return 0
    if args.mode == "edge_scan":
        run_edge_scan(WATCHLIST, logger)
        return 0
    if args.mode == "diagnose_edge":
        run_diagnose_edge(logger)
        return 0
    if args.mode == "audit_edge":
        run_audit_edge(logger)
        return 0
    if args.mode == "run_edge_lab":
        run_edge_lab(WATCHLIST, logger)
        return 0
    if args.mode == "research_ops":
        run_research_ops(WATCHLIST, env, logger)
        return 0
    if args.mode == "brief":
        run_brief(logger, telegram_env=env)
        return 0
    if args.mode == "doctor":
        report = run_doctor()
        logger.info("DOCTOR_REPORT: %s", json.dumps(report))
        return 0 if report.get("status") == "ok" else 1

    # dry_run, research_scan, or live scan path
    summary = run_watchlist_scan(WATCHLIST, args.mode, env, logger)
    if args.mode == "dry_run" and summary["pass"] == 0:
        logger.info("DRY_RUN_ALERT_TEMPLATE:\n%s", _sample_alert_template())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
