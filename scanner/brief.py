"""Operator daily brief: one command, plain English, verdict first.

Reads the latest report artifacts (no network, no model loads) and renders
what changed, how far each evidence gate is from unlocking, and the single
highest-leverage next action.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as scanner_config
from .alerts.telegram import send_telegram_message
from .config import REPORT_DIR

# Plain-English translations for the audit's blocker/warning codes:
# (phone-sized label, full explanation, the fix, needs_a_human).
#
# That last flag is the point of this table. A fault means the operator has to
# do something; a finding is the system honestly reporting what it learned or
# what the market looks like. Both used to render identically, so a normal quiet
# day - the designed state - read like an outage every single morning.
_ISSUE_GUIDE: dict[str, tuple[str, str, str, bool]] = {
    "validation_threshold_55_unsupported": (
        "Score-55 gate is unsupported",
        "The absolute score-55 gate does not satisfy its out-of-sample evidence requirements",
        "keep live off; collect only while under-sampled, otherwise test a pre-registered entry-selection change",
        False,
    ),
    "ranking_evidence_unsupported": (
        "Score doesn't rank winners yet",
        "The score does not rank outcomes strongly enough out-of-sample",
        "keep live off; collect only while under-sampled, otherwise redesign entry selection before rescoring",
        False,
    ),
    "validation_not_walk_forward": (
        "Validation is leak-contaminated",
        "Validation is not purged walk-forward, so the numbers may be leak-contaminated",
        "do not trust these results; rebuild validation on the purged walk-forward path",
        True,
    ),
    "future_analogs_allowed": (
        "Future data leaked into validation",
        "Analogs from the future leaked into validation",
        "do not trust these results; rebuild the retrieval index with the purge enabled",
        True,
    ),
    "validation_cost_model_unsupported": (
        "Validation cost basis is unsafe",
        "Validation used less than the pre-registered 25 bps/side floor or has incomplete cost provenance",
        "restore KRONOS_COST_BPS_PER_SIDE to at least 25 and rerun the full Edge lab",
        True,
    ),
    "options_no_liquid_contract": (
        "No liquid options chain",
        "Some names have no option strike that clears the spread and open-interest gates",
        "nothing to fix - the chain is genuinely untradeable, which is normal for small caps",
        False,
    ),
    "options_provider_unavailable": (
        "Options data did not come from Tradier",
        "Options data did not come from real-time Tradier: the scanner fell back to the "
        "indicative pipeline, or the lookup failed outright",
        "check TRADIER_API_TOKEN is the PRODUCTION token (not sandbox) and that Tradier is "
        "reachable; scanner/logs/scanner.log has the underlying error",
        True,
    ),
    "options_data_not_execution_grade": (
        "Options quotes are stale",
        "Real Tradier contracts were found, but their quotes were stale at scan time",
        "run research_ops during market hours so Tradier quotes are fresh",
        True,
    ),
    "options_liquidity_missing": (
        "Some OI/volume fields are zero",
        "Open interest / volume / spread fields are missing or zero on some candidates",
        "usually zero day-volume early in the session; resolves on an intraday scan",
        False,
    ),
    "low_feed_confidence": (
        "Equity bars on the free IEX feed",
        "Equity bars come from the free IEX-only feed",
        "acceptable for research; full-SIP data (Alpaca ATP or Polygon Starter) clears it",
        False,
    ),
    "delayed_equity_feed": (
        "Equity bars are consolidated but delayed",
        "At least one current candidate uses a deliberately delayed consolidated equity snapshot",
        "acceptable for research; real-time consolidated bars are required for paper/live alerts",
        False,
    ),
    "no_current_actionable_candidates": (
        "No setups today (normal)",
        "Nothing on the watchlist is near a qualifying setup today",
        "normal; the scanner is supposed to be quiet most days",
        False,
    ),
    "bearish_edge_negative": (
        "Bearish blocked (loses money)",
        "Bearish setups have negative expectancy in validation",
        "bearish promotion stays blocked until bearish evidence turns positive",
        False,
    ),
    "bullish_edge_negative": (
        "Bullish blocked (loses money)",
        "Bullish setups have negative expectancy in validation",
        "bullish promotion stays blocked until bullish evidence turns positive",
        False,
    ),
    "promoted_candidates_direction_blocked": (
        "Promotions sit in an unproven direction",
        "Promotions exist only in directions without proven positive expectancy (negative, under-sampled, or absent validation cohort)",
        "treated as research-only until that direction proves itself",
        False,
    ),
}

# Codes whose audit summary names the affected tickers; naming them turns a
# vague warning into something checkable at a glance.
_ISSUE_TICKERS = {
    "options_no_liquid_contract": "no_liquid_options_contract_candidates",
    "options_provider_unavailable": "options_provider_unavailable_candidates",
    "options_data_not_execution_grade": "stale_options_quote_candidates",
    "options_liquidity_missing": "missing_options_liquidity_candidates",
    "low_feed_confidence": "low_feed_confidence_candidates",
    "delayed_equity_feed": "delayed_equity_feed_candidates",
}

# The phone brief's UNLOCK block states these two in full, with the actual
# numbers; repeating them under FYI just said the same thing twice.
_UNLOCK_CODES = {"validation_threshold_55_unsupported", "ranking_evidence_unsupported"}

_READINESS_LINE = {
    "blocked": "NOT live-ready. Evidence gates are failing; live alerting stays off.",
    "watch_only": "Evidence gates pass but nothing is actionable today. Watch only.",
    "research_only": "Evidence gates pass; research candidates only. No live alerts.",
    "paper_trade_only": "Evidence supports PAPER trading the promoted candidates. Still not real money.",
}

# The phone version: same truth, told as status rather than as a failure.
_LIVE_LINE = {
    "blocked": "Live alerts: off - evidence gates not met (expected).",
    "watch_only": "Live alerts: off - gates pass, nothing actionable today.",
    "research_only": "Live alerts: off - research candidates only.",
    "paper_trade_only": "Live alerts: off - paper trading supported. Still not real money.",
}


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_json_lines(path: Path) -> list[dict]:
    """Best-effort JSONL reader for operator context.

    The brief must remain available even if one journal line is malformed; the
    journal-integrity stage owns repair and will surface the underlying issue.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 2, missing: str = "n/a") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return f"{number:.{digits}f}"


def _int(value: Any, default: int = 0) -> int:
    # Reports can carry explicit nulls (hand-edited or older formats), which
    # .get() defaults do not guard against.
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _percentage(value: Any, digits: int = 0) -> str:
    try:
        return _fmt(float(value) * 100, digits)
    except (TypeError, ValueError):
        return "n/a"


# A win rate over a handful of rows carries no information, but rendered as a
# bare percentage it reads like a finding - "agree 100% (n=1)" invites the
# operator to trust a filter that has been tested exactly once. Below this many
# rows the brief reports the count and withholds the rate.
MIN_ROWS_FOR_WIN_RATE = 10


def _win_rate_summary(block: dict) -> str:
    count = _int(block.get("signal_count"))
    if count <= 0:
        return "n/a (n=0)"
    if count < MIN_ROWS_FOR_WIN_RATE:
        return f"too few to rate (n={count})"
    return f"{_fmt(_num(block.get('win_rate')) * 100, 0)}% (n={count})"


def _negative_evidence_gate(
    check: dict,
    *,
    evidence_key: str,
    average_key: str,
    t_stat_key: str,
) -> bool:
    """Whether a gate has enough dependence-aware evidence and it is negative.

    A failed gate is not always immature. Once its independent-day minimum is
    met, a non-positive average with a non-positive t-stat is evidence against
    the current design. Calling that state "not yet" and asking for more rows
    hides the difference between an experiment still collecting and one whose
    tested mechanism is losing.
    """
    if check.get("passed") is True:
        return False
    value = check.get("value")
    if not isinstance(value, dict) or value.get("dependence_metrics_present") is not True:
        return False
    if not all(
        _is_finite_number(value.get(key))
        for key in (evidence_key, "min_signals", average_key, t_stat_key)
    ):
        return False
    return (
        _int(value.get(evidence_key)) >= _int(value.get("min_signals"), 20)
        and _num(value.get(average_key)) <= 0.0
        and _num(value.get(t_stat_key)) <= 0.0
    )


def _mature_negative_edge(audit: dict) -> bool:
    checks = audit.get("checks")
    if not isinstance(checks, dict):
        return False
    threshold = checks.get("validation_threshold")
    ranking = checks.get("ranking_evidence")
    if not isinstance(threshold, dict) or not isinstance(ranking, dict):
        return False
    return _negative_evidence_gate(
        threshold,
        evidence_key="signal_count",
        average_key="average_r_multiple",
        t_stat_key="t_stat_r_multiple",
    ) and _negative_evidence_gate(
        ranking,
        evidence_key="top_decile_evidence_days",
        average_key="top_decile_average_r",
        t_stat_key="top_decile_t_stat",
    )


def _gate_status(
    check: dict,
    *,
    evidence_key: str,
    average_key: str,
    t_stat_key: str,
) -> str:
    if check.get("passed") is True:
        return "PASS"
    if _negative_evidence_gate(
        check,
        evidence_key=evidence_key,
        average_key=average_key,
        t_stat_key=t_stat_key,
    ):
        return "NEGATIVE"
    value = check.get("value")
    if isinstance(value, dict) and _int(value.get(evidence_key)) < _int(value.get("min_signals"), 20):
        return "COLLECTING"
    return "UNSUPPORTED"


def _gate_progress(audit: dict, validation: dict) -> list[str]:
    lines = []
    cost_model = validation.get("cost_model")
    cost_check = audit.get("checks", {}).get("costs_charged", {})
    cost_value = cost_check.get("value", {}) if isinstance(cost_check.get("value"), dict) else {}
    if cost_check.get("passed") is True and isinstance(cost_model, dict):
        lines.append(
            f"- Cost basis: all figures NET of {_fmt(cost_model.get('bps_per_side'), 0)} bps/side "
            f"round trip ({_fmt(cost_model.get('round_trip_return_pct_charged'), 2)} pct pts per trade)"
        )
    elif isinstance(cost_model, dict):
        lines.append(
            f"- Cost basis: INVALID -- {_fmt(cost_model.get('bps_per_side'), 2)} bps/side "
            f"does not satisfy the {_fmt(cost_value.get('minimum_bps_per_side'), 0)} bps/side floor "
            "and complete-provenance gate"
        )
    else:
        lines.append("- Cost basis: MISSING -- gates are blocked until validation charges a round-trip cost")
    ranking = audit.get("checks", {}).get("ranking_evidence", {})
    value = ranking.get("value", {}) if isinstance(ranking.get("value"), dict) else {}
    status = _gate_status(
        ranking,
        evidence_key="top_decile_evidence_days",
        average_key="top_decile_average_r",
        t_stat_key="top_decile_t_stat",
    )
    raw_top_signals = _int(value.get("top_decile_signals"))
    top_evidence_days = value.get("top_decile_evidence_days")
    if top_evidence_days is None:
        top_evidence = f"top-decile signals {raw_top_signals}/{_int(value.get('min_signals'), 20)}"
    else:
        top_evidence = (
            f"top-decile evidence days {_int(top_evidence_days)}/{_int(value.get('min_signals'), 20)} "
            f"({raw_top_signals} raw signals)"
        )
    precision_lower_bound = value.get(
        "top_decile_precision_lower_bound",
        value.get("top_decile_wilson_lb_precision"),
    )
    precision_label = "dependence-aware precision LB" if top_evidence_days is not None else "Wilson-LB precision"
    lines.append(
        f"- Ranking gate ({status}): rank IC {_fmt(value.get('rank_ic'), 3)} "
        f"(need >= {_fmt(value.get('min_rank_ic'), 2)}, p {_fmt(value.get('rank_ic_p_value'), 3)}), "
        f"{top_evidence}, "
        f"avg R {_fmt(value.get('top_decile_average_r'))}, t {_fmt(value.get('top_decile_t_stat'))}, "
        f"{precision_label} {_fmt(precision_lower_bound)} (need >= 0.45)"
    )
    legacy = audit.get("checks", {}).get("validation_threshold", {})
    legacy_value = legacy.get("value", {}) if isinstance(legacy.get("value"), dict) else {}
    legacy_status = _gate_status(
        legacy,
        evidence_key="signal_count",
        average_key="average_r_multiple",
        t_stat_key="t_stat_r_multiple",
    )
    lines.append(
        f"- Legacy threshold-{legacy_value.get('threshold', 55)} gate "
        f"({legacy_status}): "
        f"{_int(legacy_value.get('signal_count'))}/{_int(legacy_value.get('min_signals'), 20)} "
        f"{'evidence days' if legacy_value.get('raw_signal_count') is not None else 'signals'}, "
        f"avg R {_fmt(legacy_value.get('average_r_multiple'))}, "
        f"t {_fmt(legacy_value.get('t_stat_r_multiple'))}, "
        f"precision LB {_fmt(legacy_value.get('precision_lower_bound'))} "
        f"(need >= {_fmt(legacy_value.get('min_precision'), 2)})"
    )
    directions = validation.get("by_direction", {})
    if isinstance(directions, dict) and directions:
        parts = []
        blocked = set(audit.get("summary", {}).get("blocked_directions", []))
        unproven = set(audit.get("summary", {}).get("unproven_directions", []))
        for name in ("bullish", "bearish"):
            block = directions.get(name)
            if not isinstance(block, dict):
                continue
            tag = " BLOCKED" if name in blocked else " UNPROVEN" if name in unproven else ""
            clustered = block.get("t_stat_r_day_clustered")
            if isinstance(clustered, dict):
                sample_label = f"days={_int(clustered.get('n_days'))}"
                average_r = clustered.get("mean_of_day_means")
            else:
                sample_label = f"n={_int(block.get('signal_count'))}"
                average_r = block.get("average_r_multiple")
            parts.append(
                f"{name} {sample_label} "
                f"avgR {_fmt(average_r)}{tag}"
            )
        if parts:
            lines.append(f"- Directions: {'; '.join(parts)}")
    return lines


def _scan_summary(scan: dict) -> list[str]:
    candidates = [row for row in scan.get("candidates", []) if isinstance(row, dict)]
    if not candidates:
        return ["- No scan data yet; run research_ops or edge_scan."]
    counts: dict[str, int] = {}
    for row in candidates:
        key = str(row.get("recommendation") or row.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    ordered = ", ".join(f"{count} {name}" for name, count in sorted(counts.items(), key=lambda kv: -kv[1]))
    lines = [f"- {len(candidates)} tickers scanned: {ordered}"]
    scored = [row for row in candidates if row.get("edge_score") is not None]
    for row in scored[:3]:
        blockers = row.get("blocking_reasons") or []
        suffix = f" -- blocked by {', '.join(blockers[:3])}" if blockers else ""
        lines.append(
            f"- {row.get('ticker')}: {row.get('direction', '?')} edge {_fmt(row.get('edge_score'))}{suffix}"
        )
    return lines


def _today_research_samples(base: Path, today: str) -> list[dict]:
    """Return today's accepted counterfactual samples, separate from Edge trades."""
    samples: dict[tuple[str, str], dict] = {}
    for row in _read_json_lines(base / "scan_decisions.jsonl"):
        diagnostics = row.get("research_diagnostics")
        is_candidate = row.get("skip_reason") == "research_candidate" or (
            isinstance(diagnostics, dict) and diagnostics.get("passed") is True
        )
        if (
            row.get("mode") != "research_scan"
            or row.get("source_session_date") != today
            or row.get("outcome_status") != "pending"
            or not is_candidate
        ):
            continue
        ticker = str(row.get("ticker") or "?")
        direction = str(row.get("direction") or "unknown")
        samples[(ticker, direction)] = {
            "ticker": ticker,
            "direction": direction,
            "research_score": row.get("research_score"),
            "doctrine_v2_score": row.get("doctrine_v2_score"),
            "doctrine_v2_passed": bool(row.get("doctrine_v2_passed")),
            "kronos_directional_agreement": row.get("kronos_directional_agreement"),
            "kronos_passed": bool(row.get("kronos_passed")),
        }
    return list(samples.values())


def _research_sample_summary(samples: list[dict]) -> list[str]:
    if not samples:
        return []
    lines = [
        f"- {len(samples)} counterfactual sample{'s' if len(samples) != 1 else ''} accepted for outcome tracking; "
        "never alerted or traded."
    ]
    for row in samples:
        doctrine_status = "pass" if row.get("doctrine_v2_passed") else "fail"
        kronos_status = "pass" if row.get("kronos_passed") else "fail"
        lines.append(
            f"- {row.get('ticker')}: {row.get('direction')} research {_fmt(row.get('research_score'), 0)}; "
            f"Doctrine v2 {_fmt(row.get('doctrine_v2_score'), 0)} {doctrine_status}; "
            f"Kronos {_percentage(row.get('kronos_directional_agreement'))}% {kronos_status}."
        )
    return lines


def _learning_summary(policy: dict, diagnostic: dict, audit: dict | None = None) -> list[str]:
    lines = []
    # The research-threshold cohort and the validation cohort are different
    # populations, and a direction can look strong in the former while being
    # promotion-blocked by the latter. Printed side by side without this label,
    # "bearish daily-WR=68%" reads as a green light for a blocked direction.
    promotable = set()
    if isinstance(audit, dict):
        summary = audit.get("summary", {})
        if isinstance(summary, dict):
            promotable = {str(d) for d in summary.get("promotable_directions", []) or []}
    research = policy.get("research_candidates", {})
    lines.append(
        f"- Journal: {_int(research.get('resolved'))} resolved research candidates "
        f"({_int(research.get('resolved_outcomes', {}).get('win'))}W/"
        f"{_int(research.get('resolved_outcomes', {}).get('loss'))}L, "
        f"{_fmt(_num(research.get('resolved_win_rate')) * 100, 1)}% WR), "
        f"{_int(diagnostic.get('research_candidates', {}).get('pending'))} pending"
    )
    recommendation = policy.get("recommendation", {})
    lines.append(
        f"- Policy: {recommendation.get('status', 'unknown')} "
        f"(threshold {research.get('current_threshold', '?')}) -- {recommendation.get('reason', '')}"
    )
    direction_blocks = research.get("current_threshold_by_direction", {})
    if isinstance(direction_blocks, dict):
        direction_parts = []
        for direction in ("bullish", "bearish", "unknown"):
            block = direction_blocks.get(direction)
            if not isinstance(block, dict) or _int(block.get("signal_count")) <= 0:
                continue
            status = "" if direction in promotable or direction == "unknown" else " [NOT PROMOTABLE]"
            if block.get("evidence_day_count") is not None:
                direction_parts.append(
                    f"{direction} rows={_int(block.get('signal_count'))} "
                    f"days={_int(block.get('evidence_day_count'))} "
                    f"daily-WR={_fmt(_num(block.get('mean_daily_win_rate')) * 100, 1)}% "
                    f"LB={_fmt(_num(block.get('dependence_adjusted_win_rate_lower_bound')) * 100, 1)}% "
                    f"avg-day={_fmt(block.get('average_daily_return_pct'))}%{status}"
                )
            else:
                direction_parts.append(
                    f"{direction} n={_int(block.get('signal_count'))} "
                    f"{_int(block.get('wins'))}W/{_int(block.get('losses'))}L "
                    f"avg {_fmt(block.get('average_return_pct'))}%{status}"
                )
        if direction_parts:
            lines.append("- Threshold cohort by direction: " + "; ".join(direction_parts))
    lift = policy.get("kronos_lift", {})
    if _int(lift.get('rows_with_kronos')) > 0:
        agree = lift.get("agree", {})
        disagree = lift.get("disagree", {})
        lines.append(
            f"- Kronos lift: {_int(lift.get('rows_with_kronos'))} scored -- agree "
            f"{_win_rate_summary(agree)} vs disagree {_win_rate_summary(disagree)}"
        )
    else:
        eval_errors = _int(lift.get("rows_with_eval_errors"))
        if eval_errors > 0:
            lines.append(
                f"- Kronos lift: MODEL ERRORS on {eval_errors} resolved candidates (check KRONOS_RESEARCH_EVAL_FAILED in scanner.log)"
            )
        else:
            lines.append("- Kronos lift: no scored research candidates yet (accumulating from today forward)")
    doctrine = policy.get("doctrine_v2", {})
    if _int(doctrine.get('resolved')) > 0:
        current = doctrine.get("current_threshold", {})
        lines.append(
            f"- Doctrine v2: {_int(doctrine.get('resolved'))} resolved, baseline cohort "
            f"{_int(current.get('wins'))}W/{_int(current.get('losses'))}L avg {_fmt(current.get('average_return_pct'))}%"
        )
    return lines


def _ticker_suffix(audit: dict, code: str, limit: int = 4) -> str:
    key = _ISSUE_TICKERS.get(code)
    if not key:
        return ""
    tickers = audit.get("summary", {}).get(key)
    if not isinstance(tickers, list) or not tickers:
        return ""
    shown = ", ".join(str(t) for t in tickers[:limit])
    extra = len(tickers) - limit
    return f" ({shown}{f' +{extra} more' if extra > 0 else ''})"


def _classify_issues(audit: dict, policy: dict) -> tuple[list[tuple], list[tuple]]:
    """Split every reported code into (faults, findings).

    Each entry is (code, short, explanation, fix). Faults need a human; findings
    are the system reporting reality and need nothing.
    """
    faults: list[tuple] = []
    findings: list[tuple] = []
    tagged = [(str(code), True) for code in audit.get("blockers", [])]
    tagged += [(str(code), False) for code in audit.get("warnings", [])]
    for code, is_blocker in tagged:
        if code in _ISSUE_GUIDE:
            short, explanation, fix, is_fault = _ISSUE_GUIDE[code]
        else:
            # A code we have no translation for. An unrecognised BLOCKER is
            # serious until proven otherwise - defaulting it to a finding would
            # let a newly added gate render as "Nothing is broken".
            short = explanation = code
            fix = "unrecognised code - see scanner/README.md"
            is_fault = is_blocker
        suffix = _ticker_suffix(audit, code)
        entry = (code, f"{short}{suffix}", f"{explanation}{suffix}", fix)
        (faults if is_fault else findings).append(entry)

    # Kronos "eval errors" surface only in the policy report, and the field
    # holds both real exceptions and ordinary skips ("need 60 synthetic bars"),
    # so the count alone cannot prove a fault. Report it as a finding and let
    # the log say which it was, rather than calling a thin-history skip a
    # model failure.
    eval_errors = _int(policy.get("kronos_lift", {}).get("rows_with_eval_errors"))
    if eval_errors > 0:
        findings.append(
            (
                "kronos_no_forecast",
                f"Kronos produced no forecast on {eval_errors} resolved candidates",
                f"Kronos produced no forecast on {eval_errors} resolved candidates",
                "usually too little bar history; if the count climbs, check "
                "KRONOS_RESEARCH_EVAL_FAILED in scanner/logs/scanner.log",
            )
        )
    return faults, findings


def _issues(audit: dict, policy: dict) -> list[str]:
    faults, findings = _classify_issues(audit, policy)
    lines = [f"- NEEDS ACTION -- {code}: {explanation}. Fix: {fix}." for code, _short, explanation, fix in faults]
    lines += [f"- {code}: {explanation}. No action: {fix}." for code, _short, explanation, fix in findings]
    return lines or ["- None. All gates green."]


def _next_action(audit: dict, policy: dict) -> str:
    recommendation = policy.get("recommendation", {})
    if recommendation.get("status") == "loosen_research_threshold":
        return (
            "Confirm the pending research-threshold loosening on tomorrow's research_ops run "
            "so the journal starts refilling."
        )
    faults, _findings = _classify_issues(audit, policy)
    if faults:
        _code, _short, explanation, fix = faults[0]
        return f"{explanation}. Fix: {fix}."
    if _mature_negative_edge(audit):
        return (
            "Keep live off. The score-55 and top-decile cohorts are already adequately sampled and negative; "
            "pre-register a new entry-selection hypothesis before changing production scoring."
        )
    if audit.get("blockers"):
        return (
            "Nothing. Keep the daily research_ops cadence; the evidence gates need more "
            "resolved samples."
        )
    return "Review promoted candidates and paper-trade them per the audit."


def build_daily_brief(report_dir: Path | None = None) -> tuple[str, dict]:
    base = Path(report_dir) if report_dir is not None else REPORT_DIR
    audit = _read_json(base / "edge_audit_report.json")
    validation = _read_json(base / "edge_validation_report.json")
    policy = _read_json(base / "adaptive_policy_report.json")
    diagnostic = _read_json(base / "zero_result_diagnostic.json")
    scan = _read_json(base / "edge_scan_report.json")

    readiness = str(audit.get("readiness", "unknown"))
    today = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    research_samples = _today_research_samples(base, today)
    research_section = (
        ["", "## Accepted research samples", *_research_sample_summary(research_samples)]
        if research_samples
        else []
    )

    lines = [
        f"# Kronos Daily Brief -- {today}",
        "",
        "## Verdict",
        f"**{readiness}** -- {_READINESS_LINE.get(readiness, 'No audit found; run research_ops first.')}",
        "",
        "## Evidence progress",
        *_gate_progress(audit, validation),
        "",
        "## Today's scan",
        *_scan_summary(scan),
        *research_section,
        "",
        "## Learning loop",
        *_learning_summary(policy, diagnostic, audit),
        "",
        "## Open issues",
        *_issues(audit, policy),
        "",
        "## Next action",
        f"{_next_action(audit, policy)}",
        "",
    ]
    markdown = "\n".join(lines)
    payload = {
        "mode": "brief",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "readiness": readiness,
        "next_action": _next_action(audit, policy),
        "research_samples": research_samples,
        "telegram_text": _telegram_text(
            today,
            readiness,
            audit,
            policy,
            diagnostic,
            scan,
            research_samples,
        ),
    }
    return markdown, payload


def _telegram_text(
    today: str,
    readiness: str,
    audit: dict,
    policy: dict,
    diagnostic: dict,
    scan: dict,
    research_samples: list[dict],
) -> str:
    """Condensed phone-sized brief. Status report only, never a trade alert.

    Ordered by what the operator actually needs: whether today requires them at
    all, then trades, then progress, then everything that needs no action. The
    old version opened with "Verdict: BLOCKED" every morning, which buried the
    real answer ("nothing to do") under a word that reads like a breakage.
    """
    faults, findings = _classify_issues(audit, policy)

    candidates = [row for row in scan.get("candidates", []) if isinstance(row, dict)]
    actionable = [row for row in candidates if row.get("recommendation") in {"research", "promote"}]
    summary = audit.get("summary", {}) if isinstance(audit.get("summary"), dict) else {}
    execution_ready = {
        str(ticker)
        for ticker in (summary.get("execution_ready_promoted_candidates", []) or [])
    }
    paper_authorized = [
        row
        for row in actionable
        if readiness == "paper_trade_only"
        and row.get("recommendation") == "promote"
        and str(row.get("ticker")) in execution_ready
    ]

    # Emoji only ever reach Telegram: the markdown brief (the one print()ed to a
    # cp1252 Windows console) stays ASCII, and every report that embeds this text
    # is logged through json.dumps, which escapes non-ASCII.
    if readiness not in _LIVE_LINE:
        # No readable audit. "No warnings" here means "we know nothing", which is
        # not the same as "nothing is wrong" - saying OK would be a lie told
        # precisely when the pipeline is most likely broken.
        lines = [f"⚠️ KRONOS - {today}", "Can't tell - no readable audit."]
    elif faults:
        headline = f"{len(faults)} thing{'s' if len(faults) > 1 else ''} need{'' if len(faults) > 1 else 's'} you."
        lines = [f"⚠️ KRONOS - {today}", headline]
    elif _mature_negative_edge(audit):
        lines = [f"✅ KRONOS - {today}", "No operator action. Strategy evidence is negative."]
    else:
        lines = [f"✅ KRONOS - {today}", "Nothing to do. Nothing is broken."]
    lines.append(_LIVE_LINE.get(readiness, "Live alerts: off - run research_ops to rebuild the reports."))

    if faults:
        lines += ["", "DO THIS"]
        for _code, short, _explanation, fix in faults:
            lines.append(f"- {short}")
            lines.append(f"  -> {fix}")

    # Edge recommendations are not executions. In particular, a scan-level
    # ``promote`` is still research-only when the audit blocks its direction,
    # evidence route, or execution quality. Calling those rows "LIVE TRADES"
    # can invite an operator to act against the same fail-closed audit printed
    # two lines above.
    lines += ["", "LIVE TRADES - none"]
    if actionable:
        section = "PAPER CANDIDATES" if len(paper_authorized) == len(actionable) else "EDGE RESEARCH"
        lines += ["", f"{section} - {len(actionable)}"]
        lines.append(
            f"{len(candidates)} scanned, {len(actionable)} Edge recommendations, "
            f"{len(paper_authorized)} paper-authorized"
        )
        paper_tickers = {str(row.get("ticker")) for row in paper_authorized}
        for row in actionable[:3]:
            status = "paper candidate" if str(row.get("ticker")) in paper_tickers else "research only"
            lines.append(
                f"- {row.get('ticker')} {row.get('direction', '?')} edge "
                f"{_fmt(row.get('edge_score'), 1)} ({status})"
            )
    else:
        lines.append(f"{len(candidates)} scanned, 0 Edge recommendations (quiet by design)")

    if research_samples:
        lines += ["", f"RESEARCH SAMPLES - {len(research_samples)} counterfactual only"]
        for row in research_samples[:4]:
            doctrine_status = "pass" if row.get("doctrine_v2_passed") else "fail"
            kronos_status = "pass" if row.get("kronos_passed") else "fail"
            lines.append(
                f"- {row.get('ticker')} {row.get('direction')} score {_fmt(row.get('research_score'), 0)} "
                f"(Doctrine {_fmt(row.get('doctrine_v2_score'), 0)} {doctrine_status}, "
                f"Kronos {_percentage(row.get('kronos_directional_agreement'))}% {kronos_status})"
            )
        if len(research_samples) > 4:
            lines.append(f"- (+{len(research_samples) - 4} more in daily_brief.md)")

    lines += ["", *_unlock_block(audit, policy, diagnostic)]

    notes = [entry for entry in findings if entry[0] not in _UNLOCK_CODES]
    if notes:
        lines += ["", "FYI - no action"]
        lines += [f"- {short}" for _code, short, _explanation, _fix in notes[:5]]
        # Say so rather than letting a trimmed list read as the whole story.
        if len(notes) > 5:
            lines.append(f"- (+{len(notes) - 5} more in daily_brief.md)")

    return "\n".join(lines)


def _unlock_block(audit: dict, policy: dict, diagnostic: dict) -> list[str]:
    """How far the evidence gates are from unlocking, in one glance."""
    ranking = audit.get("checks", {}).get("ranking_evidence", {})
    value = ranking.get("value", {}) if isinstance(ranking.get("value"), dict) else {}
    research = policy.get("research_candidates", {})
    lift = policy.get("kronos_lift", {})

    passed = bool(ranking.get("passed"))
    mature_negative = _mature_negative_edge(audit)
    if passed:
        unlock_status = "gates pass"
        ranking_suffix = ""
    elif mature_negative:
        # Adequately sampled negative evidence is a completed falsification of
        # the current design, not an immature gate that is merely "not yet"
        # ready. The phone brief must preserve the same distinction as the
        # detailed markdown because this is the surface an operator sees.
        unlock_status = "current design negative"
        ranking_suffix = " - ranking evidence is negative out of sample"
    else:
        unlock_status = "not yet"
        ranking_suffix = " - score doesn't rank winners yet"
    lines = [f"UNLOCK - {unlock_status}"]
    lines.append(
        f"Rank IC {_fmt(value.get('rank_ic'), 3)} (needs {_fmt(value.get('min_rank_ic'), 2)})"
        + ranking_suffix
    )
    lines.append(
        f"Journal {_int(research.get('resolved'))} resolved, "
        f"{_fmt(_num(research.get('resolved_win_rate')) * 100, 0)}% WR, "
        f"{_int(diagnostic.get('research_candidates', {}).get('pending'))} pending"
    )
    summary = audit.get("summary", {})
    promotable = {str(d) for d in (summary.get("promotable_directions", []) or [])} if isinstance(summary, dict) else set()
    direction_blocks = research.get("current_threshold_by_direction", {})
    if isinstance(direction_blocks, dict):
        direction_parts = []
        for direction, label in (("bullish", "bull"), ("bearish", "bear")):
            block = direction_blocks.get(direction)
            if not isinstance(block, dict) or _int(block.get("signal_count")) <= 0:
                continue
            # Without this, a blocked direction's research cohort can read as a
            # green light in the one place the operator actually looks.
            status = "" if direction in promotable else " BLOCKED"
            if block.get("evidence_day_count") is not None:
                direction_parts.append(
                    f"{label} rows={_int(block.get('signal_count'))} "
                    f"days={_int(block.get('evidence_day_count'))} "
                    f"dailyWR={_fmt(_num(block.get('mean_daily_win_rate')) * 100, 0)}% "
                    f"LB={_fmt(_num(block.get('dependence_adjusted_win_rate_lower_bound')) * 100, 0)}% "
                    f"avgDay={_fmt(block.get('average_daily_return_pct'), 1)}%{status}"
                )
            else:
                direction_parts.append(
                    f"{label} n={_int(block.get('signal_count'))} "
                    f"{_int(block.get('wins'))}W/{_int(block.get('losses'))}L "
                    f"avg {_fmt(block.get('average_return_pct'), 1)}%{status}"
                )
        if direction_parts:
            lines.append(
                f"Threshold {research.get('current_threshold', '?')} split: " + "; ".join(direction_parts)
            )
    if _int(lift.get("rows_with_kronos")) > 0:
        agree = lift.get("agree", {})
        disagree = lift.get("disagree", {})
        lines.append(
            f"Kronos agree {_win_rate_summary(agree)}"
            f" vs disagree {_win_rate_summary(disagree)}"
        )
    return lines


def run_brief(logger, report_dir: Path | None = None, telegram_env: dict | None = None) -> dict:
    base = Path(report_dir) if report_dir is not None else REPORT_DIR
    markdown, payload = build_daily_brief(base)
    base.mkdir(parents=True, exist_ok=True)
    output_path = base / "daily_brief.md"
    output_path.write_text(markdown, encoding="utf-8")
    payload["path"] = str(output_path.resolve())
    print(markdown)
    if logger is not None:
        logger.info("DAILY_BRIEF_SAVED: %s", payload["path"])
    payload["telegram"] = _deliver_telegram(payload, telegram_env, logger)
    return payload


def _deliver_telegram(payload: dict, telegram_env: dict | None, logger) -> dict:
    """Best-effort status delivery; never raises, never gates anything."""
    if not scanner_config.BRIEF_TELEGRAM_ENABLED:
        return {"status": "disabled"}
    env = telegram_env or {}
    token = str(env.get("telegram_token") or "").strip()
    chat_id = str(env.get("telegram_chat_id") or "").strip()
    if not token or not chat_id:
        return {"status": "no_credentials"}
    try:
        sent = send_telegram_message(token, chat_id, payload.get("telegram_text", ""), logger)
    except Exception as exc:
        if logger is not None:
            logger.warning("BRIEF_TELEGRAM_FAILED: %s", exc)
        return {"status": "failed", "error": str(exc)}
    return {"status": "sent" if sent else "failed"}
