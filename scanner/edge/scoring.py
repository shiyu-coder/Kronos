from __future__ import annotations

import math
from statistics import median
from typing import Any

import numpy as np

from .. import config as scanner_config
from .validation import apply_transaction_costs


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ordered_unique(reasons: list[str]) -> list[str]:
    out: list[str] = []
    for reason in reasons:
        if reason not in out:
            out.append(reason)
    return out


def _analog_summary(analogs: list[dict]) -> dict[str, float]:
    if not analogs:
        return {
            "count": 0,
            "win_rate": 0.0,
            "average_return_pct": 0.0,
            "median_return_pct": 0.0,
            "average_r_multiple": 0.0,
            "average_mae_pct": 0.0,
            "average_mfe_pct": 0.0,
            "return_std_pct": 0.0,
        }
    returns = [_finite_float(a.get("outcome_return_pct")) for a in analogs]
    r_mult = [_finite_float(a.get("r_multiple")) for a in analogs]
    mae = [_finite_float(a.get("mae_pct")) for a in analogs]
    mfe = [_finite_float(a.get("mfe_pct")) for a in analogs]
    wins = [1.0 if a.get("outcome_label") == "win" or _finite_float(a.get("outcome_return_pct")) > 0 else 0.0 for a in analogs]
    return {
        "count": len(analogs),
        "win_rate": float(np.mean(wins)),
        "average_return_pct": float(np.mean(returns)),
        "median_return_pct": float(median(returns)),
        "average_r_multiple": float(np.mean(r_mult)),
        "average_mae_pct": float(np.mean(mae)),
        "average_mfe_pct": float(np.mean(mfe)),
        "return_std_pct": float(np.std(returns)),
    }


def score_edge_candidate(features: dict, analogs: list[dict], min_analogs: int = 5) -> dict:
    """Score a candidate with transparent components and conservative promotion rules.

    Stored retrieval outcomes are gross. Charge the same committed transaction
    cost used by validation before analog expectancy affects a score or a
    promotion; otherwise candidate selection and readiness evaluate different
    strategies.
    """
    gross_summary = _analog_summary(analogs)
    cost_bps_per_side = abs(_finite_float(scanner_config.EDGE_COST_BPS_PER_SIDE))
    net_analogs = apply_transaction_costs(analogs, cost_bps_per_side)
    summary = _analog_summary(net_analogs)
    cost_risk_rows = sum(1 for row in analogs if _finite_float(row.get("risk_pct_used")) > 0.0)
    cost_basis_complete = bool(analogs) and cost_risk_rows == len(analogs)
    research_score = _finite_float(features.get("research_score"))
    rr_ratio = _finite_float(features.get("rr_ratio"))
    empty_space_score = _finite_float(features.get("empty_space_score"))
    potter_passed = bool(features.get("potter_passed"))
    if "empty_space_passed" in features:
        empty_space_passed = bool(features.get("empty_space_passed"))
    else:
        # Older evidence rows predate the explicit pass flag. Reconstruct the
        # same gate used by score_empty_space instead of treating a weak
        # one-point score as a valid setup.
        empty_space_passed = (
            empty_space_score >= float(scanner_config.MIN_EMPTY_SPACE_SCORE)
            and rr_ratio >= float(scanner_config.MIN_RR)
        )
    setup_gate_passed = potter_passed or empty_space_passed
    data_quality = _finite_float(features.get("data_quality_score"), 1.0)
    feed_confidence = _finite_float(features.get("feed_confidence"), 0.5)
    options_spread = _finite_float(features.get("options_spread_pct"))
    options_data_quality = _finite_float(features.get("options_data_quality"), 0.45)
    options_spread_limit = float(scanner_config.MAX_ATM_BID_ASK_SPREAD_PCT)
    options_passed = _finite_float(features.get("options_passed"), 1.0) >= 1.0
    options_provider = str(features.get("options_data_provider") or "").strip()
    doctrine_v2 = 0.0
    if "doctrine_v2_score" in features:
        doctrine_score = _finite_float(features.get("doctrine_v2_score"))
        doctrine_passed = bool(features.get("doctrine_v2_passed"))
        doctrine_failed = bool(features.get("doctrine_v2_failed_reentry"))
        doctrine_v2 = _clamp(
            ((doctrine_score - float(scanner_config.DOCTRINE_V2_SCORE_BASELINE)) * 0.35)
            + (6.0 if doctrine_passed else 0.0)
            - (15.0 if doctrine_failed else 0.0),
            -15.0,
            15.0,
        )

    scorecard = {
        "base": 30.0,
        "setup_quality": _clamp((research_score * 0.15) + (5.0 if potter_passed else 0.0) + (empty_space_score * 2.0), 0.0, 24.0),
        "setup_gate": 0.0 if setup_gate_passed else -25.0,
        "doctrine_v2": doctrine_v2,
        "reward_risk": _clamp(rr_ratio * 4.0, 0.0, 12.0),
        "analog_expectancy": _clamp((summary["average_r_multiple"] * 25.0) + ((summary["win_rate"] - 0.5) * 20.0), -30.0, 35.0),
        # Kronos failed its preregistered production ranking gate. Keep its
        # fields in the feature/report payload for continued measurement, but
        # an advisory model must not move a setup across score thresholds.
        "kronos": 0.0,
        "uncertainty": -_clamp(summary["return_std_pct"], 0.0, 12.0),
        "sample_penalty": -_clamp(max(min_analogs - summary["count"], 0) * 5.0, 0.0, 20.0),
        # Historical rows cannot reconstruct point-in-time feed confidence or
        # option-chain quality. Letting those execution-only fields change the
        # numeric score put live candidates on a different scale from the
        # walk-forward thresholds (good live quotes gained up to 12 points
        # versus otherwise identical historical rows). Preserve the scorecard
        # keys for report compatibility, but enforce these conditions only via
        # the promotion gates and explicit blocking reasons below.
        "data_quality": 0.0,
        "options_liquidity": 0.0,
        "options_data_quality": 0.0,
    }
    raw_score = sum(scorecard.values())
    edge_score = round(_clamp(raw_score, 0.0, 100.0), 2)
    if not setup_gate_passed:
        edge_score = min(edge_score, 44.0)

    if (
        edge_score >= 65.0
        and setup_gate_passed
        and summary["count"] >= min_analogs
        and summary["average_r_multiple"] > 0.0
        and cost_bps_per_side > 0.0
        and cost_basis_complete
        and data_quality >= 0.5
        and feed_confidence >= 0.35
        and options_passed
        and options_spread <= options_spread_limit
        and options_data_quality >= 0.75
    ):
        recommendation = "promote"
    elif edge_score >= 45.0 and setup_gate_passed:
        recommendation = "research"
    else:
        recommendation = "reject"

    blocking_reasons = []
    if not setup_gate_passed:
        blocking_reasons.append("setup_gate_failed")
    if summary["count"] < min_analogs:
        blocking_reasons.append("insufficient_analogs")
    if summary["average_r_multiple"] <= 0.0:
        blocking_reasons.append("non_positive_analog_expectancy")
    if cost_bps_per_side <= 0.0:
        blocking_reasons.append("analog_costs_not_charged")
    elif analogs and not cost_basis_complete:
        blocking_reasons.append("analog_cost_basis_incomplete")
    if data_quality < 0.5:
        blocking_reasons.append("low_data_quality")
    if feed_confidence < 0.35:
        blocking_reasons.append("low_feed_confidence")
    if "options_passed" in features and not options_passed:
        blocking_reasons.append(
            "options_no_liquid_contract" if options_provider else "options_provider_unavailable"
        )
    else:
        if options_spread > options_spread_limit:
            blocking_reasons.append("wide_options_spread")
        if options_data_quality < 0.75:
            blocking_reasons.append("options_data_not_execution_grade")
    if edge_score < 45.0:
        blocking_reasons.append("edge_score_below_research_threshold")
    elif edge_score < 65.0:
        blocking_reasons.append("edge_score_below_promotion_threshold")
    blocking_reasons = _ordered_unique(blocking_reasons)
    rejection_reasons = blocking_reasons if recommendation == "reject" else []

    return {
        "edge_score": edge_score,
        "recommendation": recommendation,
        "scorecard": {key: round(float(value), 4) for key, value in scorecard.items()},
        "analog_summary": summary,
        "gross_analog_summary": gross_summary,
        "analog_cost_model": {
            "bps_per_side": cost_bps_per_side,
            "round_trip_return_pct_charged": round(2.0 * cost_bps_per_side / 100.0, 6),
            "basis": "net_of_costs" if cost_bps_per_side > 0.0 else "gross",
            "risk_coverage_rows": cost_risk_rows,
            "analog_rows": len(analogs),
            "risk_coverage_complete": cost_basis_complete,
            "applies_to": ["returns", "r_multiple", "win_loss_label"],
        },
        "blocking_reasons": blocking_reasons,
        "rejection_reasons": rejection_reasons,
    }
