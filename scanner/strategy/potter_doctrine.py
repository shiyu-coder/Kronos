from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd


DOCTRINE_V2_EVIDENCE_VERSION = 2


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    return {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if pd.notna(out) else default


def _control_levels(pb: dict) -> tuple[float, float, float]:
    diagnostics = pb.get("diagnostics") if isinstance(pb.get("diagnostics"), dict) else {}
    top = _finite_float(diagnostics.get("control_top", pb.get("box_top")))
    bottom = _finite_float(diagnostics.get("control_bottom", pb.get("box_bottom")))
    cost_basis = _finite_float(pb.get("cost_basis"), (top + bottom) / 2.0 if top and bottom else 0.0)
    return top, bottom, cost_basis


def _infer_direction(pb: dict, latest_close: float, top: float, bottom: float) -> str | None:
    direction = pb.get("direction")
    if direction in {"bullish", "bearish"}:
        return direction
    if top and latest_close > top:
        return "bullish"
    if bottom and latest_close < bottom:
        return "bearish"
    return None


def _punchback_state(bars: pd.DataFrame, direction: str | None, top: float, bottom: float, tolerance: float) -> str:
    if direction not in {"bullish", "bearish"} or bars is None or len(bars) < 2:
        return "unknown"
    recent = bars.tail(3)
    latest = float(recent["Close"].iloc[-1])
    prior = recent.iloc[:-1]
    if direction == "bullish":
        if latest <= top:
            return "failed_reentry"
        breakout_seen = False
        for _, bar in prior.iterrows():
            close = float(bar["Close"])
            intersects = float(bar["Low"]) <= top + tolerance and float(bar["High"]) >= top - tolerance
            # A control touch is a retest only after an earlier close already
            # established the breakout. The old any-touch heuristic treated a
            # normal consolidation touch as a post-breakout reclaim and gave
            # initial breaks the stronger doctrine bonus.
            if breakout_seen and intersects and close > top:
                return "reclaim"
            if close > top:
                breakout_seen = True
        return "fresh_breakout"
    if latest >= bottom:
        return "failed_reentry"
    breakdown_seen = False
    for _, bar in prior.iterrows():
        close = float(bar["Close"])
        intersects = float(bar["High"]) >= bottom - tolerance and float(bar["Low"]) <= bottom + tolerance
        if breakdown_seen and intersects and close < bottom:
            return "reclaim"
        if close < bottom:
            breakdown_seen = True
    return "fresh_breakout"


def _cost_basis_state(bars: pd.DataFrame, direction: str | None, cost_basis: float) -> str:
    if direction not in {"bullish", "bearish"} or bars is None or bars.empty or cost_basis <= 0:
        return "unknown"
    recent = bars.tail(5)
    latest = float(recent["Close"].iloc[-1])
    prior = recent["Close"].iloc[:-1]
    if direction == "bullish":
        if latest < cost_basis:
            return "lost"
        if bool((prior < cost_basis).any()):
            return "reclaimed"
        return "held"
    if latest > cost_basis:
        return "lost"
    if bool((prior > cost_basis).any()):
        return "reclaimed"
    return "held"


def _box_stack_score(bars: pd.DataFrame, top: float, bottom: float) -> float:
    if bars is None or bars.empty or top <= bottom:
        return 0.0
    latest = _finite_float(bars["Close"].iloc[-1], 1.0)
    control_width_pct = ((top - bottom) / max(latest, 1e-9)) * 100.0
    score = 0.0
    for lookback in (15, 30, 60):
        if len(bars) < max(lookback, 5):
            continue
        window = bars.tail(lookback)
        width_pct = ((float(window["High"].max()) - float(window["Low"].min())) / max(latest, 1e-9)) * 100.0
        if width_pct <= max(control_width_pct * 2.5, 8.0):
            score += 5.0
    return min(score, 15.0)


def score_potter_doctrine_v2(ticker: str, bars: pd.DataFrame, potter_box: Any, empty_space: Any = None) -> dict:
    """Score public Potter-style mechanics as research-only evidence."""
    pb = _as_dict(potter_box)
    es = _as_dict(empty_space)
    if bars is None or bars.empty:
        return {
            "version": DOCTRINE_V2_EVIDENCE_VERSION,
            "ticker": ticker,
            "passed": False,
            "score": 0,
            "direction": None,
            "reason": "missing bars",
            "reasons": [],
            "risk_flags": ["missing_bars"],
        }

    top, bottom, cost_basis = _control_levels(pb)
    latest_close = _finite_float(pb.get("breakout_close"), _finite_float(bars["Close"].iloc[-1]))
    direction = _infer_direction(pb, latest_close, top, bottom)
    diagnostics = pb.get("diagnostics") if isinstance(pb.get("diagnostics"), dict) else {}
    tolerance = max(_finite_float(diagnostics.get("touch_tolerance")), abs(top - bottom) * 0.05, latest_close * 0.001)

    punchback = _punchback_state(bars, direction, top, bottom, tolerance)
    cost_state = _cost_basis_state(bars, direction, cost_basis)
    stack_score = _box_stack_score(bars, top, bottom)

    score = 0.0
    reasons: list[str] = []
    risk_flags: list[str] = []

    if direction in {"bullish", "bearish"}:
        score += 10
        reasons.append(f"{direction}_bias")

    top_touches = _finite_float(diagnostics.get("top_touches"))
    bottom_touches = _finite_float(diagnostics.get("bottom_touches"))
    if top_touches >= 2 and bottom_touches >= 2:
        score += 15
        reasons.append("balanced_box_touches")

    if punchback == "reclaim":
        score += 30
        reasons.append("punchback_reclaim")
    elif punchback == "fresh_breakout":
        score += 18
        reasons.append("fresh_breakout")
    elif punchback == "failed_reentry":
        score -= 20
        risk_flags.append("failed_reentry")

    if cost_state in {"held", "reclaimed"}:
        score += 15
        reasons.append(f"cost_basis_{cost_state}")
    elif cost_state == "lost":
        score -= 15
        risk_flags.append("cost_basis_lost")

    if stack_score > 0:
        score += stack_score
        reasons.append("box_stack_alignment")

    empty_space_score = _finite_float(es.get("score"))
    if es.get("passed") or empty_space_score >= 2:
        score += 10
        reasons.append("empty_space_available")

    final_score = int(max(0.0, min(100.0, round(score))))
    passed = final_score >= 70 and "failed_reentry" not in risk_flags and "cost_basis_lost" not in risk_flags
    return {
        "version": DOCTRINE_V2_EVIDENCE_VERSION,
        "ticker": ticker,
        "passed": passed,
        "score": final_score,
        "direction": direction,
        "punchback_state": punchback,
        "cost_basis_state": cost_state,
        "box_stack_score": round(float(stack_score), 4),
        "reasons": reasons,
        "risk_flags": risk_flags,
        "reason": "potter_doctrine_v2_candidate" if passed else "potter_doctrine_v2_below_threshold",
        "control_top": top,
        "control_bottom": bottom,
        "cost_basis": cost_basis,
    }
