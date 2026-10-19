from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


FEATURE_VERSION = 4


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    return {
        key: getattr(obj, key)
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _flag(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) <= 1e-12:
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def _volume_expansion(bars: pd.DataFrame, lookback: int = 15) -> float:
    if bars is None or bars.empty or "Volume" not in bars.columns or len(bars) < 2:
        return 0.0
    latest = _finite_float(bars["Volume"].iloc[-1])
    prior = bars["Volume"].iloc[-(lookback + 1) : -1].replace(0, np.nan).dropna()
    if prior.empty:
        return 0.0
    return _safe_ratio(latest, float(prior.mean()))


def _volume_percentile(bars: pd.DataFrame, lookback: int = 60) -> float:
    if bars is None or bars.empty or "Volume" not in bars.columns or len(bars) < 2:
        return 0.0
    latest = _finite_float(bars["Volume"].iloc[-1])
    prior = bars["Volume"].iloc[:-1].tail(lookback).dropna()
    if prior.empty:
        return 0.0
    return float((prior <= latest).mean())


def _realized_volatility_pct(bars: pd.DataFrame, lookback: int = 20) -> float:
    if bars is None or bars.empty or "Close" not in bars.columns or len(bars) < 3:
        return 0.0
    returns = bars["Close"].pct_change().dropna().tail(lookback)
    if returns.empty:
        return 0.0
    return _finite_float(returns.std(ddof=0) * 100.0)


def _recent_return_pct(bars: pd.DataFrame, lookback: int = 20) -> float:
    if bars is None or bars.empty or "Close" not in bars.columns or len(bars) < 2:
        return 0.0
    start_idx = max(0, len(bars) - lookback - 1)
    start = _finite_float(bars["Close"].iloc[start_idx])
    end = _finite_float(bars["Close"].iloc[-1])
    return _safe_ratio(end - start, start) * 100.0


def extract_edge_features(
    ticker: str,
    bars: pd.DataFrame,
    potter_box: Any,
    empty_space: Any = None,
    doctrine_v2: Any = None,
    event_risk: Any = None,
    options_contract: Any = None,
    kronos: Any = None,
    data_quality: dict | None = None,
) -> dict[str, Any]:
    """Return a stable, JSON-safe feature vector for ranking and retrieval."""
    pb = _as_dict(potter_box)
    es = _as_dict(empty_space)
    doctrine = _as_dict(doctrine_v2)
    ev = _as_dict(event_risk)
    opt = _as_dict(options_contract)
    kr = _as_dict(kronos)
    dq = data_quality or {}
    diagnostics = pb.get("diagnostics") or {}
    doctrine_risk_flags = doctrine.get("risk_flags") if isinstance(doctrine.get("risk_flags"), list) else []
    punchback_state = str(doctrine.get("punchback_state") or "unknown")
    cost_basis_state = str(doctrine.get("cost_basis_state") or "unknown")

    latest_close = _finite_float(pb.get("breakout_close"))
    if latest_close == 0.0 and bars is not None and not bars.empty and "Close" in bars.columns:
        latest_close = _finite_float(bars["Close"].iloc[-1])

    control_top = _finite_float(diagnostics.get("control_top", pb.get("box_top")))
    control_bottom = _finite_float(diagnostics.get("control_bottom", pb.get("box_bottom")))
    box_width = max(control_top - control_bottom, 0.0)
    direction = pb.get("direction")
    if direction not in {"bullish", "bearish"}:
        direction = None

    if direction == "bullish":
        breakout_distance_pct = _safe_ratio(latest_close - control_top, control_top) * 100.0
    elif direction == "bearish":
        breakout_distance_pct = _safe_ratio(control_bottom - latest_close, control_bottom) * 100.0
    elif control_top and control_bottom:
        breakout_distance_pct = max(
            _safe_ratio(latest_close - control_top, control_top) * 100.0,
            _safe_ratio(control_bottom - latest_close, control_bottom) * 100.0,
        )
    else:
        breakout_distance_pct = 0.0

    close_position = _safe_ratio(latest_close - control_bottom, box_width)
    latest_ts = None
    if bars is not None and not bars.empty:
        latest_ts = pd.Timestamp(bars.index[-1]).isoformat()

    return {
        "feature_version": FEATURE_VERSION,
        "ticker": ticker,
        "timestamp": latest_ts,
        "direction": direction or "unknown",
        "bar_count": int(len(bars)) if bars is not None else 0,
        "latest_close": latest_close,
        "potter_passed": _flag(pb.get("passed")),
        "empty_space_passed": _flag(es.get("passed")),
        "event_risk_passed": _flag(ev.get("passed", True)),
        "options_passed": _flag(opt.get("passed", True)),
        "kronos_passed": _flag(kr.get("passed", False)),
        "box_top": control_top,
        "box_bottom": control_bottom,
        "box_width_pct": _safe_ratio(box_width, latest_close) * 100.0,
        "close_position_in_box": close_position,
        "breakout_distance_pct": breakout_distance_pct,
        "abs_breakout_distance_pct": abs(breakout_distance_pct),
        "breakout_strength_pct": _finite_float(pb.get("breakout_strength_pct")),
        "atr_value": _finite_float(pb.get("atr_value")),
        "range_compression_ratio": _finite_float(pb.get("range_compression_ratio"), 1.0),
        "no_trend_score": _finite_float(pb.get("no_trend_score")),
        "top_touches": _finite_float(diagnostics.get("top_touches")),
        "bottom_touches": _finite_float(diagnostics.get("bottom_touches")),
        "touch_tolerance": _finite_float(diagnostics.get("touch_tolerance")),
        "volume_expansion": _finite_float(diagnostics.get("volume_expansion"), _volume_expansion(bars)),
        "volume_percentile": _volume_percentile(bars),
        "realized_volatility_pct": _realized_volatility_pct(bars),
        "recent_return_pct": _recent_return_pct(bars),
        "empty_space_score": _finite_float(es.get("score")),
        "rr_ratio": _finite_float(es.get("rr_ratio")),
        "distance_to_target_pct": _finite_float(es.get("distance_to_target_pct")),
        "risk_pct": _finite_float(es.get("risk_pct")),
        "doctrine_v2_passed": _flag(doctrine.get("passed")),
        "doctrine_v2_version": _finite_float(doctrine.get("version")),
        "doctrine_v2_score": _finite_float(doctrine.get("score")),
        "doctrine_v2_box_stack_score": _finite_float(doctrine.get("box_stack_score")),
        "doctrine_v2_punchback_reclaim": _flag(punchback_state == "reclaim"),
        "doctrine_v2_failed_reentry": _flag(
            punchback_state == "failed_reentry" or "failed_reentry" in doctrine_risk_flags
        ),
        "punchback_state": punchback_state,
        "cost_basis_state": cost_basis_state,
        "options_spread_pct": _finite_float(opt.get("spread_pct"), 1.0 if opt else 0.0),
        "options_open_interest": _finite_float(opt.get("open_interest")),
        "options_volume": _finite_float(opt.get("volume")),
        "options_data_provider": opt.get("data_provider"),
        "options_data_feed": opt.get("data_feed"),
        "options_quote_age_minutes": _finite_float(opt.get("quote_age_minutes"), 9999.0 if opt else 0.0),
        "options_source_disagreement_pct": _finite_float(opt.get("source_disagreement_pct")),
        "options_data_quality": _finite_float(opt.get("options_data_quality"), 0.45 if opt else 0.0),
        # None (not 0.0) when Kronos was never consulted: 0.0 reads as "maximum
        # disagreement" downstream and silently penalized every candidate that
        # skipped the Kronos stage.
        "kronos_directional_agreement": _finite_float(kr.get("directional_agreement")) if kr else None,
        "kronos_median_forecast_return_pct": _finite_float(kr.get("median_forecast_return_pct")) if kr else None,
        "kronos_worst_sampled_return_pct": _finite_float(kr.get("worst_sampled_return_pct")) if kr else None,
        "kronos_sample_count": _finite_float(kr.get("sample_count")),
        "data_missing_bars": _finite_float(dq.get("missing_bars")),
        "data_stale_minutes": _finite_float(dq.get("stale_minutes")),
        "data_quality_score": _finite_float(dq.get("quality_score"), 1.0),
        "feed_confidence": _finite_float(dq.get("feed_confidence"), 0.5),
        "data_provider": dq.get("provider"),
        "data_feed": dq.get("feed"),
        "data_delay_minutes": _finite_float(dq.get("delay_minutes")),
        "skip_reason": pb.get("skip_reason") or es.get("skip_reason") or ev.get("skip_reason") or opt.get("skip_reason"),
    }
