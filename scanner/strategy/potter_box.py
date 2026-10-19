from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as scanner_config
from ..config import (
    ATR_PERIOD,
    BOX_TOUCH_TOLERANCE_PCT,
    CONSOLIDATION_BARS,
    MIN_BOX_BOTTOM_TOUCHES,
    MIN_BOX_TOP_TOUCHES,
    RESEARCH_MIN_VOLUME_EXPANSION,
    RESEARCH_NEAR_BREAKOUT_PCT,
    USE_CLOSE_BASED_CONTROL,
)
from ..utils.validation import PotterBoxResult


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _count_touches(closes: pd.Series, level: float, tolerance: float, side: str) -> int:
    if side == "top":
        return int((closes >= (level - tolerance)).sum())
    return int((closes <= (level + tolerance)).sum())


def detect_potter_box(ticker: str, synthetic_bars: pd.DataFrame) -> PotterBoxResult:
    # The source-derived Potter setup needs a complete consolidation window,
    # its prior close, and the breakout candle. ATR/range compression remain
    # useful diagnostics, but missing extra lookback must not reject an
    # otherwise documented control/touch/break setup.
    min_needed = CONSOLIDATION_BARS + 2
    if synthetic_bars is None or len(synthetic_bars) < min_needed:
        return PotterBoxResult(
            ticker=ticker,
            passed=False,
            direction=None,
            box_top=None,
            box_bottom=None,
            cost_basis=None,
            prior_close=None,
            breakout_close=None,
            breakout_strength_pct=None,
            atr_value=None,
            range_compression_ratio=None,
            no_trend_score=None,
            skip_reason=f"not enough synthetic bars ({len(synthetic_bars) if synthetic_bars is not None else 0})",
            diagnostics={},
        )

    df = synthetic_bars.copy().sort_index()
    breakout = df.iloc[-1]
    prior_close = float(df.iloc[-2]["Close"])

    # Consolidation window excludes the latest breakout candle.
    cons = df.iloc[-(CONSOLIDATION_BARS + 1):-1].copy()
    box_top = float(cons["High"].max())
    box_bottom = float(cons["Low"].min())
    close_top_control = float(cons["Close"].max())
    close_bottom_control = float(cons["Close"].min())
    control_top = close_top_control if USE_CLOSE_BASED_CONTROL else box_top
    control_bottom = close_bottom_control if USE_CLOSE_BASED_CONTROL else box_bottom
    cost_basis = (control_top + control_bottom) / 2.0

    cons_ranges = cons["High"] - cons["Low"]
    avg_cons_range = float(cons_ranges.mean())

    pre_breakout_df = df.iloc[:-1].copy()
    atr_series = _atr(pre_breakout_df, ATR_PERIOD)
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None

    prior_window = pre_breakout_df.iloc[-(CONSOLIDATION_BARS + ATR_PERIOD):-CONSOLIDATION_BARS]
    prior_ranges = (prior_window["High"] - prior_window["Low"]).dropna()
    prior_avg_range = float(prior_ranges.mean()) if not prior_ranges.empty else None

    range_compression_ratio = (
        avg_cons_range / prior_avg_range
        if prior_avg_range is not None and prior_avg_range > 0
        else None
    )

    cons_closes = cons["Close"].values
    x = np.arange(len(cons_closes))
    slope = np.polyfit(x, cons_closes, 1)[0] / (np.mean(cons_closes) + 1e-9)
    no_trend_score = abs(float(slope))

    breakout_close = float(breakout["Close"])
    breakout_open = float(breakout["Open"])
    direction = None
    breakout_strength_pct = None
    box_range = max(control_top - control_bottom, 1e-9)
    touch_tol = max(box_range * 0.10, breakout_close * BOX_TOUCH_TOLERANCE_PCT)
    top_touches = _count_touches(cons["Close"], control_top, touch_tol, "top")
    bottom_touches = _count_touches(cons["Close"], control_bottom, touch_tol, "bottom")

    bullish = breakout_close > control_top and prior_close > cost_basis
    bearish = breakout_close < control_bottom and prior_close < cost_basis

    if bullish:
        direction = "bullish"
        breakout_strength_pct = ((breakout_close - control_top) / max(control_top, 1e-9)) * 100
    elif bearish:
        direction = "bearish"
        breakout_strength_pct = ((control_bottom - breakout_close) / max(control_bottom, 1e-9)) * 100

    checks = {
        # Compression values are advisory research/ranking features. The
        # source-derived doctrine does not make them setup pass conditions.
        "atr_compressed": (
            atr_value is not None
            and prior_avg_range is not None
            and prior_avg_range > 0
            and atr_value <= scanner_config.ATR_COMPRESSION * prior_avg_range
        ),
        "range_compressed": (
            range_compression_ratio is not None
            and range_compression_ratio <= scanner_config.RANGE_COMPRESSION
        ),
        "no_trend": no_trend_score <= scanner_config.NO_TREND_SLOPE_ABS_MAX,
        "top_touches_ok": top_touches >= MIN_BOX_TOP_TOUCHES,
        "bottom_touches_ok": bottom_touches >= MIN_BOX_BOTTOM_TOUCHES,
        "bullish_breakout": bullish,
        "bearish_breakdown": bearish,
    }

    passed = (
        checks["top_touches_ok"]
        and checks["bottom_touches_ok"]
        and (bullish or bearish)
    )
    reason = None if passed else "no valid Potter Box breakout/breakdown"

    return PotterBoxResult(
        ticker=ticker,
        passed=passed,
        direction=direction,
        box_top=box_top,
        box_bottom=box_bottom,
        cost_basis=cost_basis,
        prior_close=prior_close,
        breakout_close=breakout_close,
        breakout_strength_pct=breakout_strength_pct,
        atr_value=atr_value,
        range_compression_ratio=range_compression_ratio,
        no_trend_score=no_trend_score,
        skip_reason=reason,
        diagnostics={
            **checks,
            "prior_avg_range": prior_avg_range,
            "avg_consolidation_range": avg_cons_range,
            "control_top": control_top,
            "control_bottom": control_bottom,
            "touch_tolerance": touch_tol,
            "top_touches": top_touches,
            "bottom_touches": bottom_touches,
            "breakout_open": breakout_open,
            "breakout_open_outside_box": (
                breakout_open > control_top if bullish else breakout_open < control_bottom if bearish else False
            ),
        },
    )


def score_potter_research_candidate(pb: PotterBoxResult, synthetic_bars: pd.DataFrame) -> dict:
    """Grade near-miss setups for research logging without relaxing live alerts."""
    if synthetic_bars is None or synthetic_bars.empty or pb.breakout_close is None:
        return {"passed": False, "score": 0, "direction": None, "reason": "missing bars or entry"}

    diagnostics = dict(pb.diagnostics or {})
    control_top = diagnostics.get("control_top", pb.box_top)
    control_bottom = diagnostics.get("control_bottom", pb.box_bottom)
    if control_top is None or control_bottom is None or control_top <= control_bottom:
        return {"passed": False, "score": 0, "direction": None, "reason": "missing box controls"}

    close = float(pb.breakout_close)
    top = float(control_top)
    bottom = float(control_bottom)
    near_pct = float(RESEARCH_NEAR_BREAKOUT_PCT)
    direction = pb.direction
    breakout_state = "inside"
    distance_to_breakout_pct = None

    if close > top:
        direction = "bullish"
        breakout_state = "confirmed_breakout"
        distance_to_breakout_pct = ((close - top) / max(top, 1e-9)) * 100.0
    elif close < bottom:
        direction = "bearish"
        breakout_state = "confirmed_breakdown"
        distance_to_breakout_pct = ((bottom - close) / max(bottom, 1e-9)) * 100.0
    elif close >= top * (1.0 - near_pct):
        direction = "bullish"
        breakout_state = "near_breakout"
        distance_to_breakout_pct = ((close - top) / max(top, 1e-9)) * 100.0
    elif close <= bottom * (1.0 + near_pct):
        direction = "bearish"
        breakout_state = "near_breakdown"
        distance_to_breakout_pct = ((bottom - close) / max(bottom, 1e-9)) * 100.0

    if direction not in {"bullish", "bearish"}:
        return {
            "passed": False,
            "score": 0,
            "direction": None,
            "reason": "not near box edge",
            "breakout_state": breakout_state,
        }

    score = 0
    reasons: list[str] = []
    if breakout_state.startswith("confirmed"):
        score += 35
        reasons.append(breakout_state)
    else:
        score += 22
        reasons.append(breakout_state)

    if diagnostics.get("atr_compressed"):
        score += 10
        reasons.append("atr_compressed")
    if diagnostics.get("range_compressed"):
        score += 10
        reasons.append("range_compressed")
    if diagnostics.get("no_trend"):
        score += 10
        reasons.append("no_trend")
    if diagnostics.get("top_touches_ok"):
        score += 5
        reasons.append("top_touches_ok")
    if diagnostics.get("bottom_touches_ok"):
        score += 5
        reasons.append("bottom_touches_ok")

    prior_close = pb.prior_close
    if prior_close is not None and pb.cost_basis is not None:
        if direction == "bullish" and float(prior_close) > float(pb.cost_basis):
            score += 8
            reasons.append("cost_basis_bias")
        elif direction == "bearish" and float(prior_close) < float(pb.cost_basis):
            score += 8
            reasons.append("cost_basis_bias")

    df = synthetic_bars.sort_index()
    cons = df.iloc[-(CONSOLIDATION_BARS + 1):-1]
    breakout = df.iloc[-1]
    volume_expansion = None
    if "Volume" in df.columns and not cons.empty:
        avg_volume = float(cons["Volume"].replace(0, pd.NA).dropna().mean()) if not cons["Volume"].dropna().empty else 0.0
        breakout_volume = float(breakout.get("Volume", 0.0) or 0.0)
        if avg_volume > 0:
            volume_expansion = breakout_volume / avg_volume
            if volume_expansion >= RESEARCH_MIN_VOLUME_EXPANSION:
                score += 10
                reasons.append("volume_expansion")

    bar_range = max(float(breakout["High"]) - float(breakout["Low"]), 1e-9)
    close_location = (close - float(breakout["Low"])) / bar_range
    if (direction == "bullish" and close_location >= 0.60) or (direction == "bearish" and close_location <= 0.40):
        score += 5
        reasons.append("strong_close_location")

    min_score = int(scanner_config.RESEARCH_CANDIDATE_MIN_SCORE)
    passed = score >= min_score
    return {
        "passed": passed,
        "score": int(score),
        "direction": direction,
        "entry_price": close,
        "reason": "research_candidate" if passed else "score below research threshold",
        "reasons": reasons,
        "breakout_state": breakout_state,
        "distance_to_breakout_pct": distance_to_breakout_pct,
        "volume_expansion": volume_expansion,
        "min_score": min_score,
    }
