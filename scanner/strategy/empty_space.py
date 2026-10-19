from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as scanner_config
from ..strategy.risk_reward import compute_rr
from ..utils.validation import EmptySpaceResult


def score_empty_space(
    bars: pd.DataFrame,
    direction: str,
    breakout_close: float,
    cost_basis: float,
    lookback_bars: int = 120,
) -> EmptySpaceResult:
    hist = bars.iloc[:-1].tail(lookback_bars)
    if hist.empty:
        return EmptySpaceResult(False, 0, None, None, cost_basis, None, None, "historical bars", "insufficient history", {})

    next_target: float | None = None
    if direction == "bullish":
        resistances = hist[hist["High"] > breakout_close]["High"]
        nearest_target = float(resistances.min()) if not resistances.empty else float(hist["High"].max())
        further = resistances[resistances > nearest_target]
        if not further.empty:
            next_target = float(further.min())
    else:
        supports = hist[hist["Low"] < breakout_close]["Low"]
        nearest_target = float(supports.max()) if not supports.empty else float(hist["Low"].min())
        further = supports[supports < nearest_target]
        if not further.empty:
            next_target = float(further.max())

    rr, reward_abs, risk_abs = compute_rr(
        entry=breakout_close,
        target=nearest_target,
        invalidation=cost_basis,
        direction=direction,
    )

    risk_pct = (risk_abs / breakout_close) * 100 if breakout_close else None
    dist_pct = (reward_abs / breakout_close) * 100 if breakout_close else None

    score = 0
    if rr >= 2.5:
        score = 3
    elif rr >= 1.5:
        score = 2
    elif rr >= 1.0:
        score = 1

    passed = score >= scanner_config.MIN_EMPTY_SPACE_SCORE and rr >= scanner_config.MIN_RR
    reason = None if passed else f"empty space score {score} / rr {rr:.2f} below thresholds"

    return EmptySpaceResult(
        passed=passed,
        score=score,
        nearest_target=nearest_target,
        distance_to_target_pct=dist_pct,
        invalidation_level=cost_basis,
        risk_pct=risk_pct,
        rr_ratio=rr,
        support_resistance_source="rolling_swing_levels",
        skip_reason=reason,
        diagnostics={
            "reward_abs": reward_abs,
            "risk_abs": risk_abs,
            "lookback_bars": lookback_bars,
            "next_target": next_target,
            "distance_to_next_target_pct": (
                abs(next_target - breakout_close) / breakout_close * 100
                if next_target is not None and breakout_close
                else None
            ),
        },
    )
