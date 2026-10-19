"""Independent second-source verification of daily evidence bars.

Alpaca's split adjustment has documented lapses (missing/wrong split factors
fixed reactively as recently as Nov 2025). A corrupted daily series walks
straight into triple-barrier outcomes, so the evidence index cross-checks
each ticker's daily RETURNS against Tradier's history endpoint whenever a
production token is configured. Returns (not levels) are compared because the
two vendors sit on different adjustment bases - a constant scale factor
cancels in return space, while a missed split shows up as one enormous
single-day divergence.

The check is opportunistic hardening: no token or a transport failure means
"skipped", never a block. A genuine disagreement is authoritative and fails
the ticker out of the index for the run.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from .options_data import _tradier_get

# A missed split shows a >=33% one-day return divergence; ordinary
# adjustment-basis noise (dividends on the ex-date, closing-auction rounding)
# stays under ~2-3%. Flag days over the noise band; verdict "disagreement"
# needs either one split-scale day or a systemic flagged-day rate.
RETURN_DIFF_FLAG_PP = 3.0
RETURN_DIFF_SPLIT_PP = 20.0
FLAGGED_RATE_FAIL = 0.05
MIN_COMPARED_SESSIONS = 20


def _tradier_daily_closes(ticker: str, start: str, end: str, token: str, logger: logging.Logger) -> dict[str, float]:
    payload = _tradier_get(
        "/markets/history",
        {"symbol": ticker, "interval": "daily", "start": start, "end": end, "session_filter": "open"},
        token,
        logger,
    )
    if not isinstance(payload, dict):
        return {}
    history = payload.get("history")
    if not isinstance(history, dict):
        return {}
    days = history.get("day")
    if days is None:
        return {}
    if isinstance(days, dict):
        days = [days]
    closes: dict[str, float] = {}
    for row in days:
        if not isinstance(row, dict):
            continue
        try:
            closes[str(row["date"])] = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
    return closes


def cross_check_daily_bars(
    ticker: str,
    daily: pd.DataFrame,
    logger: logging.Logger,
    *,
    lookback_sessions: int = 120,
) -> dict:
    """Compare the tail of a daily bar frame against Tradier daily history.

    Returns a dict with `status` in {"ok", "disagreement", "skipped"} plus
    comparison metrics for the index report.
    """
    token = os.getenv("TRADIER_API_TOKEN", "").strip()
    if not token:
        return {"status": "skipped", "reason": "no_tradier_token"}
    if daily is None or daily.empty or "Close" not in daily.columns:
        return {"status": "skipped", "reason": "no_primary_bars"}

    tail = daily.tail(lookback_sessions)
    start = pd.Timestamp(tail.index[0]).date().isoformat()
    end = pd.Timestamp(tail.index[-1]).date().isoformat()
    reference = _tradier_daily_closes(ticker, start, end, token, logger)
    if not reference:
        return {"status": "skipped", "reason": "tradier_unavailable"}

    primary = {pd.Timestamp(ts).date().isoformat(): float(close) for ts, close in tail["Close"].items()}
    common = sorted(set(primary) & set(reference))
    if len(common) < MIN_COMPARED_SESSIONS + 1:
        return {"status": "skipped", "reason": f"only {max(len(common) - 1, 0)} overlapping sessions"}

    flagged_days: list[str] = []
    worst_diff_pp = 0.0
    for prev_day, day in zip(common, common[1:]):
        if primary[prev_day] <= 0 or reference[prev_day] <= 0:
            continue
        primary_ret = (primary[day] / primary[prev_day] - 1.0) * 100.0
        reference_ret = (reference[day] / reference[prev_day] - 1.0) * 100.0
        diff = abs(primary_ret - reference_ret)
        worst_diff_pp = max(worst_diff_pp, diff)
        if diff > RETURN_DIFF_FLAG_PP:
            flagged_days.append(day)

    compared = len(common) - 1
    flagged_rate = len(flagged_days) / compared if compared else 0.0
    result = {
        "compared_sessions": compared,
        "flagged_days": flagged_days[:10],
        "flagged_rate": round(flagged_rate, 4),
        "worst_return_diff_pp": round(worst_diff_pp, 3),
    }
    if worst_diff_pp > RETURN_DIFF_SPLIT_PP:
        result["status"] = "disagreement"
        result["reason"] = (
            f"split-scale return divergence vs Tradier ({worst_diff_pp:.1f}pp on one session)"
        )
    elif flagged_rate > FLAGGED_RATE_FAIL:
        result["status"] = "disagreement"
        result["reason"] = (
            f"{len(flagged_days)}/{compared} sessions diverge >{RETURN_DIFF_FLAG_PP:g}pp vs Tradier"
        )
    else:
        result["status"] = "ok"
    return result
