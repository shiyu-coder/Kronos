from __future__ import annotations

from datetime import datetime, timezone
import logging
import pandas as pd
import yfinance as yf

from ..config import (
    BLOCK_ON_EX_DIVIDEND,
    BLOCK_ON_UNKNOWN_EARNINGS,
    EARNINGS_BLOCK_DAYS,
    EARNINGS_ESTIMATE_CUSHION_DAYS,
    TIMEZONE,
)
from ..data.market_data import parse_date_like
from ..utils.validation import EventRiskResult


def _extract_earnings_date(calendar_obj) -> datetime | None:
    if calendar_obj is None:
        return None
    if isinstance(calendar_obj, pd.DataFrame) and not calendar_obj.empty:
        for key in ("Earnings Date", "Earnings", "Earnings Date Range"):
            if key in calendar_obj.index:
                value = calendar_obj.loc[key].iloc[0]
                return parse_date_like(value)
        first_val = calendar_obj.iloc[0].iloc[0]
        return parse_date_like(first_val)
    if isinstance(calendar_obj, dict):
        for key in ("Earnings Date", "Earnings", "earningsDate"):
            if key in calendar_obj:
                value = calendar_obj[key]
                if isinstance(value, (list, tuple)) and value:
                    value = value[0]
                return parse_date_like(value)
    return None


def _extract_ex_dividend(info: dict) -> datetime | None:
    value = info.get("exDividendDate")
    if isinstance(value, (int, float)):
        # Yahoo sends midnight UTC of the ex-date; naive fromtimestamp()
        # renders it in local time (Central), shifting the date back a day.
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None
    return parse_date_like(value)


def assess_event_risk(ticker: str, logger: logging.Logger) -> EventRiskResult:
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
        calendar_obj = getattr(yf_ticker, "calendar", None)
        earnings_dt = _extract_earnings_date(calendar_obj)
        ex_dividend_dt = _extract_ex_dividend(info)

        now = pd.Timestamp.now(tz=TIMEZONE)
        if earnings_dt is None:
            if BLOCK_ON_UNKNOWN_EARNINGS:
                return EventRiskResult(
                    passed=False,
                    earnings_date=None,
                    days_to_earnings=None,
                    ex_dividend_date=ex_dividend_dt,
                    status="blocked",
                    skip_reason="earnings data unavailable (fail-closed)",
                )
            return EventRiskResult(True, None, None, ex_dividend_dt, "warning", "earnings data unavailable")

        earnings_ts = pd.Timestamp(earnings_dt)
        if earnings_ts.tz is None:
            earnings_ts = earnings_ts.tz_localize(TIMEZONE)
        else:
            earnings_ts = earnings_ts.tz_convert(TIMEZONE)

        days_to = int((earnings_ts.date() - now.date()).days)
        if days_to < 0:
            # A past date means the provider has not published the NEXT
            # earnings date yet - that is unknown-earnings, not all-clear.
            return EventRiskResult(
                passed=False,
                earnings_date=earnings_ts.to_pydatetime(),
                days_to_earnings=days_to,
                ex_dividend_date=ex_dividend_dt,
                status="blocked",
                skip_reason="stale past earnings date from provider (fail-closed)",
            )
        # Provider dates are estimates; the cushion absorbs off-by-one errors.
        if days_to <= EARNINGS_BLOCK_DAYS + EARNINGS_ESTIMATE_CUSHION_DAYS:
            return EventRiskResult(
                passed=False,
                earnings_date=earnings_ts.to_pydatetime(),
                days_to_earnings=days_to,
                ex_dividend_date=ex_dividend_dt,
                status="blocked",
                skip_reason=(
                    f"earnings within {EARNINGS_BLOCK_DAYS} days "
                    f"(+{EARNINGS_ESTIMATE_CUSHION_DAYS}d estimate cushion)"
                ),
            )

        if ex_dividend_dt is not None and BLOCK_ON_EX_DIVIDEND:
            return EventRiskResult(
                passed=False,
                earnings_date=earnings_ts.to_pydatetime(),
                days_to_earnings=days_to,
                ex_dividend_date=ex_dividend_dt,
                status="blocked",
                skip_reason="ex-dividend date blocked by config",
            )

        status = "pass"
        if ex_dividend_dt is not None:
            status = "pass_ex_dividend_flagged"
        return EventRiskResult(
            passed=True,
            earnings_date=earnings_ts.to_pydatetime(),
            days_to_earnings=days_to,
            ex_dividend_date=ex_dividend_dt,
            status=status,
            skip_reason=None,
        )
    except Exception as exc:
        logger.error("%s event risk error: %s", ticker, exc)
        return EventRiskResult(
            passed=False,
            earnings_date=None,
            days_to_earnings=None,
            ex_dividend_date=None,
            status="blocked",
            skip_reason=f"event data error: {exc}",
        )
