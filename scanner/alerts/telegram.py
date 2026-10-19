from __future__ import annotations

import logging
import time
import requests


def _fmt_num(value, precision: int = 4, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct_ratio(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def render_alert_message(candidate) -> str:
    pb = candidate.potter_box
    es = candidate.empty_space
    ev = candidate.event_risk
    op = candidate.options_contract
    kr = candidate.kronos
    ai = candidate.ai_insight or {}
    ai_block = ""
    if ai:
        ai_block = (
            f"\nMiniMax AI:\n"
            f"Status: {ai.get('status')}\n"
            f"Score Band: {ai.get('score_band')}\n"
            f"Confidence: {ai.get('confidence')}\n"
            f"Rationale: {ai.get('rationale')}\n"
            f"Flags: {', '.join(ai.get('red_flags', [])) if ai.get('red_flags') else 'None'}\n"
        )

    return (
        f"${candidate.ticker} — POTTER BOX TRADE CANDIDATE\n\n"
        f"Direction: {candidate.direction}\n"
        f"Box Top: {_fmt_num(pb.box_top, 4)}\n"
        f"Box Bottom: {_fmt_num(pb.box_bottom, 4)}\n"
        f"Cost Basis: {_fmt_num(pb.cost_basis, 4)}\n"
        f"Breakout Close: {_fmt_num(pb.breakout_close, 4)}\n"
        f"Breakout Strength: {_fmt_num(pb.breakout_strength_pct, 2, '%')}\n\n"
        f"Empty Space:\n"
        f"Score: {es.score}\n"
        f"Nearest Target: {_fmt_num(es.nearest_target, 4)}\n"
        f"Distance to Target: {_fmt_num(es.distance_to_target_pct, 2, '%')}\n"
        f"Invalidation: {_fmt_num(es.invalidation_level, 4)}\n"
        f"R/R: {_fmt_num(es.rr_ratio, 2)}\n\n"
        f"Event Risk:\n"
        f"Earnings: {ev.earnings_date}\n"
        f"Ex-Dividend: {ev.ex_dividend_date}\n\n"
        f"Options:\n"
        f"Expiration: {op.expiration}\n"
        f"Strike: {op.strike}\n"
        f"Bid: {op.bid}\n"
        f"Ask: {op.ask}\n"
        f"Spread: {_fmt_pct_ratio(op.spread_pct)}\n"
        f"Open Interest: {op.open_interest}\n"
        f"Volume: {op.volume}\n"
        f"IV: {op.implied_volatility}\n\n"
        f"Kronos:\n"
        f"Directional Agreement: {_fmt_pct_ratio(kr.directional_agreement)}\n"
        f"Median 5-Day Forecast: {_fmt_num(kr.median_forecast_return_pct, 2, '%')}\n"
        f"Worst Sampled Forecast: {_fmt_num(kr.worst_sampled_return_pct, 2, '%')}\n"
        f"{ai_block}\n"
        "Rule:\n"
        "Setup invalid if synthetic 24h close returns inside box or closes back through cost basis."
    )


def send_telegram_message(token: str, chat_id: str, message: str, logger: logging.Logger) -> bool:
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                json={"chat_id": chat_id, "text": message},
                timeout=20,
            )
            req_id = response.headers.get("X-Request-ID")
            if req_id:
                logger.info("Telegram request-id: %s", req_id)
            if response.status_code == 200:
                return True
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(0.8 * (attempt + 1))
                continue
            logger.error("Telegram send failed: %s %s", response.status_code, response.text)
            return False
        except Exception as exc:
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
                continue
            logger.error("Telegram request error: %s", exc)
            return False
    return False
