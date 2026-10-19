from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..backtest.metrics import compute_backtest_metrics
from ..config import PRED_DAYS, REPORT_DIR
from ..data.market_data import fetch_daily_bars, fetch_intraday_bars
from ..data.synthetic_sessions import build_synthetic_sessions
from ..strategy.empty_space import score_empty_space
from ..strategy.potter_box import detect_potter_box


def _simulate_over_series(ticker: str, bars: pd.DataFrame, skip_reasons: dict[str, int]) -> list[dict]:
    trades = []
    if bars is None or len(bars) < 50:
        skip_reasons["insufficient_history"] = skip_reasons.get("insufficient_history", 0) + 1
        return trades

    for idx in range(30, len(bars) - PRED_DAYS):
        window = bars.iloc[: idx + 1]
        pb = detect_potter_box(ticker, window)
        if not pb.passed or pb.direction is None:
            skip_reasons[pb.skip_reason or "potter_fail"] = skip_reasons.get(pb.skip_reason or "potter_fail", 0) + 1
            continue

        es = score_empty_space(window, pb.direction, pb.breakout_close, pb.cost_basis)
        if not es.passed:
            skip_reasons[es.skip_reason or "empty_space_fail"] = skip_reasons.get(es.skip_reason or "empty_space_fail", 0) + 1
            continue

        fwd = bars.iloc[idx + 1 : idx + 1 + PRED_DAYS]
        if len(fwd) < PRED_DAYS:
            continue

        entry = pb.breakout_close
        close_5 = float(fwd.iloc[-1]["Close"])
        if pb.direction == "bullish":
            ret_5 = ((close_5 - entry) / entry) * 100.0
            mae = ((float(fwd["Low"].min()) - entry) / entry) * 100.0
            mfe = ((float(fwd["High"].max()) - entry) / entry) * 100.0
        else:
            ret_5 = ((entry - close_5) / entry) * 100.0
            mae = ((entry - float(fwd["High"].max())) / entry) * 100.0
            mfe = ((entry - float(fwd["Low"].min())) / entry) * 100.0

        risk = max(abs((entry - pb.cost_basis) / entry), 1e-9)
        r_multiple = (ret_5 / 100.0) / risk

        trades.append(
            {
                "ticker": ticker,
                "direction": pb.direction,
                "ret_5": ret_5,
                "mae": mae,
                "mfe": mfe,
                "r_multiple": r_multiple,
                "win": 1 if ret_5 > 0 else 0,
            }
        )
    return trades


def _write_report(filename: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / filename
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def run_intraday_60d_backtest(watchlist: list[str], logger: logging.Logger) -> dict:
    all_trades = []
    skip_reasons: dict[str, int] = {}

    for ticker in watchlist:
        try:
            intraday = fetch_intraday_bars(ticker)
            synthetic, _ = build_synthetic_sessions(intraday, 20, 0, "30m", True)
            trades = _simulate_over_series(ticker, synthetic, skip_reasons)
            all_trades.extend(trades)
        except Exception as exc:
            reason = f"data_error:{type(exc).__name__}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    metrics = compute_backtest_metrics(all_trades)
    payload = {
        "mode": "backtest_intraday_60d",
        "label": "synthetic intraday yfinance (30m prepost=true)",
        "metrics": metrics.__dict__,
        "skipped_ticker_count": int(sum(skip_reasons.values())),
        "skip_reasons": skip_reasons,
    }
    report_path = _write_report("intraday_60d_report.json", payload)
    logger.info("INTRADAY_BACKTEST_REPORT: %s", json.dumps(payload))
    logger.info("Intraday report saved: %s", str(report_path.resolve()))
    return payload


def run_daily_proxy_2y_backtest(watchlist: list[str], logger: logging.Logger) -> dict:
    all_trades = []
    skip_reasons: dict[str, int] = {}

    for ticker in watchlist:
        try:
            daily = fetch_daily_bars(ticker)
            trades = _simulate_over_series(ticker, daily, skip_reasons)
            all_trades.extend(trades)
        except Exception as exc:
            reason = f"data_error:{type(exc).__name__}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    metrics = compute_backtest_metrics(all_trades)
    payload = {
        "mode": "backtest_daily_proxy_2y",
        "label": "daily proxy only (not true 24h ETH validation)",
        "metrics": metrics.__dict__,
        "skipped_ticker_count": int(sum(skip_reasons.values())),
        "skip_reasons": skip_reasons,
    }
    report_path = _write_report("daily_proxy_2y_report.json", payload)
    logger.info("DAILY_PROXY_BACKTEST_REPORT: %s", json.dumps(payload))
    logger.info("Daily proxy report saved: %s", str(report_path.resolve()))
    return payload
