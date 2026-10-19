from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class BacktestMetrics:
    signal_count: int
    win_rate: float
    average_5bar_return: float
    median_5bar_return: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    average_r_multiple: float


def compute_backtest_metrics(trades: list[dict]) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = np.array([t["ret_5"] for t in trades], dtype=float)
    wins = np.array([t["win"] for t in trades], dtype=float)
    mae = np.array([t["mae"] for t in trades], dtype=float)
    mfe = np.array([t["mfe"] for t in trades], dtype=float)
    r_mult = np.array([t.get("r_multiple", 0.0) for t in trades], dtype=float)

    return BacktestMetrics(
        signal_count=len(trades),
        win_rate=float(wins.mean()),
        average_5bar_return=float(returns.mean()),
        median_5bar_return=float(np.median(returns)),
        max_adverse_excursion=float(mae.min()),
        max_favorable_excursion=float(mfe.max()),
        average_r_multiple=float(r_mult.mean()),
    )
