from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class MarketState:
    market_id: str
    yes_price: float
    no_price: float
    liquidity: float
    history_yes_price: list[float]


class PolyMarketCollector:
    """Mock collector; thay bằng PolyMarket API ở bước tích hợp thật."""

    def fetch_market_state(self, market_id: str, lookback: int = 256) -> MarketState:
        base = 0.55
        history = [max(0.01, min(0.99, base + random.uniform(-0.03, 0.03))) for _ in range(lookback)]
        yes_price = history[-1]
        no_price = 1.0 - yes_price
        liquidity = random.uniform(6_000, 20_000)
        return MarketState(
            market_id=market_id,
            yes_price=yes_price,
            no_price=no_price,
            liquidity=liquidity,
            history_yes_price=history,
        )
