from __future__ import annotations

from dataclasses import dataclass

from polymarket_bot.config.settings import RiskConfig, StrategyConfig


@dataclass
class TradeDecision:
    should_trade: bool
    side: str | None
    size_fraction: float
    expected_value: float
    reason: str


class DecisionEngine:
    def __init__(self, risk: RiskConfig, strategy: StrategyConfig):
        self.risk = risk
        self.strategy = strategy

    def evaluate(self, prob_yes: float, yes_price: float, liquidity: float) -> TradeDecision:
        if liquidity < self.risk.min_liquidity:
            return TradeDecision(False, None, 0.0, 0.0, "liquidity_too_low")

        if prob_yes < self.risk.min_probability:
            return TradeDecision(False, None, 0.0, 0.0, "probability_below_threshold")

        b = (1 - yes_price) / yes_price
        q = 1 - prob_yes
        kelly_f = max(0.0, (b * prob_yes - q) / b) if b > 0 else 0.0
        size = min(kelly_f, self.strategy.kelly_fraction_cap, self.risk.max_risk_per_trade)

        ev = prob_yes * (1 - yes_price) - (1 - prob_yes) * yes_price - self.strategy.fee_rate
        if ev <= 0 or size <= 0:
            return TradeDecision(False, None, 0.0, ev, "non_positive_ev_or_size")

        return TradeDecision(True, "YES", size, ev, "ok")
