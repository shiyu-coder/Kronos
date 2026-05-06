from __future__ import annotations

import sys
from pathlib import Path


# Allow running via `python main.py` when cwd is `Kronos/polymarket_bot`.
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from polymarket_bot.config.settings import RiskConfig, RuntimeConfig, StrategyConfig
from polymarket_bot.data.collector import PolyMarketCollector
from polymarket_bot.decision.engine import DecisionEngine
from polymarket_bot.execution.executor import ExecutionEngine
from polymarket_bot.model.kronos_adapter import KronosAdapter
from polymarket_bot.monitoring.logger import TradeLogger


def run_once(market_id: str = "demo_market_1") -> dict:
    runtime = RuntimeConfig()
    collector = PolyMarketCollector()
    predictor = KronosAdapter(device=runtime.device)
    decision_engine = DecisionEngine(RiskConfig(), StrategyConfig())
    executor = ExecutionEngine(dry_run=runtime.dry_run)
    logger = TradeLogger()

    state = collector.fetch_market_state(market_id=market_id, lookback=runtime.lookback)
    prob_yes = predictor.predict_yes_probability(state.history_yes_price, pred_len=runtime.pred_len)
    decision = decision_engine.evaluate(prob_yes=prob_yes, yes_price=state.yes_price, liquidity=state.liquidity)

    result = {
        "market_id": state.market_id,
        "prob_yes": prob_yes,
        "yes_price": state.yes_price,
        "liquidity": state.liquidity,
        "decision": decision.__dict__,
    }

    if decision.should_trade:
        result["execution"] = executor.place_order(
            market_id=state.market_id,
            side=decision.side or "YES",
            size_fraction=decision.size_fraction,
        )

    logger.log(result)
    return result


if __name__ == "__main__":
    print(run_once())
