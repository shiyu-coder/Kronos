from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    min_liquidity: float = 5000.0
    min_probability: float = 0.68


@dataclass
class StrategyConfig:
    kelly_fraction_cap: float = 0.25
    fee_rate: float = 0.01


@dataclass
class RuntimeConfig:
    dry_run: bool = True
    device: str = "cpu"
    lookback: int = 256
    pred_len: int = 1
