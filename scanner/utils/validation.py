"""Dataclasses and validation helpers for scanner decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TickerValidationResult:
    ticker: str
    is_active: bool
    price: float | None
    is_above_min_price: bool
    has_options: bool
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PotterBoxResult:
    ticker: str
    passed: bool
    direction: str | None
    box_top: float | None
    box_bottom: float | None
    cost_basis: float | None
    prior_close: float | None
    breakout_close: float | None
    breakout_strength_pct: float | None
    atr_value: float | None
    range_compression_ratio: float | None
    no_trend_score: float | None
    skip_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmptySpaceResult:
    passed: bool
    score: int
    nearest_target: float | None
    distance_to_target_pct: float | None
    invalidation_level: float | None
    risk_pct: float | None
    rr_ratio: float | None
    support_resistance_source: str
    skip_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRiskResult:
    passed: bool
    earnings_date: datetime | None
    days_to_earnings: int | None
    ex_dividend_date: datetime | None
    status: str
    skip_reason: str | None = None


@dataclass
class OptionsContractResult:
    passed: bool
    expiration: str | None
    dte: int | None
    contract_type: str | None
    strike: float | None
    bid: float | None
    ask: float | None
    midpoint: float | None
    spread_pct: float | None
    open_interest: int | None
    volume: int | None
    implied_volatility: float | None
    skip_reason: str | None = None
    data_provider: str | None = None
    data_feed: str | None = None
    quote_source: str | None = None
    open_interest_source: str | None = None
    quote_timestamp: str | None = None
    quote_age_minutes: float | None = None
    greeks_available: bool | None = None
    source_disagreement_pct: float | None = None
    options_data_quality: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None


@dataclass
class KronosResult:
    passed: bool
    output_mode: str
    directional_agreement: float | None
    median_forecast_return_pct: float | None
    worst_sampled_return_pct: float | None
    sample_count: int
    skip_reason: str | None = None
    output_type: str | None = None
    output_shape: Any = None


@dataclass
class AlertCandidate:
    ticker: str
    direction: str
    potter_box: PotterBoxResult
    empty_space: EmptySpaceResult
    event_risk: EventRiskResult
    options_contract: OptionsContractResult
    kronos: KronosResult
    final_decision: str
    timestamp: str
    ai_insight: dict[str, Any] | None = None
