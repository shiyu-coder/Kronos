from datetime import datetime

from scanner.ai.minimax_adapter import MiniMaxAdapter
from scanner.alerts.telegram import render_alert_message
from scanner.utils.validation import (
    AlertCandidate,
    EmptySpaceResult,
    EventRiskResult,
    KronosResult,
    OptionsContractResult,
    PotterBoxResult,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def test_render_alert_message_handles_nullable_fields():
    candidate = AlertCandidate(
        ticker="TEST",
        direction="bullish",
        potter_box=PotterBoxResult(
            ticker="TEST",
            passed=True,
            direction="bullish",
            box_top=None,
            box_bottom=None,
            cost_basis=None,
            prior_close=None,
            breakout_close=None,
            breakout_strength_pct=None,
            atr_value=None,
            range_compression_ratio=None,
            no_trend_score=None,
        ),
        empty_space=EmptySpaceResult(
            passed=True,
            score=2,
            nearest_target=None,
            distance_to_target_pct=None,
            invalidation_level=None,
            risk_pct=None,
            rr_ratio=None,
            support_resistance_source="test",
        ),
        event_risk=EventRiskResult(
            passed=True,
            earnings_date=None,
            days_to_earnings=None,
            ex_dividend_date=None,
            status="clear",
        ),
        options_contract=OptionsContractResult(
            passed=True,
            expiration=None,
            dte=None,
            contract_type=None,
            strike=None,
            bid=None,
            ask=None,
            midpoint=None,
            spread_pct=None,
            open_interest=None,
            volume=None,
            implied_volatility=None,
        ),
        kronos=KronosResult(
            passed=True,
            output_mode="forecast_alignment",
            directional_agreement=None,
            median_forecast_return_pct=None,
            worst_sampled_return_pct=None,
            sample_count=1,
        ),
        final_decision="pass",
        timestamp=datetime.now().isoformat(),
    )
    msg = render_alert_message(candidate)
    assert "N/A" in msg
    assert "$TEST" in msg


def test_minimax_fallback_confidence_regex_parses_decimal():
    adapter = MiniMaxAdapter(DummyLogger())
    parsed = adapter._extract_structured_fallback("score band A confidence 0.87")
    assert parsed["score_band"] == "A"
    assert parsed["confidence"] == 0.87
