from scanner import config
from scanner.edge.scoring import score_edge_candidate


def _features():
    return {
        "ticker": "TEST",
        "direction": "bullish",
        "potter_passed": 1.0,
        "research_score": 70.0,
        "rr_ratio": 2.2,
        "empty_space_score": 3.0,
        "kronos_directional_agreement": 0.72,
        "kronos_median_forecast_return_pct": 2.0,
        "kronos_worst_sampled_return_pct": -0.5,
        "data_quality_score": 1.0,
        "feed_confidence": 0.9,
        "options_spread_pct": 0.06,
        "options_data_quality": 0.9,
    }


def test_score_edge_candidate_promotes_positive_analog_expectancy():
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "risk_pct_used": 2.142857, "mae_pct": -0.6, "mfe_pct": 4.0},
        {"outcome_label": "win", "outcome_return_pct": 2.0, "r_multiple": 1.0, "risk_pct_used": 2.0, "mae_pct": -0.8, "mfe_pct": 3.0},
        {"outcome_label": "loss", "outcome_return_pct": -0.7, "r_multiple": -0.3, "risk_pct_used": 2.333333, "mae_pct": -1.2, "mfe_pct": 1.0},
    ]

    result = score_edge_candidate(_features(), analogs, min_analogs=3)

    assert result["edge_score"] >= 65
    assert result["recommendation"] == "promote"
    assert result["scorecard"]["analog_expectancy"] > 0
    assert result["analog_summary"]["count"] == 3


def test_execution_only_fields_do_not_change_validated_edge_score():
    """Live and historical rows must share one strategy-score scale.

    The historical retrieval index cannot reconstruct point-in-time equity
    feed confidence or option-chain quality. Those fields still gate execution,
    but they must not move the score whose thresholds are validated on that
    historical index.
    """
    analogs = [
        {
            "outcome_label": "win",
            "outcome_return_pct": 3.0,
            "r_multiple": 1.4,
            "risk_pct_used": 2.142857,
            "mae_pct": -0.6,
            "mfe_pct": 4.0,
        },
        {
            "outcome_label": "win",
            "outcome_return_pct": 2.0,
            "r_multiple": 1.0,
            "risk_pct_used": 2.0,
            "mae_pct": -0.8,
            "mfe_pct": 3.0,
        },
        {
            "outcome_label": "loss",
            "outcome_return_pct": -0.7,
            "r_multiple": -0.3,
            "risk_pct_used": 2.333333,
            "mae_pct": -1.2,
            "mfe_pct": 1.0,
        },
    ]
    live_features = _features()
    historical_features = {
        **live_features,
        "feed_confidence": 0.5,
        "options_spread_pct": 0.0,
        "options_data_quality": 0.0,
    }

    live = score_edge_candidate(live_features, analogs, min_analogs=3)
    historical = score_edge_candidate(historical_features, analogs, min_analogs=3)

    assert live["edge_score"] == historical["edge_score"]
    assert live["recommendation"] == "promote"
    assert historical["recommendation"] != "promote"
    assert "options_data_not_execution_grade" in historical["blocking_reasons"]


def test_score_edge_candidate_charges_costs_before_analog_expectancy():
    # Three tiny gross wins are all net losses after the strategy's committed
    # 25 bps/side execution basis. Candidate scoring must see the same net
    # economics as readiness validation instead of promoting a gross-only edge.
    analogs = [
        {
            "outcome_label": "win",
            "outcome_return_pct": 0.2,
            "r_multiple": 0.2,
            "risk_pct_used": 1.0,
            "mae_pct": -0.2,
            "mfe_pct": 0.3,
        }
        for _ in range(3)
    ]

    result = score_edge_candidate(_features(), analogs, min_analogs=3)

    assert result["analog_summary"]["average_return_pct"] == -0.3
    assert result["analog_summary"]["average_r_multiple"] == -0.3
    assert result["analog_summary"]["win_rate"] == 0.0
    assert result["recommendation"] != "promote"
    assert "non_positive_analog_expectancy" in result["blocking_reasons"]


def test_score_edge_candidate_blocks_promotion_without_analog_risk_basis():
    analogs = [
        {
            "outcome_label": "win",
            "outcome_return_pct": 3.0,
            "r_multiple": 1.4,
            "mae_pct": -0.6,
            "mfe_pct": 4.0,
        }
        for _ in range(3)
    ]

    result = score_edge_candidate(_features(), analogs, min_analogs=3)

    assert result["edge_score"] >= 65
    assert result["recommendation"] != "promote"
    assert result["analog_cost_model"]["risk_coverage_complete"] is False
    assert "analog_cost_basis_incomplete" in result["blocking_reasons"]


def test_kronos_advisory_evidence_cannot_change_edge_score_or_promotion():
    features = {
        "potter_passed": 1.0,
        "research_score": 55.0,
        "rr_ratio": 1.0,
        "empty_space_score": 1.0,
        "data_quality_score": 1.0,
        "feed_confidence": 0.9,
        "options_spread_pct": 0.06,
        "options_data_quality": 0.9,
    }
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 1.0, "r_multiple": 0.2, "mae_pct": -0.5, "mfe_pct": 1.0}
        for _ in range(3)
    ]
    disagreement = score_edge_candidate(
        {
            **features,
            "kronos_directional_agreement": 0.0,
            "kronos_median_forecast_return_pct": -5.0,
        },
        analogs,
        min_analogs=3,
    )
    agreement = score_edge_candidate(
        {
            **features,
            "kronos_directional_agreement": 1.0,
            "kronos_median_forecast_return_pct": 5.0,
        },
        analogs,
        min_analogs=3,
    )

    assert disagreement["scorecard"]["kronos"] == 0.0
    assert agreement["scorecard"]["kronos"] == 0.0
    assert disagreement["edge_score"] == agreement["edge_score"]
    assert disagreement["recommendation"] == agreement["recommendation"]


def test_score_edge_candidate_rejects_negative_expectancy():
    analogs = [
        {"outcome_label": "loss", "outcome_return_pct": -3.0, "r_multiple": -1.2, "mae_pct": -4.0, "mfe_pct": 0.5},
        {"outcome_label": "loss", "outcome_return_pct": -1.0, "r_multiple": -0.6, "mae_pct": -2.0, "mfe_pct": 0.7},
        {"outcome_label": "win", "outcome_return_pct": 0.3, "r_multiple": 0.1, "mae_pct": -1.5, "mfe_pct": 1.0},
    ]

    result = score_edge_candidate(_features(), analogs, min_analogs=3)

    assert result["edge_score"] < 55
    assert result["recommendation"] in {"reject", "research"}
    assert result["scorecard"]["analog_expectancy"] < 0


def test_score_edge_candidate_blocks_thin_or_low_quality_evidence():
    features = _features()
    features["data_quality_score"] = 0.4
    features["feed_confidence"] = 0.25
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "mae_pct": -0.6, "mfe_pct": 4.0}
    ]

    result = score_edge_candidate(features, analogs, min_analogs=5)

    assert result["recommendation"] != "promote"
    assert result["scorecard"]["sample_penalty"] < 0
    assert result["scorecard"]["data_quality"] == 0.0
    assert "low_data_quality" in result["blocking_reasons"]
    assert "low_feed_confidence" in result["blocking_reasons"]


def test_score_edge_candidate_rejects_when_core_setup_gates_fail():
    features = _features()
    features["potter_passed"] = 0.0
    features["empty_space_passed"] = 0.0
    features["empty_space_score"] = 0.0
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 12.0, "r_multiple": 3.0, "mae_pct": -0.6, "mfe_pct": 14.0},
        {"outcome_label": "win", "outcome_return_pct": 9.0, "r_multiple": 2.4, "mae_pct": -0.8, "mfe_pct": 10.0},
        {"outcome_label": "win", "outcome_return_pct": 8.0, "r_multiple": 2.1, "mae_pct": -1.0, "mfe_pct": 9.0},
    ]

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["edge_score"] < 45
    assert result["recommendation"] == "reject"
    assert result["scorecard"]["setup_gate"] < 0


def test_score_edge_candidate_does_not_infer_setup_pass_from_subthreshold_empty_space():
    features = _features()
    features["potter_passed"] = 0.0
    features["empty_space_passed"] = 0.0
    features["empty_space_score"] = config.MIN_EMPTY_SPACE_SCORE - 1
    features["rr_ratio"] = config.MIN_RR - 0.1
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 12.0, "r_multiple": 3.0, "mae_pct": -0.6, "mfe_pct": 14.0},
        {"outcome_label": "win", "outcome_return_pct": 9.0, "r_multiple": 2.4, "mae_pct": -0.8, "mfe_pct": 10.0},
        {"outcome_label": "win", "outcome_return_pct": 8.0, "r_multiple": 2.1, "mae_pct": -1.0, "mfe_pct": 9.0},
    ]

    explicit_result = score_edge_candidate(features, analogs, min_analogs=3)
    legacy_features = dict(features)
    legacy_features.pop("empty_space_passed")
    legacy_result = score_edge_candidate(legacy_features, analogs, min_analogs=3)

    for result in (explicit_result, legacy_result):
        assert result["edge_score"] < 45
        assert result["recommendation"] == "reject"
        assert "setup_gate_failed" in result["blocking_reasons"]


def test_score_edge_candidate_explains_reject_reasons():
    features = _features()
    features["potter_passed"] = 0.0
    features["empty_space_passed"] = 0.0
    features["empty_space_score"] = 0.0
    features["options_data_quality"] = 0.45
    features["options_spread_pct"] = 0.22
    analogs = [
        {"outcome_label": "loss", "outcome_return_pct": -3.0, "r_multiple": -1.2, "risk_pct_used": 2.5, "mae_pct": -4.0, "mfe_pct": 0.5},
        {"outcome_label": "loss", "outcome_return_pct": -1.0, "r_multiple": -0.6, "risk_pct_used": 1.666667, "mae_pct": -2.0, "mfe_pct": 0.7},
        {"outcome_label": "win", "outcome_return_pct": 0.3, "r_multiple": 0.1, "risk_pct_used": 3.0, "mae_pct": -1.5, "mfe_pct": 1.0},
    ]

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["recommendation"] == "reject"
    assert result["blocking_reasons"] == [
        "setup_gate_failed",
        "non_positive_analog_expectancy",
        "wide_options_spread",
        "options_data_not_execution_grade",
        "edge_score_below_research_threshold",
    ]
    assert result["rejection_reasons"] == result["blocking_reasons"]


def test_score_edge_candidate_does_not_promote_indicative_options():
    features = _features()
    features["options_data_quality"] = 0.6
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "mae_pct": -0.6, "mfe_pct": 4.0},
        {"outcome_label": "win", "outcome_return_pct": 2.0, "r_multiple": 1.0, "mae_pct": -0.8, "mfe_pct": 3.0},
        {"outcome_label": "loss", "outcome_return_pct": -0.7, "r_multiple": -0.3, "mae_pct": -1.2, "mfe_pct": 1.0},
    ]

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["recommendation"] != "promote"


def test_score_edge_candidate_does_not_promote_spread_above_current_limit(monkeypatch):
    features = _features()
    features["options_spread_pct"] = 0.10
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "risk_pct_used": 2.142857, "mae_pct": -0.6, "mfe_pct": 4.0},
        {"outcome_label": "win", "outcome_return_pct": 2.0, "r_multiple": 1.0, "risk_pct_used": 2.0, "mae_pct": -0.8, "mfe_pct": 3.0},
        {"outcome_label": "loss", "outcome_return_pct": -0.7, "r_multiple": -0.3, "risk_pct_used": 2.333333, "mae_pct": -1.2, "mfe_pct": 1.0},
    ]
    monkeypatch.setattr(config, "MAX_ATM_BID_ASK_SPREAD_PCT", 0.08)

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["recommendation"] != "promote"
    assert "wide_options_spread" in result["blocking_reasons"]


def test_score_edge_candidate_labels_no_liquid_contract_without_false_data_faults():
    features = _features()
    features.update(
        {
            "options_passed": 0.0,
            "options_data_provider": "tradier",
            "options_data_feed": "opra-consolidated",
            "options_spread_pct": 1.0,
            "options_data_quality": 0.45,
        }
    )
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "mae_pct": -0.6, "mfe_pct": 4.0},
        {"outcome_label": "win", "outcome_return_pct": 2.0, "r_multiple": 1.0, "mae_pct": -0.8, "mfe_pct": 3.0},
        {"outcome_label": "loss", "outcome_return_pct": -0.7, "r_multiple": -0.3, "mae_pct": -1.2, "mfe_pct": 1.0},
    ]

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["recommendation"] != "promote"
    assert "options_no_liquid_contract" in result["blocking_reasons"]
    assert "wide_options_spread" not in result["blocking_reasons"]
    assert "options_data_not_execution_grade" not in result["blocking_reasons"]


def test_score_edge_candidate_uses_doctrine_v2_without_bypassing_quality_gates():
    features = _features()
    features["doctrine_v2_score"] = 86.0
    features["doctrine_v2_passed"] = 1.0
    features["doctrine_v2_failed_reentry"] = 0.0
    features["options_data_quality"] = 0.6
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "mae_pct": -0.6, "mfe_pct": 4.0},
        {"outcome_label": "win", "outcome_return_pct": 2.0, "r_multiple": 1.0, "mae_pct": -0.8, "mfe_pct": 3.0},
        {"outcome_label": "loss", "outcome_return_pct": -0.7, "r_multiple": -0.3, "mae_pct": -1.2, "mfe_pct": 1.0},
    ]

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["scorecard"]["doctrine_v2"] > 0
    assert result["recommendation"] != "promote"


def test_score_edge_candidate_uses_current_doctrine_v2_baseline(monkeypatch):
    features = _features()
    features["doctrine_v2_score"] = 86.0
    features["doctrine_v2_passed"] = 1.0
    features["doctrine_v2_failed_reentry"] = 0.0
    analogs = [
        {"outcome_label": "win", "outcome_return_pct": 3.0, "r_multiple": 1.4, "mae_pct": -0.6, "mfe_pct": 4.0},
        {"outcome_label": "win", "outcome_return_pct": 2.0, "r_multiple": 1.0, "mae_pct": -0.8, "mfe_pct": 3.0},
        {"outcome_label": "loss", "outcome_return_pct": -0.7, "r_multiple": -0.3, "mae_pct": -1.2, "mfe_pct": 1.0},
    ]

    baseline_score = score_edge_candidate(features, analogs, min_analogs=3)["scorecard"]["doctrine_v2"]
    monkeypatch.setattr(config, "DOCTRINE_V2_SCORE_BASELINE", 90)

    result = score_edge_candidate(features, analogs, min_analogs=3)

    assert result["scorecard"]["doctrine_v2"] < baseline_score
