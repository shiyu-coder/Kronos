from scanner.edge.audit import compute_edge_audit_report


_NET_COST_MODEL = {
    "bps_per_side": 25.0,
    "round_trip_return_pct_charged": 0.5,
    "basis": "net_of_costs",
    "candidate_rows": 1000,
    "risk_coverage_rows": 1000,
    "risk_coverage_complete": True,
    "applies_to": ["returns", "r_multiple", "win_loss_label"],
}


def test_edge_audit_blocks_when_walk_forward_validation_has_no_supported_threshold():
    validation = {
        "samples": 1000,
        "validation_method": "purged_walk_forward",
        "future_analogs_allowed": False,
        "thresholds": {
            "45": {"signal_count": 1, "precision": 0.0, "average_r_multiple": -0.9},
            "55": {"signal_count": 0, "precision": 0.0, "average_r_multiple": 0.0},
        },
    }
    scan = {
        "candidates": [
            {
                "ticker": "CHPT",
                "status": "candidate",
                "recommendation": "reject",
                "features": {
                    "feed_confidence": 0.5,
                    "options_open_interest": 0.0,
                    "options_volume": 0.0,
                    "options_spread_pct": 0.0,
                },
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["readiness"] == "blocked"
    assert "validation_threshold_55_unsupported" in report["blockers"]
    assert "options_liquidity_missing" in report["warnings"]
    assert report["checks"]["future_analogs_blocked"]["passed"] is True


def test_edge_audit_allows_research_only_when_validation_and_candidate_quality_pass():
    validation = {
        "samples": 1000,
        "validation_method": "purged_walk_forward",
        "future_analogs_allowed": False,
        "cost_model": dict(_NET_COST_MODEL),
        "thresholds": {
            "55": {
                "signal_count": 25,
                "precision": 0.64,
                "average_r_multiple": 0.8,
                "precision_day_clustered": {
                    "n_days": 25,
                    "mean_daily_precision": 0.64,
                    "lower_bound": 0.56,
                },
                "t_stat_r_day_clustered": {
                    "n_days": 25,
                    "mean_of_day_means": 0.8,
                    "t_stat": 2.1,
                },
            },
        },
    }
    scan = {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "recommendation": "research",
                "features": {
                    "feed_confidence": 0.9,
                    "options_open_interest": 250.0,
                    "options_volume": 80.0,
                    "options_spread_pct": 0.05,
                },
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["readiness"] == "research_only"
    assert report["blockers"] == []
    assert report["summary"]["research_candidates"] == 1


def test_edge_audit_blocks_legacy_threshold_without_dependence_aware_stats():
    validation = {
        "validation_method": "purged_walk_forward",
        "future_analogs_allowed": False,
        "thresholds": {
            "55": {
                "signal_count": 25,
                "precision": 0.64,
                "average_r_multiple": 0.8,
            },
        },
    }
    scan = {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "recommendation": "research",
                "features": {},
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["checks"]["validation_threshold"]["passed"] is False
    assert report["checks"]["validation_threshold"]["value"]["dependence_metrics_present"] is False
    assert report["readiness"] == "blocked"


def _ranking_validation(within_bullish_ic=None):
    by_direction = {}
    if within_bullish_ic is not None:
        by_direction["bullish"] = {
            "signal_count": 500,
            "average_r_multiple": 0.2,
            "rank_ic_r": {"ic": within_bullish_ic, "p_value": 0.01, "p_value_day_clustered": 0.02, "n": 500},
        }
    return {
        "samples": 1000,
        "validation_method": "purged_walk_forward",
        "future_analogs_allowed": False,
        "cost_model": dict(_NET_COST_MODEL),
        "thresholds": {"55": {"signal_count": 0, "precision": 0.0, "average_r_multiple": 0.0}},
        "rank_ic_r": {
            "ic": 0.12,
            "p_value": 0.001,
            "p_value_day_clustered": 0.002,
            "n": 1000,
        },
        "percentiles": {
            "top_10_pct": {
                "signal_count": 100,
                "average_r_multiple": 0.4,
                "t_stat_r_multiple": 3.0,
                "wilson_lb_precision": 0.5,
                "precision_day_clustered": {
                    "n_days": 50,
                    "mean_daily_precision": 0.6,
                    "lower_bound": 0.5,
                },
                "t_stat_r_day_clustered": {
                    "n_days": 50,
                    "mean_of_day_means": 0.4,
                    "t_stat": 3.0,
                },
            }
        },
        "by_direction": by_direction,
    }


_EMPTY_SCAN = {"candidates": []}


def test_ranking_gate_rejects_pooled_ic_without_within_direction_skill():
    # Pooled IC of 0.12 driven purely by direction separation must NOT pass:
    # no per-direction block clears the bar (fail-closed when absent too).
    report = compute_edge_audit_report(_ranking_validation(within_bullish_ic=None), _EMPTY_SCAN)
    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert "ranking_evidence_unsupported" in report["blockers"]

    report = compute_edge_audit_report(_ranking_validation(within_bullish_ic=0.01), _EMPTY_SCAN)
    assert report["checks"]["ranking_evidence"]["passed"] is False


def test_ranking_gate_passes_with_within_direction_skill():
    report = compute_edge_audit_report(_ranking_validation(within_bullish_ic=0.10), _EMPTY_SCAN)
    assert report["checks"]["ranking_evidence"]["passed"] is True
    assert report["checks"]["ranking_evidence"]["value"]["within_direction_passed"] is True
    assert "ranking_evidence_unsupported" not in report["blockers"]


def test_otherwise_passing_ranking_evidence_is_blocked_when_costs_were_not_charged():
    """Zeroing the cost model must not become a way to open the gate."""
    for cost_model in ({"bps_per_side": 0.0, "basis": "net_of_costs"}, None):
        validation = _ranking_validation(within_bullish_ic=0.10)
        if cost_model is None:
            validation.pop("cost_model")  # a report predating the cost model
        else:
            validation["cost_model"] = cost_model

        report = compute_edge_audit_report(validation, _EMPTY_SCAN)

        assert report["checks"]["costs_charged"]["passed"] is False
        assert report["checks"]["ranking_evidence"]["passed"] is False
        assert report["readiness"] == "blocked"


def test_positive_cost_stamp_needs_complete_consistent_net_provenance():
    """A positive number alone must not certify that gate metrics are net."""
    malformed_models = [
        {**_NET_COST_MODEL, "basis": "gross"},
        {key: value for key, value in _NET_COST_MODEL.items() if key != "round_trip_return_pct_charged"},
        {key: value for key, value in _NET_COST_MODEL.items() if key != "risk_coverage_complete"},
        {**_NET_COST_MODEL, "round_trip_return_pct_charged": 0.05},
        {**_NET_COST_MODEL, "applies_to": ["returns"]},
        {**_NET_COST_MODEL, "candidate_rows": 999},
    ]
    for cost_model in malformed_models:
        validation = _ranking_validation(within_bullish_ic=0.10)
        validation["cost_model"] = cost_model

        report = compute_edge_audit_report(validation, _EMPTY_SCAN)

        assert report["checks"]["costs_charged"]["passed"] is False
        assert report["checks"]["ranking_evidence"]["passed"] is False
        assert report["readiness"] == "blocked"


def test_cost_gate_enforces_the_preregistered_25_bps_per_side_floor():
    """A token positive charge must not bypass the production cost basis."""
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["cost_model"].update(
        {
            "bps_per_side": 0.01,
            "round_trip_return_pct_charged": 0.0002,
        }
    )

    report = compute_edge_audit_report(validation, _EMPTY_SCAN)

    cost_check = report["checks"]["costs_charged"]
    assert cost_check["passed"] is False
    assert cost_check["value"]["minimum_bps_per_side"] == 25.0
    assert cost_check["value"]["minimum_cost_met"] is False
    assert "validation_cost_model_unsupported" in report["blockers"]
    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert report["readiness"] == "blocked"


def test_net_cost_provenance_requires_complete_risk_denominator_coverage():
    """Gross R rows without a stop denominator must never open readiness."""
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["cost_model"].update(
        {
            "candidate_rows": 1000,
            "risk_coverage_rows": 999,
            "risk_coverage_complete": False,
        }
    )

    report = compute_edge_audit_report(validation, _EMPTY_SCAN)

    assert report["checks"]["costs_charged"]["passed"] is False
    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert report["readiness"] == "blocked"


def test_ranking_gate_uses_clustered_p_value_instead_of_raw_iid_p_value():
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["rank_ic_r"]["p_value_day_clustered"] = 0.40

    report = compute_edge_audit_report(validation, _EMPTY_SCAN)

    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert report["checks"]["ranking_evidence"]["value"]["rank_ic_p_value"] == 0.40


def test_ranking_gate_requires_complete_dependence_aware_metrics():
    missing_global_p = _ranking_validation(within_bullish_ic=0.10)
    missing_global_p["rank_ic_r"].pop("p_value_day_clustered")

    missing_top_precision = _ranking_validation(within_bullish_ic=0.10)
    missing_top_precision["percentiles"]["top_10_pct"].pop("precision_day_clustered")

    missing_top_r = _ranking_validation(within_bullish_ic=0.10)
    missing_top_r["percentiles"]["top_10_pct"].pop("t_stat_r_day_clustered")

    for validation in (missing_global_p, missing_top_precision, missing_top_r):
        report = compute_edge_audit_report(validation, _EMPTY_SCAN)
        value = report["checks"]["ranking_evidence"]["value"]
        assert report["checks"]["ranking_evidence"]["passed"] is False
        assert value["dependence_metrics_present"] is False


def test_ranking_gate_uses_entry_day_evidence_for_top_decile():
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["rank_ic_r"]["p_value_day_clustered"] = 0.01
    top = validation["percentiles"]["top_10_pct"]
    # The raw 100-row metrics pass, but they came from only 10 correlated entry
    # days and the dependence-aware precision bound does not clear the bar.
    top["precision_day_clustered"] = {
        "signal_count": 100,
        "n_days": 10,
        "mean_daily_precision": 0.60,
        "lower_bound": 0.40,
    }
    top["t_stat_r_day_clustered"] = {
        "t_stat": 2.5,
        "n_days": 10,
        "mean_of_day_means": 0.40,
    }

    report = compute_edge_audit_report(validation, _EMPTY_SCAN)

    value = report["checks"]["ranking_evidence"]["value"]
    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert value["top_decile_signals"] == 100
    assert value["top_decile_evidence_days"] == 10
    assert value["top_decile_wilson_lb_precision"] == 0.40


def test_positive_direction_needs_clustered_expectancy_evidence_before_promotion():
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["by_direction"]["bullish"]["t_stat_r_day_clustered"] = {
        "t_stat": 1.38,
        "n_days": 65,
        "mean_of_day_means": 0.20,
    }
    scan = {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "direction": "bullish",
                "recommendation": "promote",
                "features": {},
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["checks"]["ranking_evidence"]["passed"] is True
    assert report["readiness"] == "research_only"
    assert report["summary"]["promotable_directions"] == []
    assert report["summary"]["unproven_directions"] == ["bullish"]
    assert "promoted_candidates_direction_blocked" in report["warnings"]


def test_positive_direction_with_clustered_expectancy_evidence_can_promote():
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["by_direction"]["bullish"]["t_stat_r_day_clustered"] = {
        "t_stat": 2.01,
        "n_days": 65,
        "mean_of_day_means": 0.20,
    }
    scan = {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "direction": "bullish",
                "recommendation": "promote",
                "features": {
                    "feed_confidence": 0.9,
                    "data_delay_minutes": 0.0,
                    "options_passed": 1.0,
                    "options_open_interest": 250.0,
                    "options_volume": 80.0,
                    "options_spread_pct": 0.05,
                    "options_data_provider": "tradier",
                    "options_data_feed": "opra-consolidated",
                    "options_data_quality": 0.9,
                },
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["readiness"] == "paper_trade_only"
    assert report["summary"]["promotable_directions"] == ["bullish"]
    assert report["summary"]["unproven_directions"] == []
    assert "promoted_candidates_direction_blocked" not in report["warnings"]


def test_low_confidence_equity_feed_cannot_authorize_live_eligible_readiness():
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["by_direction"]["bullish"]["t_stat_r_day_clustered"] = {
        "t_stat": 2.01,
        "n_days": 65,
        "mean_of_day_means": 0.20,
    }
    scan = {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "direction": "bullish",
                "recommendation": "promote",
                "features": {
                    # Alpaca's free IEX feed is intentionally research-grade.
                    "feed_confidence": 0.7,
                    "options_passed": 1.0,
                    "options_open_interest": 250.0,
                    "options_volume": 80.0,
                    "options_spread_pct": 0.05,
                    "options_data_provider": "tradier",
                    "options_data_feed": "opra-consolidated",
                    "options_data_quality": 0.9,
                },
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["readiness"] == "research_only"
    assert "low_feed_confidence" in report["warnings"]
    assert "promoted_candidates_execution_quality_blocked" in report["warnings"]
    assert report["summary"]["execution_ready_promoted_candidates"] == []
    assert report["summary"]["execution_quality_blocked_promoted_candidates"] == ["TEST"]


def test_delayed_consolidated_equity_feed_cannot_authorize_live_eligible_readiness():
    validation = _ranking_validation(within_bullish_ic=0.10)
    validation["by_direction"]["bullish"]["t_stat_r_day_clustered"] = {
        "t_stat": 2.01,
        "n_days": 65,
        "mean_of_day_means": 0.20,
    }
    scan = {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "direction": "bullish",
                "recommendation": "promote",
                "features": {
                    # The free historical SIP route is consolidated, but its
                    # intentional 16-minute delay is not execution-grade.
                    "feed_confidence": 0.9,
                    "data_quality_score": 1.0,
                    "data_delay_minutes": 16.0,
                    "data_stale_minutes": 16.0,
                    "options_passed": 1.0,
                    "options_open_interest": 250.0,
                    "options_volume": 80.0,
                    "options_spread_pct": 0.05,
                    "options_data_provider": "tradier",
                    "options_data_feed": "opra-consolidated",
                    "options_data_quality": 0.9,
                },
            }
        ]
    }

    report = compute_edge_audit_report(validation, scan)

    assert report["readiness"] == "research_only"
    assert "delayed_equity_feed" in report["warnings"]
    assert "promoted_candidates_execution_quality_blocked" in report["warnings"]
    assert report["summary"]["delayed_equity_feed_candidates"] == ["TEST"]
    assert report["summary"]["execution_ready_promoted_candidates"] == []
    assert report["summary"]["execution_quality_blocked_promoted_candidates"] == ["TEST"]

    # Older/malformed scan artifacts that omit delay provenance must fail
    # closed too; absence cannot mean real time at an authorization boundary.
    del scan["candidates"][0]["features"]["data_delay_minutes"]
    missing_provenance_report = compute_edge_audit_report(validation, scan)
    assert missing_provenance_report["readiness"] == "research_only"
    assert "delayed_equity_feed" in missing_provenance_report["warnings"]


_PASSING_VALIDATION = {
    "validation_method": "purged_walk_forward",
    "future_analogs_allowed": False,
    "thresholds": {
        "55": {
            "signal_count": 25,
            "precision": 0.64,
            "average_r_multiple": 0.8,
            "precision_day_clustered": {
                "n_days": 25,
                "mean_daily_precision": 0.64,
                "lower_bound": 0.56,
            },
            "t_stat_r_day_clustered": {
                "n_days": 25,
                "mean_of_day_means": 0.8,
                "t_stat": 2.1,
            },
        },
    },
}


def _scan_with(features: dict, ticker: str = "TEST") -> dict:
    base = {
        "feed_confidence": 0.9,
        "options_passed": 1.0,
        "options_open_interest": 250.0,
        "options_volume": 80.0,
        "options_spread_pct": 0.05,
        "options_data_provider": "tradier",
        "options_data_feed": "opra-consolidated",
        "options_data_quality": 0.9,
    }
    base.update(features)
    return {
        "candidates": [
            {
                "ticker": ticker,
                "status": "candidate",
                "recommendation": "research",
                "features": base,
            }
        ]
    }


def test_edge_audit_reports_illiquid_chain_as_its_own_cause_not_a_feed_fault():
    # Tradier answering "no strike clears the gates" is an authoritative verdict
    # about the market. It must not read as degraded data, because the brief
    # turns that into "check TRADIER_API_TOKEN" for a perfectly healthy token.
    scan = _scan_with(
        {
            "options_passed": 0.0,
            "options_data_quality": 0.45,
            "options_open_interest": 0.0,
            "options_volume": 0.0,
            "options_spread_pct": 1.0,
            "options_quote_age_minutes": 9999.0,
        },
        ticker="RIOT",
    )

    report = compute_edge_audit_report(_PASSING_VALIDATION, scan)

    assert "options_no_liquid_contract" in report["warnings"]
    assert report["summary"]["no_liquid_options_contract_candidates"] == ["RIOT"]
    # The healthy-provider codes must stay silent.
    assert "options_data_not_execution_grade" not in report["warnings"]
    assert "options_provider_unavailable" not in report["warnings"]
    # ...and the zeroed liquidity fields are the same fact, not a second one.
    assert "options_liquidity_missing" not in report["warnings"]


def test_edge_audit_flags_provider_fallback_separately_from_stale_quotes():
    scan = _scan_with(
        {
            "options_data_quality": 0.6,
            "options_data_provider": "alpaca+yfinance",
            "options_data_feed": "indicative",
        },
        ticker="SOFI",
    )

    report = compute_edge_audit_report(_PASSING_VALIDATION, scan)

    assert "options_provider_unavailable" in report["warnings"]
    assert report["summary"]["options_provider_unavailable_candidates"] == ["SOFI"]
    assert report["summary"]["stale_options_quote_candidates"] == []
    # One root cause, one alarm: "stale quotes" is a different, false claim here.
    assert "options_data_not_execution_grade" not in report["warnings"]


def test_edge_audit_flags_dead_tradier_token_rather_than_calling_it_illiquid():
    # The shape a dead/sandbox/missing TRADIER_API_TOKEN actually produces:
    # _tradier_get 401s -> _select_via_tradier returns None -> the yfinance
    # fallback finds nothing -> OptionsContractResult(passed=False) with NO
    # data_provider (options_data.py:303/314/391/413). Only the Tradier failure
    # paths set data_provider, so its absence is the tell.
    scan = _scan_with(
        {
            "options_passed": 0.0,
            "options_data_provider": None,
            "options_data_feed": None,
            "options_data_quality": 0.45,
        },
        ticker="SOFI",
    )

    report = compute_edge_audit_report(_PASSING_VALIDATION, scan)

    assert "options_provider_unavailable" in report["warnings"]
    assert report["summary"]["options_provider_unavailable_candidates"] == ["SOFI"]
    # Calling a dead token "an untradeable chain" is the inverse of the original
    # bug and hides a real outage behind a benign, no-action finding.
    assert "options_no_liquid_contract" not in report["warnings"]


def test_edge_audit_flags_stale_tradier_quotes_without_blaming_the_provider():
    scan = _scan_with({"options_data_quality": 0.7, "options_quote_age_minutes": 400.0})

    report = compute_edge_audit_report(_PASSING_VALIDATION, scan)

    assert "options_data_not_execution_grade" in report["warnings"]
    assert report["summary"]["stale_options_quote_candidates"] == ["TEST"]
    assert report["summary"]["non_execution_grade_options_candidates"] == ["TEST"]
    assert "options_provider_unavailable" not in report["warnings"]


def test_edge_audit_makes_no_options_claim_when_the_stage_never_ran():
    scan = _scan_with({"options_data_provider": None, "options_data_feed": None,
                       "options_data_quality": 0.0, "options_open_interest": 100.0})

    report = compute_edge_audit_report(_PASSING_VALIDATION, scan)

    assert "options_data_not_execution_grade" not in report["warnings"]
    assert "options_no_liquid_contract" not in report["warnings"]
    assert "options_provider_unavailable" not in report["warnings"]


def test_edge_audit_healthy_options_raise_no_options_warning():
    report = compute_edge_audit_report(_PASSING_VALIDATION, _scan_with({}))

    assert not [code for code in report["warnings"] if code.startswith("options_")]
