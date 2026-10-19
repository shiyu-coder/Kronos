import pytest

from scanner.edge.audit import compute_edge_audit_report
from scanner.edge.stats import spearman_rank_ic, t_statistic, wilson_lower_bound
from scanner.edge.validation import compute_edge_validation_report


def test_wilson_lower_bound_matches_known_values():
    assert wilson_lower_bound(8, 22, z=1.28) == pytest.approx(0.2461, abs=1e-4)
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(10, 10, z=1.645) < 1.0


def test_spearman_rank_ic_detects_monotonic_relationship():
    scores = [float(i) for i in range(30)]
    outcomes = [float(i) * 0.5 for i in range(30)]

    result = spearman_rank_ic(scores, outcomes)

    assert result["ic"] == pytest.approx(1.0)
    assert result["p_value"] < 0.001
    assert result["n"] == 30

    inverted = spearman_rank_ic(scores, list(reversed(outcomes)))
    assert inverted["ic"] == pytest.approx(-1.0)
    assert inverted["p_value"] > 0.99


def test_spearman_rank_ic_degenerate_inputs():
    assert spearman_rank_ic([1.0, 2.0], [1.0, 2.0]) == {"ic": 0.0, "p_value": 1.0, "n": 2}
    flat = spearman_rank_ic([1.0] * 10, [float(i) for i in range(10)])
    assert flat["ic"] == 0.0
    assert flat["p_value"] == 1.0


def test_t_statistic_zero_for_tiny_or_flat_samples():
    assert t_statistic([]) == 0.0
    assert t_statistic([1.0]) == 0.0
    assert t_statistic([1.0, 1.0, 1.0]) == 0.0
    assert t_statistic([0.5, 0.6, 0.4, 0.5, 0.7]) > 0.0


def test_validation_report_includes_ranking_metrics():
    candidates = [
        {
            "ticker": f"T{i}",
            "timestamp": f"2026-03-{(i % 20) + 1:02d}T00:00:00-05:00",
            "direction": "bullish" if i % 2 == 0 else "bearish",
            "edge_score": float(i),
            "outcome_label": "win" if i >= 20 else "loss",
            "outcome_return_pct": float(i - 20),
            "r_multiple": float(i - 20) / 10.0,
            "exit_reason": "target" if i >= 20 else "stop",
        }
        for i in range(40)
    ]

    report = compute_edge_validation_report(candidates, thresholds=(50,), top_k=5)

    assert report["rank_ic_r"]["ic"] > 0.9
    assert report["rank_ic_r"]["p_value"] < 0.01
    assert report["percentiles"]["top_10_pct"]["signal_count"] == 4
    assert report["percentiles"]["top_10_pct"]["wilson_lb_precision"] > 0.3
    assert report["percentiles"]["top_10_pct"]["target_rate"] == 1.0
    assert report["decile_spread"]["spread_r"] > 0
    assert report["concentration"]["distinct_days"] == 20


def _scan(direction="bullish", recommendation="research"):
    return {
        "candidates": [
            {
                "ticker": "TEST",
                "status": "candidate",
                "direction": direction,
                "recommendation": recommendation,
                "features": {
                    "feed_confidence": 0.9,
                    "data_delay_minutes": 0.0,
                    "options_passed": 1.0,
                    "options_open_interest": 900.0,
                    "options_volume": 120.0,
                    "options_spread_pct": 0.05,
                    "options_data_provider": "tradier",
                    "options_data_feed": "opra-consolidated",
                    "options_data_quality": 0.9,
                },
            }
        ]
    }


def _validation_with_ranking(top_decile_signals=40, bearish_avg_r=0.1, bearish_n=20):
    return {
        "samples": 600,
        "validation_method": "purged_walk_forward",
        "future_analogs_allowed": False,
        "cost_model": {
            "bps_per_side": 25.0,
            "round_trip_return_pct_charged": 0.5,
            "basis": "net_of_costs",
            "candidate_rows": 600,
            "risk_coverage_rows": 600,
            "risk_coverage_complete": True,
            "applies_to": ["returns", "r_multiple", "win_loss_label"],
        },
        "thresholds": {
            "55": {"signal_count": 0, "precision": 0.0, "average_r_multiple": 0.0},
        },
        "rank_ic_r": {
            "ic": 0.12,
            "p_value": 0.002,
            "p_value_day_clustered": 0.003,
            "n": 600,
        },
        "percentiles": {
            "top_10_pct": {
                "signal_count": top_decile_signals,
                "average_r_multiple": 0.4,
                "t_stat_r_multiple": 2.6,
                "wilson_lb_precision": 0.48,
                "precision_day_clustered": {
                    "n_days": top_decile_signals,
                    "mean_daily_precision": 0.6,
                    "lower_bound": 0.48,
                },
                "t_stat_r_day_clustered": {
                    "n_days": top_decile_signals,
                    "mean_of_day_means": 0.4,
                    "t_stat": 2.6,
                },
            }
        },
        "by_direction": {
            "bullish": {
                "signal_count": 300,
                "average_r_multiple": 0.2,
                "rank_ic_r": {"ic": 0.10, "p_value": 0.01, "p_value_day_clustered": 0.02, "n": 300},
                "t_stat_r_day_clustered": {"t_stat": 2.5, "n_days": 60},
            },
            "bearish": {
                "signal_count": bearish_n,
                "average_r_multiple": bearish_avg_r,
                "rank_ic_r": {"ic": 0.0, "p_value": 0.5, "p_value_day_clustered": 0.6, "n": bearish_n},
                "t_stat_r_day_clustered": {
                    "t_stat": 2.1 if bearish_avg_r > 0 else -2.1,
                    "n_days": 20,
                },
            },
        },
    }


def test_audit_accepts_ranking_evidence_when_absolute_threshold_is_starved():
    report = compute_edge_audit_report(_validation_with_ranking(), _scan())

    assert report["checks"]["ranking_evidence"]["passed"] is True
    assert report["blockers"] == []
    assert report["readiness"] == "research_only"


def test_audit_blocks_when_both_evidence_routes_fail():
    report = compute_edge_audit_report(_validation_with_ranking(top_decile_signals=5), _scan())

    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert "validation_threshold_55_unsupported" in report["blockers"]
    assert "ranking_evidence_unsupported" in report["blockers"]
    assert report["readiness"] == "blocked"


def test_audit_flags_negative_direction_and_blocks_promotion_in_it():
    validation = _validation_with_ranking(bearish_avg_r=-0.15, bearish_n=25)

    research_report = compute_edge_audit_report(validation, _scan(direction="bearish"))
    assert "bearish_edge_negative" in research_report["warnings"]
    assert research_report["summary"]["blocked_directions"] == ["bearish"]
    assert research_report["summary"]["promotable_directions"] == ["bullish"]

    promoted_report = compute_edge_audit_report(validation, _scan(direction="bearish", recommendation="promote"))
    assert promoted_report["readiness"] == "research_only"
    assert "promoted_candidates_direction_blocked" in promoted_report["warnings"]

    bullish_promoted = compute_edge_audit_report(validation, _scan(direction="bullish", recommendation="promote"))
    assert bullish_promoted["readiness"] == "paper_trade_only"


def test_audit_treats_undersampled_direction_as_unproven_not_safe():
    # n=14 bearish with terrible avg R sits below min_direction_samples; the
    # guardrail must fail CLOSED for promotion, not open.
    validation = _validation_with_ranking(bearish_avg_r=-3.0, bearish_n=14)

    promoted_report = compute_edge_audit_report(validation, _scan(direction="bearish", recommendation="promote"))

    assert "bearish" not in promoted_report["summary"]["promotable_directions"]
    assert promoted_report["readiness"] == "research_only"
    assert "promoted_candidates_direction_blocked" in promoted_report["warnings"]


def test_audit_blocks_promotion_when_direction_history_is_absent():
    # The promoted candidate's direction has no validation history at all:
    # it must not be promotable, even while another direction's evidence
    # keeps the ranking route alive.
    validation = _validation_with_ranking()
    validation["by_direction"].pop("bearish")

    promoted_report = compute_edge_audit_report(validation, _scan(direction="bearish", recommendation="promote"))

    assert "bearish" not in promoted_report["summary"]["promotable_directions"]
    assert promoted_report["readiness"] == "research_only"
    assert "promoted_candidates_direction_blocked" in promoted_report["warnings"]


def test_audit_blocks_when_no_direction_history_exists_at_all():
    # Removing ALL per-direction evidence must fail the ranking route
    # closed - pooled IC alone can be pure direction-separation artifact.
    validation = _validation_with_ranking()
    validation.pop("by_direction")

    report = compute_edge_audit_report(validation, _scan(direction="bearish", recommendation="promote"))

    assert report["checks"]["ranking_evidence"]["passed"] is False
    assert report["readiness"] == "blocked"
