from scanner.edge.validation import compute_edge_validation_report


def test_compute_edge_validation_report_threshold_and_topk_metrics():
    candidates = [
        {"ticker": "A", "edge_score": 90, "outcome_label": "win", "outcome_return_pct": 4.0, "r_multiple": 2.0},
        {"ticker": "B", "edge_score": 80, "outcome_label": "loss", "outcome_return_pct": -2.0, "r_multiple": -1.0},
        {"ticker": "C", "edge_score": 60, "outcome_label": "win", "outcome_return_pct": 1.5, "r_multiple": 0.8},
        {"ticker": "D", "edge_score": 30, "outcome_label": "loss", "outcome_return_pct": -1.0, "r_multiple": -0.4},
    ]

    report = compute_edge_validation_report(candidates, thresholds=(50, 75), top_k=2, cost_bps_per_side=5.0)

    assert report["samples"] == 4
    assert report["thresholds"]["50"]["signal_count"] == 3
    assert report["thresholds"]["50"]["precision"] == 2 / 3
    assert report["thresholds"]["50"]["recall"] == 1.0
    assert report["thresholds"]["75"]["signal_count"] == 2
    assert report["thresholds"]["75"]["precision"] == 0.5
    assert report["top_k"]["k"] == 2
    assert report["top_k"]["precision"] == 0.5
    block = report["thresholds"]["50"]
    assert block["average_return_pct"] < block["average_gross_return_pct"]
    assert report["cost_model"]["bps_per_side"] == 5.0
    assert report["cost_model"]["round_trip_return_pct_charged"] == 0.1
    assert report["cost_model"]["basis"] == "net_of_costs"
    assert report["cost_model"]["candidate_rows"] == 4
    assert report["cost_model"]["risk_coverage_rows"] == 0
    assert report["cost_model"]["risk_coverage_complete"] is False
    assert set(report["cost_model"]["applies_to"]) == {"returns", "r_multiple", "win_loss_label"}


def test_costs_are_charged_before_every_gate_metric():
    """A cohort that is profitable gross but not net must not read as profitable."""
    # +0.20% gross on a 1% stop = +0.20R gross. A 25bps/side round trip costs
    # 0.50 percentage points, which flips both the return and the label.
    candidates = [
        {
            "ticker": f"T{i}",
            "timestamp": f"2026-03-{i + 1:02d}T15:00:00-04:00",
            "direction": "bullish",
            "edge_score": 90.0,
            "outcome_label": "win",
            "outcome_return_pct": 0.20,
            "r_multiple": 0.20,
            "risk_pct_used": 1.0,
        }
        for i in range(10)
    ]

    gross = compute_edge_validation_report(candidates, thresholds=(50,), top_k=5, cost_bps_per_side=0.0)
    net = compute_edge_validation_report(candidates, thresholds=(50,), top_k=5, cost_bps_per_side=25.0)

    assert gross["thresholds"]["50"]["precision"] == 1.0
    assert gross["thresholds"]["50"]["average_r_multiple"] > 0

    # Net of costs the same cohort is a total loss: label, R and precision all flip.
    assert net["thresholds"]["50"]["precision"] == 0.0
    assert net["thresholds"]["50"]["average_r_multiple"] < 0
    assert net["wins"] == 0
    assert net["cost_model"]["risk_coverage_rows"] == 10
    assert net["cost_model"]["risk_coverage_complete"] is True
    # Gross stays visible for diagnosis rather than being overwritten.
    assert net["thresholds"]["50"]["average_gross_return_pct"] == 0.20


def test_missing_risk_pct_does_not_fabricate_a_net_r():
    candidates = [
        {"ticker": "A", "edge_score": 90, "outcome_label": "win", "outcome_return_pct": 4.0, "r_multiple": 2.0},
    ]
    report = compute_edge_validation_report(candidates, thresholds=(50,), top_k=1, cost_bps_per_side=25.0)
    # No stop distance on record -> stored R is preserved, return is still charged.
    assert report["thresholds"]["50"]["average_r_multiple"] == 2.0
    assert report["thresholds"]["50"]["average_return_pct"] == 3.5
    assert report["cost_model"]["risk_coverage_rows"] == 0
    assert report["cost_model"]["risk_coverage_complete"] is False


def test_by_direction_blocks_carry_within_direction_ranking_metrics():
    candidates = [
        {
            "ticker": f"T{i}",
            "timestamp": f"2026-03-{(i % 25) + 1:02d}T15:00:00-04:00",
            "direction": "bullish" if i % 2 == 0 else "bearish",
            "edge_score": float(i),
            "outcome_label": "win" if i % 3 == 0 else "loss",
            "outcome_return_pct": (i % 7) - 3.0,
            "r_multiple": ((i % 7) - 3.0) / 2.0,
        }
        for i in range(80)
    ]

    report = compute_edge_validation_report(candidates)

    bullish = report["by_direction"]["bullish"]
    assert "rank_ic_r" in bullish and "n_days" in bullish["rank_ic_r"]
    assert "p_value_day_clustered" in bullish["rank_ic_r"]
    assert "tercile_lift" in bullish
    assert "tail_retention" in bullish
    assert bullish["t_stat_r_day_clustered"]["n_days"] > 0
    top_decile = report["percentiles"]["top_10_pct"]
    assert top_decile["precision_day_clustered"]["n_days"] > 0
    assert top_decile["t_stat_r_day_clustered"]["method"] == "entry_day_mean_hac_lag5"
    # Tiny subsets must degrade gracefully, not crash.
    small = compute_edge_validation_report(candidates[:4])
    assert small["by_direction"]["bullish"]["tercile_lift"]["insufficient"] is True
