import numpy as np

from scanner.edge.stats import (
    day_clustered_precision,
    day_clustered_t,
    spearman_rank_ic,
    tail_retention,
    tercile_lift,
    t_statistic,
)


def _synthetic_rows(n=120, ic_strength=0.9, seed=3):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    outcomes = ic_strength * scores + rng.normal(scale=0.5, size=n)
    days = [f"2026-01-{(i % 20) + 1:02d}" for i in range(n)]
    ids = [f"T{i}" for i in range(n)]
    return list(scores), list(outcomes), days, ids


def test_day_clustered_t_uses_days_not_trades():
    # 100 identical trades on 2 days must not manufacture a huge t-stat.
    values = [1.0] * 50 + [0.5] * 50
    days = ["2026-01-05"] * 50 + ["2026-01-06"] * 50
    result = day_clustered_t(values, days)
    assert result["n_days"] == 2
    # 2 day-means -> t is computable but tiny-n; the point is n_days honesty.
    assert result["mean_of_day_means"] == 0.75
    assert result["method"] == "entry_day_mean_hac_lag5"


def test_day_clustered_t_accounts_for_serially_overlapping_outcomes():
    # Smooth day-level returns are strongly serially correlated. Treating the
    # 30 daily means as IID materially overstates the evidence.
    day_values = [0.6 + 0.25 * np.sin(i / 4.0) for i in range(30)]
    values = [value for value in day_values for _ in range(4)]
    days = [f"2026-01-{i + 1:02d}" for i in range(30) for _ in range(4)]

    result = day_clustered_t(values, days)

    assert result["n_days"] == 30
    assert result["t_stat"] < t_statistic(day_values)


def test_day_clustered_precision_does_not_count_correlated_trades_as_iid():
    # Two entry days, not 200 independent Bernoulli trials.
    wins = [1.0] * 100 + [0.0] * 100
    days = ["2026-01-05"] * 100 + ["2026-01-06"] * 100

    result = day_clustered_precision(wins, days)

    assert result["signal_count"] == 200
    assert result["n_days"] == 2
    assert result["mean_daily_precision"] == 0.5
    assert result["lower_bound"] < 0.5
    assert result["method"] == "entry_day_mean_hac_lag5_min_wilson"


def test_spearman_day_clustered_p_is_more_conservative():
    scores, outcomes, days, _ = _synthetic_rows()
    result = spearman_rank_ic(scores, outcomes, day_keys=days)
    assert result["n"] == 120
    assert result["n_days"] == 20
    assert result["p_value_day_clustered"] >= result["p_value"]
    assert result["day_cluster_method"] == "entry_day_cluster_hac_lag5_cr1"
    assert result["p_value_alternative"] == "greater"


def test_spearman_clustered_p_uses_within_day_dependence_not_just_day_count():
    # Both datasets have the same n, n_days and global IC. In the first, each
    # day's observations move together; in the second, dependence is dispersed.
    # A real sandwich estimator can distinguish them, unlike the old n_days
    # substitution shortcut.
    rng = np.random.default_rng(41)
    n_days = 30
    per_day = 8
    day_signal = rng.normal(size=n_days)
    scores_clustered = np.repeat(day_signal, per_day) + rng.normal(scale=0.8, size=n_days * per_day)
    outcomes_clustered = 0.1 * np.repeat(day_signal, per_day) + rng.normal(scale=1.0, size=n_days * per_day)
    days = [f"d{i}" for i in range(n_days) for _ in range(per_day)]

    clustered = spearman_rank_ic(list(scores_clustered), list(outcomes_clustered), day_keys=days)
    shuffled = spearman_rank_ic(
        list(scores_clustered),
        list(outcomes_clustered),
        day_keys=list(np.asarray(days)[rng.permutation(len(days))]),
    )

    assert clustered["n_days"] == shuffled["n_days"] == n_days
    assert clustered["p_value_day_clustered"] != shuffled["p_value_day_clustered"]


def test_spearman_reports_two_sided_p_for_negative_association():
    scores = list(range(60))
    outcomes = list(reversed(scores))
    days = [f"d{i // 3}" for i in range(60)]
    result = spearman_rank_ic(scores, outcomes, day_keys=days)

    assert result["p_value"] > 0.99  # one-sided alternative is positive IC
    assert result["p_value_two_sided"] < 0.001


def test_tercile_lift_detects_real_spread():
    scores, outcomes, days, ids = _synthetic_rows()
    result = tercile_lift(scores, outcomes, days, row_ids=ids)
    assert result["insufficient"] is False
    assert result["spread_r"] > 0
    assert result["spread_ci_low"] is not None
    assert result["spread_ci_low"] > 0  # strong synthetic signal


def test_tercile_lift_insufficient_below_30():
    result = tercile_lift([1.0] * 10, [0.5] * 10, ["2026-01-01"] * 10)
    assert result["insufficient"] is True


def test_tercile_lift_is_deterministic():
    scores, outcomes, days, ids = _synthetic_rows()
    a = tercile_lift(scores, outcomes, days, row_ids=ids)
    b = tercile_lift(scores, outcomes, days, row_ids=ids)
    assert a == b
    assert a["bootstrap_method"] == "circular_moving_entry_day_blocks"
    assert a["bootstrap_block_days"] == 5


def test_tercile_lift_mass_ties_do_not_bias_by_input_order():
    # 90 zero-score rows where the LAST 30 in input order carry all the R:
    # stable sort would put early rows in the top tercile deterministically.
    scores = [0.0] * 90
    outcomes = [0.0] * 60 + [2.0] * 30
    days = [f"2026-02-{(i % 15) + 1:02d}" for i in range(90)]
    ids = [f"X{i}" for i in range(90)]
    result = tercile_lift(scores, outcomes, days, row_ids=ids)
    # With hash tie-breaking the tail spreads across buckets instead of
    # landing wholesale in one; the spread must not be the degenerate 2.0.
    assert abs(result["spread_r"]) < 2.0


def test_tail_retention_reports_top_tercile_capture():
    # Scores perfectly rank outcomes: every tail trade is in the top tercile.
    n = 90
    scores = list(range(n))
    outcomes = [s / 10.0 for s in scores]
    ids = [f"Y{i}" for i in range(n)]
    result = tail_retention(scores, outcomes, row_ids=ids, tail_r=8.0)
    assert result["insufficient"] is False
    assert result["tail_count"] == 10
    assert result["tail_in_top_tercile"] == 10
    assert result["observed_share"] == 1.0


def test_tail_retention_insufficient_below_30():
    result = tail_retention([1.0] * 5, [0.5] * 5)
    assert result["insufficient"] is True
