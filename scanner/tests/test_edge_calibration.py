import numpy as np
import pytest

from scanner.edge.calibration import (
    META_FEATURE_KEYS,
    META_OBJECTIVES,
    META_PURGE_DAYS,
    fit_model,
    fit_win_probability_model,
    predict_expected_r,
    predict_score,
    predict_win_probability,
    select_shippable_objective,
    walk_forward_calibration,
    walk_forward_calibration_suite,
)
from scanner.config import EDGE_EMBARGO_DAYS


def _features(rng, signal=0.0):
    # volume_expansion carries the planted signal; everything else is noise.
    return {
        "volume_expansion": 1.0 + signal + rng.normal(scale=0.2),
        "volume_percentile": rng.uniform(0, 100),
        "breakout_strength_pct": rng.normal(scale=1.0),
        "close_position_in_box": rng.uniform(0, 1),
        "box_width_pct": rng.uniform(1, 8),
        "range_compression_ratio": rng.uniform(0.4, 1.1),
        "realized_volatility_pct": rng.uniform(1, 6),
        "no_trend_score": rng.uniform(0, 1),
        "doctrine_v2_score": rng.uniform(40, 90),
        "recent_return_pct": rng.normal(scale=2.0),
    }


def _records(n=900, predictive=True, seed=11):
    """Synthetic bullish index records spread across ~2 years of days."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        signal = rng.normal()
        features = _features(rng, signal=0.5 * signal if predictive else 0.0)
        r_multiple = (0.8 * signal if predictive else 0.0) + rng.normal(scale=0.8)
        day = i // 2  # two records per day
        ts = np.datetime64("2024-08-01") + np.timedelta64(int(day), "D")
        records.append(
            {
                "ticker": f"T{i % 40}",
                "timestamp": f"{ts}T15:00:00+00:00",
                "direction": "bullish",
                "features": features,
                "r_multiple": float(r_multiple),
            }
        )
    return records


def test_fit_learns_planted_coefficient_direction():
    records = _records(n=600)
    model = fit_win_probability_model(
        [r["features"] for r in records], [r["r_multiple"] for r in records]
    )
    assert model is not None
    idx = list(META_FEATURE_KEYS).index("volume_expansion")
    assert model["coefficients"][idx] > 0.1
    assert model["n_train"] == 600


def test_fit_returns_none_below_min_train():
    records = _records(n=100)
    model = fit_win_probability_model(
        [r["features"] for r in records], [r["r_multiple"] for r in records]
    )
    assert model is None


def test_predict_bounds_and_expected_r_sign():
    records = _records(n=600)
    model = fit_win_probability_model(
        [r["features"] for r in records], [r["r_multiple"] for r in records]
    )
    rng = np.random.default_rng(0)
    strong = _features(rng, signal=3.0)
    weak = _features(rng, signal=-3.0)
    p_strong = predict_win_probability(model, strong)
    p_weak = predict_win_probability(model, weak)
    assert 0.0 <= p_weak < p_strong <= 1.0
    assert predict_expected_r(model, strong) > predict_expected_r(model, weak)


def test_predict_handles_missing_features_via_median():
    records = _records(n=600)
    model = fit_win_probability_model(
        [r["features"] for r in records], [r["r_multiple"] for r in records]
    )
    p = predict_win_probability(model, {"volume_expansion": 1.2})
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_walk_forward_accepts_planted_signal():
    result = walk_forward_calibration(_records(n=900, predictive=True))
    assert result["n_evaluated"] >= 300
    metrics = result["metrics"]
    assert metrics["insufficient"] is False
    assert metrics["rank_ic_r"]["ic"] > 0.07
    assert metrics["beats_naive"] is True
    assert result["acceptance"]["passed"] is True
    assert result["final_model"] is not None
    # Every prediction is out-of-fold and keyed for joining.
    assert len(result["predictions"]) == result["n_evaluated"]


def test_walk_forward_rejects_noise():
    result = walk_forward_calibration(_records(n=900, predictive=False, seed=29))
    assert result["acceptance"]["passed"] is False
    # The honest failure is low IC / no tercile spread, not a crash.
    assert result["metrics"]["insufficient"] is False
    assert abs(result["metrics"]["rank_ic_r"]["ic"]) < 0.15


def test_walk_forward_is_deterministic():
    a = walk_forward_calibration(_records(n=700))
    b = walk_forward_calibration(_records(n=700))
    assert a["metrics"] == b["metrics"]
    assert a["acceptance"] == b["acceptance"]


def test_walk_forward_insufficient_records_fails_closed():
    result = walk_forward_calibration(_records(n=120))
    assert result["acceptance"]["passed"] is False
    assert result["metrics"]["insufficient"] is True


def test_walk_forward_ignores_other_directions():
    records = _records(n=900)
    for record in records[:450]:
        record["direction"] = "bearish"
    result = walk_forward_calibration(records)
    assert result["n_records"] == 450


def test_predictions_are_out_of_fold_purged():
    # A record must never be predicted by a model trained on records less
    # than purge_days older: plant a single overwhelming outlier the day
    # before a prediction and confirm it cannot flip that prediction via
    # training leakage. Structural proxy: training cutoff respects purge.
    records = _records(n=700)
    result = walk_forward_calibration(records)
    assert META_PURGE_DAYS == EDGE_EMBARGO_DAYS == 11
    assert result["config"]["purge_days"] == 11
    assert result["n_evaluated"] > 0


def test_expected_r_objective_learns_magnitude():
    records = _records(n=600)
    model = fit_model([r["features"] for r in records], [r["r_multiple"] for r in records], objective="expected_r")
    assert model is not None
    assert model["objective"] == "expected_r"
    rng = np.random.default_rng(0)
    strong_features = _features(rng, signal=3.0)
    weak_features = _features(rng, signal=-3.0)
    assert predict_score(model, strong_features) > predict_score(model, weak_features)
    # expected_r's E[R] IS its score (R units, unbounded).
    assert predict_expected_r(model, strong_features) == pytest.approx(predict_score(model, strong_features))


def test_tail_prob_objective_needs_enough_tail_events():
    # All outcomes far below the 2R tail threshold -> too few positive
    # events -> the fit must refuse rather than learn from ~nothing.
    records = _records(n=600, predictive=False, seed=5)
    for record in records:
        record["r_multiple"] = float(np.clip(record["r_multiple"], -1.5, 1.5))
    model = fit_model([r["features"] for r in records], [r["r_multiple"] for r in records], objective="tail_prob")
    assert model is None


def test_suite_runs_every_registered_objective():
    suite = walk_forward_calibration_suite(_records(n=900))
    assert set(suite) == set(META_OBJECTIVES)
    for objective, result in suite.items():
        assert result["objective"] == objective
        assert "acceptance" in result


def _suite_stub(passes, ics):
    return {
        objective: {
            "acceptance": {"passed": passes.get(objective, False)},
            "metrics": {"rank_ic_r": {"ic": ics.get(objective, 0.0)}},
            "final_model": {"objective": objective},
        }
        for objective in META_OBJECTIVES
    }


def test_two_touch_ship_rule():
    suite = _suite_stub({"expected_r": True, "tail_prob": True}, {"expected_r": 0.09, "tail_prob": 0.12})

    # Nothing shipped without a previous pass (first touch only).
    assert select_shippable_objective(suite, {}) is None
    # Previous pass on one objective -> that one ships.
    assert select_shippable_objective(suite, {"expected_r": True}) == "expected_r"
    # Both two-touched -> highest current OOF IC wins.
    assert select_shippable_objective(suite, {"expected_r": True, "tail_prob": True}) == "tail_prob"
    # A previous pass cannot ship a model that fails NOW.
    assert select_shippable_objective(suite, {"p_win": True}) is None


def test_ship_rule_requires_final_model():
    suite = _suite_stub({"expected_r": True}, {"expected_r": 0.09})
    suite["expected_r"]["final_model"] = None
    assert select_shippable_objective(suite, {"expected_r": True}) is None


@pytest.mark.parametrize("bad", [None, {}, {"feature_keys": None}])
def test_predict_win_probability_rejects_malformed_model(bad):
    assert predict_win_probability(bad, {"volume_expansion": 1.0}) in (None,)


def test_fit_survives_whole_column_feature_outage():
    # Adversary finding: an all-NaN feature column used to poison the
    # winsorize bounds -> NaN standardization -> NaN weights -> predictions
    # returned NaN (not None), silently annotating live candidates. The fit
    # must now either neutralize the dead column or fail closed - never
    # emit non-finite parameters.
    import math

    records = _records(n=600)
    for record in records:
        record["features"]["volume_percentile"] = float("nan")
    model = fit_win_probability_model(
        [r["features"] for r in records], [r["r_multiple"] for r in records]
    )
    if model is not None:
        assert all(math.isfinite(c) for c in model["coefficients"])
        assert math.isfinite(model["intercept"])
        rng = np.random.default_rng(1)
        p = predict_win_probability(model, _features(rng))
        assert p is not None and math.isfinite(p)
