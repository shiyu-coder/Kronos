import pandas as pd

from scanner.edge.features import extract_edge_features
from scanner.edge.retrieval import EdgeAnalogIndex, EdgeRecord, find_analogs, select_recent_records
from scanner.edge.scoring import score_edge_candidate
from scanner.strategy.potter_box import detect_potter_box


def _record(ticker, timestamp, direction="bullish", features=None, **overrides):
    payload = {
        "ticker": ticker,
        "timestamp": timestamp,
        "direction": direction,
        "features": features or {},
        "outcome_return_pct": 1.0,
        "outcome_label": "win",
        "r_multiple": 0.5,
        "mae_pct": -1.0,
        "mfe_pct": 2.0,
    }
    payload.update(overrides)
    return EdgeRecord(**payload)


def _shape_features(**overrides):
    features = {
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
        "rr_ratio": 2.0,
        "box_width_pct": 4.0,
        "close_position_in_box": 0.9,
    }
    features.update(overrides)
    return features


def test_distance_ignores_price_level_features():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-01T00:00:00-05:00",
        **_shape_features(),
        "latest_close": 5.0,
        "box_top": 5.2,
        "box_bottom": 4.8,
        "atr_value": 0.15,
    }
    cheap = _record(
        "BBB",
        "2026-01-01T00:00:00-05:00",
        features={**_shape_features(), "latest_close": 5.1, "box_top": 5.3, "box_bottom": 4.9, "atr_value": 0.14},
    )
    expensive = _record(
        "CCC",
        "2026-01-01T00:00:00-05:00",
        features={**_shape_features(), "latest_close": 480.0, "box_top": 495.0, "box_bottom": 465.0, "atr_value": 12.0},
    )

    analogs = find_analogs(query, [cheap, expensive], k=2)

    assert len(analogs) == 2
    assert abs(analogs[0]["distance"] - analogs[1]["distance"]) < 1e-9


def test_direction_match_excludes_opposite_direction():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-01T00:00:00-05:00",
        "direction": "bullish",
        **_shape_features(),
    }
    bearish = _record("BBB", "2026-01-01T00:00:00-05:00", direction="bearish", features=_shape_features())
    bullish = _record("CCC", "2026-01-05T00:00:00-05:00", direction="bullish", features=_shape_features())

    matched = find_analogs(query, [bearish, bullish], k=5, direction_match=True)
    unmatched = find_analogs(query, [bearish, bullish], k=5, direction_match=False)

    assert [row["ticker"] for row in matched] == ["CCC"]
    assert len(unmatched) == 2


def test_cross_ticker_embargo_blocks_same_day_records():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-10T00:00:00-05:00",
        **_shape_features(),
    }
    same_day_other_ticker = _record("BBB", "2026-02-10T00:00:00-05:00", features=_shape_features())
    older = _record("CCC", "2026-01-10T00:00:00-05:00", features=_shape_features())

    analogs = find_analogs(query, [same_day_other_ticker, older], k=5, cross_ticker_embargo_days=1)

    assert [row["ticker"] for row in analogs] == ["CCC"]


def test_index_matches_brute_force_for_direction_and_cross_ticker_filters():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-10T00:00:00-05:00",
        "direction": "bullish",
        **_shape_features(),
    }
    records = [
        _record("BBB", "2026-02-10T00:00:00-05:00", features=_shape_features()),
        _record("CCC", "2026-01-10T00:00:00-05:00", direction="bearish", features=_shape_features()),
        _record("DDD", "2026-01-05T00:00:00-05:00", features=_shape_features(breakout_distance_pct=2.2)),
    ]

    kwargs = {"k": 5, "direction_match": True, "cross_ticker_embargo_days": 1}
    indexed = EdgeAnalogIndex(records).find_analogs(query, **kwargs)
    direct = find_analogs(query, records, **kwargs)

    assert [row["ticker"] for row in indexed] == [row["ticker"] for row in direct] == ["DDD"]


def test_purge_config_covers_outcome_horizon():
    # An analog's 5-trading-bar outcome spans up to ~9 calendar days (with a
    # holiday). If either embargo is shorter, analog outcomes that resolve
    # AFTER the query bar leak future market moves into the query's score.
    from scanner import config as scanner_config

    min_covering_days = 9  # PRED_DAYS=5 trading bars, worst-case calendar span
    assert scanner_config.PRED_DAYS == 5
    assert scanner_config.EDGE_EMBARGO_DAYS >= min_covering_days
    assert scanner_config.EDGE_CROSS_TICKER_EMBARGO_DAYS >= min_covering_days


def test_cross_ticker_embargo_excludes_unresolved_outcome_windows():
    # Analog entered 3 days before the query: its outcome window still spans
    # the query's future. Under a horizon-covering embargo it must be blocked.
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-10T00:00:00-05:00",
        **_shape_features(),
    }
    overlapping = _record("BBB", "2026-02-07T00:00:00-05:00", features=_shape_features())
    resolved = _record("CCC", "2026-01-10T00:00:00-05:00", features=_shape_features())

    analogs = find_analogs(query, [overlapping, resolved], k=5, cross_ticker_embargo_days=9)

    assert [row["ticker"] for row in analogs] == ["CCC"]


def test_select_recent_records_picks_most_recent_by_time_across_tickers():
    records = [
        _record("AAA", "2026-01-01T00:00:00-05:00"),
        _record("AAA", "2026-03-01T00:00:00-05:00"),
        _record("BBB", "2026-02-01T00:00:00-05:00"),
        _record("BBB", "2026-04-01T00:00:00-05:00"),
    ]

    selected = select_recent_records(records, 2)

    assert sorted(row.timestamp for row in selected) == [
        "2026-03-01T00:00:00-05:00",
        "2026-04-01T00:00:00-05:00",
    ]
    assert select_recent_records(records, 0) == records
    assert select_recent_records(records, 10) == records


def _bars():
    rows = []
    for i in range(40):
        if i < 24:
            rows.append([100, 104, 96, 100 + (0.2 if i % 2 == 0 else -0.2), 1000])
        elif i < 39:
            rows.append([100, 101, 99, 100 + (0.05 if i % 2 == 0 else -0.05), 1200])
        else:
            rows.append([101, 104, 100.5, 103, 2500])
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def test_missing_kronos_features_are_neutral_not_penalized():
    bars = _bars()
    pb = detect_potter_box("TEST", bars)

    features = extract_edge_features("TEST", bars, pb)

    assert features["kronos_directional_agreement"] is None
    assert features["kronos_median_forecast_return_pct"] is None
    assert features["kronos_worst_sampled_return_pct"] is None

    scoring = score_edge_candidate(features, analogs=[])
    assert scoring["scorecard"]["kronos"] == 0.0


def test_present_kronos_features_remain_advisory():
    bars = _bars()
    pb = detect_potter_box("TEST", bars)
    kronos = {
        "passed": True,
        "directional_agreement": 0.75,
        "median_forecast_return_pct": 1.2,
        "worst_sampled_return_pct": -2.0,
        "sample_count": 10,
    }

    features = extract_edge_features("TEST", bars, pb, kronos=kronos)

    assert features["kronos_directional_agreement"] == 0.75
    scoring = score_edge_candidate(features, analogs=[])
    assert scoring["scorecard"]["kronos"] == 0.0
