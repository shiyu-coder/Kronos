from scanner.edge.retrieval import EdgeAnalogIndex, EdgeRecord, find_analogs
from scanner.config import EDGE_CROSS_TICKER_EMBARGO_DAYS, EDGE_EMBARGO_DAYS


def test_find_analogs_ranks_nearest_numeric_features():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-01T00:00:00-05:00",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
        "rr_ratio": 2.0,
    }
    near = EdgeRecord(
        ticker="BBB",
        timestamp="2026-01-01T00:00:00-05:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.1, "volume_expansion": 1.45, "rr_ratio": 2.1},
        outcome_return_pct=3.0,
        outcome_label="win",
        r_multiple=1.5,
        mae_pct=-1.0,
        mfe_pct=4.0,
    )
    far = EdgeRecord(
        ticker="CCC",
        timestamp="2026-01-01T00:00:00-05:00",
        direction="bullish",
        features={"breakout_distance_pct": 9.0, "volume_expansion": 0.3, "rr_ratio": 0.4},
        outcome_return_pct=-2.0,
        outcome_label="loss",
        r_multiple=-1.0,
        mae_pct=-3.0,
        mfe_pct=1.0,
    )

    analogs = find_analogs(query, [far, near], k=2)

    assert [a["ticker"] for a in analogs] == ["BBB", "CCC"]
    assert analogs[0]["distance"] < analogs[1]["distance"]


def test_find_analogs_excludes_same_ticker_inside_embargo():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-10T00:00:00-05:00",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
    }
    leaked = EdgeRecord(
        ticker="AAA",
        timestamp="2026-02-08T00:00:00-05:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.0, "volume_expansion": 1.5},
        outcome_return_pct=5.0,
        outcome_label="win",
        r_multiple=2.0,
        mae_pct=-0.5,
        mfe_pct=5.5,
    )
    allowed = EdgeRecord(
        ticker="AAA",
        timestamp="2026-01-01T00:00:00-05:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.2, "volume_expansion": 1.4},
        outcome_return_pct=1.0,
        outcome_label="win",
        r_multiple=0.5,
        mae_pct=-1.0,
        mfe_pct=2.0,
    )

    analogs = find_analogs(query, [leaked, allowed], k=5, embargo_days=5)

    assert [a["timestamp"] for a in analogs] == ["2026-01-01T00:00:00-05:00"]


def test_default_embargo_excludes_holiday_extended_outcome_windows():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-07-13T00:00:00-04:00",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
    }
    still_resolving = EdgeRecord(
        ticker="AAA",
        timestamp="2026-07-03T00:00:00-04:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.0, "volume_expansion": 1.5},
        outcome_return_pct=4.0,
        outcome_label="win",
        r_multiple=1.5,
        mae_pct=-0.5,
        mfe_pct=4.5,
    )
    resolved = EdgeRecord(
        ticker="AAA",
        timestamp="2026-07-02T00:00:00-04:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.1, "volume_expansion": 1.4},
        outcome_return_pct=-1.0,
        outcome_label="loss",
        r_multiple=-0.4,
        mae_pct=-2.0,
        mfe_pct=1.0,
    )

    analogs = find_analogs(query, [still_resolving, resolved], k=5, embargo_days=EDGE_EMBARGO_DAYS)

    assert EDGE_EMBARGO_DAYS == 11
    assert [a["timestamp"] for a in analogs] == ["2026-07-02T00:00:00-04:00"]


def test_default_cross_ticker_embargo_excludes_same_week_market_leakage():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-07-13T00:00:00-04:00",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
    }
    still_resolving = EdgeRecord(
        ticker="BBB",
        timestamp="2026-07-03T00:00:00-04:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.0, "volume_expansion": 1.5},
        outcome_return_pct=4.0,
        outcome_label="win",
        r_multiple=1.5,
        mae_pct=-0.5,
        mfe_pct=4.5,
    )
    resolved = EdgeRecord(
        ticker="CCC",
        timestamp="2026-07-02T00:00:00-04:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.1, "volume_expansion": 1.4},
        outcome_return_pct=-1.0,
        outcome_label="loss",
        r_multiple=-0.4,
        mae_pct=-2.0,
        mfe_pct=1.0,
    )

    analogs = EdgeAnalogIndex([still_resolving, resolved]).find_analogs(
        query,
        k=5,
        embargo_days=EDGE_EMBARGO_DAYS,
        cross_ticker_embargo_days=EDGE_CROSS_TICKER_EMBARGO_DAYS,
    )

    assert EDGE_CROSS_TICKER_EMBARGO_DAYS == 11
    assert [a["timestamp"] for a in analogs] == ["2026-07-02T00:00:00-04:00"]


def test_edge_analog_index_matches_find_analogs_and_embargo():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-10T00:00:00-05:00",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
        "rr_ratio": 2.0,
    }
    records = [
        EdgeRecord(
            ticker="AAA",
            timestamp="2026-02-08T00:00:00-05:00",
            direction="bullish",
            features={"breakout_distance_pct": 2.0, "volume_expansion": 1.5, "rr_ratio": 2.0},
            outcome_return_pct=5.0,
            outcome_label="win",
            r_multiple=2.0,
            mae_pct=-0.5,
            mfe_pct=5.5,
        ),
        EdgeRecord(
            ticker="BBB",
            timestamp="2026-01-01T00:00:00-05:00",
            direction="bullish",
            features={"breakout_distance_pct": 2.1, "volume_expansion": 1.45, "rr_ratio": 2.1},
            outcome_return_pct=3.0,
            outcome_label="win",
            r_multiple=1.5,
            mae_pct=-1.0,
            mfe_pct=4.0,
        ),
        EdgeRecord(
            ticker="CCC",
            timestamp="2026-01-01T00:00:00-05:00",
            direction="bullish",
            features={"breakout_distance_pct": 9.0, "volume_expansion": 0.3, "rr_ratio": 0.4},
            outcome_return_pct=-2.0,
            outcome_label="loss",
            r_multiple=-1.0,
            mae_pct=-3.0,
            mfe_pct=1.0,
        ),
    ]

    indexed = EdgeAnalogIndex(records).find_analogs(query, k=2, embargo_days=5)
    direct = find_analogs(query, records, k=2, embargo_days=5)

    assert [row["ticker"] for row in indexed] == [row["ticker"] for row in direct]
    assert [row["timestamp"] for row in indexed] == [row["timestamp"] for row in direct]
    assert indexed[0]["distance"] == direct[0]["distance"]


def test_find_analogs_can_exclude_future_records_for_walk_forward_validation():
    query = {
        "ticker": "AAA",
        "timestamp": "2026-02-10T00:00:00-05:00",
        "breakout_distance_pct": 2.0,
        "volume_expansion": 1.5,
    }
    future_leak = EdgeRecord(
        ticker="BBB",
        timestamp="2026-02-20T00:00:00-05:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.0, "volume_expansion": 1.5},
        outcome_return_pct=8.0,
        outcome_label="win",
        r_multiple=3.0,
        mae_pct=-0.5,
        mfe_pct=9.0,
    )
    past_allowed = EdgeRecord(
        ticker="CCC",
        timestamp="2026-01-20T00:00:00-05:00",
        direction="bullish",
        features={"breakout_distance_pct": 2.3, "volume_expansion": 1.3},
        outcome_return_pct=-1.0,
        outcome_label="loss",
        r_multiple=-0.4,
        mae_pct=-2.0,
        mfe_pct=1.0,
    )

    analogs = EdgeAnalogIndex([future_leak, past_allowed]).find_analogs(
        query,
        k=2,
        embargo_days=5,
        allow_future=False,
    )

    assert [row["ticker"] for row in analogs] == ["CCC"]
