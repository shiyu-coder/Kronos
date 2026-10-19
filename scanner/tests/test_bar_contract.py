import pandas as pd

from scanner.data.bar_contract import check_ohlcv_contract, check_session_completeness


def _frame(rows, start="2026-06-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def test_clean_bars_pass():
    df = _frame([[10, 11, 9, 10.5, 100], [10.5, 12, 10, 11.5, 200]])
    violations, warnings = check_ohlcv_contract(df)
    assert violations == []
    assert warnings == []


def test_empty_frame_fails():
    violations, _ = check_ohlcv_contract(pd.DataFrame())
    assert violations == ["empty bar frame"]


def test_high_below_body_fails():
    df = _frame([[10, 9.5, 9, 10.5, 100]])
    violations, _ = check_ohlcv_contract(df)
    assert any("High below" in v for v in violations)


def test_low_above_body_fails():
    df = _frame([[10, 11, 10.2, 10.5, 100]])
    violations, _ = check_ohlcv_contract(df)
    assert any("Low above" in v for v in violations)


def test_nonpositive_price_fails():
    df = _frame([[0.0, 11, 9, 10.5, 100]])
    violations, _ = check_ohlcv_contract(df)
    assert any("non-positive" in v for v in violations)


def test_negative_volume_fails():
    df = _frame([[10, 11, 9, 10.5, -5]])
    violations, _ = check_ohlcv_contract(df)
    assert any("negative volume" in v for v in violations)


def test_nan_ohlc_fails():
    df = _frame([[10, 11, 9, float("nan"), 100]])
    violations, _ = check_ohlcv_contract(df)
    assert any("NaN OHLC" in v for v in violations)


def test_duplicate_timestamps_fail():
    idx = pd.DatetimeIndex(["2026-06-01", "2026-06-01"]).tz_localize("America/New_York")
    df = pd.DataFrame(
        [[10, 11, 9, 10.5, 100], [10.5, 12, 10, 11.5, 200]],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    violations, _ = check_ohlcv_contract(df)
    assert any("duplicate timestamps" in v for v in violations)


def test_unsorted_timestamps_fail():
    idx = pd.DatetimeIndex(["2026-06-02", "2026-06-01"]).tz_localize("America/New_York")
    df = pd.DataFrame(
        [[10, 11, 9, 10.5, 100], [10.5, 12, 10, 11.5, 200]],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    violations, _ = check_ohlcv_contract(df)
    assert any("not sorted" in v for v in violations)


def test_suspect_split_sized_move_warns_but_passes():
    # A 50% one-day drop is the fingerprint of a missed corporate action on
    # raw bars; it must be surfaced without deleting the ticker outright.
    df = _frame([[100, 101, 99, 100, 100], [50, 51, 49, 50, 100]])
    violations, warnings = check_ohlcv_contract(df)
    assert violations == []
    assert any("suspect one-bar move" in w for w in warnings)
    # An exact 2:1 gap also snaps to the split-ratio detector.
    assert any("matches split ratio ~2" in w for w in warnings)


def test_three_for_two_split_gap_warns():
    # -33% gap (3:2 split) slips under the 45% suspect-move rule; the ratio
    # snap test is what catches it.
    df = _frame([[15, 15.2, 14.8, 15, 100], [10, 10.1, 9.9, 10, 100]])
    violations, warnings = check_ohlcv_contract(df)
    assert violations == []
    assert any("matches split ratio ~1.5" in w for w in warnings)


def test_normal_gap_does_not_snap():
    # A 10% overnight gap is a real move, not a split candidate.
    df = _frame([[100, 101, 99, 100, 100], [90, 92, 89, 91, 100]])
    _, warnings = check_ohlcv_contract(df)
    assert not any("split ratio" in w for w in warnings)


def test_zero_volume_with_range_fails():
    df = _frame([[10, 11, 9, 10.5, 0]])
    violations, _ = check_ohlcv_contract(df)
    assert any("price range but zero volume" in v for v in violations)


def test_forward_filled_flat_zero_volume_bar_fails():
    df = _frame([[10, 10, 10, 10, 0]])
    violations, _ = check_ohlcv_contract(df)
    assert any("forward-filled" in v for v in violations)


def test_single_print_bar_warns():
    df = _frame([[10, 10, 10, 10, 500], [10.5, 11, 10, 10.8, 300]])
    violations, warnings = check_ohlcv_contract(df)
    assert violations == []
    assert any("single-print" in w for w in warnings)


def test_stale_close_run_warns_then_fails():
    warn_df = _frame(
        [[10, 11, 9, 10.5, 100], [10.4, 11, 10, 10.5, 90], [10.6, 11, 10, 10.5, 80], [10.2, 11, 10, 10.9, 70]]
    )
    _, warnings = check_ohlcv_contract(warn_df)
    assert any("consecutive identical closes" in w for w in warnings)

    fail_rows = [[10 + i * 0.01, 11, 9, 10.5, 100 - i] for i in range(5)]
    violations, _ = check_ohlcv_contract(_frame(fail_rows))
    assert any("consecutive identical closes" in v for v in violations)


def test_stale_identical_bar_run_fails():
    rows = [[10, 11, 9, 10.5, 100]] * 4
    violations, _ = check_ohlcv_contract(_frame(rows))
    assert any("identical OHLCV bars" in v for v in violations)


def test_session_completeness_flags_missing_sessions():
    # 2026-06-01 through 2026-06-05 are five NYSE sessions; drop Wednesday.
    dates = ["2026-06-01", "2026-06-02", "2026-06-04", "2026-06-05"]
    idx = pd.DatetimeIndex(dates).tz_localize("America/New_York")
    df = pd.DataFrame(
        [[10, 11, 9, 10.5, 100]] * len(dates),
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    violations, warnings, stats = check_session_completeness(df)
    assert violations == []
    assert any("missing" in w for w in warnings)
    assert stats["missing_sessions"] == ["2026-06-03"]


def test_session_completeness_full_week_passes():
    dates = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
    idx = pd.DatetimeIndex(dates).tz_localize("America/New_York")
    df = pd.DataFrame(
        [[10, 11, 9, 10.5, 100]] * len(dates),
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    violations, warnings, stats = check_session_completeness(df)
    assert violations == []
    assert warnings == []
    assert stats["expected_sessions"] == 5
    assert stats["present_sessions"] == 5


def test_session_completeness_many_missing_fails():
    # Only 2 bars across a full month of sessions - systemic gap.
    dates = ["2026-06-01", "2026-06-30"]
    idx = pd.DatetimeIndex(dates).tz_localize("America/New_York")
    df = pd.DataFrame(
        [[10, 11, 9, 10.5, 100]] * len(dates),
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    violations, _, stats = check_session_completeness(df)
    assert any("expected sessions missing" in v for v in violations)
    assert len(stats["missing_sessions"]) > 5
