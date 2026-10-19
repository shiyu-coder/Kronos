"""Fail-closed data contract for OHLCV bars feeding the evidence engine.

Every triple-barrier outcome and every feature in the edge index is only as
honest as the bars underneath it. This module enforces the hard per-bar
invariants before any bar set is allowed to become evidence; violations fail
the ticker rather than silently producing fake trade outcomes.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# One-day moves beyond this are almost always a missed corporate action on
# unadjusted data rather than a real repricing; they are surfaced as warnings
# (not violations) so a genuine crash cannot silently delete a ticker.
SUSPECT_MOVE_PCT = 45.0

# Overnight gaps that land within tolerance of a common split ratio are the
# signature of an unadjusted corporate action. Ratios below 1.5 (a 3:2 split
# is a -33% gap) are excluded: 20-25% overnight moves happen legitimately on
# this watchlist and would drown the signal in noise. The 45% rule above
# misses 3:2 exactly; this closes that hole.
SPLIT_SNAP_RATIOS = (1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 50.0)
SPLIT_SNAP_TOLERANCE = 0.03

# Repeated identical closes are invisible to any statistical outlier check (a
# repeated value is in-distribution by construction); only a deterministic
# run-length rule catches a stale/forward-filled feed.
STALE_CLOSE_WARN_RUN = 3
STALE_CLOSE_FAIL_RUN = 5
STALE_BAR_WARN_RUN = 2
STALE_BAR_FAIL_RUN = 4


def check_ohlcv_contract(df: pd.DataFrame, profile: str = "daily") -> tuple[list[str], list[str]]:
    """Return (violations, warnings) for a bar frame.

    Violations are hard invariant breaches - the bars must not become
    evidence. Warnings flag suspicious-but-possible data for the report.

    `profile="intraday"` downgrades the stale-run rules to warnings: identical
    consecutive closes across quiet 30-minute bars are plausible on thin
    names, while on daily bars they mean a frozen feed.
    """
    violations: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return ["empty bar frame"], warnings

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return [f"missing columns: {', '.join(missing)}"], warnings

    ohlc = df[["Open", "High", "Low", "Close"]]
    nan_rows = int(ohlc.isna().any(axis=1).sum())
    if nan_rows:
        violations.append(f"{nan_rows} bars with NaN OHLC values")

    finite = df[~ohlc.isna().any(axis=1)]
    if not finite.empty:
        nonpositive = int((finite[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
        if nonpositive:
            violations.append(f"{nonpositive} bars with non-positive prices")

        body_high = finite[["Open", "Close"]].max(axis=1)
        body_low = finite[["Open", "Close"]].min(axis=1)
        bad_high = int((finite["High"] < body_high).sum())
        bad_low = int((finite["Low"] > body_low).sum())
        if bad_high:
            violations.append(f"{bad_high} bars with High below max(Open, Close)")
        if bad_low:
            violations.append(f"{bad_low} bars with Low above min(Open, Close)")

    volume = pd.to_numeric(df["Volume"], errors="coerce")
    negative_volume = int((volume < 0).sum())
    if negative_volume:
        violations.append(f"{negative_volume} bars with negative volume")

    index = pd.DatetimeIndex(df.index)
    if index.has_duplicates:
        violations.append(f"{int(index.duplicated().sum())} duplicate timestamps")
    if not index.is_monotonic_increasing:
        violations.append("timestamps not sorted ascending")

    if not finite.empty:
        zero_vol = pd.to_numeric(finite["Volume"], errors="coerce").fillna(0.0) == 0
        has_range = finite["High"] > finite["Low"]
        flat = (
            (finite["Open"] == finite["High"])
            & (finite["High"] == finite["Low"])
            & (finite["Low"] == finite["Close"])
        )

        # Price range with no trades is internally impossible; a flat bar
        # with no trades is a vendor forward-fill of a non-trading session.
        # Both fabricate the High/Low paths that triple-barrier outcomes
        # walk, so both are hard failures.
        range_no_trades = int((zero_vol & has_range).sum())
        if range_no_trades:
            violations.append(f"{range_no_trades} bars with price range but zero volume")
        filled = int((zero_vol & flat).sum())
        if filled:
            violations.append(f"{filled} zero-volume flat bars (forward-filled non-trading sessions)")

        one_print = int((~zero_vol & flat).sum())
        if one_print:
            warnings.append(f"{one_print} single-print bars (Open==High==Low==Close with volume)")

    if not finite.empty and len(finite) > 1:
        closes = finite["Close"].astype(float)
        day_moves = closes.pct_change().abs() * 100.0
        suspects = day_moves[day_moves > SUSPECT_MOVE_PCT]
        for ts, move in suspects.items():
            warnings.append(f"suspect one-bar move {move:.0f}% at {ts} (possible unadjusted corporate action)")

        for ts, ratio in _split_snap_gaps(finite):
            warnings.append(
                f"overnight gap at {ts} matches split ratio ~{ratio:g} (possible unadjusted split)"
            )

        stale_sink = warnings if profile == "intraday" else None
        stale_close = _longest_run(closes.values)
        if stale_close >= STALE_CLOSE_FAIL_RUN:
            (stale_sink if stale_sink is not None else violations).append(
                f"stale feed: {stale_close} consecutive identical closes"
            )
        elif stale_close >= STALE_CLOSE_WARN_RUN:
            warnings.append(f"{stale_close} consecutive identical closes (possible stale feed)")

        bar_tuples = list(
            zip(
                finite["Open"].values,
                finite["High"].values,
                finite["Low"].values,
                finite["Close"].values,
                pd.to_numeric(finite["Volume"], errors="coerce").values,
            )
        )
        stale_bar = _longest_run(bar_tuples)
        if stale_bar >= STALE_BAR_FAIL_RUN:
            (stale_sink if stale_sink is not None else violations).append(
                f"stale feed: {stale_bar} consecutive identical OHLCV bars"
            )
        elif stale_bar >= STALE_BAR_WARN_RUN:
            warnings.append(f"{stale_bar} consecutive identical OHLCV bars (possible forward-fill)")

    return violations, warnings


def _longest_run(values) -> int:
    longest = 0
    run = 0
    previous = object()
    for value in values:
        run = run + 1 if value == previous else 1
        previous = value
        longest = max(longest, run)
    return longest


def _split_snap_gaps(finite: pd.DataFrame) -> list[tuple[object, float]]:
    """Overnight prev-close/open ratios that snap to a common split ratio."""
    hits: list[tuple[object, float]] = []
    prev_close = finite["Close"].shift(1)
    opens = finite["Open"].where(finite["Open"] > 0, finite["Close"])
    ratio = (prev_close / opens).dropna()
    for ts, r in ratio.items():
        if r <= 0:
            continue
        for split in SPLIT_SNAP_RATIOS:
            for candidate in (split, 1.0 / split):
                if abs(r - candidate) / candidate <= SPLIT_SNAP_TOLERANCE:
                    hits.append((ts, candidate))
                    break
            else:
                continue
            break
    return hits


def check_session_completeness(
    df: pd.DataFrame,
    *,
    calendar_name: str = "XNYS",
    max_missing_warn: int = 5,
) -> tuple[list[str], list[str], dict]:
    """Compare daily bars against the exchange calendar's expected sessions.

    Vendors emit no bar when no eligible trade printed, so a halted, delisted,
    or partially-delivered ticker silently loses sessions - and a 5-bar
    outcome horizon walked across a hidden gap spans weeks of real time. A
    few missing sessions warn; more than `max_missing_warn` fails the ticker.
    """
    stats: dict = {"expected_sessions": 0, "present_sessions": 0, "missing_sessions": []}
    if df is None or df.empty:
        return [], [], stats
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        return [], ["exchange calendar unavailable (pandas_market_calendars not installed)"], stats

    idx = pd.DatetimeIndex(df.index)
    bar_dates = {ts.date() for ts in idx}
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=min(bar_dates), end_date=max(bar_dates))
    expected = {ts.date() for ts in schedule.index}

    missing = sorted(expected - bar_dates)
    stats["expected_sessions"] = len(expected)
    stats["present_sessions"] = len(expected & bar_dates)
    stats["missing_sessions"] = [d.isoformat() for d in missing]

    violations: list[str] = []
    warnings: list[str] = []
    extra = sorted(bar_dates - expected)
    if extra:
        warnings.append(
            f"{len(extra)} bars on non-session dates (first: {extra[0].isoformat()})"
        )
    if len(missing) > max_missing_warn:
        violations.append(
            f"{len(missing)} expected sessions missing between "
            f"{min(bar_dates).isoformat()} and {max(bar_dates).isoformat()}"
        )
    elif missing:
        warnings.append(
            f"{len(missing)} expected sessions missing: {', '.join(stats['missing_sessions'])}"
        )
    return violations, warnings, stats
