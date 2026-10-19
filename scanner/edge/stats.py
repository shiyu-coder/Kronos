"""Shared small-sample statistics for edge evidence and adaptive policy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_lower_bound(wins: float, total: int, z: float = 1.28) -> float:
    """Conservative lower bound on a binomial proportion."""
    if total <= 0:
        return 0.0
    p_hat = wins / total
    z2 = z * z
    denominator = 1.0 + (z2 / total)
    center = p_hat + (z2 / (2.0 * total))
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z2 / (4.0 * total))) / total)
    return max(0.0, (center - margin) / denominator)


def t_statistic(values: list[float]) -> float:
    """One-sample t statistic against zero mean."""
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    n = len(finite)
    if n < 2:
        return 0.0
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return mean / (sd / math.sqrt(n))


def _entry_day_means(values: list[float], day_keys: list[str]) -> tuple[list[float], int]:
    buckets: dict[str, list[float]] = {}
    valid_count = 0
    for value, day in zip(values, day_keys, strict=False):
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and day:
            buckets.setdefault(str(day), []).append(float(value))
            valid_count += 1

    # Production keys are ISO dates. Sort them chronologically so HAC lags
    # describe adjacent entry days even though validation rows arrive score-
    # sorted. Synthetic/non-date keys retain insertion order.
    ordered_days = list(buckets)
    try:
        parsed = {day: pd.Timestamp(day) for day in ordered_days}
        if any(pd.isna(timestamp) for timestamp in parsed.values()):
            raise ValueError("unparseable entry day")
        ordered_days.sort(key=lambda day: parsed[day])
    except Exception:
        pass
    return [float(np.mean(buckets[day])) for day in ordered_days], valid_count


def _hac_day_mean_summary(values: list[float], day_keys: list[str], hac_lags: int = 5) -> dict:
    day_means, valid_count = _entry_day_means(values, day_keys)
    n_days = len(day_means)
    mean = float(np.mean(day_means)) if day_means else 0.0
    standard_error = 0.0
    t_stat = 0.0

    if n_days >= 2:
        centered = np.asarray(day_means, dtype=float) - mean
        gamma_zero = float(centered @ centered) / n_days
        long_run_variance = gamma_zero
        max_lag = min(max(int(hac_lags), 0), n_days - 1)
        for lag in range(1, max_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            gamma_lag = float(centered[lag:] @ centered[:-lag]) / n_days
            long_run_variance += 2.0 * weight * gamma_lag
        # Bartlett/Newey-West is non-negative in population. Numerical noise can
        # put a nearly-zero finite-sample estimate just below zero.
        long_run_variance = max(long_run_variance, 0.0)
        long_run_variance *= n_days / (n_days - 1.0)
        standard_error = math.sqrt(long_run_variance / n_days)
        if standard_error > 1e-12:
            t_stat = mean / standard_error

    return {
        "signal_count": valid_count,
        "n_days": n_days,
        "mean_of_day_means": mean,
        "standard_error": standard_error,
        "t_stat": round(t_stat, 4),
        "method": f"entry_day_mean_hac_lag{max(int(hac_lags), 0)}",
    }


def _one_sided_p_from_t(t: float) -> float:
    # Normal approximation; adequate at the sample sizes gating decisions here.
    return 0.5 * (1.0 - math.erf(t / math.sqrt(2.0)))


def _two_sided_p_from_t(t: float) -> float:
    return min(1.0, 2.0 * _one_sided_p_from_t(abs(t)))


def _cluster_robust_slope_t(
    x: np.ndarray, y: np.ndarray, clusters: list[str], hac_lags: int = 5
) -> float | None:
    """OLS slope t-statistic with entry-day clustering and HAC lags.

    Spearman correlation is the OLS slope between standardized ranks.  This
    sandwich estimator keeps the row-level rank relationship while allowing
    arbitrary dependence among records sharing an entry day. Newey-West terms
    over ordered day-score vectors also cover serial dependence from overlapping
    multi-day outcome windows. It replaces the old shortcut that merely put
    ``n_days`` into an IID correlation formula.
    """
    n = len(x)
    unique_clusters = sorted(set(clusters))
    g = len(unique_clusters)
    k = 2  # intercept + rank slope
    if n <= k or g < 3:
        return None

    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residuals = y - design @ beta

    day_scores = []
    cluster_array = np.asarray(clusters, dtype=object)
    for cluster in unique_clusters:
        mask = cluster_array == cluster
        day_scores.append(design[mask].T @ residuals[mask])

    score_matrix = np.asarray(day_scores, dtype=float)
    meat = score_matrix.T @ score_matrix
    max_lag = min(max(int(hac_lags), 0), g - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = score_matrix[lag:].T @ score_matrix[:-lag]
        meat += weight * (cross + cross.T)

    # CR1 finite-sample correction, matching common clustered-OLS defaults.
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * (xtx_inv @ meat @ xtx_inv)
    variance = float(covariance[1, 1])
    if not math.isfinite(variance) or variance <= 1e-18:
        return None
    return float(beta[1] / math.sqrt(variance))


def day_clustered_t(values: list[float], day_keys: list[str]) -> dict:
    """One-sample t against zero on entry-day means with serial HAC.

    Trades entered the same day share the market factor and (with 5-bar
    horizons) most of their outcome window; the per-trade t treats them as
    independent and overstates confidence. Entry-day aggregation handles the
    market cluster; five Newey-West lags handle overlapping outcome windows on
    adjacent entry days.
    """
    return _hac_day_mean_summary(values, day_keys, hac_lags=5)


def day_clustered_precision(
    win_values: list[float],
    day_keys: list[str],
    *,
    z: float = 1.645,
    hac_lags: int = 5,
) -> dict:
    """Dependence-aware lower bound for a selected cohort's win rate.

    Daily win-rate means prevent a crowded entry day from masquerading as many
    independent Bernoulli trials. The lower bound is the more conservative of
    a HAC normal bound (serial overlap) and a Wilson bound whose effective n is
    the number of entry days (small-sample/binomial uncertainty).
    """
    summary = _hac_day_mean_summary(win_values, day_keys, hac_lags=hac_lags)
    n_days = int(summary["n_days"])
    mean = float(summary["mean_of_day_means"])
    standard_error = float(summary["standard_error"])
    if n_days < 2:
        hac_lower = 0.0
        wilson_day_lower = 0.0
        lower_bound = 0.0
    else:
        hac_lower = max(0.0, mean - abs(float(z)) * standard_error)
        wilson_day_lower = wilson_lower_bound(mean * n_days, n_days, z=abs(float(z)))
        lower_bound = min(hac_lower, wilson_day_lower)
    return {
        "signal_count": int(summary["signal_count"]),
        "n_days": n_days,
        "mean_daily_precision": round(mean, 4),
        "lower_bound": round(lower_bound, 4),
        "hac_normal_lower_bound": round(hac_lower, 4),
        "wilson_day_lower_bound": round(wilson_day_lower, 4),
        "standard_error": round(standard_error, 6),
        "z": abs(float(z)),
        "method": f"entry_day_mean_hac_lag{max(int(hac_lags), 0)}_min_wilson",
    }


def _tie_break_key(row_id: str) -> int:
    # Deterministic pseudo-random tie order: with mass score ties (gate caps
    # collapse most scores onto a few values), stable sort would hand bucket
    # membership to input order - i.e. recency - biasing tercile/decile stats.
    import hashlib

    return int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:12], 16)


def tercile_lift(
    scores: list[float],
    outcomes: list[float],
    day_keys: list[str],
    row_ids: list[str] | None = None,
    n_boot: int = 400,
    seed: int = 7,
    block_days: int = 5,
) -> dict:
    """Mean R per score tercile with a day-block bootstrap CI on the spread.

    Terciles (not deciles) because at n~900 each decile's SE (~0.12-0.15R)
    makes decile bar charts noise generators; ~300-trade buckets resolve the
    spreads this dataset can actually support.
    """
    rows = []
    for idx, (score, outcome) in enumerate(zip(scores, outcomes, strict=False)):
        if (
            isinstance(score, (int, float))
            and isinstance(outcome, (int, float))
            and math.isfinite(float(score))
            and math.isfinite(float(outcome))
        ):
            day = str(day_keys[idx]) if idx < len(day_keys) and day_keys[idx] else ""
            row_id = str(row_ids[idx]) if row_ids and idx < len(row_ids) else str(idx)
            rows.append((float(score), float(outcome), day, row_id))

    n = len(rows)
    if n < 30:
        return {"n": n, "insufficient": True}

    def bucket_means(sample: list[tuple[float, float, str, str]]) -> tuple[float, float, float]:
        ordered = sorted(sample, key=lambda r: (-r[0], _tie_break_key(r[3])))
        size = max(len(ordered) // 3, 1)  # guard: ordered[-0:] is the whole list
        top = [r[1] for r in ordered[:size]]
        bottom = [r[1] for r in ordered[-size:]]
        middle = [r[1] for r in ordered[size : len(ordered) - size]]
        return (
            float(np.mean(top)) if top else 0.0,
            float(np.mean(middle)) if middle else 0.0,
            float(np.mean(bottom)) if bottom else 0.0,
        )

    top_mean, mid_mean, bottom_mean = bucket_means(rows)
    spread = top_mean - bottom_mean

    by_day: dict[str, list[tuple[float, float, str, str]]] = {}
    for row in rows:
        by_day.setdefault(row[2], []).append(row)
    days = sorted(by_day)

    ci_low = ci_high = None
    if len(days) >= 6 and n_boot > 0:
        rng = np.random.default_rng(seed)
        spreads = []
        block_days = min(max(int(block_days), 1), len(days))
        for _ in range(n_boot):
            drawn = []
            while len(drawn) < len(days):
                start = int(rng.integers(0, len(days)))
                drawn.extend((start + offset) % len(days) for offset in range(block_days))
            drawn = drawn[: len(days)]
            sample: list[tuple[float, float, str, str]] = []
            for day_idx in drawn:
                sample.extend(by_day[days[int(day_idx)]])
            if len(sample) < 9:
                continue
            boot_top, _, boot_bottom = bucket_means(sample)
            spreads.append(boot_top - boot_bottom)
        if spreads:
            ci_low = float(np.percentile(spreads, 2.5))
            ci_high = float(np.percentile(spreads, 97.5))

    return {
        "n": n,
        "insufficient": False,
        "top_mean_r": round(top_mean, 4),
        "middle_mean_r": round(mid_mean, 4),
        "bottom_mean_r": round(bottom_mean, 4),
        "spread_r": round(spread, 4),
        "spread_ci_low": round(ci_low, 4) if ci_low is not None else None,
        "spread_ci_high": round(ci_high, 4) if ci_high is not None else None,
        "distinct_days": len(days),
        "bootstrap_method": "circular_moving_entry_day_blocks",
        "bootstrap_block_days": min(max(int(block_days), 1), len(days)) if days else None,
    }


def tail_retention(scores: list[float], outcomes: list[float], row_ids: list[str] | None = None, tail_r: float = 2.0) -> dict:
    """Share of right-tail (>= tail_r) trades captured by the top score tercile.

    Long-breakout edges live in <10% of trades (the exit-geometry sweep
    proved this one is right-tail driven). Any ranking/filtering layer that
    lifts mean R while under-capturing the tail is silently repeating what
    profit targets did - this is the veto diagnostic.
    """
    rows = []
    for idx, (score, outcome) in enumerate(zip(scores, outcomes, strict=False)):
        if (
            isinstance(score, (int, float))
            and isinstance(outcome, (int, float))
            and math.isfinite(float(score))
            and math.isfinite(float(outcome))
        ):
            row_id = str(row_ids[idx]) if row_ids and idx < len(row_ids) else str(idx)
            rows.append((float(score), float(outcome), row_id))

    n = len(rows)
    if n < 30:
        return {"n": n, "insufficient": True}

    ordered = sorted(rows, key=lambda r: (-r[0], _tie_break_key(r[2])))
    size = max(n // 3, 1)
    top_ids = {r[2] for r in ordered[:size]}
    tail = [r for r in rows if r[1] >= tail_r]
    tail_in_top = sum(1 for r in tail if r[2] in top_ids)
    expected_share = size / n if n else 0.0
    return {
        "n": n,
        "insufficient": False,
        "tail_r_threshold": tail_r,
        "tail_count": len(tail),
        "tail_in_top_tercile": tail_in_top,
        "expected_share": round(expected_share, 4),
        "observed_share": round(tail_in_top / len(tail), 4) if tail else None,
    }


def spearman_rank_ic(scores: list[float], outcomes: list[float], day_keys: list[str] | None = None) -> dict:
    """Spearman rank correlation of score vs outcome with explicit alternatives.

    Uses every sample, so it detects ranking skill long before any absolute
    threshold accumulates enough signals (rho >= ~0.07 is detectable at
    n=600; a threshold gate needs 20+ signals it may never produce).

    The acceptance hypothesis is directional (positive rank skill), so the
    legacy ``p_value`` fields remain one-sided with alternative ``IC > 0``.
    Two-sided values are also returned so negative associations are not
    mislabeled using a one-sided significance number.

    When day_keys are provided, the day-clustered values use a sandwich
    covariance on the rank regression with arbitrary same-entry-day dependence
    and five Newey-West entry-day lags for overlapping outcome windows.
    """
    pairs = []
    pair_days = []
    for idx, (s, o) in enumerate(zip(scores, outcomes, strict=False)):
        if (
            isinstance(s, (int, float))
            and isinstance(o, (int, float))
            and math.isfinite(float(s))
            and math.isfinite(float(o))
        ):
            pairs.append((float(s), float(o)))
            if day_keys is not None and idx < len(day_keys):
                pair_days.append(str(day_keys[idx]))
    n = len(pairs)
    if n < 3:
        return {"ic": 0.0, "p_value": 1.0, "n": n}

    score_ranks = pd.Series([p[0] for p in pairs]).rank(method="average")
    outcome_ranks = pd.Series([p[1] for p in pairs]).rank(method="average")
    if score_ranks.std(ddof=0) <= 1e-12 or outcome_ranks.std(ddof=0) <= 1e-12:
        return {"ic": 0.0, "p_value": 1.0, "n": n}

    ic = float(np.corrcoef(score_ranks, outcome_ranks)[0, 1])
    if not math.isfinite(ic):
        return {"ic": 0.0, "p_value": 1.0, "n": n}
    bounded = min(max(ic, -0.999999), 0.999999)
    t = bounded * math.sqrt((n - 2) / (1.0 - bounded * bounded))
    result = {
        "ic": round(ic, 4),
        "p_value": round(_one_sided_p_from_t(t), 6),
        "p_value_two_sided": round(_two_sided_p_from_t(t), 6),
        "p_value_alternative": "greater",
        "n": n,
    }

    if day_keys is not None:
        n_days = len({d for d in pair_days if d})
        result["n_days"] = n_days
        if n_days >= 3 and len(pair_days) == n:
            x = score_ranks.to_numpy(dtype=float)
            y = outcome_ranks.to_numpy(dtype=float)
            t_clustered = _cluster_robust_slope_t(x, y, pair_days)
            if t_clustered is not None:
                result["p_value_day_clustered"] = round(_one_sided_p_from_t(t_clustered), 6)
                result["p_value_day_clustered_two_sided"] = round(_two_sided_p_from_t(t_clustered), 6)
                result["day_cluster_method"] = "entry_day_cluster_hac_lag5_cr1"
            else:
                result["p_value_day_clustered"] = 1.0
                result["p_value_day_clustered_two_sided"] = 1.0
                result["day_cluster_method"] = "unavailable"
        else:
            result["p_value_day_clustered"] = 1.0
            result["p_value_day_clustered_two_sided"] = 1.0
            result["day_cluster_method"] = "unavailable"
    return result
