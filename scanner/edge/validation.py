from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np

from .. import config as scanner_config
from .stats import (
    day_clustered_precision,
    day_clustered_t,
    spearman_rank_ic,
    t_statistic,
    tail_retention,
    tercile_lift,
    wilson_lower_bound,
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _is_win(candidate: dict) -> bool:
    if candidate.get("outcome_label") in {"win", "loss"}:
        return candidate.get("outcome_label") == "win"
    return _finite_float(candidate.get("outcome_return_pct")) > 0


def apply_transaction_costs(rows: list[dict], cost_bps_per_side: float) -> list[dict]:
    """Charge a round-trip cost against stored GROSS outcomes.

    The journal stores what the price path did, not what an account would have
    kept: `walk_triple_barrier` charges nothing. Every downstream gate metric
    therefore inherited a free-execution assumption. This converts each row to
    the net view ONCE, up front, so rank IC, average R, precision and the HAC
    t-stats are all computed on money that could actually have been kept.

    Cost is charged in return-percentage points (entry + exit) and re-divided by
    the SAME stored `risk_pct_used`, so net R stays on the original R scale. The
    win/loss label is recomputed from the net return - a horizon exit at +0.02%
    is a gross "win" and a net loss, and precision must see it as the latter.

    Gross values are preserved under `gross_*` keys for diagnosis. Rows carrying
    no usable `risk_pct_used` keep their stored R rather than inventing one; the
    return-based metrics are still charged.
    """
    charge_pct = 2.0 * abs(_finite_float(cost_bps_per_side)) / 100.0
    out: list[dict] = []
    for row in rows:
        net = dict(row)
        gross_ret = _finite_float(row.get("outcome_return_pct"))
        gross_r = _finite_float(row.get("r_multiple"))
        net["gross_outcome_return_pct"] = gross_ret
        net["gross_r_multiple"] = gross_r
        net["gross_outcome_label"] = row.get("outcome_label")

        net_ret = gross_ret - charge_pct
        net["outcome_return_pct"] = net_ret

        risk = _finite_float(row.get("risk_pct_used"))
        if risk > 0.0:
            net["r_multiple"] = min(max(net_ret / risk, -10.0), 10.0)
        # else: no stop distance on record - leave the stored R untouched
        # rather than fabricate a denominator.

        net["outcome_label"] = "win" if net_ret > 0 else "loss"
        out.append(net)
    return out


def _metric_block(candidates: list[dict], selected: list[dict], total_wins: int) -> dict:
    """All inputs are already NET of transaction costs (see apply_transaction_costs)."""
    signal_count = len(selected)
    wins = sum(1 for row in selected if _is_win(row))
    losses = signal_count - wins
    false_negatives = max(total_wins - wins, 0)
    returns = [_finite_float(row.get("outcome_return_pct")) for row in selected]
    gross_returns = [_finite_float(row.get("gross_outcome_return_pct")) for row in selected]
    r_mult = [_finite_float(row.get("r_multiple")) for row in selected]
    entry_days = [str(row.get("timestamp", ""))[:10] for row in selected]
    win_values = [1.0 if _is_win(row) else 0.0 for row in selected]
    mae = [_finite_float(row.get("mae_pct")) for row in selected if row.get("mae_pct") is not None]
    mfe = [_finite_float(row.get("mfe_pct")) for row in selected if row.get("mfe_pct") is not None]

    exit_reasons = Counter(str(row.get("exit_reason")) for row in selected if row.get("exit_reason"))

    return {
        "signal_count": signal_count,
        "wins": wins,
        "losses": losses,
        "precision": wins / signal_count if signal_count else 0.0,
        "wilson_lb_precision": round(wilson_lower_bound(wins, signal_count, z=1.645), 4) if signal_count else 0.0,
        "recall": wins / total_wins if total_wins else 0.0,
        "false_negative_rate": false_negatives / total_wins if total_wins else 0.0,
        "average_return_pct": float(np.mean(returns)) if returns else 0.0,
        "median_return_pct": float(np.median(returns)) if returns else 0.0,
        # Retained name: every return in this block is already net of costs, so
        # this is an alias kept for existing report consumers.
        "average_return_pct_after_slippage": float(np.mean(returns)) if returns else 0.0,
        "average_gross_return_pct": float(np.mean(gross_returns)) if gross_returns else 0.0,
        "average_r_multiple": float(np.mean(r_mult)) if r_mult else 0.0,
        "t_stat_r_multiple": round(t_statistic(r_mult), 4),
        "t_stat_r_day_clustered": day_clustered_t(r_mult, entry_days),
        "precision_day_clustered": day_clustered_precision(win_values, entry_days),
        "stop_rate": round(exit_reasons.get("stop", 0) / signal_count, 4) if signal_count else 0.0,
        "target_rate": round(exit_reasons.get("target", 0) / signal_count, 4) if signal_count else 0.0,
        "max_adverse_excursion": float(np.min(mae)) if mae else 0.0,
        "max_favorable_excursion": float(np.max(mfe)) if mfe else 0.0,
    }


def compute_edge_validation_report(
    candidates: Iterable[dict],
    thresholds: tuple[int, ...] = (45, 55, 65),
    top_k: int = 5,
    cost_bps_per_side: float = 0.0,
) -> dict:
    rows = [dict(row) for row in candidates]
    candidate_rows = len(rows)
    risk_coverage_rows = sum(1 for row in rows if _finite_float(row.get("risk_pct_used")) > 0.0)
    risk_coverage_complete = candidate_rows > 0 and risk_coverage_rows == candidate_rows
    # Charge costs FIRST. Everything below - precision, average R, rank IC,
    # the day-clustered t-stats - then reads the net outcome by construction,
    # so no gate can be passed on gross expectancy.
    rows = apply_transaction_costs(rows, cost_bps_per_side)
    rows.sort(key=lambda row: _finite_float(row.get("edge_score")), reverse=True)
    total_wins = sum(1 for row in rows if _is_win(row))

    threshold_blocks = {}
    for threshold in thresholds:
        selected = [row for row in rows if _finite_float(row.get("edge_score")) >= float(threshold)]
        threshold_blocks[str(threshold)] = _metric_block(rows, selected, total_wins)

    top_rows = rows[: max(top_k, 0)]
    top_block = _metric_block(rows, top_rows, total_wins)
    top_block["k"] = top_k

    by_direction: dict[str, dict] = {}
    for direction in sorted({str(row.get("direction", "unknown")) for row in rows}):
        subset = [row for row in rows if str(row.get("direction", "unknown")) == direction]
        block = _metric_block(subset, subset, sum(1 for row in subset if _is_win(row)))
        # Within-direction ranking is THE frontier metric: pooled IC can pass
        # on direction separation alone (bullish positive vs bearish negative
        # cohorts) while the score ranks nothing inside either direction.
        sub_scores = [_finite_float(row.get("edge_score")) for row in subset]
        sub_r = [_finite_float(row.get("r_multiple")) for row in subset]
        sub_days = [str(row.get("timestamp", ""))[:10] for row in subset]
        sub_ids = [f"{row.get('ticker', '')}|{row.get('timestamp', '')}" for row in subset]
        block["rank_ic_r"] = spearman_rank_ic(sub_scores, sub_r, day_keys=sub_days)
        block["rank_ic_return"] = spearman_rank_ic(
            sub_scores, [_finite_float(row.get("outcome_return_pct")) for row in subset], day_keys=sub_days
        )
        block["t_stat_r_day_clustered"] = day_clustered_t(sub_r, sub_days)
        block["tercile_lift"] = tercile_lift(sub_scores, sub_r, sub_days, row_ids=sub_ids)
        block["tail_retention"] = tail_retention(sub_scores, sub_r, row_ids=sub_ids)
        by_direction[direction] = block

    # Ranking skill over ALL samples: detectable long before any absolute
    # threshold produces 20+ signals (rows are already score-sorted).
    scores = [_finite_float(row.get("edge_score")) for row in rows]
    percentile_blocks: dict[str, dict] = {}
    for label, share in (("top_5_pct", 0.05), ("top_10_pct", 0.10), ("top_20_pct", 0.20)):
        n_top = max(int(len(rows) * share), 1) if rows else 0
        percentile_blocks[label] = _metric_block(rows, rows[:n_top], total_wins)

    decile_size = max(len(rows) // 10, 1) if rows else 0
    top_decile = rows[:decile_size]
    bottom_decile = rows[-decile_size:] if rows else []
    top_avg_r = float(np.mean([_finite_float(r.get("r_multiple")) for r in top_decile])) if top_decile else 0.0
    bottom_avg_r = float(np.mean([_finite_float(r.get("r_multiple")) for r in bottom_decile])) if bottom_decile else 0.0

    day_counts = Counter(str(row.get("timestamp", ""))[:10] for row in rows if row.get("timestamp"))
    max_day_share = (max(day_counts.values()) / len(rows)) if rows and day_counts else 0.0

    return {
        "samples": len(rows),
        "wins": total_wins,
        "losses": len(rows) - total_wins,
        "thresholds": threshold_blocks,
        "top_k": top_block,
        "by_direction": by_direction,
        "percentiles": percentile_blocks,
        "rank_ic_return": spearman_rank_ic(
            scores,
            [_finite_float(r.get("outcome_return_pct")) for r in rows],
            day_keys=[str(r.get("timestamp", ""))[:10] for r in rows],
        ),
        "rank_ic_r": spearman_rank_ic(
            scores,
            [_finite_float(r.get("r_multiple")) for r in rows],
            day_keys=[str(r.get("timestamp", ""))[:10] for r in rows],
        ),
        "decile_spread": {
            "decile_size": decile_size,
            "top_avg_r": round(top_avg_r, 4),
            "bottom_avg_r": round(bottom_avg_r, 4),
            "spread_r": round(top_avg_r - bottom_avg_r, 4),
        },
        "concentration": {
            "distinct_days": len(day_counts),
            "max_share_single_day": round(max_day_share, 4),
        },
        # Config at validation time; per-record ground truth is the index's
        # target_mode field (they agree whenever the lab rebuilt the index in
        # the same process, which run_edge_lab always does).
        "exit_geometry_config": {
            "target_mode": str(scanner_config.EDGE_EXIT_TARGET_MODE),
            "target_r_floor": _finite_float(scanner_config.EDGE_EXIT_TARGET_R_FLOOR),
            "target_atr_mult": _finite_float(scanner_config.EDGE_EXIT_TARGET_ATR_MULT),
        },
        # Every metric in this report is net of this charge. Recorded so a
        # stored report can never be re-read as a gross result later.
        "cost_model": {
            "bps_per_side": _finite_float(cost_bps_per_side),
            "round_trip_return_pct_charged": round(2.0 * abs(_finite_float(cost_bps_per_side)) / 100.0, 6),
            "basis": "net_of_costs",
            "candidate_rows": candidate_rows,
            "risk_coverage_rows": risk_coverage_rows,
            "risk_coverage_complete": risk_coverage_complete,
            "applies_to": ["returns", "r_multiple", "win_loss_label"],
        },
    }
