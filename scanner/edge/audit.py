from __future__ import annotations

import math
from typing import Any


# This is deliberately not an environment setting. 25 bps/side is the
# pre-registered production floor for this small-cap/options universe; allowing
# the same environment override that computes research scenarios to lower the
# audit floor would let a token positive charge certify effectively gross
# evidence.
MIN_AUDITED_COST_BPS_PER_SIDE = 25.0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _threshold_block(validation_report: dict, threshold: int) -> dict:
    thresholds = validation_report.get("thresholds", {})
    block = thresholds.get(str(threshold), {})
    return block if isinstance(block, dict) else {}


def _candidate_features(candidate: dict) -> dict:
    features = candidate.get("features", {})
    return features if isinstance(features, dict) else {}


def _equity_feed_delay_is_execution_grade(features: dict) -> bool:
    """Fail closed unless the current equity snapshot is effectively real time.

    Alpaca's free historical SIP route deliberately reports a 16-minute delay.
    It is excellent consolidated research evidence, but a high
    ``feed_confidence`` value alone must not make that snapshot eligible for a
    live-consumed readiness verdict. Missing or nonsensical provenance is also
    not execution grade.
    """
    value = features.get("data_delay_minutes")
    if not _is_finite_number(value):
        return False
    delay_minutes = float(value)
    return 0.0 <= delay_minutes <= 1.0


def _options_diagnosis(features: dict) -> str | None:
    """Why a candidate's options evidence is not execution grade, or None.

    Three causes that used to share one code, because the operator's response to
    each is completely different:

    - no_liquid_contract: Tradier answered and no strike clears the liquidity
      gates. An authoritative verdict about the market, not a data fault - small
      caps genuinely have untradeable chains, and blaming a credential for that
      is what made a healthy token look broken.
    - stale_quotes: a real Tradier contract whose quote had gone stale.
    - provider_unavailable: the data did not come from Tradier at all.

    The discriminator for that last one is the provider. Only the Tradier path
    records `data_provider` on a FAILED lookup (options_data.py:194, :271); the
    yfinance/Alpaca fallback and the exception handler leave it unset. So a
    failure with no provider means Tradier never answered - dead token, sandbox
    token, unreachable API, or no token at all - and that must never read as
    ordinary illiquidity.
    """
    provider = str(features.get("options_data_provider") or "").strip()
    if _finite_float(features.get("options_passed"), 1.0) < 1.0:
        return "no_liquid_contract" if provider else "provider_unavailable"
    if not provider:
        # The options stage never ran for this candidate; we know nothing about
        # its chain and must not claim its data was degraded.
        return None
    if _finite_float(features.get("options_data_quality"), 0.45) >= 0.75:
        return None
    feed = str(features.get("options_data_feed") or "").strip().lower()
    if feed not in {"opra", "opra-consolidated"}:
        return "provider_unavailable"
    return "stale_quotes"


def candidate_execution_ready(candidate: dict) -> bool:
    """Whether a promoted row has capital-grade current-market evidence.

    Edge scoring deliberately permits research on partial-market equity feeds,
    but the readiness audit is the final authorization consumed by live
    preflight. Keep that boundary stricter: IEX-only equities and indicative,
    stale, missing, or illiquid options evidence may remain research samples,
    but must never produce ``paper_trade_only`` readiness.
    """
    features = _candidate_features(candidate)
    if _finite_float(features.get("feed_confidence"), 0.5) < 0.75:
        return False
    if not _equity_feed_delay_is_execution_grade(features):
        return False
    if _finite_float(features.get("data_quality_score"), 1.0) < 0.5:
        return False
    if _finite_float(features.get("options_passed"), 0.0) < 1.0:
        return False
    provider = str(features.get("options_data_provider") or "").strip().lower()
    feed = str(features.get("options_data_feed") or "").strip().lower()
    if provider != "tradier" or feed not in {"opra", "opra-consolidated"}:
        return False
    if _finite_float(features.get("options_data_quality"), 0.0) < 0.75:
        return False
    if _options_diagnosis(features) is not None:
        return False
    return (
        _finite_float(features.get("options_open_interest")) > 0.0
        and _finite_float(features.get("options_volume")) > 0.0
        and _finite_float(features.get("options_spread_pct")) > 0.0
    )


def _check(name: str, passed: bool, detail: str, value: Any = None) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "value": value,
    }


def compute_edge_audit_report(
    validation_report: dict,
    scan_report: dict,
    *,
    expected_runtime_fingerprint: str | None = None,
    validation_threshold: int = 55,
    min_validation_signals: int = 20,
    min_precision: float = 0.55,
    min_average_r_multiple: float = 0.0,
    min_rank_ic: float = 0.07,
    max_rank_ic_p_value: float = 0.05,
    min_top_decile_t_stat: float = 2.0,
    min_top_decile_wilson_lb: float = 0.45,
    min_direction_samples: int = 15,
    min_direction_t_stat: float = 2.0,
) -> dict:
    """Summarize whether current edge evidence is research-ready or blocked."""
    candidates = [row for row in scan_report.get("candidates", []) if isinstance(row, dict)]
    active_candidates = [row for row in candidates if row.get("status") == "candidate"]
    research_candidates = [row for row in active_candidates if row.get("recommendation") == "research"]
    promoted_candidates = [row for row in active_candidates if row.get("recommendation") == "promote"]
    threshold = _threshold_block(validation_report, validation_threshold)

    validation_method = validation_report.get("validation_method")
    future_analogs_allowed = bool(validation_report.get("future_analogs_allowed", True))
    scan_runtime_fingerprint = str(scan_report.get("runtime_fingerprint") or "").strip()
    validation_runtime_fingerprint = str(validation_report.get("runtime_fingerprint") or "").strip()
    expected_runtime_fingerprint = str(expected_runtime_fingerprint or "").strip()
    runtime_source_required = bool(expected_runtime_fingerprint)
    runtime_source_matches = (
        not runtime_source_required
        or (
            bool(scan_runtime_fingerprint)
            and scan_runtime_fingerprint == validation_runtime_fingerprint
            and scan_runtime_fingerprint == expected_runtime_fingerprint
        )
    )
    threshold_precision_clustered = threshold.get("precision_day_clustered")
    threshold_precision_clustered = (
        threshold_precision_clustered if isinstance(threshold_precision_clustered, dict) else None
    )
    threshold_r_clustered = threshold.get("t_stat_r_day_clustered")
    threshold_r_clustered = threshold_r_clustered if isinstance(threshold_r_clustered, dict) else None
    threshold_precision_days = (
        int(_finite_float(threshold_precision_clustered.get("n_days")))
        if threshold_precision_clustered is not None
        else 0
    )
    threshold_r_days = (
        int(_finite_float(threshold_r_clustered.get("n_days")))
        if threshold_r_clustered is not None
        else 0
    )
    threshold_dependence_metrics_present = (
        threshold_precision_clustered is not None
        and threshold_r_clustered is not None
        and _is_finite_number(threshold_precision_clustered.get("n_days"))
        and _is_finite_number(threshold_precision_clustered.get("mean_daily_precision"))
        and _is_finite_number(threshold_precision_clustered.get("lower_bound"))
        and _is_finite_number(threshold_r_clustered.get("n_days"))
        and _is_finite_number(threshold_r_clustered.get("mean_of_day_means"))
        and _is_finite_number(threshold_r_clustered.get("t_stat"))
        and threshold_precision_days > 0
        and threshold_precision_days == threshold_r_days
    )
    raw_signal_count = int(_finite_float(threshold.get("signal_count")))
    signal_count = (
        threshold_precision_days
        if threshold_precision_clustered is not None
        else raw_signal_count
    )
    precision = (
        _finite_float(threshold_precision_clustered.get("mean_daily_precision"))
        if threshold_precision_clustered is not None
        and threshold_precision_clustered.get("mean_daily_precision") is not None
        else _finite_float(threshold.get("precision"))
    )
    precision_lower_bound = (
        _finite_float(threshold_precision_clustered.get("lower_bound"))
        if threshold_precision_clustered is not None
        and threshold_precision_clustered.get("lower_bound") is not None
        else precision
    )
    average_r = (
        _finite_float(threshold_r_clustered.get("mean_of_day_means"))
        if threshold_r_clustered is not None
        and threshold_r_clustered.get("mean_of_day_means") is not None
        else _finite_float(threshold.get("average_r_multiple"))
    )
    threshold_r_t_stat = (
        _finite_float(threshold_r_clustered.get("t_stat"))
        if threshold_r_clustered is not None
        else None
    )

    rank_ic = validation_report.get("rank_ic_r", {})
    rank_ic = rank_ic if isinstance(rank_ic, dict) else {}
    percentiles = validation_report.get("percentiles", {})
    percentiles = percentiles if isinstance(percentiles, dict) else {}
    top_decile = percentiles.get("top_10_pct", {})
    top_decile = top_decile if isinstance(top_decile, dict) else {}
    top_precision_clustered = top_decile.get("precision_day_clustered")
    top_precision_clustered = top_precision_clustered if isinstance(top_precision_clustered, dict) else None
    top_r_clustered = top_decile.get("t_stat_r_day_clustered")
    top_r_clustered = top_r_clustered if isinstance(top_r_clustered, dict) else None
    top_precision_days = (
        int(_finite_float(top_precision_clustered.get("n_days")))
        if top_precision_clustered is not None
        else 0
    )
    top_r_days = (
        int(_finite_float(top_r_clustered.get("n_days")))
        if top_r_clustered is not None
        else 0
    )
    top_dependence_metrics_present = (
        top_precision_clustered is not None
        and top_r_clustered is not None
        and _is_finite_number(top_precision_clustered.get("n_days"))
        and _is_finite_number(top_precision_clustered.get("mean_daily_precision"))
        and _is_finite_number(top_precision_clustered.get("lower_bound"))
        and _is_finite_number(top_r_clustered.get("n_days"))
        and _is_finite_number(top_r_clustered.get("mean_of_day_means"))
        and _is_finite_number(top_r_clustered.get("t_stat"))
        and top_precision_days > 0
        and top_precision_days == top_r_days
    )
    top_decile_raw_signals = int(_finite_float(top_decile.get("signal_count")))
    top_decile_evidence_days = (
        top_precision_days
        if top_precision_clustered is not None
        else top_decile_raw_signals
    )
    top_decile_average_r = (
        _finite_float(top_r_clustered.get("mean_of_day_means"))
        if top_r_clustered is not None
        and top_r_clustered.get("mean_of_day_means") is not None
        else _finite_float(top_decile.get("average_r_multiple"))
    )
    top_decile_t_stat = (
        _finite_float(top_r_clustered.get("t_stat"))
        if top_r_clustered is not None
        else _finite_float(top_decile.get("t_stat_r_multiple"))
    )
    top_decile_precision_lb = (
        _finite_float(top_precision_clustered.get("lower_bound"))
        if top_precision_clustered is not None
        and top_precision_clustered.get("lower_bound") is not None
        else _finite_float(top_decile.get("wilson_lb_precision"))
    )
    rank_ic_clustered_present = _is_finite_number(rank_ic.get("p_value_day_clustered"))
    rank_ic_p_value = _finite_float(rank_ic.get("p_value_day_clustered"), 1.0)
    ranking_dependence_metrics_present = rank_ic_clustered_present and top_dependence_metrics_present

    # Pooled IC can pass on direction separation alone (a positive bullish
    # cohort vs a negative bearish one) while the score ranks nothing INSIDE
    # either direction. Ranking evidence therefore also requires at least one
    # direction whose within-direction IC clears the bar, judged with the
    # day-clustered p-value (overlapping outcomes make the raw n
    # anti-conservative). Missing per-direction blocks fail closed.
    report_directions = validation_report.get("by_direction", {})
    report_directions = report_directions if isinstance(report_directions, dict) else {}
    within_direction_ic: dict[str, dict] = {}
    within_direction_passed = False
    for direction in ("bullish", "bearish"):
        block = report_directions.get(direction)
        if not isinstance(block, dict):
            continue
        direction_ic = block.get("rank_ic_r")
        if not isinstance(direction_ic, dict):
            continue
        ic_value = _finite_float(direction_ic.get("ic"))
        p_clustered_present = _is_finite_number(direction_ic.get("p_value_day_clustered"))
        p_clustered = _finite_float(direction_ic.get("p_value_day_clustered"), 1.0)
        within_direction_ic[direction] = {
            "ic": ic_value,
            "p_value_day_clustered": p_clustered,
            "dependence_metric_present": p_clustered_present,
            "n": int(_finite_float(direction_ic.get("n"))),
        }
        if p_clustered_present and ic_value >= min_rank_ic and p_clustered <= max_rank_ic_p_value:
            within_direction_passed = True

    # A validation report computed gross of costs describes an edge nobody can
    # capture, so it must never satisfy a promotion gate. Do not trust a bare
    # positive cost number: the report must also identify the net basis, carry
    # the exact round-trip charge implied by that number, state that return, R
    # and win/loss metrics were all recomputed, and prove that every R row had
    # the stop-distance denominator needed to recompute it. That makes malformed,
    # stale or partially migrated report artifacts fail closed instead of
    # certifying gross gate metrics with a decorative `bps_per_side` stamp.
    cost_model = validation_report.get("cost_model")
    cost_model = cost_model if isinstance(cost_model, dict) else {}
    cost_bps_per_side = _finite_float(cost_model.get("bps_per_side"))
    cost_round_trip_pct = _finite_float(cost_model.get("round_trip_return_pct_charged"))
    expected_round_trip_pct = 2.0 * cost_bps_per_side / 100.0
    applies_to = cost_model.get("applies_to")
    applies_to = (
        {str(value) for value in applies_to}
        if isinstance(applies_to, (list, tuple, set))
        else set()
    )
    required_cost_metrics = {"returns", "r_multiple", "win_loss_label"}
    cost_basis_valid = str(cost_model.get("basis") or "") == "net_of_costs"
    cost_floor_met = cost_bps_per_side >= MIN_AUDITED_COST_BPS_PER_SIDE
    cost_charge_consistent = (
        cost_bps_per_side > 0.0
        and _is_finite_number(cost_model.get("round_trip_return_pct_charged"))
        and math.isclose(cost_round_trip_pct, expected_round_trip_pct, rel_tol=1e-9, abs_tol=1e-9)
    )
    cost_metrics_complete = required_cost_metrics.issubset(applies_to)
    candidate_rows_value = cost_model.get("candidate_rows")
    risk_coverage_rows_value = cost_model.get("risk_coverage_rows")
    report_samples_value = validation_report.get("samples")
    coverage_counts_present = all(
        not isinstance(value, bool) and _is_finite_number(value)
        for value in (candidate_rows_value, risk_coverage_rows_value, report_samples_value)
    )
    candidate_rows = int(_finite_float(candidate_rows_value)) if coverage_counts_present else 0
    risk_coverage_rows = int(_finite_float(risk_coverage_rows_value)) if coverage_counts_present else 0
    report_samples = int(_finite_float(report_samples_value)) if coverage_counts_present else 0
    coverage_counts_are_whole = coverage_counts_present and all(
        value >= 0.0 and math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9)
        for value in (
            _finite_float(candidate_rows_value),
            _finite_float(risk_coverage_rows_value),
            _finite_float(report_samples_value),
        )
    )
    cost_risk_coverage_complete = (
        coverage_counts_are_whole
        and cost_model.get("risk_coverage_complete") is True
        and candidate_rows > 0
        and candidate_rows == report_samples
        and risk_coverage_rows == candidate_rows
    )
    costs_charged = (
        cost_basis_valid
        and cost_floor_met
        and cost_charge_consistent
        and cost_metrics_complete
        and cost_risk_coverage_complete
    )

    ranking_passed = (
        costs_charged
        and ranking_dependence_metrics_present
        and _finite_float(rank_ic.get("ic")) >= min_rank_ic
        and rank_ic_p_value <= max_rank_ic_p_value
        and within_direction_passed
        and top_decile_evidence_days >= min_validation_signals
        and top_decile_average_r > 0.0
        and top_decile_t_stat >= min_top_decile_t_stat
        and top_decile_precision_lb >= min_top_decile_wilson_lb
    )
    threshold_passed = (
        costs_charged
        and threshold_dependence_metrics_present
        and signal_count >= min_validation_signals
        and precision_lower_bound >= min_precision
        and average_r > min_average_r_multiple
        and threshold_r_t_stat is not None
        and threshold_r_t_stat >= min_top_decile_t_stat
    )

    checks = {
        "purged_walk_forward": _check(
            "purged_walk_forward",
            validation_method == "purged_walk_forward",
            "Historical validation must use walk-forward records only.",
            validation_method,
        ),
        "costs_charged": _check(
            "costs_charged",
            costs_charged,
            "Gate metrics must carry complete, internally consistent provenance for the pre-registered net transaction-cost floor.",
            {
                "bps_per_side": cost_bps_per_side,
                "minimum_bps_per_side": MIN_AUDITED_COST_BPS_PER_SIDE,
                "minimum_cost_met": cost_floor_met,
                "round_trip_return_pct_charged": cost_round_trip_pct,
                "expected_round_trip_return_pct": expected_round_trip_pct,
                "basis": cost_model.get("basis"),
                "applies_to": sorted(applies_to),
                "basis_valid": cost_basis_valid,
                "charge_consistent": cost_charge_consistent,
                "required_metrics_present": cost_metrics_complete,
                "candidate_rows": candidate_rows,
                "risk_coverage_rows": risk_coverage_rows,
                "report_samples": report_samples,
                "risk_coverage_complete": cost_risk_coverage_complete,
            },
        ),
        "future_analogs_blocked": _check(
            "future_analogs_blocked",
            not future_analogs_allowed,
            "Validation must not score past candidates with future analogs.",
            future_analogs_allowed,
        ),
        "runtime_source": _check(
            "runtime_source",
            runtime_source_matches,
            "Scan and validation must come from the exact production runtime being audited.",
            {
                "required": runtime_source_required,
                "expected": expected_runtime_fingerprint or None,
                "scan": scan_runtime_fingerprint or None,
                "validation": validation_runtime_fingerprint or None,
            },
        ),
        "validation_threshold": _check(
            "validation_threshold",
            threshold_passed,
            f"Threshold {validation_threshold} needs enough dependence-aware positive out-of-sample evidence.",
            {
                "threshold": validation_threshold,
                "signal_count": signal_count,
                "raw_signal_count": raw_signal_count,
                "min_signals": min_validation_signals,
                "precision": precision,
                "precision_lower_bound": precision_lower_bound,
                "min_precision": min_precision,
                "average_r_multiple": average_r,
                "t_stat_r_multiple": threshold_r_t_stat,
                "dependence_metrics_present": threshold_dependence_metrics_present,
            },
        ),
        "ranking_evidence": _check(
            "ranking_evidence",
            ranking_passed,
            "Score must rank outcomes across all samples AND within at least one direction, and the top decile must be profitable across enough independent entry days.",
            {
                "rank_ic": _finite_float(rank_ic.get("ic")),
                "rank_ic_p_value": rank_ic_p_value,
                "rank_ic_raw_p_value": _finite_float(rank_ic.get("p_value"), 1.0),
                "min_rank_ic": min_rank_ic,
                "dependence_metrics_present": ranking_dependence_metrics_present,
                "within_direction_ic": within_direction_ic,
                "within_direction_passed": within_direction_passed,
                "top_decile_signals": top_decile_raw_signals,
                "top_decile_evidence_days": top_decile_evidence_days,
                "min_signals": min_validation_signals,
                "top_decile_average_r": top_decile_average_r,
                "top_decile_t_stat": top_decile_t_stat,
                "top_decile_precision_lower_bound": top_decile_precision_lb,
                # Backward-compatible report field for older brief consumers.
                "top_decile_wilson_lb_precision": top_decile_precision_lb,
                "top_decile_dependence_method": (
                    top_precision_clustered.get("method")
                    if top_precision_clustered is not None
                    else "legacy_iid"
                ),
            },
        ),
    }

    # Either evidence route is acceptable: the absolute-threshold gate is kept
    # for continuity, but score compression can leave it permanently starved
    # of signals, so ranking skill over all samples is an equal path.
    evidence_supported = checks["validation_threshold"]["passed"] or checks["ranking_evidence"]["passed"]

    # Promotion needs a direction PROVEN positive. A positive sample mean alone
    # is not enough: today's bullish cohort, for example, can drift a few
    # hundredths of an R above zero while its day-clustered t-stat still says
    # the result is indistinguishable from noise. Under-sampled, zero-mean,
    # missing-stat, and weak-stat directions are "unproven", not "safe".
    blocked_directions: list[str] = []
    promotable_directions: list[str] = []
    unproven_directions: list[str] = []
    by_direction = validation_report.get("by_direction", {})
    if isinstance(by_direction, dict):
        for direction, block in sorted(by_direction.items()):
            if not isinstance(block, dict) or direction not in {"bullish", "bearish"}:
                continue
            clustered_t = block.get("t_stat_r_day_clustered")
            direction_samples = (
                int(_finite_float(clustered_t.get("n_days")))
                if isinstance(clustered_t, dict)
                else int(_finite_float(block.get("signal_count")))
            )
            direction_avg_r = (
                _finite_float(clustered_t.get("mean_of_day_means"))
                if isinstance(clustered_t, dict)
                and clustered_t.get("mean_of_day_means") is not None
                else _finite_float(block.get("average_r_multiple"))
            )
            direction_t_stat = (
                _finite_float(clustered_t.get("t_stat"))
                if isinstance(clustered_t, dict)
                else 0.0
            )
            if direction_samples >= min_direction_samples and direction_avg_r < 0.0:
                blocked_directions.append(direction)
            elif (
                direction_samples >= min_direction_samples
                and direction_avg_r > 0.0
                and direction_t_stat >= min_direction_t_stat
            ):
                promotable_directions.append(direction)
            else:
                unproven_directions.append(direction)

    blockers: list[str] = []
    if not checks["purged_walk_forward"]["passed"]:
        blockers.append("validation_not_walk_forward")
    if not checks["future_analogs_blocked"]["passed"]:
        blockers.append("future_analogs_allowed")
    if not checks["runtime_source"]["passed"]:
        blockers.append("evidence_runtime_source_mismatch")
    if not checks["costs_charged"]["passed"]:
        blockers.append("validation_cost_model_unsupported")
    if not evidence_supported:
        if not checks["validation_threshold"]["passed"]:
            blockers.append(f"validation_threshold_{validation_threshold}_unsupported")
        if not checks["ranking_evidence"]["passed"]:
            blockers.append("ranking_evidence_unsupported")

    warnings: list[str] = []
    low_feed_candidates = []
    delayed_equity_feed_candidates = []
    missing_liquidity_candidates = []
    non_execution_options_candidates = []
    no_liquid_contract_candidates = []
    provider_unavailable_candidates = []
    stale_quote_candidates = []
    for row in active_candidates:
        features = _candidate_features(row)
        ticker = row.get("ticker", "unknown")
        if _finite_float(features.get("feed_confidence"), 0.5) < 0.75:
            low_feed_candidates.append(ticker)
        elif not _equity_feed_delay_is_execution_grade(features):
            delayed_equity_feed_candidates.append(ticker)

        diagnosis = _options_diagnosis(features)
        if diagnosis == "no_liquid_contract":
            no_liquid_contract_candidates.append(ticker)
        elif diagnosis == "provider_unavailable":
            provider_unavailable_candidates.append(ticker)
            non_execution_options_candidates.append(ticker)
        elif diagnosis == "stale_quotes":
            stale_quote_candidates.append(ticker)
            non_execution_options_candidates.append(ticker)

        # When there is no contract at all, its zeroed open-interest/volume
        # fields are a consequence of that fact, not a second finding - the
        # cause above already reports it. Only judge liquidity when a real
        # contract exists to judge.
        if diagnosis not in {"no_liquid_contract", "provider_unavailable"}:
            open_interest = _finite_float(features.get("options_open_interest"))
            option_volume = _finite_float(features.get("options_volume"))
            spread = _finite_float(features.get("options_spread_pct"))
            if open_interest <= 0 or option_volume <= 0 or spread <= 0:
                missing_liquidity_candidates.append(ticker)

    if low_feed_candidates:
        warnings.append("low_feed_confidence")
    if delayed_equity_feed_candidates:
        warnings.append("delayed_equity_feed")
    if missing_liquidity_candidates:
        warnings.append("options_liquidity_missing")
    if no_liquid_contract_candidates:
        warnings.append("options_no_liquid_contract")
    # Keyed off stale_quote_candidates, not the union: a provider outage is one
    # root cause and must raise exactly one alarm, or the brief counts it twice
    # and claims quotes were "stale" when there were no quotes at all.
    if stale_quote_candidates:
        warnings.append("options_data_not_execution_grade")
    if provider_unavailable_candidates:
        warnings.append("options_provider_unavailable")
    if not research_candidates and not promoted_candidates:
        warnings.append("no_current_actionable_candidates")
    for direction in blocked_directions:
        warnings.append(f"{direction}_edge_negative")

    direction_eligible_promoted = [
        row for row in promoted_candidates if str(row.get("direction")) in promotable_directions
    ]
    eligible_promoted = [row for row in direction_eligible_promoted if candidate_execution_ready(row)]
    execution_quality_blocked_promoted = [
        row for row in direction_eligible_promoted if not candidate_execution_ready(row)
    ]
    if promoted_candidates and not direction_eligible_promoted:
        warnings.append("promoted_candidates_direction_blocked")
    if execution_quality_blocked_promoted:
        warnings.append("promoted_candidates_execution_quality_blocked")

    if blockers:
        readiness = "blocked"
    elif eligible_promoted:
        readiness = "paper_trade_only"
    elif research_candidates or promoted_candidates:
        readiness = "research_only"
    else:
        readiness = "watch_only"

    evidence_provenance = {
        "scan_completed_at": scan_report.get("completed_at"),
        "validation_completed_at": validation_report.get("completed_at"),
        "scan_run_id": scan_report.get("evidence_run_id"),
        "validation_run_id": validation_report.get("evidence_run_id"),
    }
    if runtime_source_required or scan_runtime_fingerprint or validation_runtime_fingerprint:
        evidence_provenance.update(
            {
                "scan_runtime_fingerprint": scan_runtime_fingerprint or None,
                "validation_runtime_fingerprint": validation_runtime_fingerprint or None,
                "runtime_fingerprint": expected_runtime_fingerprint if runtime_source_matches else None,
            }
        )

    return {
        "mode": "audit_edge",
        "readiness": readiness,
        "blockers": blockers,
        "warnings": warnings,
        "evidence_provenance": evidence_provenance,
        "checks": checks,
        "summary": {
            "candidate_count": len(candidates),
            "active_candidates": len(active_candidates),
            "research_candidates": len(research_candidates),
            "promoted_candidates": len(promoted_candidates),
            "execution_ready_promoted_candidates": [
                row.get("ticker", "unknown") for row in eligible_promoted
            ],
            "execution_quality_blocked_promoted_candidates": [
                row.get("ticker", "unknown") for row in execution_quality_blocked_promoted
            ],
            "blocked_directions": blocked_directions,
            "promotable_directions": promotable_directions,
            "unproven_directions": unproven_directions,
            "low_feed_confidence_candidates": low_feed_candidates,
            "delayed_equity_feed_candidates": delayed_equity_feed_candidates,
            "missing_options_liquidity_candidates": missing_liquidity_candidates,
            "non_execution_grade_options_candidates": non_execution_options_candidates,
            "no_liquid_options_contract_candidates": no_liquid_contract_candidates,
            "options_provider_unavailable_candidates": provider_unavailable_candidates,
            "stale_options_quote_candidates": stale_quote_candidates,
        },
    }
