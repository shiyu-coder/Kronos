import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from scanner.brief import build_daily_brief, run_brief


def _brief_for(tmp_path, *, warnings, blockers=None, summary=None, policy_extra=None, checks=None):
    """Minimal report set: enough for the brief, varied by audit codes."""
    (tmp_path / "edge_audit_report.json").write_text(
        json.dumps(
            {
                "readiness": "blocked",
                "blockers": blockers if blockers is not None else ["ranking_evidence_unsupported"],
                "warnings": warnings,
                "checks": checks
                if checks is not None
                else {
                    "ranking_evidence": {
                        "passed": False,
                        "value": {"rank_ic": -0.016, "min_rank_ic": 0.07},
                    }
                },
                "summary": summary or {},
            }
        ),
        encoding="utf-8",
    )
    policy = {"research_candidates": {"resolved": 34, "resolved_win_rate": 0.47}}
    policy.update(policy_extra or {})
    (tmp_path / "adaptive_policy_report.json").write_text(json.dumps(policy), encoding="utf-8")
    (tmp_path / "edge_scan_report.json").write_text(
        json.dumps({"candidates": [{"ticker": "RIOT", "status": "candidate", "recommendation": "reject"}]}),
        encoding="utf-8",
    )
    _markdown, payload = build_daily_brief(tmp_path)
    return payload["telegram_text"], payload["next_action"]


def test_illiquid_chain_never_blames_the_tradier_token(tmp_path):
    # The live bug: Tradier answered for every ticker with fresh OPRA quotes,
    # four small caps simply had no tradeable strike, and the brief told the
    # operator to go check a credential that was working perfectly.
    text, next_action = _brief_for(
        tmp_path,
        warnings=["options_no_liquid_contract", "no_current_actionable_candidates"],
        summary={"no_liquid_options_contract_candidates": ["RIOT", "UPST", "LUNR", "EVGO"]},
    )

    assert "TRADIER_API_TOKEN" not in text
    assert "TRADIER_API_TOKEN" not in next_action
    assert "Nothing to do. Nothing is broken." in text
    assert "No liquid options chain (RIOT, UPST, LUNR, EVGO)" in text
    assert "FYI - no action" in text


def test_provider_unavailable_is_the_only_thing_that_blames_the_token(tmp_path):
    text, next_action = _brief_for(
        tmp_path,
        warnings=["options_provider_unavailable"],
        summary={"options_provider_unavailable_candidates": ["SOFI"]},
    )

    assert "TRADIER_API_TOKEN" in text
    assert "TRADIER_API_TOKEN" in next_action
    assert "DO THIS" in text
    assert text.splitlines()[1] == "1 thing needs you."


def test_a_dead_token_never_renders_as_nothing_is_broken(tmp_path):
    # The regression guard for the inverse bug: a real credential outage must
    # not be swallowed by the benign illiquidity finding.
    text, _next_action = _brief_for(
        tmp_path,
        warnings=["options_provider_unavailable", "no_current_actionable_candidates"],
        summary={"options_provider_unavailable_candidates": ["SOFI", "PLTR"]},
    )

    assert "Nothing to do. Nothing is broken." not in text
    assert "TRADIER_API_TOKEN" in text


def test_an_unrecognised_blocker_is_treated_as_serious_not_ignored(tmp_path):
    # A gate added later, before anyone writes its _ISSUE_GUIDE entry, must not
    # render as "nothing to do" just because the brief has no translation.
    text, next_action = _brief_for(
        tmp_path,
        blockers=["some_future_gate_nobody_translated_yet"],
        warnings=[],
    )

    assert "Nothing to do. Nothing is broken." not in text
    assert "some_future_gate_nobody_translated_yet" in text
    assert "DO THIS" in text
    assert "unrecognised code" in next_action


def test_an_unrecognised_warning_stays_a_finding(tmp_path):
    text, _next_action = _brief_for(tmp_path, blockers=[], warnings=["some_new_warning"])

    assert "Nothing to do. Nothing is broken." in text
    assert "some_new_warning" in text


def test_below_floor_cost_model_is_an_actionable_fault(tmp_path):
    _brief_for(
        tmp_path,
        blockers=["validation_cost_model_unsupported"],
        warnings=[],
    )
    audit_path = tmp_path / "edge_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checks"]["costs_charged"] = {
        "passed": False,
        "value": {
            "bps_per_side": 0.01,
            "minimum_bps_per_side": 25.0,
            "minimum_cost_met": False,
        },
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    (tmp_path / "edge_validation_report.json").write_text(
        json.dumps(
            {
                "cost_model": {
                    "bps_per_side": 0.01,
                    "round_trip_return_pct_charged": 0.0002,
                }
            }
        ),
        encoding="utf-8",
    )

    markdown, payload = build_daily_brief(tmp_path)

    assert "Cost basis: INVALID" in markdown
    assert "25 bps/side floor" in markdown
    assert "Validation cost basis is unsafe" in payload["telegram_text"]
    assert "KRONOS_COST_BPS_PER_SIDE" in payload["next_action"]
    assert "DO THIS" in payload["telegram_text"]


def test_unreadable_audit_is_not_reported_as_healthy(tmp_path):
    # "No warnings" from a corrupt/absent report means "we know nothing", which
    # must never render as a green light.
    for name in ("edge_audit_report.json", "adaptive_policy_report.json", "edge_scan_report.json"):
        (tmp_path / name).write_text("{not valid json", encoding="utf-8")

    _markdown, payload = build_daily_brief(tmp_path)
    text = payload["telegram_text"]

    assert "Nothing to do. Nothing is broken." not in text
    assert "Can't tell - no readable audit." in text


def test_quiet_day_leads_with_no_action_required(tmp_path):
    text, next_action = _brief_for(
        tmp_path,
        warnings=["low_feed_confidence", "no_current_actionable_candidates", "bearish_edge_negative"],
    )

    assert text.splitlines()[1] == "Nothing to do. Nothing is broken."
    assert "DO THIS" not in text
    assert next_action.startswith("Nothing.")
    # The gate codes are spelled out under UNLOCK; FYI must not repeat them.
    assert "Score doesn't rank winners yet" not in text.split("FYI")[1]


def test_mature_negative_gates_call_for_strategy_redesign_not_more_samples(tmp_path):
    checks = {
        "ranking_evidence": {
            "passed": False,
            "value": {
                "rank_ic": 0.017,
                "min_rank_ic": 0.07,
                "rank_ic_p_value": 0.303,
                "top_decile_signals": 150,
                "top_decile_evidence_days": 56,
                "min_signals": 20,
                "top_decile_average_r": -0.16,
                "top_decile_t_stat": -2.09,
                "top_decile_precision_lower_bound": 0.27,
                "dependence_metrics_present": True,
            },
        },
        "validation_threshold": {
            "passed": False,
            "value": {
                "threshold": 55,
                "signal_count": 42,
                "raw_signal_count": 58,
                "min_signals": 20,
                "precision": 0.34,
                "precision_lower_bound": 0.21,
                "min_precision": 0.55,
                "average_r_multiple": -0.26,
                "t_stat_r_multiple": -2.85,
                "dependence_metrics_present": True,
            },
        },
    }
    text, next_action = _brief_for(
        tmp_path,
        blockers=["validation_threshold_55_unsupported", "ranking_evidence_unsupported"],
        warnings=["bullish_edge_negative", "bearish_edge_negative"],
        checks=checks,
    )

    markdown, _payload = build_daily_brief(tmp_path)

    assert text.splitlines()[1] == "No operator action. Strategy evidence is negative."
    assert "UNLOCK - current design negative" in text
    assert "not yet" not in text
    assert "ranking evidence is negative out of sample" in text
    assert "score doesn't rank winners yet" not in text
    assert "Legacy threshold-55 gate (NEGATIVE)" in markdown
    assert "Ranking gate (NEGATIVE)" in markdown
    assert "needs more resolved samples" not in next_action
    assert "already adequately sampled and negative" in next_action
    assert "pre-register a new entry-selection hypothesis" in next_action


def test_delayed_consolidated_equity_feed_is_reported_as_research_only_finding(tmp_path):
    text, next_action = _brief_for(
        tmp_path,
        warnings=["delayed_equity_feed"],
        summary={"delayed_equity_feed_candidates": ["LYFT"]},
    )

    assert "Nothing to do. Nothing is broken." in text
    assert "Equity bars are consolidated but delayed (LYFT)" in text
    assert "DO THIS" not in text
    assert next_action.startswith("Nothing.")


def test_kronos_skips_are_reported_without_manufacturing_an_alarm(tmp_path):
    # "kronos_eval_error" holds ordinary skips ("need 60 synthetic bars") as
    # well as real exceptions, so the count alone must not read as a fault.
    text, next_action = _brief_for(
        tmp_path,
        warnings=[],
        policy_extra={"kronos_lift": {"rows_with_kronos": 8, "rows_with_eval_errors": 3}},
    )

    assert "Kronos produced no forecast on 3 resolved candidates" in text
    assert "DO THIS" not in text
    assert next_action.startswith("Nothing.")


def test_empty_kronos_cohort_is_reported_as_no_sample_not_zero_win_rate(tmp_path):
    text, _next_action = _brief_for(
        tmp_path,
        warnings=[],
        policy_extra={
            "kronos_lift": {
                "rows_with_kronos": 7,
                "agree": {"signal_count": 0, "win_rate": 0.0},
                "disagree": {"signal_count": 7, "win_rate": 0.8571},
            }
        },
    )

    # n=0 is "no sample"; n=7 is a sample too small to carry a rate. Neither
    # may render as a bare percentage.
    assert "Kronos agree n/a (n=0) vs disagree too few to rate (n=7)" in text
    assert "agree 0% (n=0)" not in text


def test_single_row_cohort_never_renders_as_a_win_rate(tmp_path):
    """A 100% win rate off one row invites trust the sample cannot support."""
    text, _next_action = _brief_for(
        tmp_path,
        warnings=[],
        policy_extra={
            "kronos_lift": {
                "rows_with_kronos": 14,
                "agree": {"signal_count": 1, "win_rate": 1.0},
                "disagree": {"signal_count": 13, "win_rate": 0.54},
            }
        },
    )

    assert "100% (n=1)" not in text
    assert "too few to rate (n=1)" in text


def test_blocked_direction_cohort_is_labelled_blocked_in_the_sms(tmp_path):
    """A winning-looking cohort in a blocked direction must not read as a green light."""
    cohort = {
        "research_candidates": {
            "resolved": 34,
            "resolved_win_rate": 0.47,
            "current_threshold": 65,
            "current_threshold_by_direction": {
                "bearish": {
                    "signal_count": 21,
                    "evidence_day_count": 13,
                    "mean_daily_win_rate": 0.68,
                    "dependence_adjusted_win_rate_lower_bound": 0.502,
                    "average_daily_return_pct": 4.45,
                },
            },
        }
    }

    blocked, _ = _brief_for(tmp_path, warnings=[], summary={"promotable_directions": []}, policy_extra=cohort)
    assert "dailyWR=68% LB=50% avgDay=4.5% BLOCKED" in blocked

    promotable, _ = _brief_for(
        tmp_path, warnings=[], summary={"promotable_directions": ["bearish"]}, policy_extra=cohort
    )
    assert "avgDay=4.5%;" in promotable or promotable.rstrip().endswith("avgDay=4.5%")
    assert "BLOCKED" not in promotable.split("Threshold 65 split:")[1].split("\n")[0]


def test_cohort_at_the_sample_floor_still_reports_its_rate(tmp_path):
    text, _next_action = _brief_for(
        tmp_path,
        warnings=[],
        policy_extra={
            "kronos_lift": {
                "rows_with_kronos": 20,
                "agree": {"signal_count": 10, "win_rate": 0.60},
                "disagree": {"signal_count": 10, "win_rate": 0.40},
            }
        },
    )

    assert "agree 60% (n=10)" in text


def _write_reports(tmp_path):
    (tmp_path / "edge_audit_report.json").write_text(
        json.dumps(
            {
                "readiness": "blocked",
                "blockers": ["ranking_evidence_unsupported"],
                "warnings": ["options_data_not_execution_grade", "low_feed_confidence"],
                "checks": {
                    "ranking_evidence": {
                        "passed": False,
                        "value": {
                            "rank_ic": 0.03,
                            "rank_ic_p_value": 0.21,
                            "min_rank_ic": 0.07,
                            "top_decile_signals": 12,
                            "top_decile_evidence_days": 8,
                            "min_signals": 20,
                            "top_decile_average_r": 0.15,
                            "top_decile_t_stat": 1.1,
                            "top_decile_wilson_lb_precision": 0.38,
                        },
                    },
                    "validation_threshold": {
                        "passed": False,
                        "value": {"threshold": 55, "signal_count": 0, "min_signals": 20},
                    },
                },
                "summary": {
                    "blocked_directions": ["bearish"],
                    "unproven_directions": ["bullish"],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "edge_validation_report.json").write_text(
        json.dumps(
            {
                "by_direction": {
                    "bullish": {
                        "signal_count": 300,
                        "average_r_multiple": 0.21,
                        "t_stat_r_day_clustered": {
                            "n_days": 60,
                            "mean_of_day_means": 0.18,
                        },
                    },
                    "bearish": {
                        "signal_count": 200,
                        "average_r_multiple": -0.05,
                        "t_stat_r_day_clustered": {
                            "n_days": 55,
                            "mean_of_day_means": -0.06,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adaptive_policy_report.json").write_text(
        json.dumps(
            {
                "research_candidates": {
                    "resolved": 22,
                    "resolved_outcomes": {"win": 8, "loss": 14},
                    "resolved_win_rate": 0.3636,
                    "current_threshold": 72,
                    "current_threshold_by_direction": {
                        "bullish": {
                            "signal_count": 8,
                            "evidence_day_count": 7,
                            "wins": 4,
                            "losses": 4,
                            "average_return_pct": -2.76,
                            "mean_daily_win_rate": 0.5,
                            "dependence_adjusted_win_rate_lower_bound": 0.2939,
                            "average_daily_return_pct": -2.5,
                        },
                        "bearish": {
                            "signal_count": 14,
                            "evidence_day_count": 10,
                            "wins": 9,
                            "losses": 5,
                            "average_return_pct": 11.51,
                            "mean_daily_win_rate": 0.65,
                            "dependence_adjusted_win_rate_lower_bound": 0.4344,
                            "average_daily_return_pct": 4.84,
                        },
                    },
                },
                "recommendation": {
                    "status": "loosen_research_threshold",
                    "reason": "a lower research threshold dominates the current cohort",
                },
                "kronos_lift": {
                    "rows_with_kronos": 4,
                    "agree": {"signal_count": 2, "win_rate": 1.0},
                    "disagree": {"signal_count": 2, "win_rate": 0.0},
                },
                "doctrine_v2": {
                    "resolved": 8,
                    "current_threshold": {"wins": 2, "losses": 1, "average_return_pct": 3.9},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "zero_result_diagnostic.json").write_text(
        json.dumps({"research_candidates": {"pending": 6}}),
        encoding="utf-8",
    )
    (tmp_path / "edge_scan_report.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "ticker": "T",
                        "status": "candidate",
                        "direction": "bearish",
                        "edge_score": 7.05,
                        "recommendation": "reject",
                        "blocking_reasons": ["setup_gate_failed", "options_data_not_execution_grade"],
                    },
                    {"ticker": "SOFI", "status": "skip", "reason": "not near box edge"},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_build_daily_brief_renders_verdict_progress_and_next_action(tmp_path):
    _write_reports(tmp_path)

    markdown, payload = build_daily_brief(tmp_path)

    assert payload["readiness"] == "blocked"
    assert "## Verdict" in markdown
    assert "NOT live-ready" in markdown
    assert "rank IC 0.030" in markdown
    assert "top-decile evidence days 8/20 (12 raw signals)" in markdown
    assert "bullish days=60 avgR 0.18 UNPROVEN" in markdown
    assert "bearish days=55 avgR -0.06 BLOCKED" in markdown
    assert "T: bearish edge 7.05" in markdown
    assert "Kronos lift: 4 scored" in markdown
    assert "loosen_research_threshold" in markdown
    assert "Threshold cohort by direction: bullish rows=8 days=7 daily-WR=50.0% LB=29.4% avg-day=-2.50%" in markdown
    assert "Threshold 72 split: bull rows=8 days=7 dailyWR=50% LB=29% avgDay=-2.5%" in payload["telegram_text"]
    assert "Confirm the pending research-threshold loosening" in payload["next_action"]


def test_today_research_samples_are_not_reported_as_zero_qualified(tmp_path):
    _write_reports(tmp_path)
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    rows = [
        {
            "ticker": "LYFT",
            "mode": "research_scan",
            "source_session_date": today,
            "direction": "bullish",
            "outcome_status": "pending",
            "skip_reason": "research_candidate",
            "research_score": 68,
            "research_diagnostics": {"passed": True},
            "doctrine_v2_score": 80,
            "doctrine_v2_passed": True,
            "kronos_directional_agreement": 0.0,
            "kronos_passed": False,
        },
        {
            "ticker": "CHPT",
            "mode": "research_scan",
            "source_session_date": today,
            "direction": "bullish",
            "outcome_status": "pending",
            "skip_reason": "research_candidate",
            "research_score": 73,
            "research_diagnostics": {"passed": True},
            "doctrine_v2_score": 65,
            "doctrine_v2_passed": False,
            "kronos_directional_agreement": 0.4,
            "kronos_passed": False,
        },
    ]
    (tmp_path / "scan_decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    markdown, payload = build_daily_brief(tmp_path)
    telegram = payload["telegram_text"]

    assert "LIVE TRADES - none" in telegram
    assert "2 scanned, 0 Edge recommendations" in telegram
    assert "RESEARCH SAMPLES - 2 counterfactual only" in telegram
    assert "LYFT bullish score 68" in telegram
    assert "CHPT bullish score 73" in telegram
    assert "## Accepted research samples" in markdown
    assert "never alerted or traded" in markdown
    assert [row["ticker"] for row in payload["research_samples"]] == ["LYFT", "CHPT"]


def test_blocked_edge_promotions_are_never_labelled_live_trades(tmp_path):
    """A scan-level promotion is only research when the readiness audit blocks it."""
    _write_reports(tmp_path)
    scan_path = tmp_path / "edge_scan_report.json"
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan["candidates"] = [
        {
            "ticker": "T",
            "status": "candidate",
            "direction": "bullish",
            "edge_score": 72.45,
            "recommendation": "promote",
        },
        {
            "ticker": "AFRM",
            "status": "candidate",
            "direction": "bullish",
            "edge_score": 67.40,
            "recommendation": "promote",
        },
    ]
    scan_path.write_text(json.dumps(scan), encoding="utf-8")

    _markdown, payload = build_daily_brief(tmp_path)
    telegram = payload["telegram_text"]

    assert "LIVE TRADES - none" in telegram
    assert "LIVE TRADES - 2" not in telegram
    assert "EDGE RESEARCH - 2" in telegram
    assert "2 Edge recommendations, 0 paper-authorized" in telegram


def test_audit_authorized_promotion_is_labelled_paper_not_live(tmp_path):
    _write_reports(tmp_path)
    audit_path = tmp_path / "edge_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["readiness"] = "paper_trade_only"
    audit["blockers"] = []
    audit["summary"]["execution_ready_promoted_candidates"] = ["T"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    scan_path = tmp_path / "edge_scan_report.json"
    scan_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "ticker": "T",
                        "status": "candidate",
                        "direction": "bullish",
                        "edge_score": 72.45,
                        "recommendation": "promote",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _markdown, payload = build_daily_brief(tmp_path)
    telegram = payload["telegram_text"]

    assert "LIVE TRADES - none" in telegram
    assert "PAPER CANDIDATES - 1" in telegram
    assert "1 Edge recommendations, 1 paper-authorized" in telegram
    assert "(paper candidate)" in telegram


def test_build_daily_brief_survives_missing_reports(tmp_path):
    markdown, payload = build_daily_brief(tmp_path)

    assert payload["readiness"] == "unknown"
    assert "No scan data yet" in markdown


def test_run_brief_writes_markdown_file(tmp_path, capsys, monkeypatch):
    _write_reports(tmp_path)
    monkeypatch.setattr("scanner.brief.scanner_config.BRIEF_TELEGRAM_ENABLED", True)

    payload = run_brief(logging.getLogger("test"), report_dir=tmp_path)

    output = tmp_path / "daily_brief.md"
    assert output.exists()
    assert payload["path"] == str(output.resolve())
    assert "## Verdict" in output.read_text(encoding="utf-8")
    assert "Kronos Daily Brief" in capsys.readouterr().out
    assert payload["telegram"]["status"] == "no_credentials"


def test_run_brief_sends_condensed_telegram_when_configured(tmp_path, monkeypatch):
    _write_reports(tmp_path)
    monkeypatch.setattr("scanner.brief.scanner_config.BRIEF_TELEGRAM_ENABLED", True)
    sent = {}

    def fake_send(token, chat_id, message, logger):
        sent.update({"token": token, "chat_id": chat_id, "message": message})
        return True

    monkeypatch.setattr("scanner.brief.send_telegram_message", fake_send)

    payload = run_brief(
        logging.getLogger("test"),
        report_dir=tmp_path,
        telegram_env={"telegram_token": "tok", "telegram_chat_id": "42"},
    )

    assert payload["telegram"]["status"] == "sent"
    assert sent["chat_id"] == "42"
    assert "KRONOS" in sent["message"]
    assert "Live alerts: off" in sent["message"]
    assert "TRADES" in sent["message"]
    assert "UNLOCK" in sent["message"]
    assert len(sent["message"]) < 1500


def test_run_brief_telegram_failure_never_raises(tmp_path, monkeypatch):
    _write_reports(tmp_path)
    monkeypatch.setattr("scanner.brief.scanner_config.BRIEF_TELEGRAM_ENABLED", True)

    def boom(token, chat_id, message, logger):
        raise RuntimeError("telegram down")

    monkeypatch.setattr("scanner.brief.send_telegram_message", boom)

    payload = run_brief(
        logging.getLogger("test"),
        report_dir=tmp_path,
        telegram_env={"telegram_token": "tok", "telegram_chat_id": "42"},
    )

    assert payload["telegram"]["status"] == "failed"


def test_run_brief_telegram_respects_disable_flag(tmp_path, monkeypatch):
    _write_reports(tmp_path)
    monkeypatch.setattr("scanner.brief.scanner_config.BRIEF_TELEGRAM_ENABLED", False)

    def fail_send(*args, **kwargs):
        raise AssertionError("must not send when disabled")

    monkeypatch.setattr("scanner.brief.send_telegram_message", fail_send)

    payload = run_brief(
        logging.getLogger("test"),
        report_dir=tmp_path,
        telegram_env={"telegram_token": "tok", "telegram_chat_id": "42"},
    )

    assert payload["telegram"]["status"] == "disabled"
