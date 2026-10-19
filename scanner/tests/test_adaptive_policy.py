import json

from scanner.learning.adaptive_policy import build_adaptive_policy_report, apply_adaptive_overrides
from scanner.strategy.potter_doctrine import DOCTRINE_V2_EVIDENCE_VERSION


def _research_record(ticker, score, label, ret, day=1, doctrine_score=None, punchback_state=None):
    record = {
        "ticker": ticker,
        "mode": "research_scan",
        "decision_ts": f"2026-06-{day:02d}T10:00:00-04:00",
        "final_pass": False,
        "stage_failed": "potter_box_research",
        "skip_reason": "research_candidate",
        "outcome_status": "resolved",
        "outcome_label": label,
        "outcome_ret_5bar_pct": ret,
        "research_score": score,
        "research_diagnostics": {
            "passed": True,
            "score": score,
            "reason": "research_candidate",
            "reasons": ["confirmed_breakout", "volume_expansion"],
        },
    }
    if doctrine_score is not None:
        record.update(
            {
                "doctrine_v2_score": doctrine_score,
                "doctrine_v2_version": DOCTRINE_V2_EVIDENCE_VERSION,
                "doctrine_v2_passed": doctrine_score >= 70,
                "doctrine_v2_punchback_state": punchback_state or "fresh_breakout",
                "doctrine_v2_cost_basis_state": "held",
                "doctrine_v2_risk_flags": ["failed_reentry"] if punchback_state == "failed_reentry" else [],
            }
        )
    return record


def test_adaptive_policy_tightens_loss_heavy_research_candidates():
    records = [
        _research_record("L1", 63, "loss", -2.2, 1),
        _research_record("L2", 65, "loss", -1.8, 2),
        _research_record("L3", 68, "loss", -4.1, 3),
        _research_record("L4", 63, "loss", -0.6, 4),
        _research_record("L5", 65, "loss", -3.4, 5),
        _research_record("L6", 68, "loss", -1.1, 6),
        _research_record("W1", 63, "win", 0.7, 7),
        _research_record("W2", 65, "win", 0.4, 8),
    ]

    report = build_adaptive_policy_report(records, current_research_score=62, min_research_samples=8)

    assert report["research_candidates"]["resolved"] == 8
    assert report["research_candidates"]["resolved_outcomes"] == {"loss": 6, "win": 2}
    assert report["research_candidates"]["resolved_win_rate"] == 0.25
    assert report["recommendation"]["status"] == "tighten_research_threshold"
    assert report["recommendation"]["auto_apply_safe"] is True
    assert report["recommendation"]["proposed_overrides"] == {"RESEARCH_CANDIDATE_MIN_SCORE": 67}


def test_adaptive_policy_selects_supported_higher_score_threshold():
    records = []
    for i in range(8):
        records.append(_research_record(f"LOW{i}", 63 + (i % 2), "loss", -1.5, i + 1))
    for i in range(12):
        label = "loss" if i == 0 else "win"
        ret = -0.3 if label == "loss" else 1.4
        records.append(_research_record(f"HIGH{i}", 72 + (i % 3), label, ret, i + 10))

    report = build_adaptive_policy_report(records, current_research_score=62, min_research_samples=8)

    assert report["recommendation"]["status"] == "improve_research_threshold"
    assert report["recommendation"]["selected_threshold"] == 70
    assert report["recommendation"]["proposed_overrides"] == {"RESEARCH_CANDIDATE_MIN_SCORE": 70}
    selected = next(row for row in report["threshold_candidates"] if row["threshold"] == 70)
    assert selected["signal_count"] == 12
    assert selected["win_rate"] > 0.9
    assert selected["average_return_pct"] > 1.0


def test_adaptive_policy_holds_supported_current_research_threshold_without_noop():
    records = [_research_record(f"W{i}", 65, "win", 1.0, i + 1) for i in range(8)]

    report = build_adaptive_policy_report(records, current_research_score=65, min_research_samples=8)

    assert report["recommendation"]["status"] == "hold_supported_research_threshold"
    assert report["recommendation"]["auto_apply_safe"] is False
    assert report["recommendation"]["proposed_overrides"] == {}


def test_adaptive_policy_does_not_ratcheting_tighten_without_current_threshold_samples():
    records = [
        _research_record("L1", 63, "loss", -2.2, 1),
        _research_record("L2", 65, "loss", -1.8, 2),
        _research_record("L3", 68, "loss", -4.1, 3),
        _research_record("L4", 63, "loss", -0.6, 4),
        _research_record("L5", 65, "loss", -3.4, 5),
        _research_record("L6", 68, "loss", -1.1, 6),
        _research_record("W1", 63, "win", 0.7, 7),
        _research_record("W2", 65, "win", 0.4, 8),
    ]

    report = build_adaptive_policy_report(records, current_research_score=67, min_research_samples=8)

    assert report["research_candidates"]["signal_count"] == 2
    assert report["recommendation"]["status"] == "hold_current_threshold_pending_samples"
    assert report["recommendation"]["auto_apply_safe"] is False
    assert report["recommendation"]["proposed_overrides"] == {}


def test_adaptive_policy_can_tighten_doctrine_v2_baseline_from_losses():
    records = [
        _research_record("D1", 65, "loss", -2.2, 1, doctrine_score=71, punchback_state="failed_reentry"),
        _research_record("D2", 65, "loss", -1.8, 2, doctrine_score=72, punchback_state="failed_reentry"),
        _research_record("D3", 65, "loss", -4.1, 3, doctrine_score=73, punchback_state="failed_reentry"),
        _research_record("D4", 65, "loss", -0.6, 4, doctrine_score=74, punchback_state="failed_reentry"),
        _research_record("D5", 65, "loss", -3.4, 5, doctrine_score=71, punchback_state="fresh_breakout"),
        _research_record("D6", 65, "loss", -1.1, 6, doctrine_score=72, punchback_state="fresh_breakout"),
        _research_record("D7", 65, "win", 0.7, 7, doctrine_score=73, punchback_state="reclaim"),
        _research_record("D8", 65, "win", 0.4, 8, doctrine_score=74, punchback_state="reclaim"),
    ]

    report = build_adaptive_policy_report(
        records,
        current_research_score=80,
        current_doctrine_score_baseline=70,
        min_research_samples=8,
        min_doctrine_samples=8,
    )

    assert report["doctrine_v2"]["resolved"] == 8
    assert report["doctrine_v2"]["current_baseline"] == 70
    assert report["doctrine_v2"]["punchback_states"]["failed_reentry"]["losses"] == 4
    assert report["doctrine_v2"]["recommendation"]["status"] == "tighten_doctrine_v2_baseline"
    assert report["recommendation"]["proposed_overrides"] == {"DOCTRINE_V2_SCORE_BASELINE": 75}


def test_adaptive_policy_excludes_legacy_doctrine_scores_after_semantic_change():
    current = _research_record("CURRENT", 65, "win", 1.0, 1, doctrine_score=75)
    legacy = _research_record("LEGACY", 65, "loss", -5.0, 2, doctrine_score=80)
    legacy.pop("doctrine_v2_version")

    report = build_adaptive_policy_report(
        [current, legacy],
        current_research_score=80,
        current_doctrine_score_baseline=75,
        min_research_samples=8,
        min_doctrine_samples=8,
    )

    doctrine = report["doctrine_v2"]
    assert doctrine["evidence_version"] == DOCTRINE_V2_EVIDENCE_VERSION
    assert doctrine["legacy_rows_excluded"] == 1
    assert doctrine["resolved"] == 1


def test_adaptive_policy_holds_supported_current_doctrine_baseline_without_noop():
    records = [
        _research_record(f"D{i}", 65, "win", 1.0, i + 1, doctrine_score=75)
        for i in range(8)
    ]

    report = build_adaptive_policy_report(
        records,
        current_research_score=80,
        current_doctrine_score_baseline=75,
        min_research_samples=8,
        min_doctrine_samples=8,
    )

    recommendation = report["doctrine_v2"]["recommendation"]
    assert recommendation["status"] == "hold_supported_doctrine_v2_baseline"
    assert recommendation["auto_apply_safe"] is False
    assert recommendation["proposed_overrides"] == {}


def test_adaptive_policy_does_not_count_overlapping_ticker_outcomes_as_independent():
    records = [
        {
            **_research_record("IONQ", 75, "win", 10.0, day, doctrine_score=80),
            "source_session_date": f"2026-06-{day:02d}",
        }
        for day in range(1, 9)
    ]

    report = build_adaptive_policy_report(
        records,
        current_research_score=65,
        current_doctrine_score_baseline=70,
        min_research_samples=8,
        min_doctrine_samples=8,
    )

    research = report["research_candidates"]
    assert research["resolved"] == 8
    assert research["independent_resolved"] == 1
    assert research["evidence_independence"]["overlap_rows_excluded"] == 7
    assert research["signal_count"] == 1
    assert report["recommendation"]["auto_apply_safe"] is False
    assert report["doctrine_v2"]["independent_resolved"] == 1
    assert report["doctrine_v2"]["recommendation"]["auto_apply_safe"] is False


def test_adaptive_policy_does_not_count_same_day_cross_ticker_outcomes_as_independent():
    records = [
        {
            **_research_record(f"TICKER{i}", 75, "win", 10.0, 1, doctrine_score=80),
            "source_session_date": "2026-06-01",
        }
        for i in range(8)
    ]

    report = build_adaptive_policy_report(
        records,
        current_research_score=65,
        current_doctrine_score_baseline=70,
        min_research_samples=8,
        min_doctrine_samples=8,
    )

    research = report["research_candidates"]
    assert research["independent_resolved"] == 8
    assert research["evidence_independence"]["evidence_days"] == 1
    assert research["signal_count"] == 8
    assert research["evidence_day_count"] == 1
    assert report["recommendation"]["auto_apply_safe"] is False
    assert report["doctrine_v2"]["evidence_independence"]["evidence_days"] == 1
    assert report["doctrine_v2"]["recommendation"]["auto_apply_safe"] is False


def test_adaptive_policy_keeps_non_overlapping_ticker_outcomes():
    records = [
        _research_record("IONQ", 70, "win", 2.0, 1),
        _research_record("IONQ", 70, "loss", -1.0, 12),
    ]

    report = build_adaptive_policy_report(records, current_research_score=65, min_research_samples=99)

    research = report["research_candidates"]
    assert research["independent_resolved"] == 2
    assert research["evidence_independence"]["overlap_rows_excluded"] == 0
    assert research["signal_count"] == 2


def test_adaptive_policy_reports_current_threshold_evidence_by_direction():
    records = [
        {**_research_record("BULL1", 70, "win", 2.0, 1), "direction": "bullish"},
        {**_research_record("BULL2", 70, "loss", -1.0, 2), "direction": "bullish"},
        {**_research_record("BEAR1", 70, "win", 4.0, 3), "direction": "bearish"},
        {**_research_record("BEAR2", 70, "win", 2.0, 4), "direction": "bearish"},
    ]

    report = build_adaptive_policy_report(records, current_research_score=65, min_research_samples=99)

    by_direction = report["research_candidates"]["current_threshold_by_direction"]
    assert by_direction["bullish"]["signal_count"] == 2
    assert by_direction["bullish"]["wins"] == 1
    assert by_direction["bullish"]["losses"] == 1
    assert by_direction["bullish"]["average_return_pct"] == 0.5
    assert by_direction["bearish"]["signal_count"] == 2
    assert by_direction["bearish"]["wins"] == 2
    assert by_direction["bearish"]["losses"] == 0
    assert by_direction["bearish"]["average_return_pct"] == 3.0


def test_metric_returns_prefer_barrier_outcome_over_close_horizon():
    records = [
        {**_research_record("A", 70, "loss", 3.0, 1), "outcome_return_pct": -2.0},  # stopped out, drifted green
        _research_record("B", 70, "loss", -1.0, 2),  # legacy row, close metric only
    ]

    report = build_adaptive_policy_report(records, current_research_score=62, min_research_samples=99)

    assert report["research_candidates"]["average_return_pct"] == -1.5


def _patch_apply_env(monkeypatch, tmp_path, current_score=60):
    overrides_path = tmp_path / "overrides.json"
    monkeypatch.setattr("scanner.learning.adaptive_policy.OVERRIDES_PATH", overrides_path)
    monkeypatch.setattr("scanner.learning.adaptive_policy.TUNING_DIR", tmp_path)
    monkeypatch.setattr("scanner.learning.adaptive_policy.record_trial", lambda kind, payload: None)
    monkeypatch.setattr(
        "scanner.learning.adaptive_policy.scanner_config.RESEARCH_CANDIDATE_MIN_SCORE",
        current_score,
    )
    return overrides_path


def test_apply_adaptive_overrides_merges_existing_tuning(monkeypatch, tmp_path):
    overrides_path = _patch_apply_env(monkeypatch, tmp_path, current_score=60)
    overrides_path.write_text(json.dumps({"MIN_RR": 1.7}), encoding="utf-8")

    result = apply_adaptive_overrides(
        {
            "recommendation": {
                "auto_apply_safe": True,
                "proposed_overrides": {"RESEARCH_CANDIDATE_MIN_SCORE": 67},
            }
        },
        logger=None,
    )

    assert result["status"] == "applied"
    written = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert written["MIN_RR"] == 1.7
    assert written["RESEARCH_CANDIDATE_MIN_SCORE"] == 67
    assert "last_auto_change_at" in written["_meta"]


def test_apply_adaptive_overrides_refreshes_runtime_config(monkeypatch, tmp_path):
    _patch_apply_env(monkeypatch, tmp_path, current_score=60)
    calls = []
    monkeypatch.setattr(
        "scanner.learning.adaptive_policy.scanner_config.reload_overrides",
        lambda: calls.append("reload"),
    )

    result = apply_adaptive_overrides(
        {
            "recommendation": {
                "auto_apply_safe": True,
                "proposed_overrides": {"RESEARCH_CANDIDATE_MIN_SCORE": 72},
            }
        },
        logger=None,
    )

    assert result["status"] == "applied"
    assert calls == ["reload"]


def test_apply_adaptive_overrides_does_not_record_noop_as_change(monkeypatch, tmp_path):
    overrides_path = _patch_apply_env(monkeypatch, tmp_path, current_score=65)
    original = {
        "RESEARCH_CANDIDATE_MIN_SCORE": 65,
        "_meta": {"last_auto_change_at": "2026-07-01T00:00:00+00:00"},
    }
    overrides_path.write_text(json.dumps(original), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "scanner.learning.adaptive_policy.scanner_config.reload_overrides",
        lambda: calls.append("reload"),
    )

    result = apply_adaptive_overrides(
        {
            "recommendation": {
                "auto_apply_safe": True,
                "proposed_overrides": {"RESEARCH_CANDIDATE_MIN_SCORE": 65},
            }
        },
        logger=None,
        now="2026-07-18T18:00:00Z",
    )

    assert result["status"] == "no_overrides_applied"
    assert json.loads(overrides_path.read_text(encoding="utf-8")) == original
    assert calls == []


def _loosen_deadlock_records():
    """The observed deadlock: threshold 72 starves (n=5, loss-heavy) while 65
    holds three times the samples with positive, regime-stable returns."""
    records = []
    day = 1
    mid_labels = ["win", "loss"] * 7 + ["win"] * 4
    for i, label in enumerate(mid_labels):
        ret = 1.8 if label == "win" else -1.2
        records.append(_research_record(f"MID{i}", 66, label, ret, (day := day + 1) % 28 or 1))
    for i in range(5):
        label = "win" if i == 0 else "loss"
        ret = 0.5 if label == "win" else -1.8
        records.append(_research_record(f"TOP{i}", 73, label, ret, (day := day + 1) % 28 or 1))
    return records


def test_adaptive_policy_loosens_when_lower_threshold_dominates():
    report = build_adaptive_policy_report(
        _loosen_deadlock_records(),
        current_research_score=72,
        min_research_samples=8,
    )

    assert report["recommendation"]["status"] == "loosen_research_threshold"
    assert report["recommendation"]["selected_threshold"] == 65
    assert report["recommendation"]["auto_apply_safe"] is True
    assert report["recommendation"]["proposed_overrides"] == {"RESEARCH_CANDIDATE_MIN_SCORE": 65}


def test_adaptive_policy_does_not_loosen_on_thin_challenger():
    records = _loosen_deadlock_records()[12:]  # challenger cohort now below 12 evidence days

    report = build_adaptive_policy_report(records, current_research_score=72, min_research_samples=8)

    assert report["recommendation"]["status"] != "loosen_research_threshold"


def test_loosening_requires_next_day_confirmation_then_applies(monkeypatch, tmp_path):
    overrides_path = _patch_apply_env(monkeypatch, tmp_path, current_score=72)
    monkeypatch.setattr("scanner.learning.adaptive_policy.scanner_config.reload_overrides", lambda: None)
    report = {
        "recommendation": {
            "status": "loosen_research_threshold",
            "auto_apply_safe": True,
            "proposed_overrides": {"RESEARCH_CANDIDATE_MIN_SCORE": 65},
        }
    }

    first = apply_adaptive_overrides(report, logger=None, now="2026-07-02T18:00:00Z")
    assert first["status"] == "pending_confirmation"
    written = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert "RESEARCH_CANDIDATE_MIN_SCORE" not in written
    assert written["_meta"]["pending_loosen"]["threshold"] == 65

    same_day = apply_adaptive_overrides(report, logger=None, now="2026-07-02T21:00:00Z")
    assert same_day["status"] == "pending_confirmation"

    next_day = apply_adaptive_overrides(report, logger=None, now="2026-07-03T18:00:00Z")
    assert next_day["status"] == "applied"
    written = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert written["RESEARCH_CANDIDATE_MIN_SCORE"] == 65
    assert "pending_loosen" not in written["_meta"]
    assert "last_auto_change_at" in written["_meta"]


def test_loosening_blocked_by_cooldown_after_recent_change(monkeypatch, tmp_path):
    overrides_path = _patch_apply_env(monkeypatch, tmp_path, current_score=72)
    overrides_path.write_text(
        json.dumps({"_meta": {"last_auto_change_at": "2026-07-01T00:00:00+00:00"}}),
        encoding="utf-8",
    )
    report = {
        "recommendation": {
            "status": "loosen_research_threshold",
            "auto_apply_safe": True,
            "proposed_overrides": {"RESEARCH_CANDIDATE_MIN_SCORE": 65},
        }
    }

    result = apply_adaptive_overrides(report, logger=None, now="2026-07-03T00:00:00Z")

    assert result["status"] == "cooldown_active"
    assert "RESEARCH_CANDIDATE_MIN_SCORE" not in json.loads(overrides_path.read_text(encoding="utf-8"))


def test_tightening_still_applies_immediately(monkeypatch, tmp_path):
    overrides_path = _patch_apply_env(monkeypatch, tmp_path, current_score=62)
    monkeypatch.setattr("scanner.learning.adaptive_policy.scanner_config.reload_overrides", lambda: None)

    result = apply_adaptive_overrides(
        {
            "recommendation": {
                "status": "tighten_research_threshold",
                "auto_apply_safe": True,
                "proposed_overrides": {"RESEARCH_CANDIDATE_MIN_SCORE": 67},
            }
        },
        logger=None,
        now="2026-07-02T18:00:00Z",
    )

    assert result["status"] == "applied"
    assert json.loads(overrides_path.read_text(encoding="utf-8"))["RESEARCH_CANDIDATE_MIN_SCORE"] == 67
