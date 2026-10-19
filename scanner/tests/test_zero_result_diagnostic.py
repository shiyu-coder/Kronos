import json

import scanner.main as scanner_main


def test_zero_result_diagnostic_quantifies_strict_gate_starvation_and_research_outcomes(monkeypatch, tmp_path):
    rows = [
        {
            "ticker": "AAA",
            "mode": "dry_run",
            "final_pass": False,
            "stage_failed": "potter_box",
            "skip_reason": "no valid Potter Box breakout/breakdown",
            "outcome_status": "resolved",
            "outcome_label": "loss",
            "outcome_ret_5bar_pct": -3.2,
            "research_score": 68,
            "research_diagnostics": {
                "passed": True,
                "reason": "research_candidate",
                "reasons": ["confirmed_breakdown", "volume_expansion"],
            },
        },
        {
            "ticker": "BBB",
            "mode": "dry_run",
            "final_pass": False,
            "stage_failed": "potter_box",
            "skip_reason": "no valid Potter Box breakout/breakdown",
            "outcome_status": "resolved",
            "outcome_label": "win",
            "outcome_ret_5bar_pct": 1.1,
            "research_score": 65,
            "research_diagnostics": {
                "passed": True,
                "reason": "research_candidate",
                "reasons": ["confirmed_breakout"],
            },
        },
        {
            "ticker": "CCC",
            "mode": "dry_run",
            "final_pass": False,
            "stage_failed": "potter_box",
            "skip_reason": "no valid Potter Box breakout/breakdown",
            "outcome_status": "not_applicable",
            "research_score": 42,
            "research_diagnostics": {
                "passed": False,
                "reason": "not near box edge",
                "reasons": [],
            },
        },
        {
            "ticker": "DDD",
            "mode": "research_scan",
            "final_pass": False,
            "stage_failed": "potter_box_research",
            "skip_reason": "research_candidate",
            "outcome_status": "resolved",
            "outcome_label": "loss",
            "outcome_ret_5bar_pct": -0.8,
            "research_score": 73,
            "research_diagnostics": {
                "passed": True,
                "reason": "research_candidate",
                "reasons": ["confirmed_breakout", "strong_close_location"],
            },
        },
        {
            "ticker": "EEE",
            "mode": "dry_run",
            "final_pass": False,
            "stage_failed": "validation",
            "skip_reason": "price below $5.00",
            "outcome_status": "not_applicable",
        },
    ]
    monkeypatch.setattr(scanner_main, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(scanner_main, "load_decisions", lambda: rows)

    report = scanner_main._write_zero_result_diagnostic(scanner_main.setup_logging(tmp_path))

    assert report["final_pass_counts"] == {"fail": 5}
    assert report["strict_path"]["records"] == 4
    assert report["strict_path"]["stage_counts"]["potter_box"] == 3
    assert report["research_candidates"]["records"] == 3
    assert report["research_candidates"]["resolved_outcomes"] == {"loss": 2, "win": 1}
    assert report["research_candidates"]["resolved_win_rate"] == 1 / 3
    assert report["research_score_buckets"]["62_to_69"] == 2
    assert report["research_award_counts"]["confirmed_breakout"] == 2
    assert report["potter_research_reason_counts"]["research_candidate"] == 2
    assert report["diagnostic_summary"]["primary_bottleneck_stage"] == "potter_box"
    assert report["diagnostic_summary"]["research_edge_status"] == "loss_heavy"
    assert report["diagnostic_summary"]["recommended_live_gate_action"] == "do_not_loosen_without_validated_edge"

    saved = json.loads((tmp_path / "zero_result_diagnostic.json").read_text(encoding="utf-8"))
    assert saved["diagnostic_summary"] == report["diagnostic_summary"]


def test_zero_result_diagnostic_prioritizes_strict_path_for_primary_bottleneck(monkeypatch, tmp_path):
    rows = [
        {
            "ticker": "AAA",
            "mode": "dry_run",
            "final_pass": False,
            "stage_failed": "potter_box",
            "skip_reason": "no valid Potter Box breakout/breakdown",
            "outcome_status": "not_applicable",
        },
        {
            "ticker": "BBB",
            "mode": "research_scan",
            "final_pass": False,
            "stage_failed": "potter_box_research",
            "skip_reason": "score below research threshold",
            "outcome_status": "not_applicable",
        },
        {
            "ticker": "CCC",
            "mode": "research_scan",
            "final_pass": False,
            "stage_failed": "potter_box_research",
            "skip_reason": "not near box edge",
            "outcome_status": "not_applicable",
        },
    ]
    monkeypatch.setattr(scanner_main, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(scanner_main, "load_decisions", lambda: rows)

    report = scanner_main._write_zero_result_diagnostic(scanner_main.setup_logging(tmp_path))

    assert report["stage_counts"]["potter_box_research"] == 2
    assert report["strict_path"]["stage_counts"] == {"potter_box": 1}
    assert report["diagnostic_summary"]["primary_bottleneck_stage"] == "potter_box"
