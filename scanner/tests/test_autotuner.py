from scanner.learning.autotuner import propose_overrides


def test_autotune_holds_when_missed_winners_equal_correct_skips():
    records = []
    for i in range(28):
        records.append(
            {
                "ticker": f"W{i}",
                "final_pass": False,
                "stage_failed": "potter_box",
                "outcome_status": "resolved",
                "outcome_label": "win",
            }
        )
        records.append(
            {
                "ticker": f"L{i}",
                "final_pass": False,
                "stage_failed": "potter_box",
                "outcome_status": "resolved",
                "outcome_label": "loss",
            }
        )

    proposal = propose_overrides(records)
    assert proposal["status"] == "hold_no_edge"
    assert proposal["overrides"] == {}


def test_autotune_can_loosen_when_missed_win_rate_has_edge():
    records = []
    for i in range(40):
        records.append(
            {
                "ticker": f"W{i}",
                "final_pass": False,
                "stage_failed": "potter_box",
                "outcome_status": "resolved",
                "outcome_label": "win",
            }
        )
    for i in range(10):
        records.append(
            {
                "ticker": f"L{i}",
                "final_pass": False,
                "stage_failed": "potter_box",
                "outcome_status": "resolved",
                "outcome_label": "loss",
            }
        )

    proposal = propose_overrides(records)
    assert proposal["status"] == "ok"
    assert proposal["missed_win_rate"] == 0.8
    assert proposal["overrides"]["ATR_COMPRESSION"] > 0.75


def test_autotune_does_not_count_duplicate_resolved_setups():
    records = []
    for i in range(20):
        record = {
            "ticker": f"W{i}",
            "mode": "research_scan",
            "decision_ts": "2026-06-06T10:00:00-04:00",
            "direction": "bullish",
            "entry_price": float(i + 1),
            "final_pass": False,
            "stage_failed": "potter_box",
            "outcome_status": "resolved",
            "outcome_label": "win",
        }
        records.extend([record, dict(record)])

    proposal = propose_overrides(records)

    assert proposal["samples"] == 20
    assert proposal["duplicate_records_ignored"] == 20
