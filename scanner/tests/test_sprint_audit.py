import importlib.util
from pathlib import Path


def _audit_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "experiments"
        / "20260710_sprint"
        / "audit_sprint.py"
    )
    spec = importlib.util.spec_from_file_location("sprint_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _e4_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "experiments"
        / "20260710_sprint"
        / "e4_decision"
        / "run_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("sprint_e4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sprint_declared_and_serialized_cell_sets_match_exactly():
    result = _audit_module().audit()

    assert result["cell_integrity"]["all_cell_sets_match"] is True
    assert result["cell_integrity"]["serialized_verdict_count"] == 48


def test_sprint_count_reconciles_degenerate_kronos_cell():
    result = _audit_module().audit()

    assert result["count_reconciliation"]["serialized_verdicts"] == 48
    assert result["count_reconciliation"]["nondegenerate_verdicts"] == 47
    assert "kronos_only" in result["count_reconciliation"]["degenerate_verdict"]


def test_sprint_audit_does_not_claim_historical_exact_reproduction():
    result = _audit_module().audit()

    assert result["reproduction"]["historical_exact_rerun_possible"] is False
    assert result["provenance"]["preregistrations_and_results_added_in_one_git_commit"] is True


def test_e4_power_formula_honors_parameters_and_reproduces_registered_sizes():
    e4 = _e4_module()
    p0 = 13 / 29

    assert e4.one_sample_proportion_n(p0, p0 + 0.10) == 195
    assert e4.one_sample_proportion_n(p0, p0 + 0.05) == 780
    assert e4.one_sample_proportion_n(p0, p0 + 0.10, power=0.90) > 195


def test_e4_bootstrap_uses_five_day_moving_blocks():
    e4 = _e4_module()
    days = [f"2026-01-{day:02d}" for day in range(1, 16)]
    rows_by_day = {
        day: [(float(idx), float((idx % 5) - 2), day)]
        for idx, day in enumerate(days)
    }

    result = e4._bootstrap_deltas(rows_by_day, days, 0.33, 2.0, n_boot=20, seed=42)

    assert result["n_boot_used"] == 20
    assert result["bootstrap_method"] == "circular_moving_entry_day_blocks"
    assert result["bootstrap_block_days"] == 5
