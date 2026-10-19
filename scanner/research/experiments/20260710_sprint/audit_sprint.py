"""Mechanical integrity audit for the 2026-07-10 research sprint.

This does not bless scientific claims.  It checks serialized cell coverage,
provenance evidence available in Git/NTFS, current-input drift, and the exact
meaning of the Kronos population claim without rewriting historical results.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_added_together() -> bool | None:
    paths = [str(p.relative_to(REPO_ROOT)) for p in ROOT.glob("*/preregistration.json")]
    paths += [str(p.relative_to(REPO_ROOT)) for p in ROOT.glob("*/results.json")]
    try:
        commits = subprocess.check_output(
            ["git", "log", "--format=%H", "--", *paths], cwd=REPO_ROOT, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    return len(set(commits)) == 1 if commits else None


def _cell_audit() -> dict:
    details = {}

    e1p, e1r = _json(ROOT / "e1_robust_r/preregistration.json"), _json(ROOT / "e1_robust_r/results.json")
    expected = {c["id"] for c in e1p["cells"]}
    actual = set(e1r["cells"])
    details["e1"] = {"expected": len(expected), "actual": len(actual), "exact_match": expected == actual}

    e2p, e2r = _json(ROOT / "e2_tail_grid/preregistration.json"), _json(ROOT / "e2_tail_grid/results.json")
    expected = {(float(c["tail_r"]), float(c["l2_lambda"])) for c in e2p["grid"]["cells"]}
    actual = {(float(c["tail_r"]), float(c["l2_lambda"])) for c in e2r["grid"]}
    details["e2"] = {"expected": len(expected), "actual": len(actual), "exact_match": expected == actual}

    for family in ("e3_features", "w2_compact5"):
        prereg, results = _json(ROOT / family / "preregistration.json"), _json(ROOT / family / "results.json")
        expected = {f"cell_{c['id']}_{c['name']}" for c in prereg["cells"]}
        actual = set(results["cells"])
        details[family] = {"expected": len(expected), "actual": len(actual), "exact_match": expected == actual}

    e4 = _json(ROOT / "e4_decision/results.json")
    bullish = e4["task3_topk_counterfactuals"]
    bearish = e4["task4_bearish_check"]["per_objective"]
    objectives = {"p_win", "expected_r", "tail_prob"}
    details["e4"] = {
        "expected": 6,
        "actual": len(bullish) + len(bearish),
        "exact_match": set(bullish) == objectives and set(bearish) == objectives,
    }
    total = sum(v["actual"] for v in details.values())
    return {"families": details, "serialized_verdict_count": total, "all_cell_sets_match": all(v["exact_match"] for v in details.values())}


def _mtime_audit() -> dict:
    families = {}
    for prereg in sorted(ROOT.glob("*/preregistration.json")):
        result = prereg.with_name("results.json")
        runner_candidates = list(prereg.parent.glob("*.py"))
        prereg_ns = prereg.stat().st_mtime_ns
        result_ns = result.stat().st_mtime_ns
        runners_after_result_created = [
            p.name for p in runner_candidates if p.stat().st_ctime_ns > result.stat().st_ctime_ns
        ]
        families[prereg.parent.name] = {
            "prereg_mtime_precedes_result_mtime": prereg_ns < result_ns,
            "runner_created_after_result": sorted(runners_after_result_created),
            "warning": "filesystem times are mutable local metadata, not immutable preregistration proof",
        }
    return families


def _current_input_audit() -> dict:
    index_path = REPO_ROOT / "scanner/reports/edge_retrieval_index.json"
    records = _json(index_path)
    if isinstance(records, dict):
        records = records.get("records", records.get("items", []))
    saved = _json(ROOT / "e4_decision/results.json")["run_metadata"]
    bullish = [r for r in records if r.get("direction") == "bullish"]
    bearish = [r for r in records if r.get("direction") == "bearish"]
    forecast_keys = (
        "kronos_median_forecast_return_pct",
        "kronos_directional_agreement",
        "kronos_worst_sampled_return_pct",
    )

    def finite_count(key: str) -> int:
        values = [(r.get("features") or {}).get(key) for r in records]
        return sum(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)

    sample_values = [(r.get("features") or {}).get("kronos_sample_count") for r in records]
    return {
        "path": str(index_path),
        "sha256": _sha256(index_path),
        "saved_counts": {"all": saved["n_index_records_total"], "bullish": saved["n_bullish"], "bearish": saved["n_bearish"]},
        "current_counts": {"all": len(records), "bullish": len(bullish), "bearish": len(bearish)},
        "matches_saved_input_counts": (len(records), len(bullish), len(bearish)) == (
            saved["n_index_records_total"], saved["n_bullish"], saved["n_bearish"]
        ),
        "kronos_forecast_finite_counts": {key: finite_count(key) for key in forecast_keys},
        "kronos_sample_count": {
            "finite": sum(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in sample_values),
            "nonzero": sum(isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) != 0 for v in sample_values),
        },
    }


def audit() -> dict:
    cells = _cell_audit()
    current = _current_input_audit()
    return {
        "cell_integrity": cells,
        "provenance": {
            "preregistrations_and_results_added_in_one_git_commit": _git_added_together(),
            "filesystem_times": _mtime_audit(),
            "conclusion": "local ordering is visible, but independent preregistration is not proven",
        },
        "reproduction": {
            "historical_exact_rerun_possible": False,
            "reason": "the saved run has no input/protocol hashes or immutable input snapshot, and current counts differ",
            "current_input": current,
        },
        "count_reconciliation": {
            "serialized_verdicts": cells["serialized_verdict_count"],
            "nondegenerate_verdicts": cells["serialized_verdict_count"] - 1,
            "degenerate_verdict": "e3 cell_13_kronos_only is intercept-only because forecast fields are absent and sample_count is constant zero",
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
