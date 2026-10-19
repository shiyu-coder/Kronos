"""Append every tested cell to the trial registry (multiple-testing honesty).

All 56 cells are logged, not just the interesting ones - the registry exists so
a future reader can see the full denominator behind any claim made from this
sweep. Atomic append, matching the rest of the lab's registry writes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).parent
REGISTRY = REPO_ROOT / "scanner" / "reports" / "trial_registry.jsonl"

T_SUGGESTIVE = 2.0
T_BONFERRONI = 3.2  # two-sided 0.05 / 28 pre-registered cells per direction


def verdict(cell: dict, robustness: dict | None) -> str:
    net = cell.get("net_r")
    t = cell.get("t_day_clustered", 0.0)
    margin_t = cell.get("margin_t", 0.0)
    margin = cell.get("margin", 0.0)
    if net is None or net <= 0:
        return "fail_net_negative"
    if t < T_SUGGESTIVE:
        return "fail_not_significant"
    if margin <= 0:
        return "exposure_not_edge"
    if t < T_BONFERRONI:
        return "suggestive_only"
    if margin_t < T_SUGGESTIVE:
        return "exposure_not_edge"
    return "pass"


def main() -> int:
    results = json.loads((HERE / "results.json").read_text())
    robustness = {
        (r["horizon"], r["target"]): r
        for r in json.loads((HERE / "robustness_return_space.json").read_text())
    }
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    counts: dict[str, int] = {}
    for cell in results["cells"]:
        v = verdict(cell, robustness.get((cell["horizon"], cell["target"])))
        counts[v] = counts.get(v, 0) + 1
        rob = robustness.get((cell["horizon"], cell["target"]))
        lines.append(json.dumps({
            "recorded_at": now,
            "kind": "horizon_geometry_trial",
            "experiment": "20260813_horizon_sweep",
            "preregistration": "scanner/research/experiments/20260813_horizon_sweep/preregistration.json",
            "variant": {"horizon_bars": cell["horizon"], "target_r": cell["target"],
                        "direction": cell["direction"]},
            "cohort": "fixed_resolvable_at_max_horizon",
            "metrics": {
                "n": cell["n"], "n_days": cell["n_days"],
                "target_hit_pct": round(cell["target_hit_pct"], 3),
                "win_rate_pct": round(cell["win_rate_pct"], 3),
                "gross_r": round(cell["gross_r"], 5),
                "net_r": round(cell["net_r"], 5),
                "t_day_clustered": cell["t_day_clustered"],
                "hac_method": cell["hac_method"],
                "placebo_net_r": round(cell["placebo_net_r"], 5),
                "margin_vs_placebo_r": round(cell["margin"], 5),
                "margin_t": cell["margin_t"],
                "margin_vs_placebo_return_pct": (
                    round(rob["margin_pct"], 5) if rob else None),
                "margin_return_pct_t": rob["margin_t"] if rob else None,
            },
            "gates": {"t_suggestive": T_SUGGESTIVE, "t_bonferroni": T_BONFERRONI,
                      "requires_positive_placebo_margin": True},
            "outcome": v,
        }))

    with REGISTRY.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"appended {len(lines)} cells to {REGISTRY}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
