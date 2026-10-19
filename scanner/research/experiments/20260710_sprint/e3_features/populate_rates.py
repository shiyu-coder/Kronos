"""Inspect populate-rates for candidate features on bullish edge-index records.

Read-only inspection script. Does not write anything under scanner/ except
this experiment folder's own output files.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4].parent))

from scanner.edge.retrieval import load_edge_index
from scanner.config import EDGE_INDEX_PATH

CANDIDATES = [
    # standard 10
    "volume_expansion",
    "volume_percentile",
    "breakout_strength_pct",
    "close_position_in_box",
    "box_width_pct",
    "range_compression_ratio",
    "realized_volatility_pct",
    "no_trend_score",
    "doctrine_v2_score",
    "recent_return_pct",
    # extended candidate list from the task
    "abs_breakout_distance_pct",
    "atr_value",
    "bar_count",
    "bottom_touches",
    "top_touches",
    "breakout_distance_pct",
    "distance_to_target_pct",
    "doctrine_v2_box_stack_score",
    "empty_space_score",
    "kronos_median_forecast_return_pct",
    "kronos_directional_agreement",
    "kronos_sample_count",
    "kronos_worst_sampled_return_pct",
    "research_score",
    "data_quality_score",
    "feed_confidence",
    # excluded always
    "rr_ratio",
]


def _finite(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def main() -> None:
    records = load_edge_index(EDGE_INDEX_PATH)
    bullish = [r for r in records if r.direction == "bullish"]
    print(f"total records: {len(records)}")
    print(f"bullish records: {len(bullish)}")

    rows = []
    for key in CANDIDATES:
        finite_count = sum(1 for r in bullish if _finite(r.features.get(key)))
        rate = finite_count / len(bullish) if bullish else 0.0
        # constant check among finite values
        values = [float(r.features[key]) for r in bullish if _finite(r.features.get(key))]
        distinct = len(set(round(v, 8) for v in values)) if values else 0
        rows.append(
            {
                "key": key,
                "finite_count": finite_count,
                "n_bullish": len(bullish),
                "populate_rate": round(rate, 4),
                "distinct_values": distinct,
                "constant": distinct <= 1,
            }
        )

    for row in sorted(rows, key=lambda r: -r["populate_rate"]):
        print(
            f"{row['key']:40s} rate={row['populate_rate']:.4f} "
            f"n_finite={row['finite_count']:5d}/{row['n_bullish']:5d} "
            f"distinct={row['distinct_values']:5d} constant={row['constant']}"
        )

    out_path = Path(__file__).resolve().parent / "populate_rates.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
