from __future__ import annotations


def compute_rr(entry: float, target: float, invalidation: float, direction: str) -> tuple[float, float, float]:
    if direction == "bullish":
        reward = max(target - entry, 0.0)
        risk = max(entry - invalidation, 1e-9)
    else:
        reward = max(entry - target, 0.0)
        risk = max(invalidation - entry, 1e-9)
    rr = reward / risk if risk > 0 else 0.0
    return rr, reward, risk
