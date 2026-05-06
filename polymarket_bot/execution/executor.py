from __future__ import annotations

import time


class ExecutionEngine:
    def __init__(self, dry_run: bool = True, max_retries: int = 3):
        self.dry_run = dry_run
        self.max_retries = max_retries

    def place_order(self, market_id: str, side: str, size_fraction: float) -> dict:
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.dry_run:
                    return {
                        "status": "simulated_filled",
                        "market_id": market_id,
                        "side": side,
                        "size_fraction": size_fraction,
                        "attempt": attempt,
                    }
                # TODO: gọi API thật của PolyMarket tại đây
                raise NotImplementedError("Live execution not implemented yet")
            except Exception as exc:  # retry policy
                if attempt == self.max_retries:
                    return {"status": "failed", "error": str(exc), "attempt": attempt}
                time.sleep(0.5 * attempt)
        return {"status": "failed", "error": "unknown"}
