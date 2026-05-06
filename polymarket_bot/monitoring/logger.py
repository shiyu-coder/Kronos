from __future__ import annotations

import json
from datetime import datetime, timezone


class TradeLogger:
    def __init__(self, log_path: str = "polymarket_bot_trades.log"):
        self.log_path = log_path

    def log(self, payload: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
