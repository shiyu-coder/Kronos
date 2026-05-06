from __future__ import annotations

import pandas as pd

from model import Kronos, KronosPredictor, KronosTokenizer


class KronosAdapter:
    """Adapter gói inference từ shiyu-coder/Kronos."""

    def __init__(self, device: str = "cpu", max_context: int = 512):
        self.device = device
        self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=device, max_context=max_context)

    def predict_yes_probability(self, price_series: list[float], pred_len: int = 1) -> float:
        df = pd.DataFrame({"close": price_series})
        pred = self.predictor.predict(df, x_timestamp=None, pred_len=pred_len)
        current = float(price_series[-1])
        next_price = float(pred[0]) if len(pred) else current

        # Heuristic mapping price delta -> probability adjustment
        delta = next_price - current
        p = max(0.01, min(0.99, current + delta))
        return p
