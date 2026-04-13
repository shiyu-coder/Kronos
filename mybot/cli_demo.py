"""
CLI demo: one-shot forecast + matplotlib chart.

Uses the same pipeline as the API and web UI: :mod:`mybot.prediction`.

From the repository root::

    python -m mybot.cli_demo
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mybot.prediction import PredictParams, run_prediction


def main() -> None:
    params = PredictParams(
        symbol="BTC/USDT",
        timeframe="1d",
        limit=300,
        window_size=200,
        pred_len=30,
        rsi_period=14,
        ema_fast=50,
        ema_slow=200,
        model_id="NeoQuasar/Kronos-base",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        verbose=False,
    )

    out = run_prediction(params)

    print("Symbol:", out["symbol"])
    print("Last Price:", out["last_price"])
    print("Predicted end close:", out["pred_end_price"])
    print("Trend:", out["trend"])
    print("RSI:", out["rsi"])
    print("EMA fast / slow:", out["ema_fast"], "/", out["ema_slow"])
    print("SIGNAL:", out["signal"])

    hist_close = [h["close"] for h in out["history"]]
    pred_close = [h["close"] for h in out["forecast"]]

    plt.figure()
    plt.plot(range(len(hist_close)), hist_close, label="Real Price")
    plt.plot(range(len(hist_close), len(hist_close) + len(pred_close)), pred_close, label="Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
