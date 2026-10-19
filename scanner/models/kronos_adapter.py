from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config as scanner_config
from ..config import KRONOS_LOOKBACK_BARS, KRONOS_MIN_BARS, KRONOS_SAMPLE_COUNT, PRED_DAYS
from ..data.market_data import compute_future_timestamps
from ..utils.validation import KronosResult


class KronosAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._predictor = None

    def _load_once(self):
        if self._predictor is not None:
            return self._predictor

        scanner_root = Path(__file__).resolve().parents[1]
        parent_root = scanner_root.parent
        if str(parent_root) not in sys.path:
            sys.path.insert(0, str(parent_root))

        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
            from ..config import KRONOS_MODEL_NAME, KRONOS_TOKENIZER_NAME

            tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_NAME)
            model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
            predictor = KronosPredictor(model, tokenizer, max_context=512)
            self._predictor = predictor
            self.logger.info("Kronos loaded successfully.")
            return self._predictor
        except Exception as exc:
            raise RuntimeError(f"failed to load Kronos: {exc}") from exc

    @staticmethod
    def _format_features(synthetic_bars: pd.DataFrame) -> pd.DataFrame:
        bars = synthetic_bars[["Open", "High", "Low", "Close", "Volume"]].copy()
        bars.columns = ["open", "high", "low", "close", "volume"]
        bars["amount"] = bars["close"] * bars["volume"].fillna(0.0)
        return bars

    def evaluate(self, ticker: str, synthetic_bars: pd.DataFrame, direction: str) -> KronosResult:
        try:
            predictor = self._load_once()
        except Exception as exc:
            self.logger.error("Kronos load failure for %s: %s", ticker, exc)
            return KronosResult(
                passed=False,
                output_mode="error",
                directional_agreement=None,
                median_forecast_return_pct=None,
                worst_sampled_return_pct=None,
                sample_count=0,
                skip_reason=str(exc),
            )

        try:
            if len(synthetic_bars) < KRONOS_MIN_BARS:
                return KronosResult(False, "insufficient_context", None, None, None, 0, f"need {KRONOS_MIN_BARS} synthetic bars")

            features = self._format_features(synthetic_bars.tail(KRONOS_LOOKBACK_BARS))
            # KronosPredictor uses the .dt accessor, so timestamps must be
            # Series, not DatetimeIndex. This never surfaced before because
            # the strict pipeline never reached the Kronos stage.
            x_timestamp = pd.Series(features.index)
            y_timestamp = pd.Series(compute_future_timestamps(features.index[-1], PRED_DAYS))

            paths: list[pd.DataFrame] = []
            for _ in range(KRONOS_SAMPLE_COUNT):
                pred_df = predictor.predict(
                    df=features,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=PRED_DAYS,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False,
                )
                if not isinstance(pred_df, pd.DataFrame) or "close" not in pred_df.columns:
                    return KronosResult(
                        passed=False,
                        output_mode="unknown",
                        directional_agreement=None,
                        median_forecast_return_pct=None,
                        worst_sampled_return_pct=None,
                        sample_count=0,
                        skip_reason="Kronos output format unknown",
                        output_type=str(type(pred_df)),
                        output_shape=getattr(pred_df, "shape", None),
                    )
                paths.append(pred_df)

            latest_close = float(features["close"].iloc[-1])
            final_returns = []
            agree = []
            for path in paths:
                ret = ((float(path["close"].iloc[-1]) - latest_close) / latest_close) * 100.0
                final_returns.append(ret)
                agree.append(ret > 0 if direction == "bullish" else ret < 0)

            if len(final_returns) > 1:
                directional_agreement = float(np.mean(agree))
                median_ret = float(np.median(final_returns))
                worst_ret = float(np.min(final_returns)) if direction == "bullish" else float(np.max(final_returns))
                passed = directional_agreement >= scanner_config.MIN_KRONOS_AGREEMENT
                return KronosResult(
                    passed=passed,
                    output_mode="multi_path_agreement",
                    directional_agreement=directional_agreement,
                    median_forecast_return_pct=median_ret,
                    worst_sampled_return_pct=worst_ret,
                    sample_count=len(final_returns),
                    skip_reason=None if passed else f"directional agreement {directional_agreement:.2%} < {scanner_config.MIN_KRONOS_AGREEMENT:.0%}",
                    output_type="list[pd.DataFrame]",
                    output_shape=(len(paths), len(paths[0]), len(paths[0].columns)),
                )

            single_ret = final_returns[0]
            aligned = (single_ret > 0 and direction == "bullish") or (single_ret < 0 and direction == "bearish")
            return KronosResult(
                passed=aligned,
                output_mode="forecast_alignment",
                directional_agreement=1.0 if aligned else 0.0,
                median_forecast_return_pct=single_ret,
                worst_sampled_return_pct=single_ret,
                sample_count=1,
                skip_reason=None if aligned else "single-path forecast misaligned",
                output_type="pd.DataFrame",
                output_shape=(len(paths[0]), len(paths[0].columns)),
            )

        except Exception as exc:
            self.logger.error("Kronos inference error for %s: %s", ticker, exc)
            return KronosResult(
                passed=False,
                output_mode="error",
                directional_agreement=None,
                median_forecast_return_pct=None,
                worst_sampled_return_pct=None,
                sample_count=0,
                skip_reason=f"Kronos error: {exc}",
                output_type="exception",
                output_shape=None,
            )
