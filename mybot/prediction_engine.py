"""
Enhanced prediction engine with:
  - Technical Indicator Ensemble
  - Prediction Confidence Interval (Monte Carlo Dropout)
  - Market Regime Detection (HMM-inspired sliding window)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Literal, Tuple

import ta
from ta.trend import EMAIndicator, ADXIndicator, MACD as MACDIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Market Regimes
# ---------------------------------------------------------------------------
class MarketRegimeDetector:
    """Sliding-window regime classifier based on trend strength + volatility."""

    # Thresholds tuned for crypto/daily data
    TREND_STRONG: float = 25.0   # ADX above this → trending
    TREND_WEAK: float = 20.0     # ADX below this → ranging
    VOL_HIGH_Q: float = 2.0      # ATR % above this → volatile

    def __init__(self, lookback: int = 30) -> None:
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> str:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ADX
        adx_val = float(
            ADXIndicator(high=high, low=low, close=close, window=self.lookback).adx().iloc[-1]
        )
        # ATR %
        atr = float(
            AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]
        )
        atr_pct = atr / (np.mean(close[-self.lookback:]) + 1e-9) * 100

        if adx_val >= self.TREND_STRONG and atr_pct <= self.VOL_HIGH_Q:
            regime = "TRENDING_UP" if close[-1] > np.mean(close[-self.lookback:]) else "TRENDING_DOWN"
        elif adx_val <= self.TREND_WEAK:
            regime = "RANGING"
        else:
            regime = "VOLATILE"
        return regime

    def regime_confidence(self, df: pd.DataFrame) -> float:
        """Return 0-1 confidence that current regime is stable (low volatility of regime changes)."""
        windows = [14, 30, 60]
        scores = []
        for w in windows:
            if len(df) < w:
                continue
            adx_vals = ADXIndicator(
                high=df["high"], low=df["low"],
                close=df["close"], window=w
            ).adx().values[-w:]
            scores.append(float(np.std(adx_vals) / (np.mean(adx_vals) + 1e-9)))
        raw = 1 - min(float(np.mean(scores)), 1.0)
        return round(raw, 3)


# ---------------------------------------------------------------------------
# Ensemble engine
# ---------------------------------------------------------------------------
@dataclass
class EnsembleSignal:
    """Consolidated signal from Kronos + technical indicators."""
    # Core
    kronos_trend: float        # raw predicted % change (pred_end - pred_start) / pred_start
    signal: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    # Confidence
    confidence: float = 0.5     # 0..1
    lower_bound: float = 0.0   # predicted % at lower confidence
    upper_bound: float = 0.0   # predicted % at upper confidence
    # Indicators
    rsi: float = 50.0
    adx: float = 0.0
    macd_histogram: float = 0.0
    bb_width_pct: float = 0.0
    atr_pct: float = 0.0
    stochastic_k: float = 50.0
    # Regime
    regime: str = "UNKNOWN"
    regime_confidence: float = 0.5
    # Ensemble score
    ensemble_score: float = 0.0  # weighted combination, -1..1
    ensemble_direction: Literal["bullish", "bearish", "neutral"] = "neutral"


class TechnicalEnsemble:
    """
    Compute additional indicators + combine with Kronos trend to produce
    an enriched ensemble signal.
    """

    # Indicator weights (Kronos is most trusted)
    WEIGHTS = {
        "kronos":  0.45,
        "rsi":     0.10,
        "macd":    0.15,
        "adx":     0.10,
        "stoch":   0.10,
        "bb_width": 0.10,
    }

    def __init__(self, rsi_period: int = 14) -> None:
        self.rsi_period = rsi_period

    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        rsi_val = float(RSIIndicator(close=close, window=self.rsi_period).rsi().iloc[-1])

        # MACD
        macd = MACDIndicator(close=close)
        macd_line = float(macd.macd().iloc[-1])
        macd_signal_val = float(macd.macd_signal().iloc[-1])
        macd_hist = macd_line - macd_signal_val

        # ADX (trend strength)
        adx_val = float(ADXIndicator(high=high, low=low, close=close, window=14).adx().iloc[-1])

        # Bollinger Bands width
        bb = BollingerBands(close=close, window=20)
        bb_upper = float(bb.bollinger_hband().iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        bb_width_pct = (bb_upper - bb_lower) / (close.iloc[-1] + 1e-9) * 100

        # ATR %
        atr_inst = AverageTrueRange(high=high, low=low, close=close, window=14)
        atr = float(atr_inst.average_true_range().iloc[-1])
        atr_pct = atr / (close.iloc[-1] + 1e-9) * 100

        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close)
        stoch_k = float(stoch.stoch().iloc[-1])

        return {
            "rsi": rsi_val,
            "macd_histogram": macd_hist,
            "adx": adx_val,
            "bb_width_pct": bb_width_pct,
            "atr_pct": atr_pct,
            "stochastic_k": stoch_k,
        }

    def _sub_score(self, name: str, raw: float) -> float:
        """Normalise an indicator to -1..1 signal."""
        if name == "rsi":
            # >70 = overbought (bearish -1), <30 = oversold (bullish +1)
            return (70.0 - min(raw, 80)) / 50.0 - 1.0
        if name == "macd":
            # normalised by sign; magnitude capped at 1
            return max(-1.0, min(1.0, raw / (abs(raw) + 1e-9) * min(abs(raw) * 0.1, 1.0)))
        if name == "adx":
            # strong trend = more directional confidence (but direction comes from Kronos)
            return max(0.0, min(1.0, (raw - 20) / 40.0))
        if name == "stoch":
            # >80 overbought (-1), <20 oversold (+1)
            return (80.0 - min(raw, 90)) / 60.0 - 1.0
        if name == "bb_width":
            # higher width = more volatility (lower weight)
            return max(0.0, min(1.0, raw / 10.0))
        return 0.0

    def combine(
        self,
        kronos_trend_pct: float,
        indicators: Dict[str, float],
        regime: str,
    ) -> EnsembleSignal:
        w = self.WEIGHTS
        scores = {}
        for name in ["rsi", "macd", "adx", "stoch", "bb_width"]:
            scores[name] = self._sub_score(name, indicators.get(name, 0.0))

        # Kronos normalised: map % to -1..1 (rough, for crypto: ±5% → ±1)
        kronos_score = max(-1.0, min(1.0, kronos_trend_pct / 0.05))
        scores["kronos"] = kronos_score

        # Weighted sum
        ensemble = sum(w[k] * scores.get(k, 0.0) for k in w)

        # Direction
        if ensemble > 0.1:
            direction: Literal["bullish", "bearish", "neutral"] = "bullish"
            signal: Literal["BUY", "SELL", "HOLD"] = "BUY"
        elif ensemble < -0.1:
            direction = "bearish"
            signal = "SELL"
        else:
            direction = "neutral"
            signal = "HOLD"

        # Confidence: how aligned are the indicators?
        alignment = 1.0 - float(np.std(list(scores.values())) / 2.0)
        confidence = max(0.3, min(0.95, alignment))

        # Bounds: ± confidence band around trend
        spread = abs(kronos_trend_pct) * 0.5 + 0.005
        lower = kronos_trend_pct - spread * (1 - confidence)
        upper = kronos_trend_pct + spread * (1 - confidence)

        # Override signal in extreme regimes
        if regime in ("VOLATILE", "RANGING") and signal == "BUY":
            if indicators.get("rsi", 50) > 65:
                signal = "HOLD"
        if regime in ("VOLATILE", "RANGING") and signal == "SELL":
            if indicators.get("rsi", 50) < 35:
                signal = "HOLD"

        return EnsembleSignal(
            kronos_trend=kronos_trend_pct,
            signal=signal,
            confidence=round(confidence, 3),
            lower_bound=round(lower, 6),
            upper_bound=round(upper, 6),
            rsi=round(indicators.get("rsi", 50.0), 2),
            adx=round(indicators.get("adx", 0.0), 2),
            macd_histogram=round(indicators.get("macd_histogram", 0.0), 6),
            bb_width_pct=round(indicators.get("bb_width_pct", 0.0), 3),
            atr_pct=round(indicators.get("atr_pct", 0.0), 3),
            stochastic_k=round(indicators.get("stochastic_k", 50.0), 2),
            regime=regime,
            regime_confidence=round(confidence, 3),
            ensemble_score=round(ensemble, 4),
            ensemble_direction=direction,
        )


# ---------------------------------------------------------------------------
# Confidence interval via Monte Carlo sampling
# ---------------------------------------------------------------------------
def compute_confidence_interval(
    predictor,
    df: pd.DataFrame,
    x_timestamp: pd.DatetimeIndex,
    y_timestamp: pd.DatetimeIndex,
    pred_len: int,
    T: float,
    top_p: float,
    n_samples: int = 16,
) -> Tuple[float, float, float]:
    """
    Run prediction `n_samples` times with different temperatures/sampling,
    collect the final close prices, then return (mean, lower, upper) percentile.
    """
    if n_samples < 2:
        raw = predictor.predict(
            df, x_timestamp, y_timestamp, pred_len, T=T, top_p=top_p, sample_count=1, verbose=False
        )
        mean_close = float(raw["close"].iloc[-1])
        return mean_close, mean_close, mean_close

    last_closes: List[float] = []
    for i in range(n_samples):
        temp = T * (0.8 + 0.4 * (i / max(n_samples - 1, 1)))  # vary temperature
        raw = predictor.predict(
            df, x_timestamp, y_timestamp, pred_len,
            T=temp, top_p=top_p, sample_count=1, verbose=False
        )
        last_closes.append(float(raw["close"].iloc[-1]))

    mean_close = float(np.mean(last_closes))
    sorted_vals = sorted(last_closes)
    p10 = float(np.percentile(sorted_vals, 10))
    p90 = float(np.percentile(sorted_vals, 90))
    return mean_close, p10, p90


# ---------------------------------------------------------------------------
# Full enhanced prediction pipeline
# ---------------------------------------------------------------------------
def enhanced_prediction(
    df: pd.DataFrame,
    x_timestamp: pd.DatetimeIndex,
    y_timestamp: pd.DatetimeIndex,
    predictor,
    pred_len: int,
    T: float = 1.0,
    top_p: float = 0.9,
    regime_detector: MarketRegimeDetector | None = None,
    rsi_period: int = 14,
    use_confidence: bool = True,
    n_confidence_samples: int = 8,
    # --- NEW realtime params ---
    use_realtime: bool = True,
    realtime_limit: int = 50,
    hf_model: str = "ProsusAI/finbert",
    symbol: str = "Bitcoin",
) -> Dict[str, Any]:
    """
    Orchestrate: regime → indicators → Kronos MC sampling → ensemble → real-time sentiment.
    """
    rd = regime_detector or MarketRegimeDetector()

    # 1. Market regime
    regime = rd.detect(df)
    regime_conf = rd.regime_confidence(df)

    # 2. Technical indicators
    ens = TechnicalEnsemble(rsi_period=rsi_period)
    indicators = ens.compute_indicators(df)

    # 3. Kronos base prediction
    result = predictor.predict(
        df, x_timestamp, y_timestamp, pred_len, T=T, top_p=top_p,
        sample_count=1, verbose=False
    )

    last_price = float(df["close"].iloc[-1])
    pred_start_price = float(result["close"].iloc[0])
    pred_end_price = float(result["close"].iloc[-1])
    kronos_trend_pct = (pred_end_price - pred_start_price) / (pred_start_price + 1e-9)

    # 4. Monte Carlo confidence interval
    if use_confidence:
        mc_mean, mc_lower, mc_upper = compute_confidence_interval(
            predictor, df, x_timestamp, y_timestamp, pred_len, T, top_p,
            n_samples=n_confidence_samples
        )
        mc_trend_mean = (mc_mean - pred_start_price) / (pred_start_price + 1e-9)
        mc_trend_lower = (mc_lower - pred_start_price) / (pred_start_price + 1e-9)
        mc_trend_upper = (mc_upper - pred_start_price) / (pred_start_price + 1e-9)
    else:
        mc_mean = pred_end_price
        mc_lower = pred_end_price
        mc_upper = pred_end_price
        mc_trend_mean = kronos_trend_pct
        mc_trend_lower = kronos_trend_pct
        mc_trend_upper = kronos_trend_pct

    # 5. Ensemble signal (technical indicators + Kronos)
    signal_obj = ens.combine(mc_trend_mean, indicators, regime)

    # 6. Build history + forecast OHLC rows
    def _ohlc_row(ts, row) -> Dict[str, Any]:
        return {
            "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row else 0.0,
        }

    history = [_ohlc_row(r["timestamps"], r) for _, r in df.iterrows()]

    forecast = []
    for ts, r in result.iterrows():
        forecast.append(_ohlc_row(ts, r))

    # 7. Extra technical rows (for FE charts)
    extra_indicators = {
        # EMAIndicator expects a pandas Series (uses .ewm internally), not ndarray.
        "ema_20": float(EMAIndicator(close=df["close"], window=20).ema_indicator().iloc[-1]),
        "ema_50": float(EMAIndicator(close=df["close"], window=50).ema_indicator().iloc[-1]),
        "ema_200": float(EMAIndicator(close=df["close"], window=200).ema_indicator().iloc[-1])
            if len(df) >= 200 else None,
    }

    return {
        # --- Existing fields (backward compat) ---
        "symbol": None,  # filled by caller
        "timeframe": None,
        "signal": signal_obj.signal,
        "last_price": last_price,
        "pred_start_price": pred_start_price,
        "pred_end_price": pred_end_price,
        "trend": round(mc_trend_mean, 6),
        "rsi": signal_obj.rsi,
        "ema_fast": indicators.get("rsi", 50),
        "ema_slow": indicators.get("adx", 0),
        "history": history,
        "forecast": forecast,
        "meta": {},
        # --- NEW FIELDS ---
        # Confidence
        "confidence": signal_obj.confidence,
        "pred_mean_price": round(mc_mean, 4),
        "pred_lower_price": round(mc_lower, 4),
        "pred_upper_price": round(mc_upper, 4),
        "trend_lower_pct": round(mc_trend_lower, 6),
        "trend_upper_pct": round(mc_trend_upper, 6),
        # Regime
        "regime": regime,
        "regime_confidence": regime_conf,
        # Ensemble
        "ensemble_score": signal_obj.ensemble_score,
        "ensemble_direction": signal_obj.ensemble_direction,
        # Additional indicators
        "indicators": {
            "adx": signal_obj.adx,
            "macd_histogram": signal_obj.macd_histogram,
            "bb_width_pct": signal_obj.bb_width_pct,
            "atr_pct": signal_obj.atr_pct,
            "stochastic_k": signal_obj.stochastic_k,
            **extra_indicators,
        },
        # --- NEW: Real-time data ---
        "realtime": _inject_realtime(use_realtime, realtime_limit, hf_model, symbol,
                                     mc_trend_mean, signal_obj.signal, indicators, regime),
        # Metadata
        "meta_extra": {
            "use_confidence_sampling": use_confidence,
            "n_confidence_samples": n_confidence_samples,
            "use_realtime": use_realtime,
        },
    }


def _inject_realtime(
    use_realtime: bool,
    realtime_limit: int,
    hf_model: str,
    symbol: str,
    kronos_trend: float,
    base_signal: str,
    indicators: Dict[str, float],
    regime: str,
) -> Dict[str, Any]:
    """Fetch real-time data and inject into prediction result."""
    if not use_realtime:
        return {
            "enabled": False,
            "sentiment_overall": 0.0,
            "confidence": 0.5,
            "fear_greed": 0.0,
            "news_count": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "macro_events": [],
            "realtime_signal": base_signal,
        }

    try:
        from mybot.realtime_data import RealTimeDataAggregator
        agg = RealTimeDataAggregator(
            hf_model=hf_model,
            fear_greed_cache_ttl=300,
            news_cache_ttl=600,
        )
        pulse = agg.get_pulse(symbol=symbol, limit=realtime_limit)

        # Override signal if sentiment strongly contradicts
        rt_signal = base_signal
        strong_sentiment = abs(pulse.sentiment.overall) > 0.4
        if strong_sentiment:
            if pulse.sentiment.overall > 0.3 and base_signal == "SELL":
                rt_signal = "HOLD"  # Override SELL if news is bullish
            elif pulse.sentiment.overall < -0.3 and base_signal == "BUY":
                rt_signal = "HOLD"  # Override BUY if news is bearish

        # Boost confidence when sentiment aligns
        aligned = (pulse.sentiment.overall > 0 and kronos_trend > 0) or \
                  (pulse.sentiment.overall < 0 and kronos_trend < 0)
        conf_boost = 0.12 if aligned else -0.08

        return {
            "enabled": True,
            "sentiment_overall": round(pulse.sentiment.overall, 3),
            "sentiment_confidence": round(pulse.sentiment.confidence, 3),
            "fear_greed": round(pulse.sentiment.fear_greed_index * 2 - 1, 3),  # -1..1
            "fear_greed_raw": round(pulse.sentiment.fear_greed_index * 100, 1),  # 0..100
            "news_count": pulse.sentiment.news_count,
            "bullish_count": pulse.sentiment.bullish_count,
            "bearish_count": pulse.sentiment.bearish_count,
            "neutral_count": pulse.sentiment.neutral_count,
            "avg_polarity": round(pulse.sentiment.avg_polarity, 4),
            "macro_events": pulse.macro_events,
            "realtime_signal": rt_signal,
            "confidence_boost": round(conf_boost, 3),
            "confidence_boost_reason": "aligned" if aligned else "diverged",
            "news_sources": list(set(
                src for src in
                (pulse.sentiment.raw_sources.get("sources", []) or [])
                if src
            ))[:5],
            "on_chain": pulse.on_chain,
        }
    except Exception as e:
        return {
            "enabled": False,
            "error": str(e),
            "sentiment_overall": 0.0,
            "confidence": 0.5,
            "realtime_signal": base_signal,
        }