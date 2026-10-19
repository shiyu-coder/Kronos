# Kronos Edge Engine Design

Date: 2026-05-14
Status: Approved for implementation planning

## Purpose

Kronos Predictor should become an evidence-driven market research and alert engine. Its job is not to create more alerts by loosening rules. Its job is to discover trade candidates, estimate whether similar setups have historically produced positive expectancy, quantify uncertainty, and only promote ideas that survive repeatable validation.

This design does not promise trading profits. It creates a stronger research system for finding, rejecting, and ranking candidate trades before any live alert is trusted.

## Current Baseline

The scanner test suite passes, but the trading system is not performing well enough:

- Scanner tests: `16 passed`.
- Intraday 60 day report: `0` signals, `750` skipped windows, all skipped by Potter Box.
- Daily 2 year proxy report: `0` signals, `13,950` skipped windows, nearly all skipped by Potter Box.
- Replay report: `0.0` precision and `0.0` recall on the current sample.
- Decision journal: `28` missed winners and `28` correct skips, so current autotune correctly refuses to loosen thresholds.
- Root pytest currently has environment/model issues: missing `matplotlib` during collection and NaN logits in Kronos MSE regression tests.

The primary root cause is a brittle binary signal gate with too little validated evidence. The secondary root cause is insufficient model/runtime hardening.

## Guiding Principle

The project should optimize for validated expectancy, not signal count. A candidate should advance only when the system can explain:

- Why this setup exists now.
- Which historical analogs resemble it.
- What happened after those analogs.
- How uncertain the forecast is.
- Whether data quality and liquidity are good enough to act on.
- Whether the edge survives walk-forward validation.

## Architecture

### 1. Stability Foundation

Fix the current execution and test baseline before adding new model logic.

- Make root test collection reproducible in the local venv.
- Harden Kronos inference against NaN/Inf logits and invalid probability tensors.
- Improve replay datasets so they contain enough bars to exercise Potter, Empty Space, and future-outcome paths.
- Preserve fail-closed behavior for live alerts.

### 2. Feature Engine

Convert each scan window into a stable feature vector that can be logged, retrieved, ranked, and validated.

Feature families:

- Potter geometry: box width, close position, breakout distance, top/bottom touches, cost-basis bias.
- Compression: ATR compression, range compression, slope/no-trend score, realized volatility.
- Volume and participation: breakout volume expansion, recent volume percentile, abnormal volume score.
- Relative behavior: return versus watchlist, return versus broad market proxy when available, momentum regime.
- Risk and structure: empty-space reward/risk, invalidation distance, recent support/resistance density.
- Data quality: provider, feed type, missing bars, stale data, synthetic session anchor, calibration status.
- Options quality: spread, open interest, volume, DTE, indicative versus OPRA feed when known.
- Model signals: Kronos agreement, median forecast return, worst sampled return, inference status.

The feature engine should be deterministic and unit-tested.

### 3. Retrieval-Augmented Forecasting

Add a local analog search layer inspired by retrieval-augmented time-series forecasting. For every current setup:

- Build a query vector from the feature engine.
- Search historical candidate windows for similar market states.
- Exclude future-leaking windows with purged/embargoed rules.
- Return nearest analogs with their forward returns, max adverse excursion, max favorable excursion, and R-multiple.
- Estimate empirical expectancy and downside risk from analog outcomes.

This gives the scanner a memory: "What happened after setups like this?"

### 4. Calibrated Edge Scoring

Replace the all-or-nothing Potter decision with a ranked candidate model while keeping strict live gates.

The edge score should combine:

- Potter/research setup quality.
- Empty-space reward/risk.
- Historical analog expectancy.
- Analog sample size and dispersion.
- Kronos directional agreement and forecast distribution.
- Uncertainty penalty.
- Options liquidity penalty.
- Data-quality penalty.
- Event-risk penalty.

The output should include a transparent scorecard, not only a final boolean.

### 5. Uncertainty and Risk Layer

Do not treat a model forecast as truth. Estimate whether the system knows enough to act.

Initial implementation should use practical uncertainty proxies:

- Kronos sample dispersion.
- Analog outcome dispersion.
- Analog sample count.
- Data-feed quality penalty.
- Recent volatility regime shift penalty.

Future extension can evaluate explicit probabilistic methods such as evidential uncertainty or conformal intervals once a larger local dataset exists.

### 6. Validation Lab

Add a validation mode that measures ranked candidate quality, not only final pass/fail counts.

Required metrics:

- Signal count by threshold.
- Precision, recall, and false-negative rate.
- Precision at top-K candidates per scan.
- Average forward return.
- Median forward return.
- Average R-multiple.
- Max adverse excursion.
- Max favorable excursion.
- Drawdown proxy.
- Expectancy after configurable slippage.
- Breakdown by ticker, direction, regime, and failure stage.

Validation must support walk-forward or purged splits so the retrieval layer cannot cheat by seeing nearby future data.

### 7. Live Promotion Rules

Live alerts remain fail-closed. A signal can be promoted only if:

- Hard validation passes.
- Data feed is acceptable for the intended use.
- Options liquidity passes.
- Event risk passes or is explicitly allowed.
- Edge score clears a configured threshold.
- Historical analog sample size is large enough.
- Expected R-multiple and downside risk clear configured thresholds.
- Kronos inference is healthy or explicitly marked as unavailable with a conservative penalty.

Research candidates may be logged even when live promotion fails.

## Data Strategy

The current free/limited data stack is useful for development but should be treated as lower-confidence:

- Alpaca Basic live equities data uses IEX rather than full SIP.
- Alpaca free options data can be indicative rather than true OPRA.
- yfinance is convenient but not a trading-grade source of truth.

The system should record feed provenance and penalize uncertain data. If better data credentials are available later, the same pipeline should improve without changing strategy code.

## Modes

Add or evolve scanner modes:

- `research_scan`: collect candidates and feature vectors without live alerts.
- `build_retrieval_index`: generate historical feature/outcome windows.
- `edge_scan`: rank current candidates using analog retrieval and model scorecards.
- `validate_edge`: run walk-forward validation and write reports.
- `diagnose_edge`: explain bottlenecks, missing data, weak scores, and promotion failures.

Existing dry-run/live behavior should remain compatible.

## Testing Strategy

Tests should cover:

- Feature extraction on synthetic bullish, bearish, invalid, missing-data, and low-liquidity cases.
- Retrieval index construction without future leakage.
- Analog ranking determinism.
- Edge score behavior when analog expectancy is positive, negative, uncertain, or unavailable.
- NaN-safe Kronos inference fallback.
- Replay evaluation with enough bars to exercise a complete candidate.
- Validation metrics on known toy datasets.
- Existing scanner gates to prevent accidental live-alert relaxation.

## Implementation Boundaries

This implementation should not:

- Turn on live trading.
- Send live Telegram alerts unless existing live gates already allow them.
- Claim profitability from toy backtests.
- Replace rigorous validation with an LLM narrative.
- Depend on paid data being present.

This implementation should:

- Improve the project locally with deterministic code and tests.
- Produce richer reports that can guide real trading research.
- Keep the system honest when evidence is weak.

## Success Criteria

The first implementation is successful when:

- Scanner tests pass.
- Root test collection is either fixed or documented with a precise remaining dependency/model blocker.
- A valid replay dataset exercises the full candidate path.
- Historical candidate windows can be indexed into feature vectors.
- `edge_scan` produces ranked candidates with scorecards.
- `validate_edge` reports top-K and threshold metrics.
- Existing zero-signal diagnosis is replaced by actionable evidence: whether the system lacks candidates, lacks edge, lacks data quality, or lacks model confidence.

## Research Inputs

The design is informed by:

- Retrieval-Augmented Time Series Forecasting (RAFT): retrieval adds historical pattern memory to forecasting.
- Retrieval-augmented financial time-series forecasting / StockLLM-style work: financial prediction benefits from specialized historical retrieval.
- Probabilistic time-series foundation model research: uncertainty matters as much as point forecasts.
- Trade-flow and market microstructure foundation model research: scale-invariant market representations are a frontier direction, but current data access limits direct adoption.
- Alpaca market data documentation: IEX/indicative feeds require explicit data-quality handling.
