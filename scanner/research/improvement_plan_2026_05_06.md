# Improvement Plan - 2026-05-06

## Current Evidence
- Decision records reviewed: 180.
- Resolved outcomes: 56.
- Resolved counterfactual split: 28 wins / 28 losses.
- Strict alert signals remain zero.
- Primary bottleneck: Potter Box gate.
- New research scan produced 6 pending graded candidates: MARA, T, NIO, CLSK, PLTR, IONQ.

## Research Takeaways
- Alpaca free stock data uses IEX rather than full SIP for live coverage, so volume and prints can differ materially from consolidated market data. This matters for breakout and volume-expansion filters.
- Alpaca option chain snapshots expose OPRA when subscribed and free indicative data otherwise; option liquidity checks should be treated as point-in-time, not ground truth.
- VCP/breakout literature emphasizes progressive volatility contraction, constructive consolidation, and volume expansion on breakout. A binary detector is too brittle for early research.
- Support/resistance research supports counting repeated bounces/touches, but level decay and time since bounce matter. A graded level quality score is better than a hard touch count alone.
- Trading ML validation should use walk-forward or purged/embargoed validation to reduce leakage and overfit risk.

## Implemented Changes
- Added `research_scan` mode for graded near-miss candidate collection.
- Added `diagnose_zero_results` mode to summarize bottlenecks and outcome quality.
- Added Potter research candidate scoring with edge proximity, compression, touch, cost-basis, volume, and close-location components.
- Fixed autotune logic so tied missed winners/correct skips do not loosen live gates.
- Extended config overrides to include Potter and research-threshold parameters.

## Operating Loop
1. Run `research_scan` daily to collect graded candidates.
2. Run `review_outcomes` after candidates age past `OUTCOME_MIN_AGE_DAYS`.
3. Run `diagnose_zero_results` to inspect bottlenecks and candidate quality.
4. Run `autotune`; apply only when the resolved research-candidate win rate shows an edge.
5. Keep `live` strict until research candidates beat a clear win-rate and expectancy threshold.

## Sources
- Alpaca Market Data FAQ: https://docs.alpaca.markets/docs/market-data-faq
- Alpaca Option Chain API: https://docs.alpaca.markets/reference/optionchain
- VCP breakout/volume checklist: https://www.finermarketpoints.com/post/what-is-a-vcp-pattern-mark-minervini-s-volatility-contraction-pattern-explained
- Support/resistance evidence paper: https://arxiv.org/abs/2101.07410
- Purged cross-validation overview: https://en.wikipedia.org/wiki/Purged_cross-validation
- GT-Score anti-overfitting paper: https://arxiv.org/abs/2602.00080
