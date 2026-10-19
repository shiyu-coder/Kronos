# Zero-Cost Market Data Ensemble Design

## Goal

Improve Kronos research-scan and evidence-lab input quality without purchasing Alpaca Algo Trader Plus, while keeping live alerts fail-closed.

## Design

Historical research workflows will request Alpaca SIP bars only when the request end time is at least 15 minutes old. Current scans continue using the configured real-time feed, which is IEX on the Basic plan. Every returned bar set carries provider/feed metadata so scoring and audit reports can distinguish delayed consolidated research data from current single-exchange data.

Options selection will use an ensemble:

- Alpaca indicative snapshots supply current indicative bid/ask, trade volume, implied volatility, Greeks availability, and quote timestamps.
- yfinance supplies expiration discovery and open interest.
- Contracts are joined by OCC contract symbol.
- The selector records both sources, feed type, quote age, and source disagreement.
- Indicative options remain research-grade and cannot independently make live mode eligible.

## Data Quality Rules

- Delayed SIP receives higher research confidence than IEX because it covers consolidated US exchanges, but it is labeled delayed and never treated as live.
- Alpaca indicative option quotes are explicitly labeled modified/delayed.
- Missing or stale quotes, missing open interest, wide spreads, and material source disagreement reduce quality or reject the contract.
- If Alpaca is unavailable, the selector falls back to the existing yfinance behavior and records the fallback source.

## Boundaries

- `scanner/data/market_data.py`: provider-aware bar retrieval and metadata.
- `scanner/data/options_data.py`: Alpaca/yfinance option ensemble.
- `scanner/utils/validation.py`: option-result provenance fields.
- `scanner/edge/features.py`: stable provenance and quality features.
- `scanner/main.py`: request delayed SIP for research/history and pass metadata into scoring.
- `scanner/edge/audit.py`: preserve conservative readiness semantics.

## Verification

- Unit tests prove delayed SIP requests use an end time older than 15 minutes.
- Unit tests prove current scans remain on IEX.
- Unit tests prove Alpaca indicative snapshots enrich yfinance open interest and preserve provenance.
- Unit tests prove failures fall back safely.
- Full edge lab refresh measures whether missing-liquidity and low-feed warnings improve without changing validation thresholds.
