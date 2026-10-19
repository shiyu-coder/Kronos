from __future__ import annotations

from datetime import datetime
import logging
import os
import pandas as pd
import requests
import yfinance as yf

from .. import config as scanner_config
from .market_data import _alpaca_credentials, _alpaca_get
from ..utils.validation import OptionsContractResult

TRADIER_API_BASE = "https://api.tradier.com/v1"


def _spread_pct(bid: float, ask: float) -> float:
    if ask <= 0:
        return 1.0
    return (ask - bid) / ask


def _pick_contract_row(chain_df: pd.DataFrame, underlying_price: float) -> pd.Series | None:
    if chain_df is None or chain_df.empty:
        return None
    df = chain_df.copy()
    df = df.dropna(subset=["strike", "bid", "ask", "openInterest"], how="any")
    if df.empty:
        return None
    df["dist"] = (df["strike"] - underlying_price).abs()
    df = df.sort_values(["dist", "openInterest"], ascending=[True, False])
    return df.iloc[0]


def _fetch_alpaca_option_snapshots(ticker: str, logger: logging.Logger) -> dict:
    key, secret = _alpaca_credentials()
    if not key or not secret:
        return {}
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    feed = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip().lower() or "indicative"
    snapshots: dict = {}
    page_token = None
    while True:
        params = {"feed": feed, "limit": 1000}
        if page_token:
            params["page_token"] = page_token
        response = _alpaca_get(
            f"https://data.alpaca.markets/v1beta1/options/snapshots/{ticker}",
            headers=headers,
            params=params,
            timeout=30,
            retries=3,
        )
        if response.status_code != 200:
            logger.warning("%s Alpaca options snapshot unavailable: %s", ticker, response.status_code)
            return {}
        payload = response.json()
        snapshots.update(payload.get("snapshots", {}) or {})
        page_token = payload.get("next_page_token")
        if not page_token:
            return snapshots


def _quote_age_minutes(timestamp) -> float | None:
    if not timestamp:
        return None
    try:
        quote_ts = pd.Timestamp(timestamp)
        if quote_ts.tzinfo is None:
            quote_ts = quote_ts.tz_localize("UTC")
        age = float((pd.Timestamp.now(tz="UTC") - quote_ts.tz_convert("UTC")).total_seconds() / 60.0)
    except Exception:
        return None
    if age < -5.0:
        # Future-dated quote: clock skew or a corrupt feed timestamp must
        # not read as a fresh (execution-grade) quote.
        return None
    return max(0.0, age)


def _relative_disagreement(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or first <= 0 or second <= 0:
        return None
    return abs(first - second) / max(first, second)


def _options_quality(feed: str, quote_age: float | None, disagreement: float | None) -> float:
    if feed == "opra":
        quality = 1.0
    elif feed == "opra-consolidated":
        # Tradier: real OPRA-consolidated NBBO with sizes and native OI.
        quality = 0.9
    elif feed == "indicative":
        quality = 0.6
    else:
        quality = 0.45
    if quote_age is None or quote_age > 30:
        quality -= 0.2
    if disagreement is not None:
        quality -= min(disagreement, 0.3)
    return round(max(0.0, min(1.0, quality)), 4)


def _tradier_token() -> str:
    return os.getenv("TRADIER_API_TOKEN", "").strip()


def _tradier_get(path: str, params: dict, token: str, logger: logging.Logger, retries: int = 2) -> dict | None:
    """GET against the production Tradier API; None means infrastructure
    failure (auth, transport, non-200) and the caller may fall back."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = requests.get(f"{TRADIER_API_BASE}{path}", headers=headers, params=params, timeout=15)
        except Exception as exc:
            last_error = exc
            continue
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                last_error = exc
                continue
            return payload if isinstance(payload, dict) else None
        if response.status_code in {401, 403}:
            logger.warning(
                "Tradier auth failed (%s); TRADIER_API_TOKEN must be the PRODUCTION token, not sandbox",
                response.status_code,
            )
            return None
        last_error = RuntimeError(f"tradier http {response.status_code}")
    logger.warning("Tradier request failed: %s (%s)", path, last_error)
    return None


def _tradier_list(payload_section) -> list:
    # Tradier collapses single-item lists to a bare object/string.
    if payload_section is None:
        return []
    if isinstance(payload_section, list):
        return payload_section
    return [payload_section]


def _tradier_quote_timestamp(row: dict):
    stamps = [row.get("bid_date"), row.get("ask_date")]
    stamps = [s for s in stamps if isinstance(s, (int, float)) and s > 0]
    if not stamps:
        return None
    return pd.Timestamp(max(stamps), unit="ms", tz="UTC")


def _select_via_tradier(
    ticker: str,
    direction: str,
    breakout_price: float,
    logger: logging.Logger,
    min_dte: int,
    max_dte: int,
    token: str,
) -> OptionsContractResult | None:
    """ATM contract from real-time OPRA-consolidated Tradier chains.

    Returns None only on infrastructure failure so the caller can fall back
    to the indicative pipeline. A chain that fails the liquidity gates is an
    authoritative fail - falling back to lower-grade data to force a pass
    would defeat the point of execution-grade quotes.
    """
    payload = _tradier_get(
        "/markets/options/expirations",
        {"symbol": ticker, "includeAllRoots": "true", "strikes": "false"},
        token,
        logger,
    )
    if payload is None:
        return None
    expirations_section = payload.get("expirations")
    dates = _tradier_list(expirations_section.get("date")) if isinstance(expirations_section, dict) else []

    contract_type = "call" if direction == "bullish" else "put"
    # Expirations are exchange (NY) dates; the machine clock is Central.
    today = pd.Timestamp.now(tz=scanner_config.TIMEZONE).date()
    valid_exp = []
    for exp in dates:
        try:
            dte = (pd.Timestamp(exp).date() - today).days
        except (TypeError, ValueError):
            continue
        if min_dte <= dte <= max_dte:
            valid_exp.append((str(exp), dte))

    if not valid_exp:
        return OptionsContractResult(
            False, None, None, contract_type, None, None, None, None, None, None, None, None,
            f"no expirations in {min_dte}-{max_dte} DTE",
            data_provider="tradier",
            data_feed="opra-consolidated",
        )

    for exp, dte in valid_exp:
        chain_payload = _tradier_get(
            "/markets/options/chains",
            {"symbol": ticker, "expiration": exp, "greeks": "true"},
            token,
            logger,
        )
        if chain_payload is None:
            return None
        options_section = chain_payload.get("options")
        rows = _tradier_list(options_section.get("option")) if isinstance(options_section, dict) else []
        candidates = []
        for row in rows:
            if not isinstance(row, dict) or row.get("option_type") != contract_type:
                continue
            try:
                strike = float(row.get("strike"))
                bid = float(row.get("bid") or 0.0)
                ask = float(row.get("ask") or 0.0)
            except (TypeError, ValueError):
                continue
            if not (strike > 0) or bid <= 0 or ask <= bid:
                # Unquotable rows can't anchor the ATM pick; a dead nearest
                # strike should not disqualify the whole expiration.
                continue
            candidates.append((abs(strike - breakout_price), -float(row.get("open_interest") or 0), row, strike, bid, ask))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, row, strike, bid, ask = candidates[0]

        open_interest = int(row.get("open_interest") or 0)
        volume = int(row.get("volume") or 0)
        spread = _spread_pct(bid, ask)
        if spread > scanner_config.MAX_ATM_BID_ASK_SPREAD_PCT:
            continue
        if open_interest < scanner_config.MIN_ATM_OPEN_INTEREST:
            continue

        quote_ts = _tradier_quote_timestamp(row)
        quote_age = _quote_age_minutes(quote_ts) if quote_ts is not None else None
        greeks = row.get("greeks") if isinstance(row.get("greeks"), dict) else {}
        implied_vol = greeks.get("mid_iv") if greeks.get("mid_iv") is not None else greeks.get("smv_vol")
        return OptionsContractResult(
            passed=True,
            expiration=exp,
            dte=dte,
            contract_type=contract_type,
            strike=strike,
            bid=bid,
            ask=ask,
            midpoint=(bid + ask) / 2.0,
            spread_pct=spread,
            open_interest=open_interest,
            volume=volume,
            implied_volatility=float(implied_vol) if implied_vol is not None else None,
            skip_reason=None,
            data_provider="tradier",
            data_feed="opra-consolidated",
            quote_source="tradier",
            open_interest_source="tradier",
            quote_timestamp=quote_ts.isoformat() if quote_ts is not None else None,
            quote_age_minutes=quote_age,
            greeks_available=bool(greeks),
            source_disagreement_pct=None,
            options_data_quality=_options_quality("opra-consolidated", quote_age, None),
            bid_size=int(row.get("bidsize") or 0) or None,
            ask_size=int(row.get("asksize") or 0) or None,
        )

    return OptionsContractResult(
        False, None, None, contract_type, None, None, None, None, None, None, None, None,
        (
            f"no {contract_type} contract passed: bid>0, ask>bid, "
            f"spread<={scanner_config.MAX_ATM_BID_ASK_SPREAD_PCT:.0%}, OI>={scanner_config.MIN_ATM_OPEN_INTEREST}"
        ),
        data_provider="tradier",
        data_feed="opra-consolidated",
    )


def select_options_contract(
    ticker: str,
    direction: str,
    breakout_price: float,
    logger: logging.Logger,
    min_dte: int = 30,
    max_dte: int = 60,
) -> OptionsContractResult:
    try:
        token = _tradier_token()
        if token:
            tradier_result = _select_via_tradier(ticker, direction, breakout_price, logger, min_dte, max_dte, token)
            if tradier_result is not None:
                return tradier_result
            logger.warning("%s Tradier unavailable; falling back to indicative options pipeline", ticker)

        yf_ticker = yf.Ticker(ticker)
        alpaca_snapshots = _fetch_alpaca_option_snapshots(ticker, logger)
        alpaca_feed = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip().lower() or "indicative"
        expirations = list(yf_ticker.options or [])
        if not expirations:
            return OptionsContractResult(False, None, None, None, None, None, None, None, None, None, None, None, "empty options chain")

        today = pd.Timestamp.now(tz=scanner_config.TIMEZONE).date()
        valid_exp = []
        for exp in expirations:
            exp_dt = pd.Timestamp(exp).date()
            dte = (exp_dt - today).days
            if min_dte <= dte <= max_dte:
                valid_exp.append((exp, dte))

        if not valid_exp:
            return OptionsContractResult(False, None, None, None, None, None, None, None, None, None, None, None, "no expirations in 30-60 DTE")

        contract_type = "call" if direction == "bullish" else "put"

        best: OptionsContractResult | None = None
        for exp, dte in valid_exp:
            chain = yf_ticker.option_chain(exp)
            chain_df = chain.calls if contract_type == "call" else chain.puts
            row = _pick_contract_row(chain_df, breakout_price)
            if row is None:
                continue

            yf_bid = float(row["bid"])
            yf_ask = float(row["ask"])
            contract_symbol = str(row.get("contractSymbol", ""))
            snapshot = alpaca_snapshots.get(contract_symbol, {}) if contract_symbol else {}
            quote = snapshot.get("latestQuote", {}) or {}
            alpaca_bid = float(quote.get("bp", 0.0) or 0.0)
            alpaca_ask = float(quote.get("ap", 0.0) or 0.0)
            has_alpaca_quote = alpaca_bid > 0 and alpaca_ask > alpaca_bid
            bid = alpaca_bid if has_alpaca_quote else yf_bid
            ask = alpaca_ask if has_alpaca_quote else yf_ask
            oi = int(row.get("openInterest", 0))
            vol_val = row.get("volume", 0)
            yf_volume = int(vol_val) if pd.notna(vol_val) else 0
            alpaca_volume = int((snapshot.get("dailyBar", {}) or {}).get("v", 0) or 0)
            volume = max(yf_volume, alpaca_volume)

            if bid <= 0:
                continue
            if ask <= bid:
                continue
            spread = _spread_pct(bid, ask)
            if spread > scanner_config.MAX_ATM_BID_ASK_SPREAD_PCT:
                continue
            if oi < scanner_config.MIN_ATM_OPEN_INTEREST:
                continue

            yf_midpoint = (yf_bid + yf_ask) / 2.0 if yf_bid > 0 and yf_ask > yf_bid else None
            midpoint = (bid + ask) / 2.0
            disagreement = _relative_disagreement(midpoint, yf_midpoint) if has_alpaca_quote else None
            quote_age = _quote_age_minutes(quote.get("t")) if has_alpaca_quote else None
            data_feed = alpaca_feed if has_alpaca_quote else "yfinance"
            candidate = OptionsContractResult(
                passed=True,
                expiration=exp,
                dte=dte,
                contract_type=contract_type,
                strike=float(row["strike"]),
                bid=bid,
                ask=ask,
                midpoint=midpoint,
                spread_pct=spread,
                open_interest=oi,
                volume=volume,
                implied_volatility=(
                    float(snapshot["impliedVolatility"])
                    if snapshot.get("impliedVolatility") is not None
                    else float(row["impliedVolatility"])
                    if pd.notna(row.get("impliedVolatility"))
                    else None
                ),
                skip_reason=None,
                data_provider="alpaca+yfinance" if has_alpaca_quote else "yfinance",
                data_feed=data_feed,
                quote_source="alpaca" if has_alpaca_quote else "yfinance",
                open_interest_source="yfinance",
                quote_timestamp=quote.get("t") if has_alpaca_quote else None,
                quote_age_minutes=quote_age,
                greeks_available=bool(snapshot.get("greeks")) if has_alpaca_quote else False,
                source_disagreement_pct=disagreement,
                options_data_quality=_options_quality(data_feed, quote_age, disagreement),
            )
            best = candidate
            break

        if best is None:
            return OptionsContractResult(
                passed=False,
                expiration=None,
                dte=None,
                contract_type=contract_type,
                strike=None,
                bid=None,
                ask=None,
                midpoint=None,
                spread_pct=None,
                open_interest=None,
                volume=None,
                implied_volatility=None,
                skip_reason=(
                    f"no {contract_type} contract passed: bid>0, ask>bid, "
                    f"spread<={scanner_config.MAX_ATM_BID_ASK_SPREAD_PCT:.0%}, OI>={scanner_config.MIN_ATM_OPEN_INTEREST}"
                ),
            )

        return best
    except Exception as exc:
        logger.error("%s options error: %s", ticker, exc)
        return OptionsContractResult(
            passed=False,
            expiration=None,
            dte=None,
            contract_type="call" if direction == "bullish" else "put",
            strike=None,
            bid=None,
            ask=None,
            midpoint=None,
            spread_pct=None,
            open_interest=None,
            volume=None,
            implied_volatility=None,
            skip_reason=f"options lookup error: {exc}",
        )
