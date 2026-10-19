from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import time
import pandas as pd
import requests
import yfinance as yf

from ..config import (
    ALPACA_FEED,
    DAILY_PROXY_LOOKBACK,
    INTRADAY_INTERVAL,
    INTRADAY_LOOKBACK,
    MARKET_DATA_PROVIDER_DEFAULT,
    MIN_STOCK_PRICE,
    TIMEZONE,
    LOG_DIR,
)
from ..utils.validation import TickerValidationResult


def _to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if out.index.tz is None:
        # Naive midnight-only indexes are calendar DATES (yfinance daily bars),
        # not UTC instants - localizing them to UTC would shift every session
        # to the prior NY evening. Localize dates to NY directly.
        idx = pd.DatetimeIndex(out.index)
        if len(idx) and (idx == idx.normalize()).all():
            out.index = idx.tz_localize(TIMEZONE)
            return out
        out.index = idx.tz_localize("UTC")
    out.index = out.index.tz_convert(TIMEZONE)
    return out


def _extract_price(info: dict, fast_info: dict) -> float | None:
    for key in ("regularMarketPrice", "currentPrice", "previousClose"):
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    for key in ("lastPrice", "regularMarketPreviousClose"):
        value = fast_info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _alpaca_credentials() -> tuple[str, str]:
    return (
        os.getenv("ALPACA_API_KEY", "").strip(),
        os.getenv("ALPACA_SECRET_KEY", "").strip(),
    )


def _alpaca_enabled() -> bool:
    key, secret = _alpaca_credentials()
    return bool(key and secret)


def _provider_choice() -> str:
    return os.getenv("MARKET_DATA_PROVIDER", MARKET_DATA_PROVIDER_DEFAULT).strip().lower()


def _persist_request_id(service: str, endpoint: str, request_id: str | None, status_code: int | None):
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        trace_path = Path(LOG_DIR) / "request_ids.log"
        stamp = pd.Timestamp.utcnow().isoformat()
        rid = request_id if request_id else "none"
        line = f"{stamp} | {service} | {endpoint} | status={status_code} | request_id={rid}\n"
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        return


def _alpaca_get(url: str, headers: dict, params: dict | None = None, timeout: int = 30, retries: int = 3) -> requests.Response:
    last_resp = None
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as exc:
            # Transport blips (timeout, reset) deserve the same retry budget
            # as retryable HTTP statuses instead of instantly falling through
            # to the degraded provider.
            last_exc = exc
            _persist_request_id("alpaca", url, None, None)
            if attempt < retries - 1:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise
        req_id = resp.headers.get("X-Request-ID")
        _persist_request_id("alpaca", url, req_id, resp.status_code)
        last_resp = resp
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt < retries - 1:
                time.sleep(0.8 * (attempt + 1))
                continue
        return resp
    if last_resp is None and last_exc is not None:
        raise last_exc
    return last_resp


def _interval_to_alpaca(interval: str) -> str:
    mapping = {"30m": "30Min", "1d": "1Day"}
    if interval not in mapping:
        raise ValueError(f"unsupported interval for Alpaca: {interval}")
    return mapping[interval]


def _fetch_alpaca_bars(
    ticker: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    feed: str | None = None,
    delay_minutes: int = 0,
    adjustment: str = "raw",
) -> pd.DataFrame:
    key, secret = _alpaca_credentials()
    if not key or not secret:
        raise RuntimeError("missing Alpaca credentials")

    timeframe = _interval_to_alpaca(interval)
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    feed = (feed or os.getenv("ALPACA_FEED", ALPACA_FEED)).strip().lower() or "iex"
    base_url = "https://data.alpaca.markets/v2/stocks"
    url = f"{base_url}/{ticker}/bars"

    bars = []
    page_token = None
    while True:
        params = {
            "timeframe": timeframe,
            "start": start.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            "end": end.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            "adjustment": adjustment,
            "sort": "asc",
            "limit": 10000,
            "feed": feed,
        }
        if page_token:
            params["page_token"] = page_token

        resp = _alpaca_get(url, headers=headers, params=params, timeout=30, retries=3)
        if resp.status_code != 200:
            req_id = resp.headers.get("X-Request-ID")
            raise RuntimeError(f"alpaca bars error {resp.status_code} request_id={req_id} body={resp.text[:200]}")
        payload = resp.json()
        bars.extend(payload.get("bars", []))
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "timestamp", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    result = _to_ny_index(df[["Open", "High", "Low", "Close", "Volume"]])
    result.attrs.update(
        {
            "data_provider": "alpaca",
            "data_feed": feed,
            "data_delay_minutes": int(delay_minutes),
            "data_adjustment": adjustment,
        }
    )
    return result


def validate_ticker(ticker: str, logger: logging.Logger) -> TickerValidationResult:
    try:
        info = {}
        price = None
        is_active_alpaca = None

        # Prefer Alpaca tradability + price when credentials are available.
        if _alpaca_enabled():
            key, secret = _alpaca_credentials()
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
            asset_resp = _alpaca_get(f"https://paper-api.alpaca.markets/v2/assets/{ticker}", headers=headers, timeout=20, retries=2)
            if asset_resp.status_code == 200:
                asset = asset_resp.json()
                is_active_alpaca = bool(asset.get("status") == "active" and asset.get("tradable", False))
            quote_resp = _alpaca_get(
                f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest",
                headers=headers,
                params={"feed": os.getenv("ALPACA_FEED", ALPACA_FEED).strip().lower() or "iex"},
                timeout=20,
                retries=2,
            )
            if quote_resp.status_code == 200:
                q = quote_resp.json().get("quote", {}) or {}
                if q.get("ap") is not None and q.get("bp") is not None:
                    price = (float(q["ap"]) + float(q["bp"])) / 2.0
                elif q.get("ap") is not None:
                    price = float(q["ap"])
                elif q.get("bp") is not None:
                    price = float(q["bp"])

        yf_ticker = yf.Ticker(ticker)
        yf_info = yf_ticker.info or {}
        fast_info = getattr(yf_ticker, "fast_info", {}) or {}
        if price is None:
            price = _extract_price(yf_info, fast_info)

        expirations = []
        try:
            expirations = list(yf_ticker.options or [])
        except Exception as opt_exc:
            logger.warning("%s options lookup warning: %s", ticker, opt_exc)

        is_active = bool(price is not None)
        if is_active_alpaca is not None:
            is_active = is_active and is_active_alpaca
        is_above_min = bool(price is not None and price >= MIN_STOCK_PRICE)
        has_options = len(expirations) > 0

        skip_reason = None
        if not is_active:
            skip_reason = "missing current price"
        elif not is_above_min:
            skip_reason = f"price below ${MIN_STOCK_PRICE:.2f}"
        elif not has_options:
            skip_reason = "no active listed options"

        return TickerValidationResult(
            ticker=ticker,
            is_active=is_active,
            price=price,
            is_above_min_price=is_above_min,
            has_options=has_options,
            skip_reason=skip_reason,
            metadata={"exchange": yf_info.get("exchange"), "quote_type": yf_info.get("quoteType"), "alpaca_active": is_active_alpaca},
        )
    except Exception as exc:
        return TickerValidationResult(
            ticker=ticker,
            is_active=False,
            price=None,
            is_above_min_price=False,
            has_options=False,
            skip_reason=f"validation error: {exc}",
            metadata={},
        )


def fetch_intraday_bars(
    ticker: str,
    interval: str = INTRADAY_INTERVAL,
    period: str = INTRADAY_LOOKBACK,
    *,
    research: bool = False,
    now: pd.Timestamp | None = None,
    adjustment: str = "split",
) -> pd.DataFrame:
    provider = _provider_choice()
    if provider in {"auto", "alpaca"} and _alpaca_enabled():
        try:
            days = int(period.rstrip("d")) if period.endswith("d") else 60
            end = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
            delay_minutes = 16 if research else 0
            if delay_minutes:
                end -= pd.Timedelta(minutes=delay_minutes)
            start = end - pd.Timedelta(days=days + 5)
            feed = "sip" if research else (os.getenv("ALPACA_FEED", ALPACA_FEED).strip().lower() or "iex")
            # "split" adjusts price AND volume on every timeframe, so a split
            # inside the 60d window no longer reads as a giant gap in the
            # synthetic sessions (and volume ratios stay comparable across it).
            result = _fetch_alpaca_bars(
                ticker=ticker,
                interval=interval,
                start=start,
                end=end,
                feed=feed,
                delay_minutes=delay_minutes,
                adjustment=adjustment,
            )
            result.attrs.update({"data_provider": "alpaca", "data_feed": feed, "data_delay_minutes": delay_minutes})
            return result
        except Exception as exc:
            if provider == "alpaca":
                raise
            logger = logging.getLogger("scanner.market_data")
            logger.warning("PROVIDER_FALLBACK: %s intraday bars alpaca->yfinance (%s)", ticker, exc)

    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        prepost=True,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    result = _to_ny_index(data)
    # Yahoo intraday prints are as-traded (auto_adjust is a no-op there); a
    # split inside the window shows up as a gap the bar contract must catch.
    result.attrs.update(
        {"data_provider": "yfinance", "data_feed": None, "data_delay_minutes": 0, "data_adjustment": "raw"}
    )
    return result


def fetch_daily_bars(
    ticker: str,
    period: str = DAILY_PROXY_LOOKBACK,
    *,
    research: bool = False,
    adjustment: str = "raw",
) -> pd.DataFrame:
    provider = _provider_choice()
    if provider in {"auto", "alpaca"} and _alpaca_enabled():
        try:
            years = int(period.rstrip("y")) if period.endswith("y") else 2
            end = pd.Timestamp.now(tz=TIMEZONE)
            start = end - pd.Timedelta(days=365 * years + 10)
            delay_minutes = 16 if research else 0
            if delay_minutes:
                end -= pd.Timedelta(minutes=delay_minutes)
            feed = "sip" if research else (os.getenv("ALPACA_FEED", ALPACA_FEED).strip().lower() or "iex")
            return _fetch_alpaca_bars(
                ticker=ticker,
                interval="1d",
                start=start,
                end=end,
                feed=feed,
                delay_minutes=delay_minutes,
                adjustment=adjustment,
            )
        except Exception as exc:
            if provider == "alpaca":
                raise
            logger = logging.getLogger("scanner.market_data")
            logger.warning("PROVIDER_FALLBACK: %s daily bars alpaca->yfinance (%s)", ticker, exc)

    data = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        prepost=False,
        auto_adjust=False,
        # Yahoo intermittently fails to apply splits at all (yfinance#1531);
        # repair detects and fixes unapplied splits and 100x glitches on
        # daily bars, marking touched rows in a "Repaired?" column.
        repair=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    repaired_bars = 0
    if "Repaired?" in data.columns:
        repaired_bars = int(data["Repaired?"].fillna(False).astype(bool).sum())
        data = data.drop(columns=["Repaired?"])
    result = _to_ny_index(data)
    # Yahoo's unadjusted series is already split-adjusted (never truly raw);
    # stamp the actual basis so evidence provenance stays honest.
    result.attrs.update(
        {
            "data_provider": "yfinance",
            "data_feed": None,
            "data_delay_minutes": 0,
            "data_adjustment": "split",
            "data_repaired_bars": repaired_bars,
        }
    )
    return result


def drop_in_progress_daily_bar(df: pd.DataFrame, *, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove the current session's still-forming daily bar.

    Both Alpaca and yfinance include today's partial bar in daily history
    when fetched during market hours. The scheduled research run fires
    13:30 CT (14:30 ET), so without this the newest evidence bar every day
    is ~60% of a session: volume reads low, the close is provisional, and
    the freshest index records inherit both.
    """
    if df is None or df.empty:
        return df
    now_ts = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
    now_ts = now_ts.tz_localize(TIMEZONE) if now_ts.tzinfo is None else now_ts.tz_convert(TIMEZONE)
    last_ts = pd.Timestamp(df.index[-1])
    last_ts = last_ts.tz_localize(TIMEZONE) if last_ts.tzinfo is None else last_ts.tz_convert(TIMEZONE)
    if last_ts.date() != now_ts.date():
        return df
    # 16:00 ET close + settle buffer. On early-close days this holds a
    # completed bar out until 16:15; it re-enters on the next rebuild.
    session_close = now_ts.normalize() + pd.Timedelta(hours=16)
    if now_ts >= session_close + pd.Timedelta(minutes=15):
        return df
    out = df.iloc[:-1].copy()
    out.attrs.update(df.attrs)
    out.attrs["dropped_in_progress_bar"] = last_ts.isoformat()
    return out


def drop_vendor_placeholder_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-volume flat bars vendors emit for non-traded sessions.

    Alpaca and Yahoo both forward-fill a session with no prints (e.g. CLSK's
    full-day Nasdaq halt on 2024-11-08) as Open==High==Low==Close at the
    prior close with zero volume; Tradier omits the session entirely. No
    trades happened, so the bar carries no market information and its
    High/Low path is fabricated. Dropping it converts the halt into a
    missing session, which the session-completeness check reports honestly —
    instead of the bar contract failing the ticker for as long as the
    placeholder stays inside the lookback window.
    """
    if df is None or df.empty:
        return df
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        return df
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    finite = ~df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    flat = (
        (df["Open"] == df["High"])
        & (df["High"] == df["Low"])
        & (df["Low"] == df["Close"])
    )
    placeholder = finite & flat & (volume == 0)
    if not placeholder.any():
        return df
    out = df[~placeholder].copy()
    out.attrs.update(df.attrs)
    out.attrs["dropped_placeholder_bars"] = [
        pd.Timestamp(ts).isoformat() for ts in df.index[placeholder]
    ]
    return out


def compute_future_timestamps(last_ts: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    base = pd.Timestamp(last_ts)
    if base.tz is None:
        base = base.tz_localize(TIMEZONE)
    idx = pd.date_range(start=base + pd.Timedelta(days=1), periods=periods, freq="D", tz=TIMEZONE)
    return idx


def parse_date_like(value) -> datetime | None:
    if value is None:
        return None
    try:
        dt = pd.to_datetime(value)
        if pd.isna(dt):
            return None
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is not None:
                dt = dt.tz_convert(TIMEZONE)
            return dt.to_pydatetime()
        return dt
    except Exception:
        return None
