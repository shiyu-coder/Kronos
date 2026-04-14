"""
Real-time financial data aggregator for Kronos prediction enrichment.
Aggregates: news sentiment, fear/greed index, social volume, on-chain metrics.
"""
from __future__ import annotations

import os
import re
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import numpy as np

try:
    from transformers import pipeline as _hf_pipeline, AutoModelForSequenceClassification, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    import finnhub
    _FINNHUB_AVAILABLE = True
except ImportError:
    _FINNHUB_AVAILABLE = False

# Optional: newsapi-python, gnews, newspaper3k
try:
    from newsapi import NewsApiClient as _NewsApi
    _NEWSAPI_AVAILABLE = True
except ImportError:
    _NEWSAPI_AVAILABLE = False

try:
    from gnews import GNews as _GNews
    _GNEWS_AVAILABLE = True
except ImportError:
    _GNEWS_AVAILABLE = False

try:
    from newspaper import Article as _NewspaperArticle
    _NEWSPAPER_AVAILABLE = True
except ImportError:
    _NEWSPAPER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SentimentScore:
    """Aggregated sentiment from multiple sources."""
    overall: float          # -1 (bearish) .. +1 (bullish)
    confidence: float       # 0..1
    news_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_polarity: float    # raw avg from model
    fear_greed_index: float # 0..100 (alternative.me)
    social_volume: float    # relative, 0..inf
    last_updated: datetime
    raw_sources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketPulse:
    """Combined real-time market pulse fed into Kronos ensemble."""
    sentiment: SentimentScore
    macro_events: List[Dict[str, Any]]
    on_chain: Dict[str, float]
    confidence_boost: float  # +0.1 if sentiment aligned with Kronos signal


# ---------------------------------------------------------------------------
# Sentiment scoring — rule-based + optional HF model
# ---------------------------------------------------------------------------
class SentimentAnalyzer:
    """
    Score news headlines for sentiment.
    Priority: FinBERT/HF > keyword-based fallback.
    """

    # Keyword dictionaries (no API needed)
    BULLISH_KEYWORDS = [
        "bull", "pump", "surge", "soar", "rally", "gain", "rise", "high",
        "adoption", "upgrade", "breakout", "buy", "long", "bullish",
        "record high", "all-time", "whale", "institutional", "ETF", "approval",
        "growth", "profit", "moon", "to the moon", "flip", "outperform",
        "upgrade", "partnership", "launch", "beat", "exceed",
    ]
    BEARISH_KEYWORDS = [
        "bear", "dump", "crash", "plunge", "sell", "short", "drop", "fall",
        "ban", "regulation", "hack", "exploit", "scam", "warn",
        "fear", "panic", "liquidation", "loss", "decline", "downgrade",
        "risk", "investigation", "lawsuit", "hack", "theft", "rug pull",
        "delist", "ban", "restrict", "SEC", "CFTC", "probe",
    ]

    def __init__(self, hf_model: str = "ProsusAI/finbert") -> None:
        self._hf_model = hf_model
        self._hf_pipe = None
        if _HF_AVAILABLE:
            try:
                self._hf_pipe = _hf_pipeline(
                    "text-classification",
                    model=hf_model,
                    top_k=None,
                )
                # warm up
                self.score_text("Bitcoin price rises on ETF approval news.")
            except Exception:
                self._hf_pipe = None

    def score_text(self, text: str) -> Dict[str, Any]:
        """Return sentiment dict for a single piece of text."""
        if self._hf_pipe:
            try:
                result = self._hf_pipe(text)
                if result and isinstance(result, list) and len(result) > 0:
                    # FinBERT returns [{'label': 'positive', 'score': 0.9}, ...]
                    scores = {r["label"].lower(): r["score"] for r in result[0]}
                    # Map to -1..1
                    pos = scores.get("positive", 0.0)
                    neg = scores.get("negative", 0.0)
                    neu = scores.get("neutral", 0.0)
                    if pos > neg and pos > neu:
                        raw = pos
                        label = "bullish"
                    elif neg > pos and neg > neu:
                        raw = -neg
                        label = "bearish"
                    else:
                        raw = 0.0
                        label = "neutral"
                    return {"label": label, "score": float(raw), "confidence": float(max(scores.values()))}
            except Exception:
                pass
        # Fallback: keyword scoring
        text_lower = text.lower()
        bull_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bear_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        total = bull_count + bear_count
        if total == 0:
            raw, label = 0.0, "neutral"
        elif bull_count > bear_count:
            raw = bull_count / (bull_count + bear_count)
            label = "bullish"
        elif bear_count > bull_count:
            raw = -bear_count / (bull_count + bear_count)
            label = "bearish"
        else:
            raw, label = 0.0, "neutral"
        return {"label": label, "score": float(raw), "confidence": 0.5}

    def score_headlines(self, headlines: List[str]) -> SentimentScore:
        """Aggregate scores from multiple headlines."""
        if not headlines:
            return SentimentScore(
                overall=0.0, confidence=0.0, news_count=0,
                bullish_count=0, bearish_count=0, neutral_count=0,
                avg_polarity=0.0, fear_greed_index=50.0, social_volume=0.0,
                last_updated=datetime.now(timezone.utc),
            )

        results = [self.score_text(h) for h in headlines]
        scores = [r["score"] for r in results]
        avg_polarity = float(np.mean(scores))
        bullish = sum(1 for r in results if r["label"] == "bullish")
        bearish = sum(1 for r in results if r["label"] == "bearish")
        neutral = sum(1 for r in results if r["label"] == "neutral")
        confidence = float(np.std(scores))  # high std = disagreement = low confidence
        confidence = max(0.0, 1.0 - confidence)

        return SentimentScore(
            overall=avg_polarity,
            confidence=confidence,
            news_count=len(headlines),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            avg_polarity=avg_polarity,
            fear_greed_index=50.0,
            social_volume=0.0,
            last_updated=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Data source connectors
# ---------------------------------------------------------------------------
class FearGreedSource:
    """Fetch Fear & Greed Index from alternative.me (free, no auth)."""
    URL = "https://api.alternative.me/fng/?limit=1"

    def fetch(self) -> Dict[str, Any]:
        try:
            r = requests.get(self.URL, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("data") and len(data["data"]) > 0:
                item = data["data"][0]
                return {
                    "value": float(item.get("value", 50)),
                    "value_classification": item.get("value_classification", "Neutral"),
                    "timestamp": int(item.get("timestamp", 0)),
                }
        except Exception:
            pass
        return {"value": 50.0, "value_classification": "Neutral", "timestamp": 0}


class NewsAggregator:
    """
    Aggregate headlines from multiple sources.
    Supports: GNews (free), NewsAPI (optional), RSS feeds, CoinGecko news.
    """

    def __init__(
        self,
        gnews_token: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        rss_feeds: Optional[List[str]] = None,
    ) -> None:
        self.gnews_token = gnews_token or os.environ.get("GNEWS_API_KEY", "")
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY", "")
        self.rss_feeds = rss_feeds or []
        self._newsapi_client = None
        self._gnews_client = None

        if _NEWSAPI_AVAILABLE and self.newsapi_key:
            self._newsapi_client = _NewsApi(api_key=self.newsapi_key)

        if _GNEWS_AVAILABLE and self.gnews_token:
            self._gnews_client = _GNews(language="en", country="US", max_results=20)

        # Default RSS feeds for crypto news (no auth needed)
        self._default_rss = [
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed",
            "https://bitcoinist.com/feed/",
            "https://cryptonews.com/news/feed",
        ]

    def fetch_headlines(
        self,
        symbol: str = "Bitcoin",
        limit: int = 50,
        hours_back: int = 24,
    ) -> List[Dict[str, Any]]:
        """Fetch recent headlines from all available sources."""
        all_headlines: List[Dict[str, Any]] = []

        # 1. GNews
        if self._gnews_client:
            try:
                results = self._gnews_client.get_news(f"{symbol} crypto OR {symbol} BTC", max_results=limit)
                for item in results:
                    all_headlines.append({
                        "title": item.get("title", ""),
                        "source": f"gnews:{item.get('publisher', {}).get('title', 'unknown')}",
                        "published": item.get("published date", ""),
                        "url": item.get("url", ""),
                    })
            except Exception:
                pass

        # 2. NewsAPI
        if self._newsapi_client:
            try:
                cutoff = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
                results = self._newsapi_client.get_everything(
                    q=f'"{symbol}" OR "{symbol.replace("/", "")}" OR BTC',
                    from_param=cutoff,
                    language="en",
                    sort_by="publishedAt",
                    page_size=limit,
                )
                for item in results.get("articles", []):
                    if item.get("title"):
                        all_headlines.append({
                            "title": item["title"],
                            "source": f"newsapi:{item.get('source', {}).get('name', 'unknown')}",
                            "published": item.get("publishedAt", ""),
                            "url": item.get("url", ""),
                        })
            except Exception:
                pass

        # 3. RSS feeds
        try:
            import feedparser
            for feed_url in (self.rss_feeds or self._default_rss):
                try:
                    feed = feedparser.parse(feed_url)
                    cutoff_ts = time.time() - hours_back * 3600
                    for entry in feed.entries[:limit]:
                        pub_ts = getattr(entry, "published_parsed", None)
                        if pub_ts and time.mktime(pub_ts) < cutoff_ts:
                            continue
                        title = getattr(entry, "title", "")
                        if title:
                            all_headlines.append({
                                "title": title,
                                "source": f"rss:{feed.feed.get('title', feed_url)}",
                                "published": getattr(entry, "published", ""),
                                "url": getattr(entry, "link", ""),
                            })
                except Exception:
                    continue
        except ImportError:
            pass

        # 4. CoinGecko free news (no auth needed)
        try:
            r = requests.get(
                "https://news.coingecko.com/news/landing",
                headers={"Accept": "application/json"},
                timeout=10,
            )
            # CoinGecko news API
            news_r = requests.get(
                "https://api.coingecko.com/api/v3/news",
                timeout=10,
            )
            if news_r.ok:
                news_data = news_r.json()
                for item in (news_data.get("data", []) or [])[:limit]:
                    all_headlines.append({
                        "title": item.get("title", ""),
                        "source": f"coingecko:{item.get('thumb_2x_url', 'coingecko')[:20]}",
                        "published": item.get("updated_at", ""),
                        "url": item.get("url", ""),
                    })
        except Exception:
            pass

        # Deduplicate by title similarity
        seen = set()
        deduped = []
        for h in all_headlines:
            key = h["title"][:80].lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(h)
        deduped = deduped[:limit]

        # Optional enrichment: pull article summary/body for better sentiment quality.
        # Keep lightweight to avoid slowing API too much.
        for item in deduped[: min(len(deduped), 20)]:
            url = item.get("url", "")
            if not url:
                continue
            summary = self._extract_article_text(url)
            if summary:
                item["summary"] = summary

        return deduped

    def _extract_article_text(self, url: str, max_chars: int = 1200) -> str:
        """
        Extract article text using newspaper3k first, then BeautifulSoup fallback.
        Returns short cleaned text (or empty string if unavailable).
        """
        # 1) newspaper3k (best extraction quality if installed)
        if _NEWSPAPER_AVAILABLE:
            try:
                article = _NewspaperArticle(url=url, language="en")
                article.download()
                article.parse()
                txt = (article.text or "").strip()
                if txt:
                    return re.sub(r"\s+", " ", txt)[:max_chars]
            except Exception:
                pass

        # 2) BeautifulSoup fallback
        if _BS4_AVAILABLE:
            try:
                r = requests.get(
                    url,
                    timeout=6,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; KronosBot/1.0)"},
                )
                if not r.ok:
                    return ""
                soup = _BeautifulSoup(r.text, "html.parser")
                # remove noisy tags
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.extract()
                paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = " ".join([p for p in paragraphs if p])
                if text:
                    return re.sub(r"\s+", " ", text)[:max_chars]
            except Exception:
                pass

        return ""


class OnChainMetrics:
    """Fetch on-chain metrics (free public endpoints)."""

    @staticmethod
    def fetch_glassnode_alternative() -> Dict[str, float]:
        """Alternative.me on-chain proxy (free, no auth)."""
        metrics = {}
        try:
            # MVRV ratio proxy via CoinGecko
            r = requests.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin?localization=false"
                "&tickers=false&community_data=false&developer_data=false",
                timeout=10,
            )
            if r.ok:
                data = r.json()
                market_data = data.get("market_data", {})
                metrics["btc_dominance"] = market_data.get("market_cap", {}).get("btc", 0)
                metrics["btc_price"] = market_data.get("current_price", {}).get("usd", 0)
                ath = market_data.get("ath", {}).get("usd", 0)
                current = market_data.get("current_price", {}).get("usd", 0)
                if ath > 0:
                    metrics["ath_distance_pct"] = (ath - current) / ath * 100
        except Exception:
            pass

        try:
            # Global stats
            r = requests.get("https://api.alternative.me/v2/ticker/?limit=1&convert=USD", timeout=10)
            if r.ok:
                data = r.json()
                tickers = data.get("data", [])
                if tickers:
                    t = list(tickers.values())[0]
                    metrics["total_market_cap"] = t.get("circulating_supply", 0) * t.get("price", 0)
        except Exception:
            pass

        return metrics


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------
class RealTimeDataAggregator:
    """
    Orchestrate all real-time sources into a single MarketPulse.
    Thread-safe caching with TTL.
    """

    def __init__(
        self,
        hf_model: str = "ProsusAI/finbert",
        fear_greed_cache_ttl: int = 300,
        news_cache_ttl: int = 600,
        gnews_token: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
    ) -> None:
        self.fear_greed_source = FearGreedSource()
        self.news_aggregator = NewsAggregator(
            gnews_token=gnews_token,
            newsapi_key=newsapi_key,
        )
        self.sentiment_analyzer = SentimentAnalyzer(hf_model=hf_model)
        self.on_chain = OnChainMetrics()

        self._fg_cache: Optional[Dict[str, Any]] = None
        self._fg_cache_time: float = 0
        self._fg_ttl = fear_greed_cache_ttl

        self._news_cache: Optional[List[Dict[str, Any]]] = None
        self._news_cache_time: float = 0
        self._news_ttl = news_cache_ttl

        self._lock = threading.Lock()

        # Optional Finnhub (for market events)
        self._finnhub_client = None
        if _FINNHUB_AVAILABLE and finnhub_key:
            self._finnhub_client = finnhub.Client(api_key=finnhub_key)

    def get_fear_greed(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            if self._fg_cache is None or (now - self._fg_cache_time) > self._fg_ttl:
                self._fg_cache = self.fear_greed_source.fetch()
                self._fg_cache_time = now
            return self._fg_cache

    def get_headlines(self, symbol: str = "Bitcoin", limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            if self._news_cache is None or (now - self._news_cache_time) > self._news_ttl:
                self._news_cache = self.news_aggregator.fetch_headlines(symbol=symbol, limit=limit)
                self._news_cache_time = now
            return self._news_cache

    def get_pulse(
        self,
        symbol: str = "Bitcoin",
        sentiment_threshold: float = 0.3,
        limit: int = 50,
    ) -> MarketPulse:
        """
        Build a complete MarketPulse from all sources.
        Call this before running Kronos prediction to enrich signals.
        """
        # Fetch in parallel (no locks needed — they are thread-safe)
        fg_data = self.get_fear_greed()
        headlines = self.get_headlines(symbol=symbol, limit=limit)
        on_chain = self.on_chain.fetch_glassnode_alternative()

        # Score sentiment (title + optional summary/article text)
        headline_texts = []
        for h in headlines:
            title = (h.get("title") or "").strip()
            summary = (h.get("summary") or "").strip()
            if title and summary:
                headline_texts.append(f"{title}. {summary}")
            elif title:
                headline_texts.append(title)

        sentiment = self.sentiment_analyzer.score_headlines(headline_texts)
        sentiment.fear_greed_index = float(fg_data.get("value", 50)) / 100.0 * 2 - 1  # -1..1
        sentiment.social_volume = float(len(headlines)) / max(limit, 1)
        sentiment.raw_sources = {
            "fear_greed": fg_data,
            "news_count": len(headlines),
            "sources": list(set(h.get("source", "") for h in headlines)),
        }

        # Macro events from Finnhub (optional)
        macro_events: List[Dict[str, Any]] = []
        if self._finnhub_client:
            try:
                # Economic calendar (free)
                eco = self._finnhub_client.market_economic()
                for event in (eco or [])[:10]:
                    macro_events.append({
                        "event": event.get("event", ""),
                        "time": event.get("time", ""),
                        "country": event.get("country", ""),
                    })
            except Exception:
                pass

        # Confidence boost: sentiment aligns with fear/greed
        fg_val = fg_data.get("value", 50)
        sentiment_aligned = abs(sentiment.overall - (fg_val / 100 * 2 - 1)) < 0.3
        confidence_boost = 0.15 if sentiment_aligned else -0.05

        return MarketPulse(
            sentiment=sentiment,
            macro_events=macro_events,
            on_chain=on_chain,
            confidence_boost=confidence_boost,
        )

    def build_ensemble_features(self, pulse: MarketPulse) -> Dict[str, float]:
        """
        Convert MarketPulse into features for Kronos ensemble.
        Returns a dict of indicator names → values (-1..1 scale).
        """
        s = pulse.sentiment
        fg = s.fear_greed_index * 2 - 1  # 0..100 → -1..1

        return {
            "news_sentiment": s.overall,
            "fear_greed": fg,
            "sentiment_confidence": s.confidence,
            "bullish_ratio": s.bullish_count / max(s.news_count, 1),
            "news_volume_score": min(s.social_volume, 1.0),
            "fg_above_50": 1.0 if fg > 0 else 0.0,
            "fg_extreme": 1.0 if abs(fg) > 0.7 else 0.0,  # fear <20 or greed >80
            "confidence_boost": pulse.confidence_boost,
        }