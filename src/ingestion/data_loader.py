"""
Multi-Source Data Ingestion Engine
Handles ingestion from heterogeneous sources:
1. Financial News Articles (e.g., Financial Times, Bloomberg, Reuters, Yahoo Finance RSS)
2. Social Media Feeds (e.g., Twitter/X FinTwit, StockTwits API)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET
import requests

logger = logging.getLogger("DataIngestion")


@dataclass
class RawItem:
    id: str
    source: str
    text: str
    ticker: Optional[str] = None
    company: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiSourceIngestionEngine:
    """Ingests, cleans, and normalizes financial unstructured text from multiple channels."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (
            Path(__file__).resolve().parent.parent.parent / "data"
        )
        self.news_file = self.data_dir / "sample_news.json"
        self.tweets_file = self.data_dir / "sample_tweets.json"

    def load_cached_news(self) -> List[RawItem]:
        """Loads curated financial news articles."""
        items: List[RawItem] = []
        if not self.news_file.exists():
            logger.warning(f"News dataset file not found: {self.news_file}")
            return items

        try:
            with open(self.news_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                full_text = f"{r.get('headline', '')}. {r.get('content', '')}".strip()
                items.append(
                    RawItem(
                        id=r.get("id", f"NEWS-{len(items)+1}"),
                        source=r.get("source", "Financial News"),
                        text=full_text,
                        ticker=r.get("ticker"),
                        company=r.get("company"),
                        timestamp=r.get(
                            "timestamp", datetime.now(timezone.utc).isoformat()
                        ),
                        metadata={
                            "headline": r.get("headline"),
                            "expected_event": r.get("expected_event"),
                            "expected_sentiment_sign": r.get("expected_sentiment_sign"),
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Failed to load news file: {e}")
        return items

    def load_cached_tweets(self) -> List[RawItem]:
        """Loads curated social media / Twitter feeds."""
        items: List[RawItem] = []
        if not self.tweets_file.exists():
            logger.warning(f"Tweets dataset file not found: {self.tweets_file}")
            return items

        try:
            with open(self.tweets_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                items.append(
                    RawItem(
                        id=r.get("id", f"TWT-{len(items)+1}"),
                        source=r.get("source", "Twitter/X"),
                        text=r.get("text", "").strip(),
                        ticker=r.get("ticker"),
                        company=r.get("company"),
                        timestamp=r.get(
                            "timestamp", datetime.now(timezone.utc).isoformat()
                        ),
                        metadata={
                            "handle": r.get("handle"),
                            "follower_count": r.get("follower_count", 0),
                            "verified": r.get("verified", False),
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Failed to load tweets file: {e}")
        return items

    def fetch_live_stocktwits(
        self, symbol: str = "AAPL", limit: int = 5
    ) -> List[RawItem]:
        """Attempts to fetch live posts from StockTwits public API. Gracefully falls back if unavailable."""
        items: List[RawItem] = []
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                messages = data.get("messages", [])[:limit]
                for idx, msg in enumerate(messages):
                    msg_body = msg.get("body", "").strip()
                    created_at = msg.get(
                        "created_at", datetime.now(timezone.utc).isoformat()
                    )
                    items.append(
                        RawItem(
                            id=f"LIVE-ST-{symbol}-{msg.get('id', idx)}",
                            source="Live StockTwits",
                            text=msg_body,
                            ticker=symbol,
                            timestamp=created_at,
                            metadata={
                                "username": msg.get("user", {}).get("username"),
                                "user_followers": msg.get("user", {}).get(
                                    "followers", 0
                                ),
                            },
                        )
                    )
        except Exception as e:
            logger.info(
                f"Live StockTwits fetch for {symbol} not accessible ({e}). Utilizing cached streams."
            )
        return items

    def fetch_live_yahoo_rss(self, limit: int = 5) -> List[RawItem]:
        """Fetches live market headlines from Yahoo Finance RSS with robust parsing."""
        items: List[RawItem] = []
        url = "https://finance.yahoo.com/news/rssindex"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for idx, item in enumerate(root.findall(".//item")[:limit]):
                    title = item.findtext("title", "").strip()
                    desc = item.findtext("description", "").strip()
                    pub_date = item.findtext("pubDate", "")
                    combined_text = f"{title}. {desc}".strip()
                    items.append(
                        RawItem(
                            id=f"LIVE-YF-{idx+1}",
                            source="Live Yahoo Finance RSS",
                            text=combined_text,
                            ticker=None,
                            timestamp=pub_date
                            or datetime.now(timezone.utc).isoformat(),
                            metadata={"headline": title},
                        )
                    )
        except Exception as e:
            logger.info(
                f"Live Yahoo RSS fetch not accessible ({e}). Utilizing cached streams."
            )
        return items

    def ingest_all(
        self, include_live: bool = False, symbols: Optional[List[str]] = None
    ) -> List[RawItem]:
        """Unified ingestion pipeline pulling from all configured sources."""
        all_items = []
        news_items = self.load_cached_news()
        tweet_items = self.load_cached_tweets()
        all_items.extend(news_items)
        all_items.extend(tweet_items)

        if include_live:
            symbols = symbols or ["NVDA", "AAPL", "JPM"]
            for sym in symbols:
                live_st = self.fetch_live_stocktwits(sym, limit=3)
                all_items.extend(live_st)
            live_yf = self.fetch_live_yahoo_rss(limit=3)
            all_items.extend(live_yf)

        logger.info(f"Ingested a total of {len(all_items)} records from sources.")
        return all_items
