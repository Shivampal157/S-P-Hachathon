"""
Core Challenge: The AI/NLP Risk Engine
Responsible for ingesting text-based unstructured data (news, social media)
and transforming it into machine-readable financial risk signals:
- Sentiment Score (-1.0 to 1.0)
- Event Classification (Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch, Regulatory/Legal)
- Impact Severity Score (1 to 10)
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("NLPRiskEngine")

# Loughran & McDonald domain-specific financial sentiment modifier weights
FINANCIAL_LEXICON_MODIFIERS: Dict[str, float] = {
    # Extreme Credit & Negative Risk
    "default": -3.8,
    "defaults": -3.8,
    "downgrade": -3.2,
    "downgraded": -3.2,
    "junk": -3.0,
    "bankruptcy": -4.0,
    "bankrupt": -4.0,
    "restructuring": -2.2,
    "insolvency": -3.8,
    "subpoena": -2.8,
    "subpoenas": -2.8,
    "probe": -2.4,
    "fraud": -3.9,
    "litigation": -2.0,
    "sanction": -3.2,
    "sanctions": -3.2,
    "blockade": -3.0,
    "war": -3.5,
    "conflict": -2.8,
    "recession": -3.0,
    "stagflation": -3.5,
    "hike": -1.8,
    "hawkish": -1.6,
    "inflation": -1.5,
    "liquidity crunch": -3.7,
    "dump": -2.5,
    "plunge": -3.0,
    "collapse": -3.8,
    "cut": -1.8,
    "curb": -1.5,
    # High Positive Financial Catalysts
    "buyback": 2.8,
    "dividend": 2.2,
    "outperform": 2.5,
    "beat": 2.2,
    "record": 2.4,
    "growth": 2.0,
    "breakthrough": 3.0,
    "unveils": 2.2,
    "launched": 2.0,
    "acquisition": 2.4,
    "accretive": 2.8,
    "clearance": 2.2,
    "approval": 2.2,
    "approved": 2.2,
    "soaring": 2.8,
    "rally": 2.5,
    "expansion": 2.1,
    "margin expansion": 3.0,
}

# Canonical Event Definitions and Weighted Keyword Associations
EVENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "Geopolitical": {
        "keywords": [
            "sanction",
            "sanctions",
            "tariff",
            "tariffs",
            "war",
            "military",
            "hormuz",
            "blockade",
            "embargo",
            "middle east",
            "sovereign conflict",
            "missile",
            "treaty",
            "geopolitical",
            "shipping corridor",
            "national security",
        ],
        "base_severity": 7.5,
        "systemic_weight": 1.25,
    },
    "Credit Event": {
        "keywords": [
            "downgrade",
            "downgraded",
            "rating",
            "moody's",
            "s&p",
            "default",
            "defaults",
            "junk",
            "covenant",
            "cds",
            "credit spread",
            "liquidity crunch",
            "debt restructuring",
            "insolvency",
            "bankruptcy",
            "delinquency",
            "distressed debt",
        ],
        "base_severity": 8.0,
        "systemic_weight": 1.30,
    },
    "Macroeconomic": {
        "keywords": [
            "federal reserve",
            "fed",
            "fomc",
            "rate hike",
            "interest rate",
            "inflation",
            "cpi",
            "pce",
            "stagflation",
            "gdp",
            "unemployment",
            "yield curve",
            "treasury",
            "basis points",
            "bps",
            "central bank",
            "tightening",
        ],
        "base_severity": 7.0,
        "systemic_weight": 1.20,
    },
    "Merger/Acquisition": {
        "keywords": [
            "acquisition",
            "acquire",
            "merger",
            "bought",
            "buyout",
            "takeover",
            "ftc",
            "antitrust approval",
            "cleared",
            "deal closed",
            "divestiture",
            "spin-off",
            "consortium",
            "tender offer",
        ],
        "base_severity": 5.0,
        "systemic_weight": 0.95,
    },
    "Product Launch": {
        "keywords": [
            "unveils",
            "unveil",
            "launch",
            "launched",
            "superchip",
            "blackwell",
            "breakthrough",
            "compute",
            "innovation",
            "new platform",
            "cloud contract",
            "prototype",
            "rollout",
            "flagship",
            "clinical trial",
        ],
        "base_severity": 4.5,
        "systemic_weight": 0.85,
    },
    "Regulatory/Legal": {
        "keywords": [
            "doj",
            "sec",
            "investigation",
            "probe",
            "subpoena",
            "subpoenas",
            "lawsuit",
            "allegations",
            "penalty",
            "fine",
            "compliance violation",
            "sanctioned",
            "court",
            "prosecutors",
        ],
        "base_severity": 6.5,
        "systemic_weight": 1.10,
    },
}

KNOWN_ENTITIES: Dict[str, Tuple[str, float]] = {
    "NVDA": ("NVIDIA Corporation", 0.95),
    "AAPL": ("Apple Inc.", 0.95),
    "MSFT": ("Microsoft Corporation", 0.95),
    "GOOGL": ("Alphabet Inc.", 0.90),
    "AMZN": ("Amazon.com Inc.", 0.90),
    "META": ("Meta Platforms Inc.", 0.85),
    "JPM": ("JPMorgan Chase & Co.", 0.92),
    "BAC": ("Bank of America", 0.85),
    "TSLA": ("Tesla Inc.", 0.88),
    "BA": ("Boeing Co.", 0.82),
    "XOM": ("Exxon Mobil Corporation", 0.85),
    "CVX": ("Chevron Corporation", 0.80),
    "UNH": ("UnitedHealth Group", 0.85),
    "V": ("Visa Inc.", 0.80),
    "JNJ": ("Johnson & Johnson", 0.80),
}


@dataclass
class RiskSignal:
    id: str
    timestamp: str
    source: str
    ticker: Optional[str]
    company: Optional[str]
    sentiment_score: float  # Normalized between -1.0 and 1.0
    sentiment_label: str  # Positive, Negative, Neutral
    event_classification: str  # Geopolitical, Macroeconomic, Credit Event, etc.
    event_confidence: float  # 0.0 to 1.0
    impact_score: int  # 1 to 10
    is_high_impact: bool  # True if impact_score >= 7
    summary: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NLPRiskEngine:
    """Production-grade AI/NLP Engine for financial risk signal extraction."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (
            Path(__file__).resolve().parent.parent.parent / "data"
        )
        self.vader = SentimentIntensityAnalyzer()
        # Augment VADER with Loughran-McDonald terms
        self.vader.lexicon.update(FINANCIAL_LEXICON_MODIFIERS)
        self.signals_cache_file = self.data_dir / "signals.json"

    def detect_entity(
        self, text: str, explicit_ticker: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], float]:
        """Detects entity ticker, corporate name, and systemic prominence weight."""
        if explicit_ticker and explicit_ticker in KNOWN_ENTITIES:
            comp_name, imp = KNOWN_ENTITIES[explicit_ticker]
            return explicit_ticker, comp_name, imp

        text_upper = text.upper()
        # Check cashtags or ticker words
        for tkr, (comp_name, imp) in KNOWN_ENTITIES.items():
            if f"${tkr}" in text_upper or re.search(rf"\b{tkr}\b", text_upper):
                return tkr, comp_name, imp
            if comp_name.upper() in text_upper or comp_name.split()[0].upper() in text_upper:
                return tkr, comp_name, imp

        # Default fallback
        return explicit_ticker, explicit_ticker, 0.50

    def compute_sentiment(self, text: str) -> Tuple[float, str]:
        """Computes financial sentiment score [-1.0 to 1.0] and categorical label."""
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]

        # Boost sentiment with domain modifiers if present
        text_lower = text.lower()
        modifier_boost = 0.0
        matches = 0
        for term, weight in FINANCIAL_LEXICON_MODIFIERS.items():
            if term in text_lower:
                modifier_boost += weight * 0.12
                matches += 1

        final_score = compound + (modifier_boost / (1 + 0.3 * matches))
        # Clamp to [-1.0, 1.0]
        final_score = max(-1.0, min(1.0, round(final_score, 4)))

        if final_score > 0.15:
            label = "Positive"
        elif final_score < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        return final_score, label

    def classify_event(self, text: str) -> Tuple[str, float]:
        """Classifies financial event into taxonomy and calculates confidence score."""
        text_lower = text.lower()
        best_event = "Macroeconomic"  # Default financial baseline
        best_score = 0.0
        scores: Dict[str, float] = {}

        for event, info in EVENT_TAXONOMY.items():
            kw_matches = sum(1.5 for kw in info["keywords"] if kw in text_lower)
            # Exact phrase bonus
            phrase_bonus = sum(
                2.0 for kw in info["keywords"] if " " in kw and kw in text_lower
            )
            total = kw_matches + phrase_bonus
            scores[event] = total
            if total > best_score:
                best_score = total
                best_event = event

        total_sum = sum(scores.values())
        if total_sum > 0:
            confidence = min(0.98, max(0.40, round(best_score / (total_sum + 0.5), 2)))
        else:
            confidence = 0.50

        return best_event, confidence

    def compute_impact_score(
        self,
        event_class: str,
        sentiment_score: float,
        entity_importance: float,
        confidence: float,
        text: str,
    ) -> int:
        """Computes predicted severity score from 1 to 10."""
        event_info = EVENT_TAXONOMY.get(
            event_class, {"base_severity": 5.0, "systemic_weight": 1.0}
        )
        base = event_info["base_severity"]
        weight = event_info["systemic_weight"]

        sentiment_intensity = abs(sentiment_score)  # [0.0, 1.0]

        # Check urgency keywords
        urgency_tokens = [
            "emergency",
            "crisis",
            "spikes",
            "bloodbath",
            "subpoena",
            "downgrade",
            "blockade",
            "unprecedented",
            "record",
            "billion",
        ]
        urgency_boost = sum(
            0.4 for token in urgency_tokens if token in text.lower()
        )

        raw_impact = (
            base * 0.45 * weight
            + sentiment_intensity * 2.8
            + entity_importance * 1.8
            + confidence * 1.2
            + urgency_boost
        )

        impact_score = int(round(max(1.0, min(10.0, raw_impact))))
        return impact_score

    def analyze_text(
        self,
        text: str,
        ticker: Optional[str] = None,
        source: str = "Direct Feed",
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RiskSignal:
        """End-to-end analysis of unstructured text to generate a RiskSignal."""
        metadata = metadata or {}
        detected_ticker, company, entity_importance = self.detect_entity(text, ticker)

        sentiment_score, sentiment_label = self.compute_sentiment(text)
        event_class, confidence = self.classify_event(text)

        impact_score = self.compute_impact_score(
            event_class=event_class,
            sentiment_score=sentiment_score,
            entity_importance=entity_importance,
            confidence=confidence,
            text=text,
        )

        signal_id = (
            item_id
            or f"SIG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:17]}"
        )
        timestamp = metadata.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )

        # Brief summary
        first_sentence = text.split(".")[0].strip()
        summary = (
            (first_sentence[:140] + "...")
            if len(first_sentence) > 140
            else first_sentence
        )

        return RiskSignal(
            id=signal_id,
            timestamp=timestamp,
            source=source,
            ticker=detected_ticker,
            company=company,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            event_classification=event_class,
            event_confidence=confidence,
            impact_score=impact_score,
            is_high_impact=(impact_score >= 7),
            summary=summary,
            raw_text=text,
            metadata=metadata,
        )

    def process_batch(self, items: List[Any]) -> List[RiskSignal]:
        """Processes a list of RawItems into RiskSignals."""
        signals: List[RiskSignal] = []
        for it in items:
            sig = self.analyze_text(
                text=it.text,
                ticker=it.ticker,
                source=it.source,
                item_id=f"SIG-{it.id}",
                metadata=it.metadata,
            )
            signals.append(sig)
        return signals

    def save_signals(
        self, signals: List[RiskSignal], filepath: Optional[Path] = None
    ) -> Path:
        """Persists structured risk signals to JSON and CSV for downstream consumption."""
        out_path = filepath or self.signals_cache_file
        data = [s.to_dict() for s in signals]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        csv_path = out_path.with_suffix(".csv")
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Persisted {len(signals)} signals to {out_path} and {csv_path}")
        return out_path

    def load_signals(self, filepath: Optional[Path] = None) -> List[RiskSignal]:
        """Loads persisted risk signals."""
        src_path = filepath or self.signals_cache_file
        if not src_path.exists():
            return []
        with open(src_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        signals = []
        for r in raw_list:
            signals.append(RiskSignal(**r))
        return signals
