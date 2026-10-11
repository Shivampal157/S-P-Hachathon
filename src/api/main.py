"""
FastAPI REST API Service for Unified AI/NLP Risk Engine and Downstream Modules.
Exposes real-time endpoints for signal consumption, index rebalancing, and portfolio stress testing.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.engine.nlp_risk_engine import NLPRiskEngine, RiskSignal
from src.ingestion.data_loader import MultiSourceIngestionEngine
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.stress_tester import WholesalePortfolioStressTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RiskEngineAPI")

app = FastAPI(
    title="S&P Global & CRISIL AI/NLP Risk Intelligence API",
    description="Unified platform for processing unstructured financial text into machine-readable risk signals, tactical index rebalancing, and wholesale portfolio stress testing.",
    version="1.0.0",
)

# Enable CORS for external dashboards and integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
ingestion_engine = MultiSourceIngestionEngine()
nlp_engine = NLPRiskEngine()
rebalancer = TacticalIndexRebalancer()
stress_tester = WholesalePortfolioStressTester()


class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        description="Unstructured financial news or tweet text to analyze",
        example="Federal Reserve signals emergency 50 bps rate hike amid stubborn inflation",
    )
    ticker: Optional[str] = Field(
        None, description="Optional company ticker (e.g. JPM, NVDA, AAPL)"
    )
    source: str = Field(
        default="User API Submission", description="Source identifier"
    )


class RebalanceRequest(BaseModel):
    tilt_factor: float = Field(
        default=0.50, ge=0.0, le=2.0, description="Sentiment sensitivity tilt alpha"
    )
    min_weight: float = Field(
        default=0.02, ge=0.0, le=0.10, description="Minimum weight constraint"
    )
    max_weight: float = Field(
        default=0.18, ge=0.05, le=0.50, description="Maximum weight constraint"
    )


class StressTestRequest(BaseModel):
    signal_id: Optional[str] = Field(
        None, description="Specific signal ID to simulate"
    )
    custom_headline: Optional[str] = Field(
        None, description="Custom event headline to evaluate and stress test"
    )
    event_classification: Optional[str] = Field(
        None,
        description="Override event class (e.g. Geopolitical, Credit Event, Macroeconomic)",
    )
    impact_score: Optional[int] = Field(
        None, ge=1, le=10, description="Override impact severity score (1-10)"
    )


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "S&P Global & CRISIL AI/NLP Risk Engine",
        "author": "Shivam Pal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "documentation": "/docs",
        "endpoints": [
            "/api/signals",
            "/api/analyze",
            "/api/rebalance",
            "/api/stress-test",
            "/api/portfolio",
            "/api/universe",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/signals")
def get_signals(
    high_impact_only: bool = Query(
        False, description="Filter for high impact signals only (score >= 7)"
    ),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    event_class: Optional[str] = Query(None, description="Filter by event type"),
):
    signals = nlp_engine.load_signals()
    if not signals:
        # Ingest and process if empty
        raw_items = ingestion_engine.ingest_all()
        signals = nlp_engine.process_batch(raw_items)
        nlp_engine.save_signals(signals)

    if high_impact_only:
        signals = [s for s in signals if s.is_high_impact]
    if ticker:
        signals = [
            s
            for s in signals
            if s.ticker and s.ticker.upper() == ticker.upper()
        ]
    if event_class:
        signals = [
            s
            for s in signals
            if s.event_classification.lower() == event_class.lower()
        ]

    return {
        "count": len(signals),
        "signals": [s.to_dict() for s in signals],
    }


@app.post("/api/analyze")
def analyze_text_endpoint(req: TextAnalysisRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    signal = nlp_engine.analyze_text(
        text=req.text,
        ticker=req.ticker,
        source=req.source,
    )
    return signal.to_dict()


@app.post("/api/rebalance")
def run_rebalancing(req: RebalanceRequest = RebalanceRequest()):
    signals = nlp_engine.load_signals()
    if not signals:
        raw_items = ingestion_engine.ingest_all()
        signals = nlp_engine.process_batch(raw_items)
        nlp_engine.save_signals(signals)

    result = rebalancer.rebalance(
        signals=signals,
        tilt_factor=req.tilt_factor,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
    )

    backtest_df = rebalancer.simulate_backtest_performance(days=30)
    analytics = rebalancer.compute_strategy_analytics(backtest_df)

    return {
        "timestamp": result.timestamp,
        "total_stocks": result.total_stocks,
        "turnover": result.turnover,
        "summary_metrics": result.summary_metrics,
        "backtest_analytics": analytics,
        "allocations": [asdict(a) for a in result.allocations],
    }


@app.post("/api/stress-test")
def run_stress_test_endpoint(req: StressTestRequest = StressTestRequest()):
    target_signal: Optional[RiskSignal] = None

    if req.custom_headline:
        target_signal = nlp_engine.analyze_text(
            text=req.custom_headline, source="Custom Stress Test Scenario"
        )
    elif req.signal_id:
        signals = nlp_engine.load_signals()
        matched = [s for s in signals if s.id == req.signal_id]
        if not matched:
            raise HTTPException(
                status_code=404, detail=f"Signal {req.signal_id} not found."
            )
        target_signal = matched[0]
    else:
        # Default to highest impact signal
        signals = nlp_engine.load_signals()
        if not signals:
            raw_items = ingestion_engine.ingest_all()
            signals = nlp_engine.process_batch(raw_items)
            nlp_engine.save_signals(signals)
        high_signals = sorted(signals, key=lambda s: s.impact_score, reverse=True)
        target_signal = high_signals[0] if high_signals else None

    if not target_signal:
        raise HTTPException(
            status_code=400, detail="No suitable signal available for stress testing."
        )

    # Allow explicit overrides if specified
    if req.event_classification:
        target_signal.event_classification = req.event_classification
    if req.impact_score:
        target_signal.impact_score = req.impact_score

    result = stress_tester.run_stress_test(target_signal, force_trigger=True)
    if not result:
        raise HTTPException(
            status_code=400, detail="Stress test failed to execute."
        )

    return asdict(result)


@app.get("/api/portfolio")
def get_portfolio():
    return stress_tester.portfolio_data


@app.get("/api/universe")
def get_universe():
    return {
        "universe_size": len(rebalancer.stocks_universe),
        "stocks": rebalancer.stocks_universe,
        "current_weights": rebalancer.current_weights,
    }
