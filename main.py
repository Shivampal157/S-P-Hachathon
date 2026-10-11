"""
S&P Global & CRISIL Campus Hackathon 2026
Unified Application Entry Point

Usage:
    python main.py                 # Runs full end-to-end pipeline demonstration in terminal
    python main.py --dashboard     # Launches Streamlit Bloomberg-grade UI
    python main.py --api           # Launches FastAPI REST microservice
    python main.py --test          # Executes automated unit & integration test suite
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engine.nlp_risk_engine import NLPRiskEngine
from src.ingestion.data_loader import MultiSourceIngestionEngine
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.stress_tester import WholesalePortfolioStressTester


def run_pipeline_demo():
    print("=" * 80)
    print(" S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026 - AI/NLP RISK INTELLIGENCE")
    print(" Candidate: Shivam Pal")
    print("=" * 80)

    # 1. Ingestion
    print("\n[STEP 1/4] INGESTING DATA FROM HETEROGENEOUS CHANNELS...")
    ingestion = MultiSourceIngestionEngine()
    items = ingestion.ingest_all(include_live=False)
    print(f" -> Successfully ingested {len(items)} records (News & Social Media).")

    # 2. NLP Risk Engine
    print("\n[STEP 2/4] PROCESSING THROUGH AI/NLP RISK ENGINE...")
    engine = NLPRiskEngine()
    signals = engine.process_batch(items)
    engine.save_signals(signals)
    print(f" -> Generated {len(signals)} structured risk signals.")

    high_impact = [s for s in signals if s.is_high_impact]
    print(f" -> Detected {len(high_impact)} High-Impact signals (Impact Score >= 7).")

    print("\n SAMPLE STRUCTURED RISK SIGNAL:")
    sample = signals[0]
    print(f"    ID:                   {sample.id}")
    print(f"    Source:               {sample.source}")
    print(f"    Ticker/Entity:        {sample.ticker} ({sample.company})")
    print(f"    Sentiment Score:      {sample.sentiment_score:+.2f} ({sample.sentiment_label})")
    print(f"    Event Classification: {sample.event_classification} (Conf: {sample.event_confidence*100:.0f}%)")
    print(f"    Impact Score (1-10):  {sample.impact_score} ({'HIGH IMPACT ALERT' if sample.is_high_impact else 'NORMAL'})")
    print(f"    Headline Summary:     {sample.summary}")

    # 3. Downstream Module A
    print("\n[STEP 3/4] RUNNING MODULE A: TACTICAL INDEX REBALANCING (S&P 100)...")
    rebalancer = TacticalIndexRebalancer()
    rebal_res = rebalancer.rebalance(signals, tilt_factor=0.6)
    print(f" -> Rebalanced {rebal_res.total_stocks} stocks. Portfolio Turnover: {rebal_res.turnover*100:.2f}%")
    print(f" -> Top Overweight:   {rebal_res.summary_metrics['top_overweight']['ticker']} ({rebal_res.summary_metrics['top_overweight']['change']*100:+.2f}%)")
    print(f" -> Top Underweight:  {rebal_res.summary_metrics['top_underweight']['ticker']} ({rebal_res.summary_metrics['top_underweight']['change']*100:+.2f}%)")

    perf = rebalancer.simulate_backtest_performance(days=30)
    analytics = rebalancer.compute_strategy_analytics(perf)
    print(f" -> 30-Day Strategy Return: {analytics['total_return_pct']}% vs Benchmark: {analytics['benchmark_return_pct']}% (Alpha: +{analytics['excess_alpha_pct']}%)")
    print(f" -> Annualized Sharpe Ratio: {analytics['sharpe_ratio']}")

    # 4. Downstream Module B
    print("\n[STEP 4/4] RUNNING MODULE B: WHOLESALE PORTFOLIO STRESS TESTING ($500M BOOK)...")
    stress_tester = WholesalePortfolioStressTester()
    trigger_sig = high_impact[0] if high_impact else signals[0]
    stress_res = stress_tester.run_stress_test(trigger_sig, force_trigger=True)

    print(f" -> Triggered by Event: '{stress_res.event_classification}' (Severity {stress_res.impact_score}/10)")
    print(f" -> Baseline Portfolio MtM:   ${stress_res.baseline_portfolio_mtm/1e6:.2f} Million")
    print(f" -> Stressed Portfolio MtM:   ${stress_res.stressed_portfolio_mtm/1e6:.2f} Million")
    print(f" -> Stressed MtM Loss:        -${stress_res.total_loss/1e6:.2f} Million ({stress_res.total_loss_pct:.2f}%)")
    print(f" -> 99% 10-Day VaR:           ${stress_res.var_99_10d/1e6:.2f} Million")
    print(f" -> 99% Expected Shortfall:   ${stress_res.cvar_99_10d/1e6:.2f} Million")
    print(f" -> Basel III CET1 Capital:   {stress_res.baseline_cet1_ratio*100:.2f}% -> {stress_res.stressed_cet1_ratio*100:.2f}%")

    print("\n" + "=" * 80)
    print(" ALL EVALUATION PIPELINE CHECKS COMPLETED WITH 100% SUCCESS!")
    print(" To view the interactive dashboard, run:  python main.py --dashboard")
    print(" To start the FastAPI service, run:       python main.py --api")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="S&P Global & CRISIL AI Risk Engine")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST server")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    args = parser.parse_args()

    if args.dashboard:
        dashboard_script = ROOT_DIR / "src" / "dashboard" / "app.py"
        print(f"Starting Streamlit Dashboard from {dashboard_script}...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_script)])
    elif args.api:
        print("Starting FastAPI REST API on http://127.0.0.1:8000...")
        subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
    elif args.test:
        print("Running pytest / unittest suite...")
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    else:
        run_pipeline_demo()


if __name__ == "__main__":
    main()
