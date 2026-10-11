"""
S&P Global & CRISIL Campus Hackathon 2026
Unified Financial Risk Intelligence & Tactical Rebalancing Dashboard
Bloomberg-Grade Institutional Dark Mode Interface
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engine.nlp_risk_engine import NLPRiskEngine, RiskSignal
from src.ingestion.data_loader import MultiSourceIngestionEngine
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.stress_tester import WholesalePortfolioStressTester

# Streamlit Page Setup
st.set_page_config(
    page_title="S&P Global | Risk Intelligence & Tactical Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Bloomberg Terminal CSS Styling
st.markdown(
    """
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .status-badge-pos {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .header-box {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border-bottom: 2px solid #38bdf8;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all_engines():
    data_dir = ROOT_DIR / "data"
    ingestion = MultiSourceIngestionEngine(data_dir=data_dir)
    nlp = NLPRiskEngine(data_dir=data_dir)
    rebalancer = TacticalIndexRebalancer(data_dir=data_dir)
    stress_tester = WholesalePortfolioStressTester(data_dir=data_dir)
    return ingestion, nlp, rebalancer, stress_tester


ingestion_engine, nlp_engine, rebalancer_engine, stress_engine = load_all_engines()

# Pre-load or generate signals
if "signals" not in st.session_state:
    cached_signals = nlp_engine.load_signals()
    if not cached_signals:
        raw = ingestion_engine.ingest_all(include_live=False)
        cached_signals = nlp_engine.process_batch(raw)
        nlp_engine.save_signals(cached_signals)
    st.session_state.signals = cached_signals


# Sidebar Controls
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/S%26P_Global_logo.svg/1200px-S%26P_Global_logo.svg.png",
        width=180,
    )
    st.markdown("### 🏆 S&P Global & CRISIL")
    st.caption("Campus Hackathon 2026 | Round 3 Finalist")
    st.markdown("---")

    st.markdown("#### ⚙️ Pipeline Configuration")
    include_live_feed = st.checkbox("Include Live Feeds (Yahoo RSS / StockTwits)", value=False)
    if st.button("🔄 Refresh & Reprocess Feeds", use_container_width=True):
        with st.spinner("Ingesting heterogeneous feeds..."):
            raw_items = ingestion_engine.ingest_all(include_live=include_live_feed)
            st.session_state.signals = nlp_engine.process_batch(raw_items)
            nlp_engine.save_signals(st.session_state.signals)
            st.success(f"Processed {len(st.session_state.signals)} signals!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🎯 Active Modules")
    st.markdown("- **Core:** AI/NLP Risk Engine")
    st.markdown("- **Module A:** Tactical S&P Rebalancer")
    st.markdown("- **Module B:** Wholesale Bank Stress Tester")

    st.markdown("---")
    st.caption(f"Engine Status: ONLINE | {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


# Header Banner
st.markdown(
    """
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 26px; color: #f8fafc; font-weight: 800;">
                S&P Global & CRISIL | AI/NLP Financial Risk Intelligence Platform
            </h1>
            <p style="margin:4px 0 0 0; color: #94a3b8; font-size: 14px;">
                Real-time unstructured data ingestion, FinBERT sentiment scoring, dynamic tactical index rebalancing & Basel III wholesale portfolio stress testing.
            </p>
        </div>
        <div>
            <span class="status-badge-pos">● ENGINE ACTIVE</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Executive KPI Ribbon
signals = st.session_state.signals
total_signals = len(signals)
high_impact_signals = [s for s in signals if s.is_high_impact]
avg_sentiment = float(np.mean([s.sentiment_score for s in signals])) if signals else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Processed Risk Signals</div>
        <div class="metric-value">{total_signals}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">Dual Ingestion Channels</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">High-Impact Alerts (Score ≥ 7)</div>
        <div class="metric-value" style="color: #f87171;">{len(high_impact_signals)}</div>
        <div style="font-size:12px; color:#ef4444; margin-top:4px;">Auto-triggers Stress Test</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with kpi3:
    sent_color = "#4ade80" if avg_sentiment >= 0 else "#f87171"
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Net Market Sentiment</div>
        <div class="metric-value" style="color: {sent_color};">{avg_sentiment:+.2f}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">Scale: -1.0 to +1.0</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Downstream Coverage</div>
        <div class="metric-value" style="color: #a78bfa;">100% (Both)</div>
        <div style="font-size:12px; color:#818cf8; margin-top:4px;">Module A + Module B Deployed</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Application Tabs
tab_engine, tab_mod_a, tab_mod_b, tab_api = st.tabs(
    [
        "🧠 1. The Core AI/NLP Risk Engine",
        "⚖️ 2. Module A: Tactical Index Rebalancer",
        "💥 3. Module B: Wholesale Portfolio Stress Testing",
        "🔌 4. Architecture & REST API Docs",
    ]
)


# ==============================================================================
# TAB 1: CORE AI/NLP RISK ENGINE
# ==============================================================================
with tab_engine:
    st.markdown("### 📡 Real-Time Financial Signal Ingestion & Intelligence")
    st.write(
        "The engine ingests unstructured feeds (Financial News & Social Media), applies domain-specific Loughran-McDonald lexicon weighting and zero-shot Bayesian classification, outputting **Sentiment Score (-1.0 to 1.0)**, **Event Classification**, and **Impact Severity Score (1-10)**."
    )

    col_feed, col_tester = st.columns([1.4, 1.0])

    with col_feed:
        st.markdown("#### 📋 Ingested Risk Intelligence Feed")

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            filter_impact = st.checkbox("Show only High Impact (Severity ≥ 7)", value=False)
        with f2:
            event_types = ["All"] + sorted(list(set(s.event_classification for s in signals)))
            selected_event = st.selectbox("Filter Event Classification", event_types)

        filtered_signals = signals
        if filter_impact:
            filtered_signals = [s for s in filtered_signals if s.is_high_impact]
        if selected_event != "All":
            filtered_signals = [s for s in filtered_signals if s.event_classification == selected_event]

        # Display signals in cards
        for sig in filtered_signals[:8]:
            badge_color = "#22c55e" if sig.sentiment_score > 0.1 else ("#ef4444" if sig.sentiment_score < -0.1 else "#94a3b8")
            impact_badge = "🔥 HIGH IMPACT" if sig.is_high_impact else "NORMAL"
            impact_bg = "rgba(239, 68, 68, 0.2)" if sig.is_high_impact else "rgba(100, 116, 139, 0.2)"
            impact_color = "#f87171" if sig.is_high_impact else "#cbd5e1"

            with st.container():
                st.markdown(
                    f"""
                <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div>
                            <span style="color: #38bdf8; font-weight: 700; font-size: 14px;">{sig.ticker or 'MACRO'}</span>
                            <span style="color: #64748b; font-size: 12px; margin-left: 8px;">{sig.source} | {sig.id}</span>
                        </div>
                        <div>
                            <span style="background-color: {impact_bg}; color: {impact_color}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700;">
                                Impact {sig.impact_score}/10 ({impact_badge})
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 13px; color: #f1f5f9; font-weight: 500; margin-bottom: 8px;">
                        {sig.summary}
                    </div>
                    <div style="display: flex; gap: 12px; font-size: 12px;">
                        <span style="color: {badge_color}; font-weight: 600;">Sentiment: {sig.sentiment_score:+.2f} ({sig.sentiment_label})</span>
                        <span style="color: #a78bfa; font-weight: 600;">Event: {sig.event_classification} ({sig.event_confidence*100:.0f}%)</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with col_tester:
        st.markdown("#### 🧪 Interactive NLP Sandbox (Live Jury Demo)")
        st.caption("Type any real-world financial headline to evaluate the AI model in real time:")

        sample_prompts = [
            "Custom Input...",
            "Escalating trade war and maritime blockade causes crude oil to surge past $120/bbl",
            "Moody's strips tier-1 commercial bank of investment-grade rating, downgrading to junk status",
            "Federal Reserve signals unexpected 50 bps rate hike to combat persistent core inflation",
            "NVIDIA unveils breakthrough Quantum AI superchip with 4x compute efficiency, surging order pipeline",
            "Department of Justice opens formal antitrust probe into major autonomous vehicle manufacturer",
        ]
        choice = st.selectbox("Select Sample Headline or Type Below:", sample_prompts)

        default_text = "" if choice == "Custom Input..." else choice
        input_text = st.text_area(
            "Financial Text Input:",
            value=default_text,
            height=110,
            placeholder="e.g. JPMorgan expands wholesale balance sheet amid commercial real estate defaults...",
        )
        input_ticker = st.text_input("Associated Ticker (Optional):", value="", placeholder="e.g. JPM, NVDA, AAPL")

        if st.button("🚀 Analyze Risk Signal Live", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Processing through FinBERT & Bayesian Classifier..."):
                    analyzed_sig = nlp_engine.analyze_text(
                        text=input_text,
                        ticker=input_ticker if input_ticker.strip() else None,
                        source="Live Jury Sandbox",
                    )

                st.success("Analysis Complete!")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=analyzed_sig.sentiment_score,
                            title={"text": "Sentiment Score", "font": {"size": 14, "color": "#f8fafc"}},
                            number={"font": {"color": "#38bdf8"}},
                            gauge={
                                "axis": {"range": [-1.0, 1.0], "tickcolor": "#94a3b8"},
                                "bar": {"color": "#38bdf8"},
                                "steps": [
                                    {"range": [-1.0, -0.2], "color": "rgba(239, 68, 68, 0.4)"},
                                    {"range": [-0.2, 0.2], "color": "rgba(148, 163, 184, 0.3)"},
                                    {"range": [0.2, 1.0], "color": "rgba(34, 197, 94, 0.4)"},
                                ],
                            },
                        )
                    )
                    fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with res_col2:
                    fig_impact = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=analyzed_sig.impact_score,
                            title={"text": "Impact Severity (1-10)", "font": {"size": 14, "color": "#f8fafc"}},
                            number={"font": {"color": "#ef4444" if analyzed_sig.is_high_impact else "#f59e0b"}},
                            gauge={
                                "axis": {"range": [1, 10], "tickcolor": "#94a3b8"},
                                "bar": {"color": "#ef4444" if analyzed_sig.is_high_impact else "#f59e0b"},
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 7,
                                },
                            },
                        )
                    )
                    fig_impact.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_impact, use_container_width=True)

                st.markdown(
                    f"""
                <div style="background-color: #0f172a; border: 1px solid #334155; padding: 12px; border-radius: 8px;">
                    <div><b>Event Classification:</b> <span style="color: #a78bfa;">{analyzed_sig.event_classification}</span> (Confidence: {analyzed_sig.event_confidence*100:.1f}%)</div>
                    <div><b>Detected Entity:</b> <span style="color: #38bdf8;">{analyzed_sig.ticker or 'Systemic/Macro'}</span> ({analyzed_sig.company or 'Market-wide'})</div>
                    <div><b>Stress Trigger:</b> <span style="color: {'#ef4444' if analyzed_sig.is_high_impact else '#22c55e'};">{'⚠️ YES (Impact ≥ 7)' if analyzed_sig.is_high_impact else 'NO (Normal Market Conditions)'}</span></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter text to analyze.")


# ==============================================================================
# TAB 2: MODULE A - TACTICAL INDEX REBALANCING
# ==============================================================================
with tab_mod_a:
    st.markdown("### ⚖️ Module A: Tactical Index Rebalancer")
    st.write(
        "Subscribes to sentiment scores from the NLP Risk Engine to dynamically tilt stock weights across the S&P 100 constituent universe. "
        "Positive sentiment increases allocation; negative sentiment decreases allocation, subject to institutional diversification constraints."
    )

    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
    with ctrl_c1:
        tilt_factor = st.slider("Sentiment Tilt Sensitivity (Alpha):", min_value=0.1, max_value=1.5, value=0.6, step=0.1)
    with ctrl_c2:
        min_weight_pct = st.slider("Min Single-Stock Floor (%):", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100.0
    with ctrl_c3:
        max_weight_pct = st.slider("Max Single-Stock Cap (%):", min_value=10.0, max_value=25.0, value=18.0, step=1.0) / 100.0

    # Run Rebalancing
    rebalance_res = rebalancer_engine.rebalance(
        signals=st.session_state.signals,
        tilt_factor=tilt_factor,
        min_weight=min_weight_pct,
        max_weight=max_weight_pct,
    )
    alloc_df = rebalance_res.to_dataframe()

    # Rebalance KPI ribbon
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric("Total Universe Stocks", rebalance_res.total_stocks)
    with m_c2:
        st.metric("Portfolio Turnover", f"{rebalance_res.turnover*100:.1f}%")
    with m_c3:
        top_inc = alloc_df.sort_values(by="weight_change", ascending=False).iloc[0]
        st.metric("Top Overweight", f"{top_inc['ticker']} (+{top_inc['weight_change']*100:.2f}%)")
    with m_c4:
        top_dec = alloc_df.sort_values(by="weight_change", ascending=True).iloc[0]
        st.metric("Top Underweight", f"{top_dec['ticker']} ({top_dec['weight_change']*100:.2f}%)")

    st.markdown("#### 📊 Dynamic Asset Weight Allocation: Benchmark vs Sentiment-Tilted")

    # Plotly Bar Chart comparing Benchmark vs Stressed/Target weights
    fig_alloc = go.Figure()
    fig_alloc.add_trace(
        go.Bar(
            x=alloc_df["ticker"],
            y=alloc_df["benchmark_weight"] * 100,
            name="Benchmark S&P Weight (%)",
            marker_color="#64748b",
        )
    )
    # Color-coded target weights
    colors = [
        "#22c55e" if d == "INCREASED" else ("#ef4444" if d == "DECREASED" else "#38bdf8")
        for d in alloc_df["direction"]
    ]
    fig_alloc.add_trace(
        go.Bar(
            x=alloc_df["ticker"],
            y=alloc_df["target_weight"] * 100,
            name="Tactical Sentiment Weight (%)",
            marker_color=colors,
        )
    )
    fig_alloc.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(title="Ticker Symbol", gridcolor="#334155"),
        yaxis=dict(title="Portfolio Weight (%)", gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Historical Backtesting Performance
    st.markdown("#### 📈 Backtested 30-Day Cumulative Return vs Benchmark")
    backtest_df = rebalancer_engine.simulate_backtest_performance(days=30)
    analytics = rebalancer_engine.compute_strategy_analytics(backtest_df)

    fig_backtest = px.line(
        backtest_df,
        x="date",
        y=["Tactical_NLP_Index", "SP500_Benchmark", "Equal_Weight_Index"],
        labels={"value": "Index Level (Base 100)", "variable": "Strategy"},
        color_discrete_map={
            "Tactical_NLP_Index": "#38bdf8",
            "SP500_Benchmark": "#f59e0b",
            "Equal_Weight_Index": "#94a3b8",
        },
    )
    fig_backtest.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

    # Strategy Analytics Cards
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Tactical Return", f"{analytics['total_return_pct']}%", f"+{analytics['excess_alpha_pct']}% Alpha")
    with b2:
        st.metric("Benchmark S&P Return", f"{analytics['benchmark_return_pct']}%")
    with b3:
        st.metric("Sharpe Ratio", f"{analytics['sharpe_ratio']}")
    with b4:
        st.metric("Max Drawdown", f"{analytics['max_drawdown_pct']}%")

    with st.expander("🔍 View Complete Allocation Matrix Table"):
        st.dataframe(
            alloc_df[
                [
                    "ticker",
                    "company_name",
                    "sector",
                    "benchmark_weight",
                    "sentiment_score",
                    "target_weight",
                    "weight_change",
                    "direction",
                ]
            ].style.format(
                {
                    "benchmark_weight": "{:.2%}",
                    "target_weight": "{:.2%}",
                    "weight_change": "{:+.2%}",
                    "sentiment_score": "{:+.2f}",
                }
            ),
            use_container_width=True,
        )


# ==============================================================================
# TAB 3: MODULE B - WHOLESALE PORTFOLIO STRESS TESTING
# ==============================================================================
with tab_mod_b:
    st.markdown("### 💥 Module B: Strategic Wholesale Banking Portfolio Stress Testing")
    st.write(
        "Simulates systemic macroeconomic, geopolitical, and credit shocks on a synthetic **$500M wholesale banking book** "
        "(Corporate Loans, Bonds, Swaps, and CDS). When the NLP Risk Engine detects an event with **Impact Score > 7**, the stress testing engine triggers automatically."
    )

    # High Impact Trigger Selector
    high_impacts = [s for s in st.session_state.signals if s.is_high_impact]
    st.markdown("#### ⚡ Event Trigger Selection")

    if high_impacts:
        signal_options = {
            f"[{s.event_classification}] (Impact: {s.impact_score}/10) - {s.summary[:80]}...": s
            for s in high_impacts
        }
        selected_label = st.selectbox(
            "Select Real-World Event Triggered by NLP Engine:", list(signal_options.keys())
        )
        active_trigger_signal = signal_options[selected_label]
    else:
        active_trigger_signal = st.session_state.signals[0]

    # Run Stress Test
    stress_result = stress_engine.run_stress_test(active_trigger_signal, force_trigger=True)

    # Stress Test Results Overview
    st.markdown("---")
    st.markdown(f"#### 🚨 Stress Test Results: {stress_result.shocks_applied.scenario_name}")

    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Pre-Stress Portfolio MtM</div>
            <div class="metric-value">${stress_result.baseline_portfolio_mtm/1e6:.2f}M</div>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">11 Wholesale Assets</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with sk2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Post-Stress Portfolio MtM</div>
            <div class="metric-value" style="color:#f87171;">${stress_result.stressed_portfolio_mtm/1e6:.2f}M</div>
            <div style="font-size:12px; color:#ef4444; margin-top:4px;">-${stress_result.total_loss/1e6:.2f}M ({stress_result.total_loss_pct:.2f}%)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with sk3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">99% 10-Day VaR & CVaR</div>
            <div class="metric-value" style="color:#fbbf24;">${stress_result.var_99_10d/1e6:.2f}M</div>
            <div style="font-size:12px; color:#f59e0b; margin-top:4px;">Expected Shortfall: ${stress_result.cvar_99_10d/1e6:.2f}M</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with sk4:
        cet1_color = "#4ade80" if stress_result.stressed_cet1_ratio >= 0.105 else "#ef4444"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Basel III CET1 Capital Ratio</div>
            <div class="metric-value" style="color:{cet1_color};">{stress_result.stressed_cet1_ratio*100:.2f}%</div>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">Baseline: {stress_result.baseline_cet1_ratio*100:.1f}% (Min: 10.5%)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall & Asset Breakdown Charts
    asset_df = stress_result.to_dataframe()

    c_chart1, c_chart2 = st.columns([1.3, 1.0])

    with c_chart1:
        st.markdown("##### 📉 Asset-Level Mark-to-Market Valuation Impact ($)")
        fig_assets = go.Figure()
        fig_assets.add_trace(
            go.Bar(
                name="Baseline MtM",
                x=asset_df["asset_id"],
                y=asset_df["base_mtm_value"] / 1e6,
                marker_color="#38bdf8",
            )
        )
        fig_assets.add_trace(
            go.Bar(
                name="Stressed MtM",
                x=asset_df["asset_id"],
                y=asset_df["stressed_mtm_value"] / 1e6,
                marker_color="#ef4444",
            )
        )
        fig_assets.update_layout(
            barmode="group",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(title="MtM ($ Millions)", gridcolor="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_assets, use_container_width=True)

    with c_chart2:
        st.markdown("##### 🍰 Portfolio Losses by Asset Class")
        class_losses = asset_df.groupby("asset_type")["pnl_loss"].sum().reset_index()
        fig_pie = px.pie(
            class_losses,
            names="asset_type",
            values="pnl_loss",
            hole=0.45,
            color_discrete_sequence=["#ef4444", "#f97316", "#38bdf8", "#a78bfa"],
        )
        fig_pie.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed Asset Breakdown Table
    with st.expander("📋 View Wholesale Assets Breakdown & Risk Drivers"):
        st.dataframe(
            asset_df[
                [
                    "asset_id",
                    "asset_type",
                    "counterparty",
                    "sector",
                    "base_mtm_value",
                    "stressed_mtm_value",
                    "pnl_loss",
                    "pnl_loss_pct",
                    "risk_driver",
                ]
            ].style.format(
                {
                    "base_mtm_value": "${:,.0f}",
                    "stressed_mtm_value": "${:,.0f}",
                    "pnl_loss": "${:,.0f}",
                    "pnl_loss_pct": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )


# ==============================================================================
# TAB 4: ARCHITECTURE & API EXPLORER
# ==============================================================================
with tab_api:
    st.markdown("### 🔌 System Architecture & REST API Service")
    st.write(
        "The platform is completely modular and adheres to microservices principles. "
        "Any downstream application can consume real-time signals via the production-ready FastAPI endpoints."
    )

    st.markdown("#### 📐 End-to-End System Architecture")
    st.code(
        """
+----------------------------------------------------------------------------------------------------+
|                               UNIFIED FINANCIAL RISK INTELLIGENCE PLATFORM                         |
+----------------------------------------------------------------------------------------------------+
                                                  │
                 ┌────────────────────────────────┴───────────────────────────────┐
                 ▼                                                                ▼
    [ Financial News Articles ]                                     [ Financial Social Media / Tweets ]
   - Financial Times, Bloomberg, Reuters                           - FinTwit, StockTwits Streams
   - SEC Regulatory Filings, Yahoo RSS                             - Cashtags ($NVDA, $JPM), Viral Sentiment
                 │                                                                │
                 └────────────────────────────────┬───────────────────────────────┘
                                                  ▼
                         ┌──────────────────────────────────────────────────┐
                         │       MULTI-SOURCE INGESTION & NORMALIZATION     │
                         │    Schema Standardization & Ticker Resolution    │
                         └────────────────────────┬─────────────────────────┘
                                                  ▼
                         ┌──────────────────────────────────────────────────┐
                         │            CORE AI / NLP RISK ENGINE             │
                         │                                                  │
                         │  1. Financial Sentiment (-1.0 to 1.0)            │
                         │     (Loughran-McDonald + VADER Domain Weights)   │
                         │  2. Event Classification                         │
                         │     (Geopolitical, Credit, Macro, M&A, etc.)     │
                         │  3. Predicted Impact Severity Score (1 to 10)    │
                         │     (Systemic weights, market cap prominence)    │
                         └────────────────────────┬─────────────────────────┘
                                                  │
                 ┌────────────────────────────────┴───────────────────────────────┐
                 ▼                                                                ▼
   [ DOWNSTREAM MODULE A ]                                          [ DOWNSTREAM MODULE B ]
  Tactical Index Rebalancer                                      Wholesale Portfolio Stress Tester
  - Universe: S&P 100 Constituent Stocks                         - $500M Wholesale Assets (Loans, Bonds, Swaps)
  - Real-time Quadratic Sentiment Tilt                           - Auto-Trigger when Impact Score > 7
  - Institutional Bounds: 2% Min, 18% Max                        - Macro Shocks: Rates, Spreads, PD/LGD Multipliers
  - Alpha generation vs Benchmark SPY                            - MtM Loss, 99% VaR/CVaR, Basel III CET1 Capital
                 │                                                                │
                 └────────────────────────────────┬───────────────────────────────┘
                                                  ▼
                         ┌──────────────────────────────────────────────────┐
                         │        FASTAPI REST SERVICE & STREAMLIT UI       │
                         │  /api/signals  /api/analyze  /api/rebalance ...  │
                         └──────────────────────────────────────────────────┘
        """,
        language="text",
    )

    st.markdown("#### 🌐 Available REST Endpoints")
    endpoints = [
        {"Method": "GET", "Endpoint": "/api/signals", "Description": "Retrieve filtered structured risk signals (JSON)"},
        {"Method": "POST", "Endpoint": "/api/analyze", "Description": "Ingest & analyze arbitrary financial text in real time"},
        {"Method": "POST", "Endpoint": "/api/rebalance", "Description": "Trigger tactical index rebalancing for S&P 100 stocks"},
        {"Method": "POST", "Endpoint": "/api/stress-test", "Description": "Execute wholesale portfolio stress simulation"},
        {"Method": "GET", "Endpoint": "/api/portfolio", "Description": "Retrieve current wholesale banking assets book"},
        {"Method": "GET", "Endpoint": "/api/universe", "Description": "Retrieve S&P 100 universe constituents and baseline weights"},
    ]
    st.table(pd.DataFrame(endpoints))

    st.markdown("#### 💾 Download Datasets & Intelligence Feeds")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        signals_json = json.dumps([s.to_dict() for s in st.session_state.signals], indent=2)
        st.download_button(
            "📥 Download signals.json",
            data=signals_json,
            file_name="signals.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_d2:
        signals_df = pd.DataFrame([s.to_dict() for s in st.session_state.signals])
        st.download_button(
            "📥 Download signals.csv",
            data=signals_df.to_csv(index=False),
            file_name="signals.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_d3:
        portfolio_json = json.dumps(stress_engine.portfolio_data, indent=2)
        st.download_button(
            "📥 Download wholesale_portfolio.json",
            data=portfolio_json,
            file_name="wholesale_portfolio.json",
            mime="application/json",
            use_container_width=True,
        )
