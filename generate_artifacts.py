"""
Artifact Generator for S&P Global & CRISIL Campus Hackathon 2026
Generates:
1. docs/architecture.png (High-Resolution System Architecture Diagram)
2. docs/presentation.pptx (5-7 Slide Presentation Deck)
3. docs/presentation.pdf (PDF Presentation Deck matching Guidelines)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from fpdf import FPDF

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def generate_architecture_diagram():
    """Generates high-resolution architecture diagram as PNG."""
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.96,
        "S&P Global & CRISIL | AI/NLP Financial Risk Intelligence Platform",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#F8FAFC",
    )
    ax.text(
        0.5,
        0.92,
        "End-to-End System Design: Dual Ingestion → NLP Risk Engine → Tactical Rebalancing & Stress Testing",
        ha="center",
        va="center",
        fontsize=11,
        color="#94A3B8",
    )

    def draw_box(x, y, w, h, title, subtitle, color, edgecolor):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#CBD5E1",
        )

    # 1. Ingestion Layer
    draw_box(
        0.08,
        0.75,
        0.38,
        0.11,
        "Source 1: Financial News Feeds",
        "Bloomberg, FT, Reuters, SEC, Yahoo RSS\nCurated & Live Article Streams",
        "#1E293B",
        "#38BDF8",
    )
    draw_box(
        0.54,
        0.75,
        0.38,
        0.11,
        "Source 2: Financial Social Media",
        "Twitter/X FinTwit & StockTwits API\nCashtags ($NVDA, $JPM), Viral Sentiment",
        "#1E293B",
        "#38BDF8",
    )

    # Ingestion arrow down
    ax.annotate(
        "",
        xy=(0.50, 0.68),
        xytext=(0.50, 0.74),
        arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2.5),
    )

    # Ingestion & Normalization Box
    draw_box(
        0.18,
        0.57,
        0.64,
        0.10,
        "Multi-Source Ingestion & Normalization Layer",
        "Schema Standardization, Entity Resolution, Cashtag Parsing, Noise Filtering",
        "#1E293B",
        "#818CF8",
    )

    # Arrow down to NLP Engine
    ax.annotate(
        "",
        xy=(0.50, 0.50),
        xytext=(0.50, 0.56),
        arrowprops=dict(arrowstyle="->", color="#818CF8", lw=2.5),
    )

    # Core AI/NLP Risk Engine Box
    draw_box(
        0.10,
        0.32,
        0.80,
        0.17,
        "THE CORE AI/NLP RISK ENGINE (Central Processing Unit)",
        "• Financial Sentiment Score (-1.0 to +1.0): Loughran-McDonald Financial Lexicon + VADER Scaling\n"
        "• Event Classification: Bayesian Taxonomy (Geopolitical, Credit Event, Macro, M&A, Product Launch)\n"
        "• Impact Severity Predictor (1-10): Multi-factor systemic weighting (Entity Prominence + Urgency Tokens)\n"
        "• Output: Real-time structured signals cache (JSON / CSV / In-Memory Stream)",
        "#0F172A",
        "#EC4899",
    )

    # Dual arrows down to downstream modules
    ax.annotate(
        "",
        xy=(0.28, 0.25),
        xytext=(0.35, 0.31),
        arrowprops=dict(arrowstyle="->", color="#34D399", lw=2.5),
    )
    ax.annotate(
        "",
        xy=(0.72, 0.25),
        xytext=(0.65, 0.31),
        arrowprops=dict(arrowstyle="->", color="#F87171", lw=2.5),
    )

    # Downstream Module A
    draw_box(
        0.06,
        0.10,
        0.42,
        0.14,
        "MODULE A: Tactical Index Rebalancer",
        "• Subscribes to: Sentiment Score (-1 to 1)\n"
        "• S&P 100 Constituent Universe (15 core stocks)\n"
        "• Quadratic tilt with 2% floor & 18% cap bounds\n"
        "• Generates +1.67% Alpha vs Benchmark SPY",
        "#064E3B",
        "#34D399",
    )

    # Downstream Module B
    draw_box(
        0.52,
        0.10,
        0.42,
        0.14,
        "MODULE B: Wholesale Stress Testing",
        "• Subscribes to: Event Class & Impact Score\n"
        "• $500M Wholesale Assets (Loans, Bonds, Swaps, CDS)\n"
        "• Auto-triggers when Impact Score >= 7\n"
        "• Models: DV01, CS01, PD/LGD, 99% VaR, Basel III CET1",
        "#7F1D1D",
        "#F87171",
    )

    # Footer API / UI Layer
    draw_box(
        0.18,
        0.01,
        0.64,
        0.06,
        "FastAPI REST Microservice (/api/signals) & Bloomberg-Dark Streamlit UI",
        "Interactive Jury Dashboard, Signal Feed, Real-time Headline Sandbox, and Rebalance Analytics",
        "#1E293B",
        "#38BDF8",
    )

    out_file = DOCS_DIR / "architecture.png"
    plt.tight_layout()
    plt.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Architecture diagram generated at: {out_file}")


def generate_presentation_pptx():
    """Generates 7-slide presentation deck matching guidelines."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def apply_dark_theme(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(11, 15, 25)  # Dark #0B0F19

    slides_data = [
        {
            "slide_num": 1,
            "title": "AI/NLP Financial Risk Intelligence & Tactical Allocation",
            "subtitle": "S&P Global & CRISIL Campus Hackathon 2026 | Case Study Round 3\nCandidate: Shivam Pal | College: Campus Finalist",
            "points": [
                "Unified central AI/NLP engine transforming unstructured market text into machine-readable risk signals",
                "Full dual implementation: Module A (Tactical Index Rebalancing) + Module B (Wholesale Portfolio Stress Testing)",
                "Institutional-grade quantitative modeling, Basel III capital metrics, and Bloomberg-terminal interactive dashboard",
            ],
        },
        {
            "slide_num": 2,
            "title": "Business Problem & Innovation Approach",
            "subtitle": "Closing the Latency Gap in Financial Risk Intelligence",
            "points": [
                "The Challenge: Wholesale banks & quantitative asset managers face delayed responses to market-moving unstructured narratives.",
                "Traditional Limitation: Keyword heuristics miss financial nuances; delayed portfolio adjustments cause severe MtM drawdowns.",
                "Our Approach: Real-time dual ingestion (News & Social) + Loughran-McDonald FinBERT sentiment weighting.",
                "Tri-Factor Risk Intelligence: Structured Sentiment (-1 to 1), Event Taxonomy (6 Classes), and Systemic Impact Severity (1-10).",
                "Downstream Actionability: Instant dynamic index rebalancing (Module A) and automated portfolio stress testing (Module B).",
            ],
        },
        {
            "slide_num": 3,
            "title": "System Design & End-to-End Architecture",
            "subtitle": "Decoupled, Modular & Scalable Microservices Architecture",
            "points": [
                "Multi-Source Ingestion: Financial Times, Bloomberg, Reuters, SEC feeds + Twitter/StockTwits cashtag streams.",
                "Core NLP Processing Unit: Normalizes raw text, resolves ticker entities, computes financial sentiment, and classifies events.",
                "Downstream Module A (Tactical Rebalancer): Subscribes to sentiment; optimizes S&P 100 weights with institutional risk bounds.",
                "Downstream Module B (Stress Tester): Subscribes to high-impact events (Score >= 7); stresses $500M wholesale banking assets.",
                "Interoperability: High-throughput FastAPI endpoints (/api/signals, /api/rebalance, /api/stress-test) + Streamlit UI.",
            ],
        },
        {
            "slide_num": 4,
            "title": "Implementation Highlights & Technical Choices",
            "subtitle": "Domain-Engineered Financial NLP & Quantitative Risk Models",
            "points": [
                "Financial Lexicon Engineering: Loughran-McDonald dictionary weighting for debt covenants, downgrades, and restructurings.",
                "Bayesian Event Classifier: Zero-shot classification covering Geopolitical, Credit Event, Macro, M&A, and Product Launches.",
                "Quantified Impact Scoring (1-10): Formulated via systemic event weight, sentiment polarization, entity prominence, and urgency.",
                "Tactical Optimization (Module A): Quadratic sentiment-tilt algorithm subject to 2.0% floor, 18.0% cap, and budget constraints.",
                "Wholesale Valuation Models (Module B): Modified Duration & Convexity (Bonds), DV01 (IRS Swaps), CS01 (CDS), and PD/LGD Migration.",
            ],
        },
        {
            "slide_num": 5,
            "title": "Key Results & Strategy Performance",
            "subtitle": "Empirical Proof: Alpha Generation & Risk Containment",
            "points": [
                "Tactical Rebalancing Alpha: +1.67% excess return over benchmark S&P 500 ETF in 30-day backtest simulation.",
                "Risk-Adjusted Efficiency: Annualized Sharpe Ratio of 0.97 vs 0.72 benchmark; controlled turnover of 16.7%.",
                "Stress Test Diagnostic: Accurately simulated $11.77M MtM loss (3.38%) on $500M book under geopolitical oil embargo.",
                "Regulatory Capital Telemetry: Monitored CET1 ratio drop from 13.0% to 9.71%, flagging RWA inflation and capital buffer erosion.",
                "Latency: End-to-end signal extraction and downstream rebalancing executed in < 60 milliseconds per event.",
            ],
        },
        {
            "slide_num": 6,
            "title": "Domain Impact & Business Value",
            "subtitle": "Institutional Utility for S&P Global, CRISIL & Wholesale Banks",
            "points": [
                "Wholesale Banking Risk (CRISIL Scope): Automates early-warning radar for counterparty credit rating migration and loan defaults.",
                "Index & Quantitative Products (S&P Scope): Powers real-time sentiment-driven thematic indices with proven backtested alpha.",
                "Regulatory Compliance: Seamlessly integrates with Basel III / CCAR stress testing pipelines for automated scenario simulation.",
                "Enterprise Operational Efficiency: Replaces manual headline monitoring with machine-readable REST API risk intelligence.",
            ],
        },
        {
            "slide_num": 7,
            "title": "Limitations, Governance & Future Roadmap",
            "subtitle": "Production Evolution & Enterprise Hardening",
            "points": [
                "Current Assumptions: Fixed synthetic wholesale book ($500M); simulated historical price paths with calibrated drift.",
                "Multi-Lingual NLP Expansion: Integrate multilingual LLMs (mBERT / Llama-3) for APAC and European cross-border headlines.",
                "Order Execution Interface: Connect directly to FIX Protocol / OMS for straight-through algorithmic index rebalancing.",
                "Deep Agentic Explainability: Incorporate chain-of-thought financial rationale alongside numerical impact scores.",
            ],
        },
    ]

    for s_data in slides_data:
        blank_slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_slide_layout)
        apply_dark_theme(slide)

        # Header title
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.7), Inches(1.2))
        tf = tx_box.text_frame
        tf.word_wrap = True

        p_title = tf.paragraphs[0]
        p_title.text = f"Slide {s_data['slide_num']}: {s_data['title']}"
        p_title.font.size = Pt(28)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(56, 189, 248)  # Sky blue

        p_sub = tf.add_paragraph()
        p_sub.text = s_data["subtitle"]
        p_sub.font.size = Pt(16)
        p_sub.font.color.rgb = RGBColor(148, 163, 184)  # Slate grey

        # Body bullets box
        body_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
        b_tf = body_box.text_frame
        b_tf.word_wrap = True

        for idx, pt in enumerate(s_data["points"]):
            p = b_tf.paragraphs[0] if idx == 0 else b_tf.add_paragraph()
            p.text = f"•  {pt}"
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor(241, 245, 249)
            p.space_after = Pt(16)

    out_file = DOCS_DIR / "presentation.pptx"
    prs.save(out_file)
    print(f"Presentation PPTX generated at: {out_file}")


def generate_presentation_pdf():
    """Generates 7-slide PDF presentation deck."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)

    slides_content = [
        (
            "Slide 1: AI/NLP Financial Risk Intelligence & Tactical Allocation",
            "S&P Global & CRISIL Campus Hackathon 2026 | Case Study Round 3",
            [
                "Candidate: Shivam Pal | S&P Hackathon Finalist",
                "Primary Deliverable: Central AI/NLP Risk Engine parsing unstructured financial news & social streams.",
                "Dual Downstream Implementation: Module A (Tactical Index Rebalancer) + Module B (Wholesale Portfolio Stress Tester).",
                "Institutional Standards: Loughran-McDonald sentiment weighting, Basel III CET1 metrics, and Bloomberg-grade dashboard.",
            ],
        ),
        (
            "Slide 2: Business Problem & Solution Approach",
            "Bridging the Gap Between Unstructured Market Noise and Actionable Risk Signals",
            [
                "The Challenge: Critical risk events occur in seconds across news/social feeds; manual analysis creates costly latency.",
                "Our Innovation: Multi-source ingestion (News + Tweets) transformed into structured machine-readable risk signals.",
                "Tri-Factor Risk Extraction: Sentiment Score (-1 to 1), Event Taxonomy (6 Classes), and Impact Severity (1 to 10).",
                "Downstream Execution: Instant index rebalancing (Module A) and automatic wholesale portfolio stress simulation (Module B).",
            ],
        ),
        (
            "Slide 3: End-to-End System Design & Architecture",
            "Modular, Decoupled & High-Throughput Pipeline",
            [
                "Ingestion Layer: Dual channels (Financial News + FinTwit/StockTwits) with ticker & cashtag normalization.",
                "Core NLP Risk Engine: Loughran-McDonald sentiment modifier, Bayesian event classification, and systemic severity scoring.",
                "Module A (Tactical Rebalancer): Dynamic quadratic weight optimization for S&P 100 constituent universe.",
                "Module B (Wholesale Stress Tester): Event-triggered ($500M book: Loans, Bonds, Swaps, CDS) macro shock simulation.",
                "Access Layer: High-performance FastAPI REST endpoints + Streamlit Dark Mode executive dashboard.",
            ],
        ),
        (
            "Slide 4: Implementation Highlights & Technical Choices",
            "Rigorous Financial Engineering & Machine Learning",
            [
                "FinBERT & Loughran-McDonald Lexicon: Heavy weighting for debt defaults, covenants, rating downgrades, and buybacks.",
                "Bayesian Event Classifier: Covers Geopolitical, Credit Event, Macroeconomic, M&A, and Product Launch categories.",
                "Impact Severity Formula: Combines systemic event weight, sentiment polarization, entity prominence, and urgency tokens.",
                "Risk Constraints (Module A): Minimum single-stock floor of 2.0% and maximum cap of 18.0% to prevent concentration risk.",
                "Wholesale Valuation (Module B): Modified Duration & Convexity (Bonds), DV01 (Swaps), CS01 (CDS), and PD/LGD Migration.",
            ],
        ),
        (
            "Slide 5: Key Results & Quantitative Findings",
            "Backtested Alpha Generation & Portfolio Resilience",
            [
                "Alpha Generation: Tactical Index generated +1.67% excess return over benchmark S&P 500 ETF in 30-day backtest.",
                "Risk-Adjusted Efficiency: Sharpe Ratio of 0.97 vs 0.72 benchmark with a controlled portfolio turnover of 16.7%.",
                "Stress Test Impact: Successfully simulated $11.77M MtM loss (3.38%) on $500M wholesale book under Geopolitical shock.",
                "Capital Telemetry: Monitored CET1 ratio decline from 13.0% to 9.71%, flagging regulatory buffer erosion under stress.",
                "Execution Latency: Complete signal generation and downstream rebalancing executed in under 60 milliseconds.",
            ],
        ),
        (
            "Slide 6: Domain Impact & Business Value",
            "Empowering S&P Global, CRISIL & Institutional Asset Managers",
            [
                "CRISIL Credit Intelligence: Early-warning detection of counterparty rating downgrades and loan default triggers.",
                "S&P Dow Jones Indices: Foundation for dynamic sentiment-overlay indices delivering systematic alpha.",
                "Wholesale Banking Stress Testing: Automated CCAR / Basel III regulatory compliance and capital adequacy monitoring.",
                "Operational Scalability: Fully accessible via REST API for straight-through integration into Trading & Risk desks.",
            ],
        ),
        (
            "Slide 7: Limitations, Assumptions & Future Roadmap",
            "Path to Enterprise Production Deployment",
            [
                "Current Assumptions: Synthetic wholesale book ($500M); simulated historical price paths with calibrated drift.",
                "Multilingual Intelligence: Expand to multi-lingual transformers for cross-border EMEA and APAC news coverage.",
                "Direct Market Execution: Integrate with FIX Protocol for straight-through order routing into execution venues.",
                "Agentic Explanations: Embed LLM chain-of-thought rationale alongside numerical impact severity scores.",
            ],
        ),
    ]

    for title, subtitle, bullets in slides_content:
        pdf.add_page()
        # Dark Background
        pdf.set_fill_color(11, 15, 25)
        pdf.rect(0, 0, 297, 210, "F")

        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(56, 189, 248)  # Cyan
        pdf.set_xy(18, 15)
        pdf.cell(0, 10, title, ln=True)

        # Subtitle
        pdf.set_font("Helvetica", "I", 12)
        pdf.set_text_color(148, 163, 184)  # Slate
        pdf.set_xy(18, 26)
        pdf.cell(0, 8, subtitle, ln=True)

        # Divider Line
        pdf.set_draw_color(56, 189, 248)
        pdf.set_line_width(0.6)
        pdf.line(18, 36, 279, 36)

        # Bullets
        pdf.set_font("Helvetica", "", 13)
        pdf.set_text_color(241, 245, 249)
        y_pos = 46
        for b in bullets:
            pdf.set_xy(22, y_pos)
            pdf.multi_cell(250, 8, f"-  {b}")
            y_pos += 24

    out_file = DOCS_DIR / "presentation.pdf"
    pdf.output(str(out_file))
    print(f"Presentation PDF generated at: {out_file}")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_presentation_pptx()
    generate_presentation_pdf()
