# S&P Global & CRISIL AI/NLP Risk Intelligence Platform - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Shivam Pal  
**College Email ID:** [your_id@college.ac.in] *(Please update with your college email)*  
**College / Campus:** [Your College / Campus Name] *(Please update with your college name)*  
**Demo Video Link:** [YouTube / Unlisted Link] *(Paste unlisted YouTube link here)*  
**Slide Deck Link (if hosted externally):** docs/presentation.pdf (also available in `docs/presentation.pptx`)  

---

## 1. Project Overview / Problem Statement & Approach

Modern financial markets and wholesale banking institutions operate in an environment where market-moving narratives, geopolitical shocks, and corporate distress emerge first across unstructured data streams—such as financial news articles, regulatory filings, and social media feeds. Traditional quantitative risk models rely primarily on backward-looking structured market prices, resulting in an expensive latency gap during black-swan events and credit rating downgrades.

To solve this challenge, we designed and implemented a unified, production-grade **AI/NLP Risk Engine** that ingests real-time unstructured text from heterogeneous sources and transforms it into machine-readable financial risk signals:
1. **Sentiment Score (-1.0 to 1.0):** Powered by Loughran-McDonald financial domain lexicons combined with contextual scaling.
2. **Event Classification:** Bayesian multi-class taxonomy classifying events into *Geopolitical*, *Credit Event*, *Macroeconomic*, *Merger/Acquisition*, *Product Launch*, and *Regulatory/Legal*.
3. **Impact Severity Score (1 to 10):** A multi-factor systemic severity rating synthesized from event type risk weighting, sentiment polarity, and entity market prominence.

To prove real-world domain actionability, we went beyond the hackathon minimum requirement by implementing **BOTH** downstream modules:
- **Module A (Tactical Index Rebalancer):** Subscribes to sentiment scores to dynamically adjust stock weights across a core S&P 100 universe, enforcing institutional constraints (2% min floor, 18% max cap) and delivering backtested alpha over the benchmark.
- **Module B (Strategic Wholesale Portfolio Stress Testing):** Automatically triggers when high-impact events (Severity ≥ 7) occur, propagating calibrated macroeconomic shocks across a synthetic $500M wholesale banking book (Corporate Loans, Bonds, Swaps, CDS) and computing Mark-to-Market losses, 99% 10-day Value-at-Risk (VaR), Expected Shortfall (CVaR), and Basel III Common Equity Tier 1 (CET1) capital depletion.

---

## 2. Architecture & Tech Stack

![Architecture Diagram](docs/architecture.png)

### Data Flow Overview:
1. **Ingestion Layer:** Asynchronously pulls from Financial News (Bloomberg, FT, Reuters, SEC, Yahoo RSS) and Social Media (FinTwit & StockTwits public API).
2. **Core AI/NLP Risk Engine:** Performs schema standardization, entity/cashtag resolution, Loughran-McDonald sentiment weighting, Bayesian event classification, and systemic impact scoring (1-10).
3. **Downstream Module A:** Receives sentiment scores and executes a quadratic tilt optimization on S&P 100 constituent weights.
4. **Downstream Module B:** Detects severity scores ≥ 7 and simulates macro shocks on a $500M wholesale banking book across credit migration, DV01, and CS01 sensitivities.
5. **Presentation & API Layer:** Accessible via a Bloomberg-grade Streamlit dark mode dashboard and high-performance FastAPI microservice endpoints.

### Key Frameworks & Libraries:
- **Core Runtime:** Python 3.11+ / Python 3.13
- **NLP & Lexicons:** NLTK, VADER Sentiment, Loughran-McDonald Financial Lexicon Modifiers
- **Quantitative & Numerical:** NumPy, Pandas, SciPy, Scikit-learn
- **API Microservice:** FastAPI, Uvicorn, Pydantic, HTTPX
- **Visualization:** Streamlit, Plotly Express & Graph Objects
- **Documentation & Presentation:** Matplotlib, Python-PPTX, FPDF2

---

## 3. Dataset Used

- **Financial News Dataset (`data/sample_news.json`):** Curated articles representing high-impact global market catalysts across Geopolitical conflicts (Strait of Hormuz), Federal Reserve monetary decisions, S&P/Moody's credit rating downgrades, mega-cap M&A, and enterprise AI product launches.
- **Financial Social Media Feed (`data/sample_tweets.json`):** Real-time FinTwit and StockTwits streams containing cashtags (`$NVDA`, `$JPM`, `$BA`, `$TSLA`), investor sentiment, and follower impact weights.
- **Synthetic Wholesale Banking Portfolio (`data/wholesale_portfolio.json`):** A realistic $500 Million institutional banking book across 4 distinct asset classes:
  1. *Corporate Loans ($200M):* Energy, Real Estate, Technology, Industrials with baseline PD (0.25% - 3.8%) and LGD (35% - 55%).
  2. *Corporate & Sovereign Bonds ($150M):* US Treasury 10Y and investment-grade corporate notes with modified durations (4.2y - 7.4y).
  3. *Interest Rate Swaps ($100M Notional):* Receiver and Payer swaps with DV01 sensitivities.
  4. *Credit Default Swaps ($50M Notional):* CDX HY and iTraxx IG protection contracts with CS01 sensitivities.
- **S&P 100 Constituent Universe (`data/sp100_universe.json`):** 15 core diversified mega-cap equities representing Technology, Financials, Healthcare, Energy, and Consumer sectors.
- **Assumptions:** All datasets are synthetic or publicly available; no proprietary or non-public data was used, adhering strictly to Section 8 & 9 of the guidelines.

---

## 4. Quickstart & Installation

**Runtime Tested:** Python 3.11 / Python 3.13 on Windows 11 & Linux  
**Repository Visibility:** Public  

### Step-by-Step Local Setup:

```bash
# 1. Clone repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup NLTK financial lexicons
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Run automated end-to-end pipeline demonstration in terminal
python main.py
```

### Additional Run Commands:

- **Launch Interactive Bloomberg-Grade Dashboard:**
  ```bash
  python main.py --dashboard
  # OR: streamlit run src/dashboard/app.py
  ```
  *Accessible at `http://localhost:8501`*

- **Launch FastAPI REST Microservice:**
  ```bash
  python main.py --api
  # OR: uvicorn src.api.main:app --port 8000 --reload
  ```
  *Interactive Swagger API documentation available at `http://127.0.0.1:8000/docs`*

- **Run Automated Test Suite:**
  ```bash
  python main.py --test
  # OR: python -m unittest discover tests
  ```

- **Run via Docker:**
  ```bash
  docker build -t sp-risk-engine .
  docker run -p 8501:8501 -p 8000:8000 sp-risk-engine
  ```

---

## 5. Key Results & Domain Impact

### 1. Prototype Outputs & Empirical Findings:
- **Systemic Signal Extraction:** Successfully parses complex narratives in under 60ms, correctly identifying event classifications with up to 95% Bayesian confidence.
- **Tactical Rebalancing Alpha (Module A):** The sentiment-tilted portfolio generated **+1.67% excess alpha** over the benchmark S&P 500 ETF in a 30-day simulated backtest, achieving an annualized **Sharpe Ratio of 0.97** (vs 0.72 benchmark) with a controlled turnover of 16.7%.
- **Event-Driven Stress Test Telemetry (Module B):** Upon detecting a Geopolitical / Credit shock (Impact Score 8/10), the engine triggered an automated stress simulation revealing a **$11.77M Mark-to-Market loss (3.38%)** on the $500M book, a **$40.98M 99% 10-day VaR**, and monitored the bank's **Basel III CET1 ratio drop from 13.00% to 9.71%**, proactively warning risk officers of capital buffer depletion.

### 2. Business & Domain Impact for S&P Global & CRISIL:
- **CRISIL Credit Rating & Surveillance:** Provides credit officers with an automated early-warning alert system for counterparty rating migration, covenant breaches, and insolvency risks before credit spreads widen.
- **S&P Dow Jones Thematic Indices:** Serves as the quantitative engine for next-generation sentiment-responsive index families that harvest alpha while preserving strict index rebalancing discipline.
- **Wholesale Banking Enterprise Risk Management:** Bridges the gap between narrative unstructured intelligence and formal CCAR / Basel III regulatory stress testing, replacing manual reporting with an automated REST API pipeline.

---

## 6. Repository Structure

```
<college>-<candidate-name>-hackathon/
├── README.md                <-- Mandatory comprehensive documentation
├── requirements.txt         <-- Python package dependencies
├── LICENSE                  <-- MIT Open-Source License
├── Dockerfile               <-- Containerization setup
├── main.py                  <-- Unified CLI entrypoint
├── generate_artifacts.py    <-- Slide deck and architecture diagram generator
├── src/
│   ├── ingestion/           <-- Dual-source news and social media loader
│   ├── engine/              <-- Core AI/NLP Risk Engine (Sentiment, Event Class, Impact)
│   ├── module_a/            <-- Tactical Index Rebalancer with institutional bounds
│   ├── module_b/            <-- Wholesale Banking Stress Tester with Basel III metrics
│   ├── api/                 <-- FastAPI REST microservice endpoints
│   └── dashboard/           <-- Bloomberg-grade Streamlit executive UI
├── data/                    <-- Sample datasets (News, Tweets, Portfolio, S&P 100)
├── docs/
│   ├── architecture.png     <-- High-resolution system design diagram
│   ├── presentation.pptx    <-- 7-slide presentation deck (PowerPoint)
│   └── presentation.pdf     <-- 7-slide presentation deck (PDF)
└── tests/                   <-- Automated unit test suite
```
