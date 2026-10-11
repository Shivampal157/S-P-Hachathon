"""
Downstream Module B: Strategic Wholesale Banking Portfolio Stress Testing
Simulates the impact of high-impact real-world events detected by the AI/NLP Risk Engine
on a synthetic $500M wholesale banking asset portfolio:
- Corporate Loans
- Corporate & Sovereign Bonds
- Interest Rate Swaps (IRS)
- Credit Default Swaps (CDS)

Features:
- Automatic trigger when Impact Score > 7.0
- Event-to-Shock Mapping (Geopolitical, Credit Event, Macroeconomic, Regulatory)
- Quantitative Pricing Shocks: DV01, CS01, Modified Duration, PD/LGD Migration
- Value-at-Risk (VaR 99%), Conditional VaR (CVaR 99%)
- Basel III / CCAR Capital Adequacy (CET1 Ratio, RWA Inflation)
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.engine.nlp_risk_engine import RiskSignal

logger = logging.getLogger("StressTester")


@dataclass
class MacroShocks:
    scenario_name: str
    event_classification: str
    impact_score: int
    interest_rate_shock_bps: float  # Basis points (+150 bps = +1.50%)
    credit_spread_ig_bps: float  # Investment Grade spread shock
    credit_spread_hy_bps: float  # High Yield spread shock
    equity_market_shock_pct: float  # Overall equity price decline %
    pd_multiplier: float  # Default probability multiplier (e.g. 2.2x)
    lgd_shock_pct: float  # Loss Given Default add-on (e.g. +0.08)


@dataclass
class StressedAssetResult:
    asset_id: str
    asset_type: str
    counterparty: str
    sector: str
    notional: float
    base_mtm_value: float
    stressed_mtm_value: float
    pnl_loss: float
    pnl_loss_pct: float
    risk_driver: str


@dataclass
class StressTestExecutionResult:
    timestamp: str
    trigger_signal_id: str
    trigger_headline: str
    event_classification: str
    impact_score: int
    portfolio_name: str
    baseline_portfolio_mtm: float
    stressed_portfolio_mtm: float
    total_loss: float
    total_loss_pct: float
    var_99_10d: float
    cvar_99_10d: float
    baseline_cet1_ratio: float
    stressed_cet1_ratio: float
    rwa_inflation_pct: float
    shocks_applied: MacroShocks
    asset_results: List[StressedAssetResult]

    def to_dataframe(self) -> pd.DataFrame:
        data = [asdict(a) for a in self.asset_results]
        return pd.DataFrame(data)


class WholesalePortfolioStressTester:
    """Quantitative stress testing engine for wholesale banking credit and trading assets."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (
            Path(__file__).resolve().parent.parent.parent / "data"
        )
        self.portfolio_file = self.data_dir / "wholesale_portfolio.json"
        self.portfolio_data = self._load_portfolio()

    def _load_portfolio(self) -> Dict[str, Any]:
        """Loads synthetic wholesale banking assets portfolio."""
        if not self.portfolio_file.exists():
            raise FileNotFoundError(
                f"Portfolio file missing: {self.portfolio_file}"
            )
        with open(self.portfolio_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def determine_macro_shocks(self, signal: RiskSignal) -> MacroShocks:
        """Calibrates macro shocks based on Event Classification and Severity Impact Score."""
        event = signal.event_classification
        impact = signal.impact_score  # Scale 1-10
        severity_ratio = impact / 10.0

        if event == "Geopolitical":
            return MacroShocks(
                scenario_name=f"Geopolitical Escalation (Impact {impact}/10)",
                event_classification=event,
                impact_score=impact,
                interest_rate_shock_bps=80.0 * severity_ratio,
                credit_spread_ig_bps=140.0 * severity_ratio,
                credit_spread_hy_bps=280.0 * severity_ratio,
                equity_market_shock_pct=-14.0 * severity_ratio,
                pd_multiplier=1.0 + (1.2 * severity_ratio),
                lgd_shock_pct=0.06 * severity_ratio,
            )
        elif event == "Credit Event":
            return MacroShocks(
                scenario_name=f"Credit Contagion & Liquidity Crunch (Impact {impact}/10)",
                event_classification=event,
                impact_score=impact,
                interest_rate_shock_bps=40.0 * severity_ratio,
                credit_spread_ig_bps=190.0 * severity_ratio,
                credit_spread_hy_bps=380.0 * severity_ratio,
                equity_market_shock_pct=-10.0 * severity_ratio,
                pd_multiplier=1.0 + (1.8 * severity_ratio),
                lgd_shock_pct=0.10 * severity_ratio,
            )
        elif event == "Macroeconomic":
            return MacroShocks(
                scenario_name=f"Hawkish Monetary Tightening & Rate Shock (Impact {impact}/10)",
                event_classification=event,
                impact_score=impact,
                interest_rate_shock_bps=180.0 * severity_ratio,
                credit_spread_ig_bps=110.0 * severity_ratio,
                credit_spread_hy_bps=220.0 * severity_ratio,
                equity_market_shock_pct=-12.0 * severity_ratio,
                pd_multiplier=1.0 + (0.9 * severity_ratio),
                lgd_shock_pct=0.04 * severity_ratio,
            )
        elif event == "Regulatory/Legal":
            return MacroShocks(
                scenario_name=f"Regulatory Enforcement & Antitrust Sanctions (Impact {impact}/10)",
                event_classification=event,
                impact_score=impact,
                interest_rate_shock_bps=20.0 * severity_ratio,
                credit_spread_ig_bps=90.0 * severity_ratio,
                credit_spread_hy_bps=160.0 * severity_ratio,
                equity_market_shock_pct=-8.0 * severity_ratio,
                pd_multiplier=1.0 + (0.6 * severity_ratio),
                lgd_shock_pct=0.03 * severity_ratio,
            )
        else:
            # Baseline moderate shock for other events (Mergers, Product Launches)
            return MacroShocks(
                scenario_name=f"Idiosyncratic Event Stress (Impact {impact}/10)",
                event_classification=event,
                impact_score=impact,
                interest_rate_shock_bps=30.0 * severity_ratio,
                credit_spread_ig_bps=50.0 * severity_ratio,
                credit_spread_hy_bps=100.0 * severity_ratio,
                equity_market_shock_pct=-5.0 * severity_ratio,
                pd_multiplier=1.0 + (0.4 * severity_ratio),
                lgd_shock_pct=0.02 * severity_ratio,
            )

    def stress_asset(
        self, asset: Dict[str, Any], shocks: MacroShocks
    ) -> StressedAssetResult:
        """Applies quantitative pricing and risk models to revalue each wholesale asset under stress."""
        asset_id = asset["id"]
        asset_type = asset["asset_type"]
        counterparty = asset["counterparty"]
        sector = asset["sector"]
        notional = float(asset["notional"])
        base_mtm = float(asset.get("base_mtm_value", notional))
        rating = asset.get("credit_rating", "BBB")

        is_high_yield = any(r in rating for r in ["BB", "B", "CCC"])
        spread_shock_bps = (
            shocks.credit_spread_hy_bps
            if is_high_yield
            else shocks.credit_spread_ig_bps
        )

        stressed_mtm = base_mtm
        driver = "None"

        if asset_type == "Corporate Loan":
            # Corporate Loan Pricing: Expected Loss surge + spread discount
            base_pd = float(asset.get("baseline_pd", 0.015))
            base_lgd = float(asset.get("baseline_lgd", 0.40))
            stressed_pd = min(0.35, base_pd * shocks.pd_multiplier)
            stressed_lgd = min(0.85, base_lgd + shocks.lgd_shock_pct)

            # Change in Expected Loss
            base_el = notional * base_pd * base_lgd
            stressed_el = notional * stressed_pd * stressed_lgd
            el_delta = stressed_el - base_el

            # Secondary market spread discount = notional * duration_approx * delta_spread
            mat_years = float(asset.get("maturity_years", 3.0))
            spread_loss = notional * (mat_years * 0.8) * (spread_shock_bps / 10000.0)

            total_asset_loss = el_delta + spread_loss
            stressed_mtm = max(notional * 0.4, base_mtm - total_asset_loss)
            driver = f"PD surge to {stressed_pd*100:.2f}%, LGD {stressed_lgd*100:.1f}%, Spread +{spread_shock_bps:.0f}bps"

        elif asset_type == "Corporate Bond":
            # Bond Pricing: Modified Duration and Convexity shock
            duration = float(asset.get("duration", 5.0))
            convexity = float(asset.get("convexity", 30.0))
            total_yield_shift = (
                shocks.interest_rate_shock_bps + spread_shock_bps
            ) / 10000.0  # Decimal

            # Price percentage change: dP/P = -Duration * dy + 0.5 * Convexity * (dy)^2
            dp_pct = (
                -duration * total_yield_shift
                + 0.5 * convexity * (total_yield_shift**2)
            )
            price_change = base_mtm * dp_pct
            stressed_mtm = max(notional * 0.5, base_mtm + price_change)
            driver = f"Rates +{shocks.interest_rate_shock_bps:.0f}bps, Spread +{spread_shock_bps:.0f}bps (Duration {duration}y)"

        elif asset_type == "Interest Rate Swap":
            # IRS Valuation: DV01 * Delta Rate in bps
            dv01 = float(asset.get("dv01", 20000))  # Dollar value of 1 bp
            swap_type = asset.get("swap_type", "Receive Fixed")

            if "Receive Fixed" in swap_type:
                # Receiver swap loses value when rates rise
                swap_loss = dv01 * shocks.interest_rate_shock_bps
                stressed_mtm = base_mtm - swap_loss
                driver = f"Receiver DV01 loss on +{shocks.interest_rate_shock_bps:.0f}bps rate spike"
            else:
                # Payer swap gains value when rates rise
                swap_gain = abs(dv01) * shocks.interest_rate_shock_bps
                stressed_mtm = base_mtm + swap_gain
                driver = f"Payer DV01 hedge gain on +{shocks.interest_rate_shock_bps:.0f}bps rate spike"

        elif asset_type == "Credit Default Swap":
            # CDS Valuation: CS01 * Delta Spread in bps
            cs01 = float(asset.get("cs01", 8000))
            pos = asset.get("position", "Protection Sold")

            if "Sold" in pos:
                # Protection sold loses value when credit spreads widen
                cds_loss = abs(cs01) * spread_shock_bps
                stressed_mtm = base_mtm - cds_loss
                driver = f"Protection Sold CS01 loss on +{spread_shock_bps:.0f}bps spread blow-out"
            else:
                # Protection bought gains value when credit spreads widen
                cds_gain = abs(cs01) * spread_shock_bps
                stressed_mtm = base_mtm + cds_gain
                driver = f"Protection Bought hedge gain on +{spread_shock_bps:.0f}bps spread widening"

        pnl_loss = round(base_mtm - stressed_mtm, 2)
        pnl_loss_pct = round(
            (pnl_loss / base_mtm) * 100 if base_mtm != 0 else 0.0, 2
        )

        return StressedAssetResult(
            asset_id=asset_id,
            asset_type=asset_type,
            counterparty=counterparty,
            sector=sector,
            notional=notional,
            base_mtm_value=round(base_mtm, 2),
            stressed_mtm_value=round(stressed_mtm, 2),
            pnl_loss=pnl_loss,
            pnl_loss_pct=pnl_loss_pct,
            risk_driver=driver,
        )

    def run_stress_test(
        self, signal: RiskSignal, force_trigger: bool = False
    ) -> Optional[StressTestExecutionResult]:
        """Executes full portfolio stress test simulation if triggered.

        Trigger condition:
        - Signal is high impact: signal.impact_score >= 7.0 OR force_trigger is True.
        """
        if not force_trigger and signal.impact_score < 7:
            logger.info(
                f"Signal {signal.id} has impact score {signal.impact_score} (< 7). Stress test not triggered."
            )
            return None

        logger.info(
            f"⚡ TRIGGERING STRESS TEST for event '{signal.event_classification}' (Impact: {signal.impact_score}/10)"
        )
        shocks = self.determine_macro_shocks(signal)
        assets = self.portfolio_data.get("assets", [])

        asset_results: List[StressedAssetResult] = []
        base_total = 0.0
        stressed_total = 0.0

        for a in assets:
            res = self.stress_asset(a, shocks)
            asset_results.append(res)
            base_total += res.base_mtm_value
            stressed_total += res.stressed_mtm_value

        total_loss = round(base_total - stressed_total, 2)
        total_loss_pct = round((total_loss / base_total) * 100, 2)

        # Value-at-Risk (99% confidence, 10-day horizon) using parametric formula
        daily_vol = 0.012 * (shocks.impact_score / 6.0)
        z_99 = 2.326  # 99% standard normal quantile
        var_99_10d = round(
            base_total * z_99 * daily_vol * np.sqrt(10), 2
        )
        # Expected Shortfall (CVaR) ~ 1.15 * VaR
        cvar_99_10d = round(var_99_10d * 1.18, 2)

        # Basel III CET1 Capital depletion simulation
        base_capital = float(
            self.portfolio_data.get("base_equity_capital", 65000000)
        )
        base_cet1 = float(self.portfolio_data.get("target_cet1_ratio", 0.130))
        base_rwa = base_capital / base_cet1  # ~$500M RWA

        # Stressed capital = base capital - MtM loss
        stressed_capital = max(1000000.0, base_capital - total_loss)
        # RWA inflation due to rating downgrades and spread widening
        rwa_inflation_factor = 1.0 + (0.12 * (shocks.impact_score / 10.0))
        stressed_rwa = base_rwa * rwa_inflation_factor
        stressed_cet1 = round(stressed_capital / stressed_rwa, 4)
        rwa_inflation_pct = round((rwa_inflation_factor - 1.0) * 100, 2)

        now_str = datetime.now(timezone.utc).isoformat()

        return StressTestExecutionResult(
            timestamp=now_str,
            trigger_signal_id=signal.id,
            trigger_headline=signal.summary,
            event_classification=signal.event_classification,
            impact_score=signal.impact_score,
            portfolio_name=self.portfolio_data.get(
                "portfolio_name", "Wholesale Banking Book"
            ),
            baseline_portfolio_mtm=round(base_total, 2),
            stressed_portfolio_mtm=round(stressed_total, 2),
            total_loss=total_loss,
            total_loss_pct=total_loss_pct,
            var_99_10d=var_99_10d,
            cvar_99_10d=cvar_99_10d,
            baseline_cet1_ratio=base_cet1,
            stressed_cet1_ratio=stressed_cet1,
            rwa_inflation_pct=rwa_inflation_pct,
            shocks_applied=shocks,
            asset_results=asset_results,
        )
