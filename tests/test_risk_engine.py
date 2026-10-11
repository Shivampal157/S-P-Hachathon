"""
Automated Unit and Integration Tests for S&P AI/NLP Risk Engine
"""

import unittest
from pathlib import Path
import sys

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.data_loader import MultiSourceIngestionEngine, RawItem
from src.engine.nlp_risk_engine import NLPRiskEngine, RiskSignal
from src.module_a.index_rebalancer import TacticalIndexRebalancer
from src.module_b.stress_tester import WholesalePortfolioStressTester


class TestNLPRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NLPRiskEngine()

    def test_positive_sentiment_detection(self):
        text = "Apple posts record revenue, crushes earnings expectations and announces $100B share buyback"
        signal = self.engine.analyze_text(text, ticker="AAPL")
        self.assertGreater(signal.sentiment_score, 0.2)
        self.assertEqual(signal.sentiment_label, "Positive")
        self.assertGreaterEqual(signal.impact_score, 1)
        self.assertLessEqual(signal.impact_score, 10)

    def test_negative_credit_event_detection(self):
        text = "S&P downgrades debt to junk following severe liquidity crunch and defaults on senior debt covenants"
        signal = self.engine.analyze_text(text, ticker="BA")
        self.assertLess(signal.sentiment_score, -0.2)
        self.assertEqual(signal.sentiment_label, "Negative")
        self.assertEqual(signal.event_classification, "Credit Event")
        self.assertTrue(signal.is_high_impact)

    def test_geopolitical_classification(self):
        text = "Escalating military conflict and maritime blockade in the Strait of Hormuz causes crude oil price spikes"
        signal = self.engine.analyze_text(text)
        self.assertEqual(signal.event_classification, "Geopolitical")
        self.assertGreaterEqual(signal.impact_score, 7)


class TestTacticalIndexRebalancer(unittest.TestCase):
    def setUp(self):
        self.rebalancer = TacticalIndexRebalancer()
        self.engine = NLPRiskEngine()

    def test_rebalancing_weights_sum_to_one(self):
        signals = [
            self.engine.analyze_text("Apple surges on record buyback and growth", ticker="AAPL"),
            self.engine.analyze_text("Tesla faces DOJ probe and supply defaults", ticker="TSLA"),
        ]
        result = self.rebalancer.rebalance(signals, tilt_factor=0.5)
        alloc_df = result.to_dataframe()

        total_weight = alloc_df["target_weight"].sum()
        self.assertAlmostEqual(total_weight, 1.0, places=3)

        # Check positive sentiment increases allocation
        aapl_row = alloc_df[alloc_df["ticker"] == "AAPL"].iloc[0]
        self.assertGreater(aapl_row["target_weight"], aapl_row["benchmark_weight"])


class TestWholesaleStressTester(unittest.TestCase):
    def setUp(self):
        self.tester = WholesalePortfolioStressTester()
        self.engine = NLPRiskEngine()

    def test_stress_test_trigger_and_capital_depletion(self):
        high_impact_signal = self.engine.analyze_text(
            "Moody's and S&P downgrade major debt to junk amid widespread debt defaults and liquidity crunch"
        )
        self.assertTrue(high_impact_signal.is_high_impact)

        result = self.tester.run_stress_test(high_impact_signal, force_trigger=False)
        self.assertIsNotNone(result)
        self.assertGreater(result.total_loss, 0)
        self.assertLess(result.stressed_portfolio_mtm, result.baseline_portfolio_mtm)
        self.assertLess(result.stressed_cet1_ratio, result.baseline_cet1_ratio)


if __name__ == "__main__":
    unittest.main()
