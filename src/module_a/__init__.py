"""
Downstream Module A: Tactical Index Rebalancing Subsystem.
Dynamic asset weighting based on NLP sentiment signals.
"""
from .index_rebalancer import TacticalIndexRebalancer, RebalanceResult

__all__ = ["TacticalIndexRebalancer", "RebalanceResult"]
