"""
Simple parameter optimizer for interval learners.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from learning.interval_learners import IntervalLearner, get_interval_learner_manager


class ParameterOptimizer:
    """Coordinates parameter adjustments across interval learners."""

    def __init__(self) -> None:
        self._learner_manager = get_interval_learner_manager()
        self.trade_outcome_weight: float = 3.0

    def set_trade_outcome_weight(self, weight: float) -> None:
        weight = max(1.0, min(weight, 5.0))
        self.trade_outcome_weight = weight
        self._learner_manager.set_trade_outcome_weight(weight)

    def analyze_predictions(self, interval: str) -> Dict[str, float]:
        learner = self._learner_manager.get_learner(interval)
        stats = learner.stats.copy()
        stats["trade_outcome_weight"] = learner.trade_outcome_weight
        return stats

    def suggest_parameter_updates(self, interval: str) -> Dict[str, float]:
        learner = self._learner_manager.get_learner(interval)
        stats = learner.stats
        pending = max(0, stats.get("total_predictions", 0) - stats.get("evaluated_predictions", 0))
        adjustment: Dict[str, float] = {}
        if pending > 5:
            adjustment["confidence_bias"] = -0.05
        else:
            adjustment["confidence_bias"] = 0.05
        learner.update_parameters(adjustment)
        return adjustment

    def optimize_parameters(self, interval: str, min_predictions: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Public API used by tests to verify the optimizer can be invoked safely.

        Returns a dictionary keyed by interval so future multi-interval
        optimizations can extend the payload without breaking callers.
        """
        learner = self._learner_manager.get_learner(interval)
        stats_snapshot = learner.stats.copy()
        total_predictions = stats_snapshot.get("total_predictions", 0)

        result: Dict[str, Dict[str, Any]] = {
            interval: {
                "status": "skipped",
                "reason": "insufficient_data",
                "stats": stats_snapshot,
                "adjustments": {},
            }
        }

        if total_predictions < max(1, min_predictions):
            return result

        adjustments = self.suggest_parameter_updates(interval)
        result[interval] = {
            "status": "applied" if adjustments else "no_change",
            "reason": None if adjustments else "no_adjustment_needed",
            "stats": stats_snapshot,
            "adjustments": adjustments,
        }
        return result


_PARAMETER_OPTIMIZER: Optional[ParameterOptimizer] = None


def get_parameter_optimizer() -> ParameterOptimizer:
    global _PARAMETER_OPTIMIZER
    if _PARAMETER_OPTIMIZER is None:
        _PARAMETER_OPTIMIZER = ParameterOptimizer()
    return _PARAMETER_OPTIMIZER


__all__ = [
    "ParameterOptimizer",
    "get_parameter_optimizer",
]

