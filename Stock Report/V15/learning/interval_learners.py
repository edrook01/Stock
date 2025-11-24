"""
Interval-specific learner management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from learning.prediction_storage import PredictionRecord, get_prediction_storage


class IntervalLearner:
    """Tracks statistics and parameters for a single interval."""

    def __init__(self, interval: str) -> None:
        self.interval = interval
        self.stats: Dict[str, float] = {
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "trade_predictions": 0,
        }
        self.parameters: Dict[str, float] = {
            "confidence_bias": 0.0,
            "risk_multiplier": 1.0,
            "volatility_buffer": 0.02,
        }
        self.parameter_history: List[Dict[str, float]] = []
        self.trade_outcome_weight: float = 3.0

    def record_prediction(self, prediction: PredictionRecord) -> None:
        self.stats["total_predictions"] += 1

    def record_evaluation(self, prediction: PredictionRecord, accuracy_result: Optional[Dict] = None) -> None:
        self.stats["evaluated_predictions"] += 1
        if prediction.source == "trade_based":
            self.stats["trade_predictions"] += 1
        if accuracy_result and "accuracy_scores" in accuracy_result:
            score = accuracy_result["accuracy_scores"].get("overall_accuracy")
            if score is not None:
                self.stats.setdefault("average_accuracy", 0.0)
                count = self.stats["evaluated_predictions"]
                prev = self.stats["average_accuracy"]
                self.stats["average_accuracy"] = round((prev * (count - 1) + score) / count, 3)

    def update_parameters(self, adjustments: Dict[str, float]) -> None:
        self.parameters.update(adjustments)
        history_entry = {"timestamp": datetime.utcnow().isoformat(), **adjustments}
        self.parameter_history.append(history_entry)

    def get_statistics(self) -> Dict[str, Any]:
        """Return a snapshot of learner statistics and parameters."""
        return {
            "interval": self.interval,
            "stats": dict(self.stats),
            "parameters": dict(self.parameters),
            "trade_outcome_weight": self.trade_outcome_weight,
            "parameter_history": list(self.parameter_history),
        }

    def learn_from_predictions(self, limit: int = 10) -> Dict[str, Any]:
        """
        Review pending predictions for this interval and adjust parameters.

        Returns a summary so tests and diagnostics can verify activity even
        when there is limited data available.
        """
        storage = get_prediction_storage()
        pending = storage.get_pending_predictions(self.interval)[: max(0, limit)]
        analyzed_ids: List[str] = []
        adjustments_applied = 0

        for record in pending:
            analyzed_ids.append(record.prediction_id)
            adjustment = self._derive_adjustment(record)
            if adjustment:
                self.update_parameters(adjustment)
                adjustments_applied += 1

        return {
            "interval": self.interval,
            "predictions_analyzed": len(analyzed_ids),
            "adjustments_applied": adjustments_applied,
            "reviewed_prediction_ids": analyzed_ids,
        }

    def _derive_adjustment(self, prediction: PredictionRecord) -> Dict[str, float]:
        """Lightweight heuristic for tuning learner parameters."""
        accuracy = prediction.accuracy_score
        if accuracy is None:
            return {}

        bias = self.parameters.get("confidence_bias", 0.0)
        adjustment: Dict[str, float] = {}

        if accuracy < 0.4:
            adjustment["confidence_bias"] = max(-0.25, bias - 0.02)
        elif accuracy > 0.7:
            adjustment["confidence_bias"] = min(0.25, bias + 0.01)

        return adjustment


class IntervalLearnerManager:
    """Ensures each interval has a dedicated learner instance."""

    def __init__(self) -> None:
        self._learners: Dict[str, IntervalLearner] = {}

    def get_learner(self, interval: str) -> IntervalLearner:
        normalized = interval.lower()
        if normalized not in self._learners:
            self._learners[normalized] = IntervalLearner(normalized)
        return self._learners[normalized]

    def record_prediction(self, prediction: PredictionRecord) -> None:
        self.get_learner(prediction.interval).record_prediction(prediction)

    def record_evaluation(self, prediction: PredictionRecord, accuracy_result: Optional[Dict] = None) -> None:
        self.get_learner(prediction.interval).record_evaluation(prediction, accuracy_result)

    def set_trade_outcome_weight(self, weight: float) -> None:
        weight = max(1.0, min(weight, 5.0))
        for learner in self._learners.values():
            learner.trade_outcome_weight = weight

    def get_all_learners(self) -> List[IntervalLearner]:
        return list(self._learners.values())

    @property
    def learners(self) -> Dict[str, IntervalLearner]:
        """Expose the internal learner registry for diagnostic tests."""
        return self._learners


_INTERVAL_LEARNER_MANAGER: Optional[IntervalLearnerManager] = None


def get_interval_learner_manager() -> IntervalLearnerManager:
    global _INTERVAL_LEARNER_MANAGER
    if _INTERVAL_LEARNER_MANAGER is None:
        _INTERVAL_LEARNER_MANAGER = IntervalLearnerManager()
    return _INTERVAL_LEARNER_MANAGER


__all__ = [
    "IntervalLearner",
    "IntervalLearnerManager",
    "get_interval_learner_manager",
]

