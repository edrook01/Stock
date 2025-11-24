"""
Learning statistics tracking for the constant learning system.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

from learning.prediction_storage import PredictionRecord


class LearningStatistics:
    """Maintains aggregate metrics for predictions and evaluations."""

    def __init__(self) -> None:
        self._overall: Dict[str, Any] = {
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "average_accuracy": 0.0,
        }
        self._intervals: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "average_accuracy": 0.0,
        })
        self._history: Dict[str, Any] = {
            "entries": [],
        }

    def record_prediction(self, prediction: PredictionRecord) -> None:
        self._overall["total_predictions"] += 1
        interval_stats = self._intervals[prediction.interval]
        interval_stats["total_predictions"] += 1
        self._history["entries"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_id": prediction.prediction_id,
            "event": "stored",
            "interval": prediction.interval,
            "ticker": prediction.ticker,
        })

    def record_evaluation(self, prediction: PredictionRecord) -> None:
        self._overall["evaluated_predictions"] += 1
        interval_stats = self._intervals[prediction.interval]
        interval_stats["evaluated_predictions"] += 1
        score = prediction.accuracy_score or 0.0
        self._overall["average_accuracy"] = self._update_average(
            self._overall["average_accuracy"],
            score,
            self._overall["evaluated_predictions"],
        )
        interval_stats["average_accuracy"] = self._update_average(
            interval_stats["average_accuracy"],
            score,
            interval_stats["evaluated_predictions"],
        )
        self._history["entries"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_id": prediction.prediction_id,
            "event": "evaluated",
            "interval": prediction.interval,
            "ticker": prediction.ticker,
            "accuracy": score,
        })

    @staticmethod
    def _update_average(previous: float, value: float, count: int) -> float:
        if count <= 0:
            return previous
        return round((previous * (count - 1) + value) / count, 3)

    def get_overall_statistics(self) -> Dict[str, Any]:
        return dict(self._overall)

    def get_interval_statistics(self, interval: str) -> Dict[str, Any]:
        return dict(self._intervals[interval])

    def get_all_statistics(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Return a consolidated snapshot of overall, per-interval, and history data.

        The refresh flag is accepted for API compatibility; the in-memory
        implementation does not require any additional work to refresh.
        """
        return {
            "overall": self.get_overall_statistics(),
            "intervals": {interval: dict(stats) for interval, stats in self._intervals.items()},
            "history": list(self._history["entries"]),
            "refreshed": bool(refresh),
        }

    def generate_report(self) -> str:
        lines = [
            "=" * 60,
            "LEARNING STATISTICS REPORT",
            "=" * 60,
            f"Total predictions: {self._overall['total_predictions']}",
            f"Evaluated predictions: {self._overall['evaluated_predictions']}",
            f"Average accuracy: {self._overall['average_accuracy']:.2f}",
            "",
            "Per-interval summary:",
        ]
        for interval, stats in sorted(self._intervals.items()):
            lines.append(
                f"  {interval}: total={stats['total_predictions']} "
                f"evaluated={stats['evaluated_predictions']} "
                f"avg_accuracy={stats['average_accuracy']:.2f}"
            )
        if not self._intervals:
            lines.append("  No interval data recorded yet.")
        return "\n".join(lines)


_LEARNING_STATISTICS: Optional[LearningStatistics] = None


def get_learning_statistics() -> LearningStatistics:
    global _LEARNING_STATISTICS
    if _LEARNING_STATISTICS is None:
        _LEARNING_STATISTICS = LearningStatistics()
    return _LEARNING_STATISTICS


__all__ = [
    "LearningStatistics",
    "get_learning_statistics",
]

