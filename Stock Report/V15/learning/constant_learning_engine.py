"""
Simplified constant learning engine for automated tests and UI tooling.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence

from learning.prediction_storage import get_prediction_storage
from learning.prediction_evaluator import get_prediction_evaluator
from learning.interval_learners import get_interval_learner_manager
from learning.parameter_optimizer import get_parameter_optimizer
from learning.prediction_generator import get_prediction_generator

try:
    from core.timeframes import CONSTANT_LEARNING_INTERVALS
    from core.ticker_universe import get_trading212_tickers
except Exception:
    CONSTANT_LEARNING_INTERVALS = ["1m", "5m", "10m", "15m", "1h", "4h", "1d", "1w", "1mo", "3mo", "1y"]
    from core.ticker_universe import get_trading212_tickers  # type: ignore


class ConstantLearningEngine:
    """Coordinates periodic prediction evaluation."""

    def __init__(self, enabled: bool = True, evaluation_frequency_seconds: float = 5.0) -> None:
        self.enabled = enabled
        self.evaluation_frequency_seconds = evaluation_frequency_seconds
        self.max_predictions_per_cycle = 10
        self.active_intervals: List[str] = list(CONSTANT_LEARNING_INTERVALS)
        self.active_tickers: List[str] = get_trading212_tickers()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.storage = get_prediction_storage()
        self.evaluator = get_prediction_evaluator()
        self.learner_manager = get_interval_learner_manager()
        self.optimizer = get_parameter_optimizer()
        self.prediction_generator = get_prediction_generator()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if enabled and not self.is_running():
            self.start()
        elif not enabled and self.is_running():
            self.stop()

    def set_active_intervals(self, intervals: Sequence[str]) -> None:
        self.active_intervals = [i.lower() for i in intervals]

    def set_active_tickers(self, tickers: Sequence[str]) -> None:
        self.active_tickers = list(tickers)

    def refresh_active_tickers(self) -> None:
        """Reload the Trading212 ticker universe into the active list."""
        self.active_tickers = get_trading212_tickers()

    def set_trade_outcome_weight(self, weight: float) -> None:
        self.optimizer.set_trade_outcome_weight(weight)

    def set_max_predictions_per_cycle(self, limit: int) -> None:
        self.max_predictions_per_cycle = max(1, limit)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.enabled:
                self.run_cycle()
            self._stop_event.wait(self.evaluation_frequency_seconds)

    def run_cycle(self) -> None:
        for interval in self.active_intervals:
            expired = self.storage.get_expired_predictions(interval)
            if not expired:
                continue
            for prediction in expired[: self.max_predictions_per_cycle]:
                result = self.evaluator.evaluate_prediction(prediction)
                self.learner_manager.record_evaluation(prediction, result)
                try:
                    self.prediction_generator.ensure_predictions(
                        tickers=[prediction.ticker],
                        intervals=[prediction.interval],
                        per_interval=1,
                    )
                except Exception:
                    pass

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "running": self.is_running(),
            "active_intervals": list(self.active_intervals),
            "active_tickers_count": len(self.active_tickers),
            "evaluation_frequency_seconds": self.evaluation_frequency_seconds,
            "max_predictions_per_cycle": self.max_predictions_per_cycle,
        }

    @property
    def running(self) -> bool:
        """Expose the running state as an attribute for legacy checks."""
        return self.is_running()


_CONSTANT_LEARNING_ENGINE: Optional[ConstantLearningEngine] = None


def get_constant_learning_engine() -> ConstantLearningEngine:
    global _CONSTANT_LEARNING_ENGINE
    if _CONSTANT_LEARNING_ENGINE is None:
        _CONSTANT_LEARNING_ENGINE = ConstantLearningEngine()
    return _CONSTANT_LEARNING_ENGINE


__all__ = [
    "ConstantLearningEngine",
    "get_constant_learning_engine",
]

