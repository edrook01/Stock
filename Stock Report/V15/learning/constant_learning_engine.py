"""
Simplified constant learning engine for automated tests and UI tooling.
Runs separate instances for each interval to enable concurrent prediction processing.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence, Dict
from collections import defaultdict

from learning.prediction_storage import get_prediction_storage
from learning.prediction_evaluator import get_prediction_evaluator
from learning.interval_learners import get_interval_learner_manager
from learning.parameter_optimizer import get_parameter_optimizer
from learning.prediction_generator import get_prediction_generator

try:
    from core.timeframes import CONSTANT_LEARNING_INTERVALS, get_prediction_update_interval
    from core.ticker_universe import get_trading212_tickers
except Exception:
    CONSTANT_LEARNING_INTERVALS = ["1m", "5m", "10m", "15m", "1h", "4h", "1d", "1w", "1mo", "3mo", "1y"]
    try:
        from core.ticker_universe import get_trading212_tickers  # type: ignore
    except Exception:
        def get_trading212_tickers():
            return []
    
    def get_prediction_update_interval(timeframe: str) -> Optional[int]:
        """Fallback update interval function."""
        return 5  # Default to 5 seconds


class IntervalProcessor:
    """Handles prediction processing for a single interval."""
    
    def __init__(
        self,
        interval: str,
        storage,
        evaluator,
        learner_manager,
        prediction_generator,
        max_predictions_per_cycle: int = 10,
        base_frequency: float = 5.0,
    ) -> None:
        self.interval = interval.lower()
        self.storage = storage
        self.evaluator = evaluator
        self.learner_manager = learner_manager
        self.prediction_generator = prediction_generator
        self.max_predictions_per_cycle = max_predictions_per_cycle
        self.base_frequency = base_frequency
        
        # Use interval-specific update frequency if available, otherwise use base frequency
        update_interval = get_prediction_update_interval(self.interval)
        if update_interval:
            # Use a fraction of the update interval to check more frequently
            self.check_frequency = min(base_frequency, max(1.0, update_interval / 10.0))
        else:
            self.check_frequency = base_frequency
        
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = True
    
    def start(self) -> None:
        """Start the interval processor thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"IntervalProcessor-{self.interval}"
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the interval processor thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
    
    def is_running(self) -> bool:
        """Check if the processor thread is running."""
        return self._thread is not None and self._thread.is_alive()
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this interval processor."""
        self.enabled = enabled
        if enabled and not self.is_running():
            self.start()
        elif not enabled and self.is_running():
            self.stop()
    
    def _run_loop(self) -> None:
        """Main loop for processing predictions for this interval."""
        while not self._stop_event.is_set():
            if self.enabled:
                self._process_interval()
            self._stop_event.wait(self.check_frequency)
    
    def _process_interval(self) -> None:
        """Process expired predictions and generate new ones for this interval."""
        try:
            # Check for expired predictions for this specific interval
            expired = self.storage.get_expired_predictions(self.interval)
            if not expired:
                return
            
            # Process expired predictions
            processed_count = 0
            for prediction in expired[:self.max_predictions_per_cycle]:
                try:
                    # Evaluate the expired prediction
                    result = self.evaluator.evaluate_prediction(prediction)
                    self.learner_manager.record_evaluation(prediction, result)
                    
                    # Generate a new prediction for the same ticker/interval
                    try:
                        self.prediction_generator.ensure_predictions(
                            tickers=[prediction.ticker],
                            intervals=[prediction.interval],
                            per_interval=1,
                        )
                    except Exception:
                        pass  # Continue even if generation fails
                    
                    processed_count += 1
                except Exception:
                    # Continue processing other predictions even if one fails
                    continue
        except Exception:
            # Log error but continue running
            pass


class ConstantLearningEngine:
    """Coordinates periodic prediction evaluation with separate instances per interval."""

    def __init__(self, enabled: bool = True, evaluation_frequency_seconds: float = 5.0) -> None:
        self.enabled = enabled
        self.evaluation_frequency_seconds = evaluation_frequency_seconds
        self.max_predictions_per_cycle = 10
        self.active_intervals: List[str] = list(CONSTANT_LEARNING_INTERVALS)
        self.active_tickers: List[str] = get_trading212_tickers()
        self._stop_event = threading.Event()
        
        # Shared components
        self.storage = get_prediction_storage()
        self.evaluator = get_prediction_evaluator()
        self.learner_manager = get_interval_learner_manager()
        self.optimizer = get_parameter_optimizer()
        self.prediction_generator = get_prediction_generator()
        
        # Separate processor for each interval
        self._interval_processors: Dict[str, IntervalProcessor] = {}
        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize a processor for each active interval."""
        for interval in self.active_intervals:
            processor = IntervalProcessor(
                interval=interval,
                storage=self.storage,
                evaluator=self.evaluator,
                learner_manager=self.learner_manager,
                prediction_generator=self.prediction_generator,
                max_predictions_per_cycle=self.max_predictions_per_cycle,
                base_frequency=self.evaluation_frequency_seconds,
            )
            self._interval_processors[interval.lower()] = processor

    def start(self) -> None:
        """Start all interval processors."""
        if self.enabled:
            for processor in self._interval_processors.values():
                processor.set_enabled(True)

    def stop(self) -> None:
        """Stop all interval processors."""
        for processor in self._interval_processors.values():
            processor.set_enabled(False)

    def is_running(self) -> bool:
        """Check if any interval processor is running."""
        return any(proc.is_running() for proc in self._interval_processors.values())

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all interval processors."""
        self.enabled = enabled
        if enabled:
            self.start()
        else:
            self.stop()

    def set_active_intervals(self, intervals: Sequence[str]) -> None:
        """Update active intervals and manage processors accordingly."""
        new_intervals = [i.lower() for i in intervals]
        
        # Stop processors for intervals that are no longer active
        for interval, processor in list(self._interval_processors.items()):
            if interval not in new_intervals:
                processor.set_enabled(False)
                del self._interval_processors[interval]
        
        # Start processors for new intervals
        for interval in new_intervals:
            if interval not in self._interval_processors:
                processor = IntervalProcessor(
                    interval=interval,
                    storage=self.storage,
                    evaluator=self.evaluator,
                    learner_manager=self.learner_manager,
                    prediction_generator=self.prediction_generator,
                    max_predictions_per_cycle=self.max_predictions_per_cycle,
                    base_frequency=self.evaluation_frequency_seconds,
                )
                self._interval_processors[interval] = processor
                if self.enabled:
                    processor.set_enabled(True)
        
        self.active_intervals = new_intervals

    def set_active_tickers(self, tickers: Sequence[str]) -> None:
        """Update active tickers list."""
        self.active_tickers = list(tickers)

    def refresh_active_tickers(self) -> None:
        """Reload the Trading212 ticker universe into the active list."""
        self.active_tickers = get_trading212_tickers()

    def set_trade_outcome_weight(self, weight: float) -> None:
        """Set trade outcome weight for optimizer."""
        self.optimizer.set_trade_outcome_weight(weight)

    def set_max_predictions_per_cycle(self, limit: int) -> None:
        """Update max predictions per cycle for all processors."""
        self.max_predictions_per_cycle = max(1, limit)
        for processor in self._interval_processors.values():
            processor.max_predictions_per_cycle = max(1, limit)

    def run_cycle(self) -> None:
        """
        Legacy method for backward compatibility.
        Individual interval processors now handle their own cycles.
        """
        # This method is kept for compatibility but does nothing
        # as each interval processor runs its own cycle independently
        pass

    def get_status(self) -> dict:
        """Get status of all interval processors."""
        interval_statuses = {}
        for interval, processor in self._interval_processors.items():
            interval_statuses[interval] = {
                "running": processor.is_running(),
                "enabled": processor.enabled,
                "check_frequency": processor.check_frequency,
            }
        
        return {
            "enabled": self.enabled,
            "running": self.is_running(),
            "active_intervals": list(self.active_intervals),
            "active_tickers_count": len(self.active_tickers),
            "evaluation_frequency_seconds": self.evaluation_frequency_seconds,
            "max_predictions_per_cycle": self.max_predictions_per_cycle,
            "interval_processors": interval_statuses,
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

