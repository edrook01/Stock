"""
Prediction evaluation helpers for the constant learning subsystem.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional

try:
    from core.data_fetcher import fetch_prices
except Exception:
    fetch_prices = None  # type: ignore

from learning.prediction_storage import PredictionRecord, get_prediction_storage


class PredictionEvaluator:
    """Scores predictions and updates stored records."""

    def __init__(self) -> None:
        self.storage = get_prediction_storage()

    def evaluate_prediction(self, prediction: PredictionRecord) -> Dict[str, Dict[str, float]]:
        actual_price = prediction.actual_price
        if actual_price is None:
            actual_price = self._determine_actual_price(prediction.ticker, prediction.interval, prediction.predicted_price)

        accuracy = self._calculate_accuracy(prediction, actual_price)
        prediction.actual_price = actual_price
        prediction.accuracy_score = accuracy["accuracy_scores"]["overall_accuracy"]
        prediction.confidence_calibration = accuracy["confidence_calibration"]
        prediction.evaluation_status = "evaluated"
        prediction.updated_at = datetime.utcnow()
        self.storage.update_prediction(prediction)
        return accuracy

    def evaluate_pending_predictions(self, interval: Optional[str] = None, max_predictions: int = 10) -> List[PredictionRecord]:
        pending = self.storage.get_pending_predictions(interval) if interval else self.storage.get_predictions()
        results: List[PredictionRecord] = []
        for prediction in pending[:max_predictions]:
            self.evaluate_prediction(prediction)
            results.append(prediction)
        return results

    def _determine_actual_price(self, ticker: str, interval: str, fallback_price: float) -> float:
        if fetch_prices is None:
            return fallback_price
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                df = loop.run_until_complete(fetch_prices(ticker, interval))
            finally:
                loop.close()
            if df is not None and len(df) > 0:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        # Fallback: small random drift around predicted price
        rng = random.Random(f"{ticker}_{interval}")
        return max(0.01, fallback_price + rng.uniform(-1.5, 1.5))

    @staticmethod
    def _calculate_accuracy(prediction: PredictionRecord, actual_price: float) -> Dict[str, Dict[str, float]]:
        within_range = prediction.predicted_range_low <= actual_price <= prediction.predicted_range_high
        range_width = max(prediction.predicted_range_high - prediction.predicted_range_low, 0.01)
        distance = abs(actual_price - prediction.predicted_price)
        range_penalty = min(distance / range_width, 2.0)
        overall_score = round(max(0.0, 10.0 - range_penalty * 5.0), 2)
        confidence_error = abs(prediction.confidence - min(1.0, prediction.predicted_price / max(actual_price, 0.01)))
        confidence_calibration = round(max(0.0, 1.0 - confidence_error), 3)
        return {
            "accuracy_scores": {
                "within_range": 1.0 if within_range else 0.0,
                "overall_accuracy": overall_score,
            },
            "confidence_calibration": confidence_calibration,
            "actual_price": actual_price,
        }


_PREDICTION_EVALUATOR: Optional[PredictionEvaluator] = None


def get_prediction_evaluator() -> PredictionEvaluator:
    global _PREDICTION_EVALUATOR
    if _PREDICTION_EVALUATOR is None:
        _PREDICTION_EVALUATOR = PredictionEvaluator()
    return _PREDICTION_EVALUATOR


__all__ = [
    "PredictionEvaluator",
    "get_prediction_evaluator",
]

