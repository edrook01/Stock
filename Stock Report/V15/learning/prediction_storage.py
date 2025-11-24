"""
Prediction storage utilities for the constant learning subsystem.

Provides a lightweight in-memory storage layer with optional hooks for
statistics/learning components so tests can exercise the full workflow
without requiring an external database.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional
from pathlib import Path
import time
import json

AGENT_LOG_PATH = Path(r"c:\Users\edwar\Documents\GitHub\.cursor\debug.log")
AGENT_SESSION_ID = "debug-session"
AGENT_RUN_ID = "pre-fix"


def _agent_log(hypothesis_id: str, location: str, message: str, data=None) -> None:
    """Append NDJSON instrumentation entry for debug workflow."""
    payload = {
        "sessionId": AGENT_SESSION_ID,
        "runId": AGENT_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(AGENT_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")
    except Exception:
        pass

try:
    from core.timeframes import get_timeframe_delta
except Exception:
    def get_timeframe_delta(interval: str) -> timedelta:
        mapping = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "10m": timedelta(minutes=10),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "1y": timedelta(days=365),
        }
        return mapping.get(interval.lower(), timedelta(days=1))


def _notify_statistics(record: "PredictionRecord", event: str) -> None:
    """Best-effort hook into learning statistics without hard dependency."""
    try:
        from learning.learning_statistics import get_learning_statistics

        stats = get_learning_statistics()
        if event == "stored":
            stats.record_prediction(record)
        elif event == "evaluated":
            stats.record_evaluation(record)
    except Exception:
        pass


def _notify_live_readout(record: "PredictionRecord", event: str) -> None:
    """Send prediction activity to the live readout feed."""
    try:
        from learning.live_readout import get_live_prediction_readout

        feed = get_live_prediction_readout()
        feed.record_event(record, event)
    except Exception:
        pass


def _notify_learners(record: "PredictionRecord", event: str) -> None:
    """Notify interval learners for per-interval tracking."""
    try:
        from learning.interval_learners import get_interval_learner_manager

        manager = get_interval_learner_manager()
        if event == "stored":
            manager.record_prediction(record)
        elif event == "evaluated":
            manager.record_evaluation(record)
    except Exception:
        pass


@dataclass
class PredictionRecord:
    prediction_id: str
    ticker: str
    interval: str
    timestamp: datetime
    predicted_price: float
    predicted_range_low: float
    predicted_range_high: float
    confidence: float
    source: str
    evaluation_status: str = "pending"
    actual_price: Optional[float] = None
    actual_high: Optional[float] = None
    actual_low: Optional[float] = None
    actual_close: Optional[float] = None
    accuracy_score: Optional[float] = None
    confidence_calibration: Optional[float] = None
    accuracy_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionRecord":
        parsed = data.copy()
        parsed["timestamp"] = datetime.fromisoformat(parsed["timestamp"])
        parsed["created_at"] = datetime.fromisoformat(parsed["created_at"])
        parsed["updated_at"] = datetime.fromisoformat(parsed["updated_at"])
        parsed.setdefault("actual_price", None)
        parsed.setdefault("actual_high", None)
        parsed.setdefault("actual_low", None)
        parsed.setdefault("actual_close", None)
        parsed.setdefault("accuracy_score", None)
        parsed.setdefault("confidence_calibration", None)
        parsed.setdefault("accuracy_breakdown", {})
        return cls(**parsed)


class PredictionStorage:
    """Simple in-memory prediction registry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._predictions: Dict[str, PredictionRecord] = {}
        self.use_database = False
        self._debug_store_logs = 0
        self._debug_get_logs = 0

    def store_prediction(self, record: PredictionRecord) -> bool:
        with self._lock:
            record.updated_at = datetime.utcnow()
            self._predictions[record.prediction_id] = record
            current_count = len(self._predictions)
        if self._debug_store_logs < 5:
            self._debug_store_logs += 1
            #region agent log
            _agent_log(
                "H1",
                "learning/prediction_storage.py:store_prediction",
                "Stored prediction",
                {
                    "prediction_id": record.prediction_id,
                    "ticker": record.ticker,
                    "interval": record.interval,
                    "total_predictions": current_count,
                },
            )
            #endregion
        _notify_statistics(record, "stored")
        _notify_learners(record, "stored")
        _notify_live_readout(record, "stored")
        return True

    def get_prediction(self, prediction_id: str) -> Optional[PredictionRecord]:
        with self._lock:
            return self._predictions.get(prediction_id)

    def update_prediction(self, record: PredictionRecord) -> bool:
        with self._lock:
            if record.prediction_id not in self._predictions:
                return False
            record.updated_at = datetime.utcnow()
            self._predictions[record.prediction_id] = record
        if record.evaluation_status == "evaluated":
            _notify_statistics(record, "evaluated")
            _notify_learners(record, "evaluated")
            _notify_live_readout(record, "evaluated")
        return True

    def get_pending_predictions(self, interval: Optional[str] = None) -> List[PredictionRecord]:
        with self._lock:
            records = list(self._predictions.values())
        pending = [p for p in records if p.evaluation_status != "evaluated"]
        if interval:
            pending = [p for p in pending if p.interval.lower() == interval.lower()]
        return pending

    def get_expired_predictions(self, interval: str) -> List[PredictionRecord]:
        expiry_delta = get_timeframe_delta(interval)
        cutoff = datetime.utcnow() - expiry_delta
        with self._lock:
            records = list(self._predictions.values())
        return [
            p for p in records
            if p.interval.lower() == interval.lower()
            and p.evaluation_status != "evaluated"
            and p.timestamp < cutoff
        ]

    def get_predictions(self, interval: Optional[str] = None) -> List[PredictionRecord]:
        with self._lock:
            records = list(self._predictions.values())
            total = len(records)
        if interval:
            filtered = [p for p in records if p.interval.lower() == interval.lower()]
        else:
            filtered = records
        if self._debug_get_logs < 5 or not filtered:
            self._debug_get_logs += 1
            #region agent log
            _agent_log(
                "H3",
                "learning/prediction_storage.py:get_predictions",
                "Fetched predictions",
                {
                    "interval_filter": interval or "all",
                    "total_available": total,
                    "returned": len(filtered),
                },
            )
            #endregion
        return filtered

    def clear(self) -> None:
        with self._lock:
            self._predictions.clear()

    def get_active_tickers(self) -> List[str]:
        with self._lock:
            return sorted({p.ticker for p in self._predictions.values()})


_PREDICTION_STORAGE: Optional[PredictionStorage] = None


def get_prediction_storage() -> PredictionStorage:
    global _PREDICTION_STORAGE
    if _PREDICTION_STORAGE is None:
        _PREDICTION_STORAGE = PredictionStorage()
        #region agent log
        _agent_log(
            "H2",
            "learning/prediction_storage.py:get_prediction_storage",
            "Created prediction storage singleton",
            {
                "module": __name__,
                "storage_id": id(_PREDICTION_STORAGE),
            },
        )
        #endregion
    return _PREDICTION_STORAGE


__all__ = [
    "PredictionRecord",
    "PredictionStorage",
    "get_prediction_storage",
]

