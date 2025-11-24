"""
Live prediction readout utilities.

Provides a lightweight, thread-safe event feed so the CLI, Streamlit UI,
and tests can verify that the constant learning system is emitting
prediction activity while it runs.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Deque, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from learning.prediction_storage import PredictionRecord


class LivePredictionReadout:
    """Tracks recent prediction storage/evaluation events."""

    def __init__(self, max_entries: int = 100) -> None:
        self._max_entries = max_entries
        self._entries: Deque[Dict[str, Optional[str]]] = deque(maxlen=max_entries)
        self._lock = Lock()

    def record_event(
        self,
        record: "PredictionRecord",
        event: str,
        accuracy: Optional[float] = None,
    ) -> None:
        """Store a new live-feed entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "prediction_id": record.prediction_id,
            "ticker": record.ticker,
            "interval": record.interval,
            "predicted_price": record.predicted_price,
            "confidence": record.confidence,
            "status": record.evaluation_status,
            "event": event,
            "source": record.source,
            "accuracy": accuracy if accuracy is not None else record.accuracy_score,
        }
        with self._lock:
            self._entries.appendleft(entry)

    def get_recent_entries(self, limit: int = 10) -> List[Dict[str, Optional[str]]]:
        """Return the most recent live entries."""
        with self._lock:
            snapshot = list(self._entries)
        return snapshot[: max(1, limit)]

    def clear(self) -> None:
        """Reset the live readout (useful for tests)."""
        with self._lock:
            self._entries.clear()

    def has_entries(self) -> bool:
        with self._lock:
            return bool(self._entries)


_LIVE_READOUT: Optional[LivePredictionReadout] = None


def get_live_prediction_readout() -> LivePredictionReadout:
    """Return the singleton live readout."""
    global _LIVE_READOUT
    if _LIVE_READOUT is None:
        _LIVE_READOUT = LivePredictionReadout()
    return _LIVE_READOUT


__all__ = ["LivePredictionReadout", "get_live_prediction_readout"]


