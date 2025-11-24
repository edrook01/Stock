"""
Prediction generation pipeline for constant learning and manual analysis.

Produces PredictionRecord instances for all required intervals so the live
feed, evaluators, and learner managers always have data to work with.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import time
import json

# Handle both relative and absolute imports for portability
try:
    from .prediction_storage import PredictionRecord, get_prediction_storage
    from ..core.timeframes import CONSTANT_LEARNING_INTERVALS
    from ..core.ticker_universe import get_trading212_tickers
    from ..core.data_fetcher import fetch_prices
    from ..model.unified_model import get_model
except (ImportError, ValueError):
    from learning.prediction_storage import PredictionRecord, get_prediction_storage
    from core.timeframes import CONSTANT_LEARNING_INTERVALS
    from core.ticker_universe import get_trading212_tickers
    from core.data_fetcher import fetch_prices
    from model.unified_model import get_model


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


class PredictionGenerator:
    """Generates and stores interval predictions for Function 3."""

    def __init__(self) -> None:
        self.storage = get_prediction_storage()
        self.default_tickers: List[str] = get_trading212_tickers()
        self.default_intervals: List[str] = list(CONSTANT_LEARNING_INTERVALS)
        self._ensure_logs = 0
        self._record_logs = 0

    def ensure_predictions(
        self,
        tickers: Optional[Sequence[str]] = None,
        intervals: Optional[Sequence[str]] = None,
        per_interval: int = 1,
    ) -> int:
        """
        Ensure at least `per_interval` pending predictions exist per ticker/interval.
        Returns the number of newly created prediction records.
        """
        tickers = [t.upper() for t in (tickers or self.default_tickers)]
        intervals = [i.lower() for i in (intervals or self.default_intervals)]
        per_interval = max(1, per_interval)

        created = 0
        for ticker in tickers:
            for interval in intervals:
                pending = [
                    p
                    for p in self.storage.get_pending_predictions(interval)
                    if p.ticker.upper() == ticker
                ]
                needed = per_interval - len(pending)
                if needed <= 0:
                    continue
                for _ in range(needed):
                    record = self._generate_prediction_sync(ticker, interval)
                    if record:
                        self.storage.store_prediction(record)
                        created += 1

        if self._ensure_logs < 5:
            self._ensure_logs += 1
            #region agent log
            _agent_log(
                "H4",
                "learning/prediction_generator.py:ensure_predictions",
                "Ensured prediction backlog",
                {
                    "tickers": tickers,
                    "intervals": intervals,
                    "per_interval": per_interval,
                    "created": created,
                },
            )
            #endregion
        return created

    def record_external_prediction(
        self,
        ticker: str,
        interval: str,
        prediction: Dict,
        source: str = "manual_analysis",
    ) -> bool:
        """Store a prediction that was generated outside the generator."""
        ticker = ticker.upper()
        interval = interval.lower()
        base_price = self._fetch_recent_price_sync(ticker, interval)
        record = self._build_record_from_prediction(
            ticker=ticker,
            interval=interval,
            base_price=base_price,
            prediction=prediction,
            source=source,
            metadata_extra={"generator": source},
        )
        if record:
            self.storage.store_prediction(record)
            return True
        return False

    def _generate_prediction_sync(self, ticker: str, interval: str) -> Optional[PredictionRecord]:
        try:
            return asyncio.run(self._generate_prediction_async(ticker, interval))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._generate_prediction_async(ticker, interval))
            finally:
                loop.close()

    async def _generate_prediction_async(self, ticker: str, interval: str) -> Optional[PredictionRecord]:
        last_price: Optional[float] = None
        df = None
        try:
            df = await fetch_prices(ticker, interval)
            if df is not None and len(df) > 0:
                last_price = float(df["Close"].iloc[-1])
        except Exception:
            df = None

        model = get_model(interval)
        prediction: Optional[Dict] = None
        try:
            prediction = await model.predict(ticker, df=df)
        except Exception:
            prediction = None

        if not prediction:
            prediction = {
                "prediction": 0.0,
                "confidence": 0.5,
                "range_low": -1.0,
                "range_high": 1.0,
                "timeframe": interval,
                "is_default": True,
            }

        base_price = last_price or self._default_price_for_ticker(ticker)
        record = self._build_record_from_prediction(
            ticker=ticker,
            interval=interval,
            base_price=base_price,
            prediction=prediction,
            source="constant_learning",
            metadata_extra={"data_source": "live" if last_price else "synthetic"},
        )
        if record and self._record_logs < 5:
            self._record_logs += 1
            #region agent log
            _agent_log(
                "H5",
                "learning/prediction_generator.py:_generate_prediction_async",
                "Generated prediction record",
                {
                    "ticker": ticker,
                    "interval": interval,
                    "base_price": base_price,
                    "movement_pct": prediction.get("prediction", 0.0),
                    "confidence": prediction.get("confidence", 0.5),
                },
            )
            #endregion
        return record

    def _build_record_from_prediction(
        self,
        ticker: str,
        interval: str,
        base_price: Optional[float],
        prediction: Dict,
        source: str,
        metadata_extra: Optional[Dict] = None,
    ) -> Optional[PredictionRecord]:
        if base_price is None or base_price <= 0:
            base_price = self._default_price_for_ticker(ticker)

        movement_pct = float(prediction.get("prediction", 0.0))
        range_low_pct = float(prediction.get("range_low", movement_pct - 1.0))
        range_high_pct = float(prediction.get("range_high", movement_pct + 1.0))
        confidence = float(prediction.get("confidence", 0.5))

        predicted_price = base_price * (1 + movement_pct / 100.0)
        low_price = base_price * (1 + range_low_pct / 100.0)
        high_price = base_price * (1 + range_high_pct / 100.0)
        low_price, high_price = sorted([low_price, high_price])

        metadata = {
            "base_price": round(base_price, 6),
            "movement_pct": movement_pct,
            "range_low_pct": range_low_pct,
            "range_high_pct": range_high_pct,
            "model_agreement": prediction.get("model_agreement"),
            "elapsed": {"high": None, "low": None, "close": None},
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        return PredictionRecord(
            prediction_id=uuid.uuid4().hex,
            ticker=ticker.upper(),
            interval=interval.lower(),
            timestamp=datetime.utcnow(),
            predicted_price=round(predicted_price, 6),
            predicted_range_low=round(low_price, 6),
            predicted_range_high=round(high_price, 6),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            metadata=metadata,
        )

    def _default_price_for_ticker(self, ticker: str) -> float:
        """Deterministic fallback price when live data is unavailable."""
        seed = sum(ord(c) for c in ticker)
        return round(50.0 + (seed % 200), 4)

    def _fetch_recent_price_sync(self, ticker: str, interval: str) -> Optional[float]:
        try:
            return asyncio.run(self._fetch_recent_price_async(ticker, interval))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._fetch_recent_price_async(ticker, interval))
            finally:
                loop.close()

    async def _fetch_recent_price_async(self, ticker: str, interval: str) -> Optional[float]:
        try:
            df = await fetch_prices(ticker, interval)
            if df is not None and len(df) > 0:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return None


_PREDICTION_GENERATOR: Optional[PredictionGenerator] = None


def get_prediction_generator() -> PredictionGenerator:
    """Return singleton prediction generator."""
    global _PREDICTION_GENERATOR
    if _PREDICTION_GENERATOR is None:
        _PREDICTION_GENERATOR = PredictionGenerator()
    return _PREDICTION_GENERATOR


__all__ = ["PredictionGenerator", "get_prediction_generator"]


