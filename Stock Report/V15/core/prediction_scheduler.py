"""
Prediction Synchronization
Schedules prediction updates aligned with timeframes.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio

# Handle both relative and absolute imports for portability
try:
    from .timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES, get_prediction_update_interval
except (ImportError, ValueError):
    # Fallback for direct execution
    from core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES, get_prediction_update_interval

# Import model with multiple fallback strategies
try:
    from ..model.unified_model import get_model
except (ImportError, ValueError):
    # Fallback for direct execution or when relative imports fail
    try:
        from model.unified_model import get_model
    except (ImportError, ValueError):
        # Last resort: try importlib
        try:
            import sys
            import importlib.util
            from pathlib import Path
            V15_ROOT = Path(__file__).parent.parent
            model_spec = importlib.util.spec_from_file_location(
                "unified_model", V15_ROOT / "model" / "unified_model.py"
            )
            model_module = importlib.util.module_from_spec(model_spec)
            sys.modules['unified_model'] = model_module
            model_spec.loader.exec_module(model_module)
            get_model = model_module.get_model
        except Exception:
            # If all imports fail, create a dummy function
            def get_model(timeframe: str):
                raise ImportError("Could not import unified_model. Please ensure model/unified_model.py exists.")


class PredictionScheduler:
    """Schedules and manages prediction updates."""
    
    def __init__(self):
        """Initialize prediction scheduler."""
        self.predictions: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}
        self.scheduler_running = False
    
    async def update_predictions(self, ticker: str, timeframes: list = None) -> Dict[str, Dict]:
        """
        Update predictions for a ticker across timeframes.
        
        Args:
            ticker: Stock ticker symbol
            timeframes: List of timeframes to update (defaults to all)
            
        Returns:
            Dictionary mapping timeframe to prediction
        """
        if timeframes is None:
            timeframes = CFD_TIMEFRAMES + INVESTMENT_TIMEFRAMES
        
        predictions = {}
        
        for timeframe in timeframes:
            # Check if update is needed
            if self._should_update(ticker, timeframe):
                try:
                    model = get_model(timeframe)
                    prediction = await model.predict(ticker)
                    predictions[timeframe] = prediction
                    
                    # Store prediction
                    key = f"{ticker}_{timeframe}"
                    self.predictions[key] = prediction
                    self.last_update[key] = datetime.now()
                except Exception:
                    # Use cached prediction if available
                    key = f"{ticker}_{timeframe}"
                    if key in self.predictions:
                        predictions[timeframe] = self.predictions[key]
        
        return predictions
    
    def _should_update(self, ticker: str, timeframe: str) -> bool:
        """
        Check if prediction should be updated.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe
            
        Returns:
            True if should update, False otherwise
        """
        key = f"{ticker}_{timeframe}"
        
        # Always update if no cached prediction
        if key not in self.last_update:
            return True
        
        # Check update interval
        update_interval = get_prediction_update_interval(timeframe)
        if update_interval is None:
            return True
        
        last_update_time = self.last_update[key]
        time_since_update = (datetime.now() - last_update_time).total_seconds()
        
        return time_since_update >= update_interval
    
    def get_prediction(self, ticker: str, timeframe: str) -> Optional[Dict]:
        """
        Get cached prediction.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe
            
        Returns:
            Prediction dictionary, or None if not cached
        """
        key = f"{ticker}_{timeframe}"
        return self.predictions.get(key)
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear prediction cache.
        
        Args:
            ticker: Ticker to clear (None = clear all)
        """
        if ticker:
            keys_to_remove = [k for k in self.predictions.keys() if k.startswith(f"{ticker}_")]
            for key in keys_to_remove:
                self.predictions.pop(key, None)
                self.last_update.pop(key, None)
        else:
            self.predictions.clear()
            self.last_update.clear()


# Global prediction scheduler instance
_prediction_scheduler: Optional[PredictionScheduler] = None


def get_prediction_scheduler() -> PredictionScheduler:
    """Get global prediction scheduler instance."""
    global _prediction_scheduler
    if _prediction_scheduler is None:
        _prediction_scheduler = PredictionScheduler()
    return _prediction_scheduler

