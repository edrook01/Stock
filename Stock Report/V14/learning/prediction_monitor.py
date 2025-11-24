"""
Missed Prediction Handling
Monitors trades that exceed predicted timeframe without outcome.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys

from .trade_tracker import get_trade_tracker

# Import with fallback for direct execution
try:
    from ..core.timeframes import get_timeframe_delta, is_valid_timeframe
except (ImportError, ValueError):
    # Fallback for direct execution or when relative imports fail
    v14_root = Path(__file__).parent.parent
    sys.path.insert(0, str(v14_root))
    from core.timeframes import get_timeframe_delta, is_valid_timeframe


class PredictionMonitor:
    """Monitors predictions for missed timeframes."""
    
    def __init__(self):
        """Initialize prediction monitor."""
        self.trade_tracker = get_trade_tracker()
        self.missed_predictions: List[Dict] = []
    
    def check_missed_predictions(self, open_trades: List[Dict]) -> List[Dict]:
        """
        Check for missed predictions in open trades.
        
        Args:
            open_trades: List of open trade dictionaries with:
                - trade_id: str
                - entry_time: datetime or ISO string
                - timeframe: str
                - target_price: float (optional)
                - stop_price: float
                - current_price: float
                
        Returns:
            List of missed prediction dictionaries
        """
        missed = []
        now = datetime.now()
        
        for trade in open_trades:
            # Parse entry time
            if isinstance(trade.get("entry_time"), str):
                entry_time = datetime.fromisoformat(trade["entry_time"])
            else:
                entry_time = trade.get("entry_time")
            
            if not entry_time:
                continue
            
            timeframe = trade.get("timeframe")
            if not timeframe or not is_valid_timeframe(timeframe):
                continue
            
            # Get timeframe duration
            timeframe_delta = get_timeframe_delta(timeframe)
            if not timeframe_delta:
                continue
            
            # Calculate expected expiration time
            expected_expiry = entry_time + timeframe_delta
            
            # Check if prediction has expired
            if now > expected_expiry:
                # Check if target or stop was hit
                target_price = trade.get("target_price")
                stop_price = trade.get("stop_price")
                current_price = trade.get("current_price")
                
                target_hit = False
                stop_hit = False
                
                direction = trade.get("direction", "LONG").upper()
                
                if target_price and current_price:
                    if direction == "LONG":
                        target_hit = current_price >= target_price
                    else:  # SHORT
                        target_hit = current_price <= target_price
                
                if stop_price and current_price:
                    if direction == "LONG":
                        stop_hit = current_price <= stop_price
                    else:  # SHORT
                        stop_hit = current_price >= stop_price
                
                # If neither target nor stop hit, it's a missed prediction
                if not target_hit and not stop_hit:
                    missed_prediction = {
                        "trade_id": trade.get("trade_id"),
                        "ticker": trade.get("ticker"),
                        "timeframe": timeframe,
                        "entry_time": entry_time.isoformat(),
                        "expected_expiry": expected_expiry.isoformat(),
                        "current_time": now.isoformat(),
                        "time_overdue": (now - expected_expiry).total_seconds(),
                        "current_price": current_price,
                        "target_price": target_price,
                        "stop_price": stop_price,
                        "confidence": trade.get("confidence"),
                        "flagged_at": now.isoformat()
                    }
                    
                    missed.append(missed_prediction)
                    
                    # Add to missed predictions list if not already there
                    if not any(mp.get("trade_id") == missed_prediction["trade_id"] for mp in self.missed_predictions):
                        self.missed_predictions.append(missed_prediction)
        
        return missed
    
    def get_missed_predictions(self, ticker: Optional[str] = None) -> List[Dict]:
        """
        Get list of missed predictions.
        
        Args:
            ticker: Filter by ticker (optional)
            
        Returns:
            List of missed prediction dictionaries
        """
        if ticker:
            return [mp for mp in self.missed_predictions if mp.get("ticker") == ticker]
        return self.missed_predictions.copy()
    
    def clear_resolved(self, trade_id: str) -> bool:
        """
        Clear a missed prediction when trade is resolved.
        
        Args:
            trade_id: Trade identifier
            
        Returns:
            True if found and removed, False otherwise
        """
        initial_count = len(self.missed_predictions)
        self.missed_predictions = [
            mp for mp in self.missed_predictions
            if mp.get("trade_id") != trade_id
        ]
        return len(self.missed_predictions) < initial_count


# Global prediction monitor instance
_prediction_monitor: Optional[PredictionMonitor] = None


def get_prediction_monitor() -> PredictionMonitor:
    """Get global prediction monitor instance."""
    global _prediction_monitor
    if _prediction_monitor is None:
        _prediction_monitor = PredictionMonitor()
    return _prediction_monitor

