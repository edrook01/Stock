"""
Trade Outcome Tracking
Tracks all executed trades and their outcomes for learning.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from ..core.portable_paths import get_path


class TradeOutcome:
    """Represents a completed trade outcome."""
    
    def __init__(
        self,
        trade_id: str,
        ticker: str,
        direction: str,
        entry_time: datetime,
        entry_price: float,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        position_size: float,
        stop_price: float,
        target_price: Optional[float],
        confidence: float,
        timeframe: str,
        predicted_outcome: Optional[float] = None,
        actual_outcome: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_percentage: Optional[float] = None
    ):
        """
        Initialize trade outcome.
        
        Args:
            trade_id: Unique trade identifier
            ticker: Stock ticker symbol
            direction: Trade direction ("LONG" or "SHORT")
            entry_time: Entry timestamp
            entry_price: Entry price
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit ("TP", "SL", "Manual", "Missed")
            position_size: Position size (number of units)
            stop_price: Stop-loss price
            target_price: Take-profit price (optional)
            confidence: Model confidence at entry (0-1)
            timeframe: Prediction timeframe
            predicted_outcome: Predicted price movement (optional)
            actual_outcome: Actual price movement (optional)
            pnl: Profit/loss amount (optional)
            pnl_percentage: Profit/loss percentage (optional)
        """
        self.trade_id = trade_id
        self.ticker = ticker
        self.direction = direction.upper()
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.position_size = position_size
        self.stop_price = stop_price
        self.target_price = target_price
        self.confidence = confidence
        self.timeframe = timeframe
        self.predicted_outcome = predicted_outcome
        self.actual_outcome = actual_outcome
        self.pnl = pnl
        self.pnl_percentage = pnl_percentage
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "position_size": self.position_size,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeOutcome':
        """Create from dictionary."""
        return cls(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            direction=data["direction"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            exit_time=datetime.fromisoformat(data["exit_time"]),
            exit_price=data["exit_price"],
            exit_reason=data["exit_reason"],
            position_size=data["position_size"],
            stop_price=data["stop_price"],
            target_price=data.get("target_price"),
            confidence=data["confidence"],
            timeframe=data["timeframe"],
            predicted_outcome=data.get("predicted_outcome"),
            actual_outcome=data.get("actual_outcome"),
            pnl=data.get("pnl"),
            pnl_percentage=data.get("pnl_percentage")
        )


class TradeTracker:
    """Tracks all trade outcomes."""
    
    def __init__(self):
        """Initialize trade tracker."""
        self.outcomes: List[TradeOutcome] = []
        self._load_outcomes()
    
    def add_outcome(self, outcome: TradeOutcome) -> None:
        """Add a trade outcome."""
        self.outcomes.append(outcome)
        self._save_outcomes()
    
    def get_outcomes(self, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> List[TradeOutcome]:
        """
        Get trade outcomes, optionally filtered.
        
        Args:
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)
            
        Returns:
            List of trade outcomes
        """
        results = self.outcomes
        
        if ticker:
            results = [o for o in results if o.ticker == ticker]
        
        if timeframe:
            results = [o for o in results if o.timeframe == timeframe]
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get statistics on trade outcomes.
        
        Returns:
            Dictionary with statistics
        """
        if not self.outcomes:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0
            }
        
        wins = [o for o in self.outcomes if o.pnl and o.pnl > 0]
        losses = [o for o in self.outcomes if o.pnl and o.pnl < 0]
        
        total_pnl = sum(o.pnl for o in self.outcomes if o.pnl)
        avg_pnl = total_pnl / len(self.outcomes) if self.outcomes else 0.0
        
        return {
            "total_trades": len(self.outcomes),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.outcomes) if self.outcomes else 0.0,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "avg_confidence": sum(o.confidence for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0
        }
    
    def _save_outcomes(self) -> None:
        """Save outcomes to file."""
        try:
            history_dir = get_path('history')
            history_dir.mkdir(parents=True, exist_ok=True)
            
            outcomes_file = history_dir / 'trade_outcomes.json'
            
            outcomes_data = [outcome.to_dict() for outcome in self.outcomes]
            
            with open(outcomes_file, 'w') as f:
                json.dump(outcomes_data, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_outcomes(self) -> None:
        """Load outcomes from file."""
        try:
            history_dir = get_path('history')
            outcomes_file = history_dir / 'trade_outcomes.json'
            
            if not outcomes_file.exists():
                return
            
            with open(outcomes_file, 'r') as f:
                outcomes_data = json.load(f)
            
            self.outcomes = [TradeOutcome.from_dict(data) for data in outcomes_data]
        except Exception:
            # Silent failure on load errors
            self.outcomes = []


# Global trade tracker instance
_trade_tracker: Optional[TradeTracker] = None


def get_trade_tracker() -> TradeTracker:
    """Get global trade tracker instance."""
    global _trade_tracker
    if _trade_tracker is None:
        _trade_tracker = TradeTracker()
    return _trade_tracker

