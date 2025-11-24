"""
Trade Failure Tracking
Flags trades that exceed 2% drawdown threshold and detects execution errors.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import sys

from .trade_tracker import get_trade_tracker, TradeOutcome

# Import with fallback for direct execution
try:
    from ..risk.equity_monitor import get_equity_monitor
except (ImportError, ValueError):
    # Fallback for direct execution or when relative imports fail
    V15_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(V15_ROOT))
    from risk.equity_monitor import get_equity_monitor


class FailureTracker:
    """Tracks failed trades that exceed thresholds."""
    
    def __init__(self, failure_threshold_pct: float = 2.0):
        """
        Initialize failure tracker.
        
        Args:
            failure_threshold_pct: Failure threshold as percentage of equity (default: 2.0%)
        """
        self.failure_threshold_pct = failure_threshold_pct
        self.trade_tracker = get_trade_tracker()
        self.equity_monitor = get_equity_monitor()
        self.failed_trades: List[Dict] = []
    
    def check_trade_failure(
        self,
        trade_id: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        direction: str,
        planned_stop_price: float
    ) -> Optional[Dict]:
        """
        Check if a trade should be flagged as a failure.
        
        Args:
            trade_id: Trade identifier
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            direction: Trade direction ("LONG" or "SHORT")
            planned_stop_price: Planned stop-loss price
            
        Returns:
            Failure dictionary if trade failed, None otherwise
        """
        # Calculate actual loss
        if direction.upper() == "LONG":
            actual_loss = (entry_price - exit_price) * position_size
        else:  # SHORT
            actual_loss = (exit_price - entry_price) * position_size
        
        # Calculate planned loss (if stop was hit)
        if direction.upper() == "LONG":
            planned_loss = (entry_price - planned_stop_price) * position_size
        else:  # SHORT
            planned_loss = (planned_stop_price - entry_price) * position_size
        
        # Get current equity
        equity = self.equity_monitor.get_current_equity()
        
        if equity <= 0:
            return None
        
        # Calculate loss as percentage of equity
        loss_pct = (actual_loss / equity) * 100.0
        
        # Check if exceeds threshold
        if loss_pct > self.failure_threshold_pct:
            # Check for slippage (actual loss > planned loss)
            slippage = actual_loss - planned_loss if actual_loss > planned_loss else 0.0
            
            failure = {
                "trade_id": trade_id,
                "timestamp": datetime.now().isoformat(),
                "loss_amount": actual_loss,
                "loss_percentage": loss_pct,
                "planned_loss": planned_loss,
                "slippage": slippage,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "planned_stop": planned_stop_price,
                "direction": direction,
                "position_size": position_size,
                "failure_type": self._determine_failure_type(actual_loss, planned_loss, slippage)
            }
            
            self.failed_trades.append(failure)
            return failure
        
        return None
    
    def _determine_failure_type(
        self,
        actual_loss: float,
        planned_loss: float,
        slippage: float
    ) -> str:
        """
        Determine the type of failure.
        
        Args:
            actual_loss: Actual loss amount
            planned_loss: Planned loss amount
            slippage: Slippage amount
            
        Returns:
            Failure type string
        """
        if slippage > planned_loss * 0.5:  # Significant slippage
            return "slippage"
        elif actual_loss > planned_loss * 1.5:  # Much larger than planned
            return "execution_error"
        elif actual_loss > planned_loss:  # Some slippage
            return "slippage_minor"
        else:
            return "threshold_exceeded"
    
    def get_failed_trades(self, ticker: Optional[str] = None) -> List[Dict]:
        """
        Get list of failed trades.
        
        Args:
            ticker: Filter by ticker (optional)
            
        Returns:
            List of failed trade dictionaries
        """
        if ticker:
            return [ft for ft in self.failed_trades if ft.get("ticker") == ticker]
        return self.failed_trades.copy()
    
    def get_failure_statistics(self) -> Dict:
        """
        Get statistics on failed trades.
        
        Returns:
            Dictionary with failure statistics
        """
        if not self.failed_trades:
            return {
                "total_failures": 0,
                "total_loss": 0.0,
                "avg_loss": 0.0,
                "failure_types": {}
            }
        
        total_loss = sum(ft["loss_amount"] for ft in self.failed_trades)
        avg_loss = total_loss / len(self.failed_trades)
        
        failure_types = {}
        for ft in self.failed_trades:
            ftype = ft.get("failure_type", "unknown")
            failure_types[ftype] = failure_types.get(ftype, 0) + 1
        
        return {
            "total_failures": len(self.failed_trades),
            "total_loss": total_loss,
            "avg_loss": avg_loss,
            "failure_types": failure_types
        }


# Global failure tracker instance
_failure_tracker: Optional[FailureTracker] = None


def get_failure_tracker() -> FailureTracker:
    """Get global failure tracker instance."""
    global _failure_tracker
    if _failure_tracker is None:
        _failure_tracker = FailureTracker()
    return _failure_tracker

