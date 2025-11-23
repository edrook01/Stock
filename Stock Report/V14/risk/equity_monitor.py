"""
Account Equity Monitoring
Fetches and tracks account equity from browser interface or simulation.
"""

from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path
import json

from ..core.portable_paths import get_path


class EquityMonitor:
    """Monitors account equity and tracks equity curve."""
    
    def __init__(self, initial_equity: float = 0.0):
        """
        Initialize equity monitor.
        
        Args:
            initial_equity: Initial account equity
        """
        self.current_equity = initial_equity
        self.equity_history: List[Dict] = []
        self.peak_equity = initial_equity
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
    
    def update_equity(self, new_equity: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update current equity and track history.
        
        Args:
            new_equity: New equity value
            timestamp: Timestamp of update (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_equity = new_equity
        
        # Track equity history
        self.equity_history.append({
            "timestamp": timestamp.isoformat(),
            "equity": new_equity
        })
        
        # Update peak equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = self.peak_equity - new_equity
            drawdown_pct = (drawdown / self.peak_equity) * 100.0
            
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_pct = drawdown_pct
    
    def get_current_equity(self) -> float:
        """Get current account equity."""
        return self.current_equity
    
    def get_drawdown(self) -> tuple:
        """
        Get current drawdown.
        
        Returns:
            Tuple of (drawdown_amount, drawdown_percentage)
        """
        if self.peak_equity > 0:
            drawdown = self.peak_equity - self.current_equity
            drawdown_pct = (drawdown / self.peak_equity) * 100.0
            return (drawdown, drawdown_pct)
        return (0.0, 0.0)
    
    def get_max_drawdown(self) -> tuple:
        """
        Get maximum drawdown seen.
        
        Returns:
            Tuple of (max_drawdown_amount, max_drawdown_percentage)
        """
        return (self.max_drawdown, self.max_drawdown_pct)
    
    def save_history(self) -> None:
        """Save equity history to file."""
        try:
            history_dir = get_path('history')
            history_dir.mkdir(parents=True, exist_ok=True)
            
            history_file = history_dir / 'equity_history.json'
            
            with open(history_file, 'w') as f:
                json.dump({
                    "current_equity": self.current_equity,
                    "peak_equity": self.peak_equity,
                    "max_drawdown": self.max_drawdown,
                    "max_drawdown_pct": self.max_drawdown_pct,
                    "history": self.equity_history
                }, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def load_history(self) -> bool:
        """
        Load equity history from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            history_dir = get_path('history')
            history_file = history_dir / 'equity_history.json'
            
            if not history_file.exists():
                return False
            
            with open(history_file, 'r') as f:
                data = json.load(f)
            
            self.current_equity = data.get("current_equity", 0.0)
            self.peak_equity = data.get("peak_equity", self.current_equity)
            self.max_drawdown = data.get("max_drawdown", 0.0)
            self.max_drawdown_pct = data.get("max_drawdown_pct", 0.0)
            self.equity_history = data.get("history", [])
            
            return True
        except Exception:
            return False


# Global equity monitor instance (will be initialized by main application)
_equity_monitor: Optional[EquityMonitor] = None


def get_equity_monitor() -> EquityMonitor:
    """Get global equity monitor instance."""
    global _equity_monitor
    if _equity_monitor is None:
        _equity_monitor = EquityMonitor()
        # Try to load existing history
        _equity_monitor.load_history()
    return _equity_monitor


def set_equity_monitor(monitor: EquityMonitor) -> None:
    """Set global equity monitor instance."""
    global _equity_monitor
    _equity_monitor = monitor

