"""
Combined Exposure Tracking
Tracks total risk exposure across all open positions and enforces limits.
"""

from typing import List, Dict, Optional
from datetime import datetime

from .profiles import RiskProfile, get_max_combined_exposure


class Position:
    """Represents an open position for exposure tracking."""
    
    def __init__(
        self,
        position_id: str,
        ticker: str,
        direction: str,
        entry_price: float,
        quantity: float,
        stop_price: float,
        current_price: float
    ):
        """
        Initialize position.
        
        Args:
            position_id: Unique position identifier
            ticker: Stock ticker symbol
            direction: Trade direction ("LONG" or "SHORT")
            entry_price: Entry price
            quantity: Position size (number of units)
            stop_price: Stop-loss price
            current_price: Current market price
        """
        self.position_id = position_id
        self.ticker = ticker
        self.direction = direction.upper()
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_price = stop_price
        self.current_price = current_price
    
    def get_risk_amount(self) -> float:
        """
        Calculate risk amount (potential loss if stop is hit).
        
        Returns:
            Risk amount in dollars
        """
        if self.direction == "LONG":
            stop_distance = self.entry_price - self.stop_price
        else:  # SHORT
            stop_distance = self.stop_price - self.entry_price
        
        return abs(stop_distance * self.quantity)
    
    def get_exposure_percentage(self, equity: float) -> float:
        """
        Calculate exposure as percentage of equity.
        
        Args:
            equity: Account equity
            
        Returns:
            Exposure percentage
        """
        if equity <= 0:
            return 0.0
        
        risk_amount = self.get_risk_amount()
        return (risk_amount / equity) * 100.0


class ExposureTracker:
    """Tracks combined exposure across all positions."""
    
    def __init__(self, equity: float, profile: RiskProfile):
        """
        Initialize exposure tracker.
        
        Args:
            equity: Current account equity
            profile: Risk profile
        """
        self.equity = equity
        self.profile = profile
        self.positions: Dict[str, Position] = {}
        self.max_exposure = get_max_combined_exposure(profile)
    
    def update_equity(self, new_equity: float) -> None:
        """Update account equity."""
        self.equity = new_equity
    
    def add_position(self, position: Position) -> None:
        """Add a position to tracking."""
        self.positions[position.position_id] = position
    
    def remove_position(self, position_id: str) -> Optional[Position]:
        """Remove a position from tracking."""
        return self.positions.pop(position_id, None)
    
    def update_position_price(self, position_id: str, current_price: float) -> bool:
        """
        Update current price for a position.
        
        Args:
            position_id: Position identifier
            current_price: New current price
            
        Returns:
            True if position found and updated, False otherwise
        """
        if position_id in self.positions:
            self.positions[position_id].current_price = current_price
            return True
        return False
    
    def get_total_exposure(self) -> float:
        """
        Calculate total exposure across all positions.
        
        Returns:
            Total exposure as percentage of equity
        """
        if self.equity <= 0:
            return 0.0
        
        total_risk = sum(pos.get_risk_amount() for pos in self.positions.values())
        return (total_risk / self.equity) * 100.0
    
    def get_worst_case_loss(self) -> float:
        """
        Calculate worst-case loss if all stops are hit.
        
        Returns:
            Worst-case loss amount in dollars
        """
        return sum(pos.get_risk_amount() for pos in self.positions.values())
    
    def can_open_new_position(self, new_position_risk: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened without exceeding exposure limits.
        
        Args:
            new_position_risk: Risk amount for the new position
            
        Returns:
            Tuple of (can_open, reason)
        """
        if self.equity <= 0:
            return (False, "Invalid equity")
        
        current_exposure = self.get_total_exposure()
        new_exposure_pct = ((self.get_worst_case_loss() + new_position_risk) / self.equity) * 100.0
        
        if new_exposure_pct > self.max_exposure:
            return (
                False,
                f"New position would exceed max exposure: {new_exposure_pct:.2f}% > {self.max_exposure:.2f}%"
            )
        
        return (True, "OK")
    
    def get_position_exposures(self) -> Dict[str, float]:
        """
        Get exposure percentage for each position.
        
        Returns:
            Dictionary mapping position_id to exposure percentage
        """
        return {
            pos_id: pos.get_exposure_percentage(self.equity)
            for pos_id, pos in self.positions.items()
        }
    
    def get_summary(self) -> Dict:
        """
        Get summary of exposure tracking.
        
        Returns:
            Dictionary with exposure summary
        """
        return {
            "equity": self.equity,
            "total_positions": len(self.positions),
            "total_exposure_pct": self.get_total_exposure(),
            "worst_case_loss": self.get_worst_case_loss(),
            "max_exposure": self.max_exposure,
            "exposure_remaining": max(0.0, self.max_exposure - self.get_total_exposure()),
            "positions": [
                {
                    "id": pos.position_id,
                    "ticker": pos.ticker,
                    "exposure_pct": pos.get_exposure_percentage(self.equity),
                    "risk_amount": pos.get_risk_amount()
                }
                for pos in self.positions.values()
            ]
        }

