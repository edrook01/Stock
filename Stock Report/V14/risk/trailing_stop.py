"""
Trailing Stop Implementation
Manages trailing stops that move with price to lock in profits.
"""

from typing import Optional, Dict
from datetime import datetime
import pandas as pd

from .volatility import calculate_atr


class TrailingStop:
    """
    Represents a trailing stop for an open position.
    """
    
    def __init__(
        self,
        entry_price: float,
        direction: str,
        initial_stop: float,
        atr: float,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize trailing stop.
        
        Args:
            entry_price: Entry price of the trade
            direction: Trade direction ("LONG" or "SHORT")
            initial_stop: Initial stop-loss price
            atr: Current ATR value
            atr_multiplier: ATR multiplier for trailing distance
        """
        self.entry_price = entry_price
        self.direction = direction.upper()
        self.current_stop = initial_stop
        self.atr = atr
        self.atr_multiplier = atr_multiplier
        self.best_price = entry_price  # Best price seen so far
        self.is_profitable = False
        self.is_breakeven = False
        
        if self.direction not in ["LONG", "SHORT"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'")
    
    
    def update(self, current_price: float, new_atr: Optional[float] = None) -> bool:
        """
        Update trailing stop based on current price.
        
        Args:
            current_price: Current market price
            new_atr: Updated ATR value (optional)
            
        Returns:
            True if stop was updated, False otherwise
        """
        # Update ATR if provided
        if new_atr is not None:
            self.atr = new_atr
        
        # Calculate trailing distance
        trailing_distance = self.atr * self.atr_multiplier
        
        updated = False
        
        if self.direction == "LONG":
            # Long trade: stop trails below price
            if current_price > self.best_price:
                # New best price - update stop
                self.best_price = current_price
                new_stop = current_price - trailing_distance
                
                # Never move stop backward (only tighten)
                if new_stop > self.current_stop:
                    self.current_stop = new_stop
                    updated = True
                    
                    # Check if we're in profit
                    if self.current_stop > self.entry_price:
                        self.is_profitable = True
                        # Check if we're at breakeven
                        if not self.is_breakeven and self.current_stop >= self.entry_price:
                            self.is_breakeven = True
        else:
            # SHORT trade: stop trails above price
            if current_price < self.best_price:
                # New best price - update stop
                self.best_price = current_price
                new_stop = current_price + trailing_distance
                
                # Never move stop backward (only tighten)
                if new_stop < self.current_stop:
                    self.current_stop = new_stop
                    updated = True
                    
                    # Check if we're in profit
                    if self.current_stop < self.entry_price:
                        self.is_profitable = True
                        # Check if we're at breakeven
                        if not self.is_breakeven and self.current_stop <= self.entry_price:
                            self.is_breakeven = True
        
        return updated
    
    
    def is_triggered(self, current_price: float) -> bool:
        """
        Check if trailing stop has been triggered.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if stop is triggered, False otherwise
        """
        if self.direction == "LONG":
            return current_price <= self.current_stop
        else:  # SHORT
            return current_price >= self.current_stop
    
    
    def get_current_stop(self) -> float:
        """Get current stop-loss price."""
        return self.current_stop
    
    
    def get_profit_locked(self) -> float:
        """
        Get amount of profit currently locked in by trailing stop.
        
        Returns:
            Profit amount (0 if not profitable)
        """
        if not self.is_profitable:
            return 0.0
        
        if self.direction == "LONG":
            return self.current_stop - self.entry_price
        else:  # SHORT
            return self.entry_price - self.current_stop


def create_trailing_stop(
    entry_price: float,
    direction: str,
    initial_stop: float,
    df: pd.DataFrame,
    atr_multiplier: float = 2.0
) -> TrailingStop:
    """
    Create a new trailing stop for a position.
    
    Args:
        entry_price: Entry price
        direction: Trade direction ("LONG" or "SHORT")
        initial_stop: Initial stop-loss price
        df: DataFrame with OHLC data for ATR calculation
        atr_multiplier: ATR multiplier for trailing distance
        
    Returns:
        TrailingStop instance
    """
    try:
        atr = calculate_atr(df, period=14)
    except ValueError:
        # Fallback ATR if calculation fails
        atr = abs(entry_price - initial_stop) / atr_multiplier
    
    return TrailingStop(
        entry_price=entry_price,
        direction=direction,
        initial_stop=initial_stop,
        atr=atr,
        atr_multiplier=atr_multiplier
    )

