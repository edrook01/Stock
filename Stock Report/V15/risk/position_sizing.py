"""
Position Sizing Calculator
Calculates position sizes based on account equity, risk percentage, and stop-loss distance.
"""

from typing import Optional, Tuple
from .profiles import RiskProfile, get_equity_risk_range, get_profile_config


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_percentage: float,
    direction: str = "LONG"
) -> Tuple[float, float]:
    """
    Calculate position size based on risk management rules.
    
    Formula: Position Size = (Risk% × Equity) / StopDistance
    
    Args:
        equity: Current account equity (balance + unrealized P/L)
        entry_price: Entry price for the trade
        stop_price: Stop-loss price
        risk_percentage: Risk percentage per trade (0.5-2.0)
        direction: Trade direction ("LONG" or "SHORT")
        
    Returns:
        Tuple of (position_size, risk_amount)
        position_size: Number of units/shares to trade
        risk_amount: Dollar amount at risk
    """
    if equity <= 0:
        raise ValueError("Equity must be positive")
    
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    
    if risk_percentage <= 0 or risk_percentage > 2.0:
        raise ValueError(f"Risk percentage must be between 0 and 2.0, got {risk_percentage}")
    
    # Calculate stop distance
    if direction.upper() == "LONG":
        stop_distance = entry_price - stop_price
    else:  # SHORT
        stop_distance = stop_price - entry_price
    
    if stop_distance <= 0:
        raise ValueError("Stop price must be valid (below entry for LONG, above for SHORT)")
    
    # Calculate risk amount
    risk_amount = equity * (risk_percentage / 100.0)
    
    # Calculate position size
    # Position Size = Risk Amount / Stop Distance
    position_size = risk_amount / stop_distance
    
    # Round to reasonable precision (avoid fractional shares for stocks)
    # For CFDs, this might be fine, but we'll round to 2 decimal places
    position_size = round(position_size, 2)
    
    return (position_size, risk_amount)


def calculate_position_size_with_profile(
    equity: float,
    entry_price: float,
    stop_price: float,
    profile: RiskProfile,
    confidence: float,
    direction: str = "LONG"
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Calculate position size using risk profile settings.
    
    Args:
        equity: Current account equity
        entry_price: Entry price
        stop_price: Stop-loss price
        profile: Risk profile
        confidence: Model confidence (0-1)
        direction: Trade direction
        
    Returns:
        Tuple of (position_size, risk_amount, reason)
        position_size: Position size, or None if trade should be skipped
        risk_amount: Risk amount, or None if trade should be skipped
        reason: Reason for result
    """
    # Get risk range for profile
    risk_min, risk_max = get_equity_risk_range(profile)
    
    # Adjust risk percentage based on confidence
    # Higher confidence = use higher end of range (up to max)
    # Lower confidence = use lower end of range
    if confidence >= 0.8:
        risk_percentage = risk_min + (risk_max - risk_min) * 0.8
    elif confidence >= 0.65:
        risk_percentage = risk_min + (risk_max - risk_min) * 0.5
    else:
        risk_percentage = risk_min
    
    try:
        position_size, risk_amount = calculate_position_size(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_percentage=risk_percentage,
            direction=direction
        )
        
        # Check minimum position size (e.g., must be at least $10 worth)
        min_position_value = 10.0
        position_value = position_size * entry_price
        
        if position_value < min_position_value:
            return (None, None, f"Position value ${position_value:.2f} below minimum ${min_position_value:.2f}")
        
        return (position_size, risk_amount, "OK")
        
    except ValueError as e:
        return (None, None, str(e))


def validate_position_size(
    position_size: float,
    entry_price: float,
    min_value: float = 10.0,
    max_value: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Validate position size meets requirements.
    
    Args:
        position_size: Position size to validate
        entry_price: Entry price
        min_value: Minimum position value in dollars
        max_value: Maximum position value in dollars (optional)
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if position_size <= 0:
        return (False, "Position size must be positive")
    
    position_value = position_size * entry_price
    
    if position_value < min_value:
        return (False, f"Position value ${position_value:.2f} below minimum ${min_value:.2f}")
    
    if max_value is not None and position_value > max_value:
        return (False, f"Position value ${position_value:.2f} exceeds maximum ${max_value:.2f}")
    
    return (True, "OK")

