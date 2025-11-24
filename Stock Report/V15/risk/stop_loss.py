"""
Dynamic Stop-Loss Calculation
Calculates stop-loss distances using ATR multipliers based on risk profile and confidence.
"""

from typing import Optional, Tuple
import pandas as pd

from .volatility import calculate_atr
from .profiles import RiskProfile, get_atr_multiplier_range, get_profile_config


def calculate_stop_loss_distance(
    df: pd.DataFrame,
    profile: RiskProfile,
    confidence: float,
    asset_risk_category: str = "medium"
) -> Tuple[float, float]:
    """
    Calculate stop-loss distance using ATR-based method.
    
    Args:
        df: DataFrame with OHLC data
        profile: Risk profile (Low, Medium, High)
        confidence: Model confidence score (0-1)
        asset_risk_category: Asset risk category ("low", "medium", "high")
        
    Returns:
        Tuple of (stop_distance, atr_value)
        stop_distance: Distance from entry price for stop-loss
        atr_value: ATR value used in calculation
    """
    # Calculate ATR
    try:
        atr = calculate_atr(df, period=14)
    except ValueError:
        # Fallback to simple percentage if ATR calculation fails
        current_price = df['Close'].iloc[-1]
        return (current_price * 0.02, 0.0)  # 2% fallback
    
    # Get ATR multiplier range for profile
    atr_min, atr_max = get_atr_multiplier_range(profile)
    
    # Adjust multiplier based on confidence
    # High confidence: slightly tighter stops (allow trade to breathe)
    # Low confidence: wider stops or skip trade
    if confidence >= 0.8:
        # High confidence - use lower end of range (tighter stops)
        atr_multiplier = atr_min + (atr_max - atr_min) * 0.3
    elif confidence >= 0.65:
        # Medium confidence - use middle of range
        atr_multiplier = (atr_min + atr_max) / 2.0
    else:
        # Low confidence - use upper end of range (wider stops)
        atr_multiplier = atr_min + (atr_max - atr_min) * 0.8
    
    # Adjust for asset risk category
    if asset_risk_category.lower() in ["high", "volatile"]:
        # High-risk assets need wider stops
        atr_multiplier *= 1.2
    elif asset_risk_category.lower() in ["low", "stable"]:
        # Low-risk assets can use tighter stops
        atr_multiplier *= 0.9
    
    # Calculate stop distance
    stop_distance = atr * atr_multiplier
    
    return (stop_distance, atr)


def calculate_stop_loss_price(
    entry_price: float,
    direction: str,
    stop_distance: float
) -> float:
    """
    Calculate stop-loss price from entry price and distance.
    
    Args:
        entry_price: Entry price of the trade
        direction: Trade direction ("LONG" or "SHORT")
        stop_distance: Stop distance (absolute price difference)
        
    Returns:
        Stop-loss price
    """
    direction_upper = direction.upper()
    
    if direction_upper == "LONG":
        # Long trade: stop below entry
        return entry_price - stop_distance
    elif direction_upper == "SHORT":
        # Short trade: stop above entry
        return entry_price + stop_distance
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'")


def should_skip_trade(
    profile: RiskProfile,
    confidence: float,
    asset_risk_category: str
) -> bool:
    """
    Determine if a trade should be skipped based on risk profile and confidence.
    
    Args:
        profile: Risk profile
        confidence: Model confidence score (0-1)
        asset_risk_category: Asset risk category
        
    Returns:
        True if trade should be skipped, False otherwise
    """
    config = get_profile_config(profile)
    confidence_threshold = config["confidence_threshold"]
    
    # Skip if confidence is too low
    if confidence < confidence_threshold:
        return True
    
    # Skip if asset is not allowed for this profile
    from .profiles import is_asset_allowed
    if not is_asset_allowed(profile, asset_risk_category):
        return True
    
    return False

