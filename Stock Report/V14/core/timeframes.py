"""
Timeframe Configuration for V14
Defines allowed prediction timeframes for CFD and investment strategies.
"""

from typing import List, Dict, Optional
from datetime import timedelta


# CFD timeframes (short-term trading)
CFD_TIMEFRAMES: List[str] = ["1m", "5m", "10m", "15m", "1h", "4h"]

# Investment timeframes (long-term)
INVESTMENT_TIMEFRAMES: List[str] = ["1d", "1w", "1mo"]  # 1mo = 1 month (to distinguish from 1m = 1 minute)

# All allowed timeframes
ALL_TIMEFRAMES: List[str] = CFD_TIMEFRAMES + INVESTMENT_TIMEFRAMES

# Constant learning intervals (Function 3)
# Note: Uses "1mo" for month to distinguish from "1m" (minute)
CONSTANT_LEARNING_INTERVALS: List[str] = ["1m", "5m", "1h", "4h", "1d", "1w", "1mo"]

# Constant learning intervals (Function 3)
# Note: Uses "1mo" for month to distinguish from "1m" (minute)
CONSTANT_LEARNING_INTERVALS: List[str] = ["1m", "5m", "1h", "4h", "1d", "1w", "1mo"]


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Check if a timeframe is valid for V14.
    
    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1d")
        
    Returns:
        True if valid, False otherwise
    """
    return timeframe.lower() in [tf.lower() for tf in ALL_TIMEFRAMES]


def is_cfd_timeframe(timeframe: str) -> bool:
    """
    Check if a timeframe is a CFD timeframe.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        True if CFD timeframe, False otherwise
    """
    return timeframe.lower() in [tf.lower() for tf in CFD_TIMEFRAMES]


def is_investment_timeframe(timeframe: str) -> bool:
    """
    Check if a timeframe is an investment timeframe.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        True if investment timeframe, False otherwise
    """
    return timeframe.lower() in [tf.lower() for tf in INVESTMENT_TIMEFRAMES]


def get_timeframe_duration_seconds(timeframe: str) -> Optional[int]:
    """
    Get the duration of a timeframe in seconds.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Duration in seconds, or None if invalid
    """
    timeframe_lower = timeframe.lower()
    
    duration_map = {
        "1m": 60,           # 1 minute
        "5m": 300,          # 5 minutes
        "10m": 600,         # 10 minutes
        "15m": 900,         # 15 minutes
        "1h": 3600,         # 1 hour
        "4h": 14400,        # 4 hours
        "1d": 86400,        # 1 day
        "1w": 604800,       # 1 week
        "1mo": 2592000,     # 1 month (30 days)
    }
    
    return duration_map.get(timeframe_lower)


def get_timeframe_delta(timeframe: str) -> Optional[timedelta]:
    """
    Get the duration of a timeframe as a timedelta.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        timedelta object, or None if invalid
    """
    seconds = get_timeframe_duration_seconds(timeframe)
    if seconds is None:
        return None
    return timedelta(seconds=seconds)


def get_prediction_update_interval(timeframe: str) -> Optional[int]:
    """
    Get the recommended update interval for predictions in seconds.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Update interval in seconds, or None if invalid
    """
    timeframe_lower = timeframe.lower()
    
    # Update predictions at the start of each new timeframe period
    update_map = {
        "1m": 60,        # Every minute
        "5m": 300,       # Every 5 minutes
        "10m": 600,      # Every 10 minutes
        "15m": 900,      # Every 15 minutes
        "1h": 3600,      # Every hour
        "4h": 14400,     # Every 4 hours
        "1d": 86400,     # Every day
        "1w": 604800,    # Every week
        "1mo": 2592000,  # Every month
    }
    
    return update_map.get(timeframe_lower)

