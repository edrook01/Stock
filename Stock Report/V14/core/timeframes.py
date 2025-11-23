"""
Timeframe Configuration for V14
Defines allowed prediction timeframes for CFD and investment strategies.
"""

from typing import List, Dict, Optional
from datetime import timedelta


# CFD timeframes (short-term trading)
CFD_TIMEFRAMES: List[str] = ["1m", "5m", "10m", "15m", "1h"]

# Investment timeframes (long-term)
INVESTMENT_TIMEFRAMES: List[str] = ["1d", "1w"]

# All allowed timeframes
ALL_TIMEFRAMES: List[str] = CFD_TIMEFRAMES + INVESTMENT_TIMEFRAMES


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
        "1m": 60,
        "5m": 300,
        "10m": 600,
        "15m": 900,
        "1h": 3600,
        "1d": 86400,
        "1w": 604800,
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
        "1m": 60,      # Every minute
        "5m": 300,     # Every 5 minutes
        "10m": 600,    # Every 10 minutes
        "15m": 900,    # Every 15 minutes
        "1h": 3600,    # Every hour
        "1d": 86400,   # Every day
        "1w": 604800,  # Every week
    }
    
    return update_map.get(timeframe_lower)

