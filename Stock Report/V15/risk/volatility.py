"""
Volatility Calculation Module
Implements Average True Range (ATR) calculation for dynamic risk management.
"""

from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json

from core.portable_paths import get_path


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for a given DataFrame.
    
    ATR measures market volatility by calculating the average of true ranges
    over a specified period. True Range is the maximum of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|
    
    Args:
        df: DataFrame with OHLC data (must have 'High', 'Low', 'Close' columns)
        period: Number of periods for ATR calculation (default: 14)
        
    Returns:
        ATR value as float
        
    Raises:
        ValueError: If DataFrame is invalid or insufficient data
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")
    
    if len(df) < period + 1:
        raise ValueError(f"Insufficient data: need at least {period + 1} rows, got {len(df)}")
    
    # Calculate True Range for each period
    true_ranges = []
    
    for i in range(1, len(df)):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        prev_close = df['Close'].iloc[i - 1]
        
        # True Range is the maximum of:
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # Calculate ATR using Wilder's smoothing method
    # Initial ATR is simple average of first 'period' true ranges
    if len(true_ranges) < period:
        # If we don't have enough data, use simple average
        atr = np.mean(true_ranges) if true_ranges else 0.0
    else:
        # Initial average
        atr = np.mean(true_ranges[:period])
        
        # Apply Wilder's smoothing: new_ATR = (old_ATR * (period - 1) + new_TR) / period
        for i in range(period, len(true_ranges)):
            atr = (atr * (period - 1) + true_ranges[i]) / period
    
    return float(atr)


def calculate_atr_multiple_periods(df: pd.DataFrame, periods: list = [14, 20, 50]) -> dict:
    """
    Calculate ATR for multiple periods.
    
    Args:
        df: DataFrame with OHLC data
        periods: List of periods to calculate ATR for
        
    Returns:
        Dictionary mapping period to ATR value
    """
    results = {}
    for period in periods:
        try:
            atr = calculate_atr(df, period)
            results[period] = atr
        except ValueError:
            # Skip if insufficient data for this period
            continue
    return results


def get_atr_cache_path(ticker: str, interval: str) -> Path:
    """Get path to ATR cache file."""
    cache_dir = get_path('cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"atr_{ticker}_{interval}.json"


def cache_atr(ticker: str, interval: str, atr_values: dict) -> None:
    """
    Cache ATR values for a ticker/interval combination.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval
        atr_values: Dictionary of ATR values (keyed by period)
    """
    try:
        cache_path = get_atr_cache_path(ticker, interval)
        with open(cache_path, 'w') as f:
            json.dump(atr_values, f)
    except Exception:
        # Silent failure on cache errors
        pass


def load_cached_atr(ticker: str, interval: str) -> Optional[dict]:
    """
    Load cached ATR values.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval
        
    Returns:
        Dictionary of ATR values, or None if not cached
    """
    try:
        cache_path = get_atr_cache_path(ticker, interval)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

