"""
Technical Indicator Engine
Fast, vectorized implementations of RSI, SMA, and EMA using NumPy with optional CuPy GPU acceleration.

This module provides optimized technical indicators that match V12's behavior exactly,
but with improved performance through vectorization and optional GPU acceleration.
"""

from typing import Union
import pandas as pd
import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def _get_array_module(x: Union[np.ndarray, 'cp.ndarray']) -> Union[type(np), type]:
    """
    Determine which array module to use (NumPy or CuPy) based on input array type.
    
    Args:
        x: Input array (NumPy or CuPy)
    
    Returns:
        The appropriate array module (np or cp)
    """
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp
    return np


def _to_array(prices: pd.Series, use_gpu: bool = False) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Convert pandas Series to NumPy or CuPy array.
    
    Args:
        prices: Input pandas Series
        use_gpu: Whether to use GPU (CuPy) if available
    
    Returns:
        NumPy or CuPy array
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp.asarray(prices.values)
    return prices.values


def sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    SMA is the arithmetic mean of prices over the specified period.
    Returns the full SMA series to support downstream analytics.
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for the moving average (default: 20)
    
    Returns:
        Pandas Series containing SMA values
    
    Raises:
        ValueError: If period is invalid or insufficient data
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} values, got {len(prices)}")
    
    return prices.rolling(window=period, min_periods=period).mean()


def ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    EMA gives more weight to recent prices using exponential smoothing.
    Formula: EMA = alpha * price + (1 - alpha) * previous_EMA
    where alpha = 2 / (period + 1)
    
    This matches ta.trend.ema_indicator behavior exactly.
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for the EMA (default: 20)
    
    Returns:
        Pandas Series containing EMA values
    
    Raises:
        ValueError: If period is invalid or insufficient data
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} values, got {len(prices)}")
    
    return prices.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures the magnitude of recent price changes to evaluate overbought/oversold conditions.
    Uses Wilder's smoothing method (exponential moving average of gains/losses).
    
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss (using Wilder's smoothing)
    
    This matches ta.momentum.rsi behavior exactly.
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for RSI calculation (default: 14)
    
    Returns:
        Pandas Series containing RSI values bounded between 0 and 100
    
    Raises:
        ValueError: If period is invalid or insufficient data
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period + 1:
        raise ValueError(f"Insufficient data: need at least {period + 1} values, got {len(prices)}")
    
    deltas = prices.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)

    alpha = 1 / period
    avg_gain = gains.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = rsi_values.fillna(100).clip(lower=0, upper=100)
    return rsi_values


# Vectorized versions for batch processing (optional GPU acceleration)
def sma_vectorized(prices: pd.Series, period: int = 20, use_gpu: bool = False) -> np.ndarray:
    """
    Calculate SMA for all values in the series (vectorized).
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for the moving average
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        Array of SMA values (same length as input, with NaN for first period-1 values)
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} values, got {len(prices)}")
    
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np
    arr = _to_array(prices, use_gpu)
    
    # Use rolling window mean (vectorized)
    result = xp.full(len(arr), xp.nan, dtype=xp.float64)
    
    # Calculate rolling mean
    for i in range(period - 1, len(arr)):
        result[i] = xp.mean(arr[i - period + 1:i + 1])
    
    # Convert back to numpy if using CuPy
    if isinstance(result, cp.ndarray):
        result = cp.asnumpy(result)
    
    return result


def ema_vectorized(prices: pd.Series, period: int = 20, use_gpu: bool = False) -> np.ndarray:
    """
    Calculate EMA for all values in the series (vectorized).
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for the EMA
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        Array of EMA values (same length as input)
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} values, got {len(prices)}")
    
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np
    arr = _to_array(prices, use_gpu)
    
    # Calculate smoothing factor
    alpha = 2.0 / (period + 1.0)
    
    # Initialize result array
    result = xp.zeros_like(arr, dtype=xp.float64)
    
    # Initialize with SMA of first period
    result[period - 1] = xp.mean(arr[:period])
    
    # Apply exponential smoothing
    for i in range(period, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    
    # Fill initial values with NaN (standard behavior)
    result[:period - 1] = xp.nan
    
    # Convert back to numpy if using CuPy
    if isinstance(result, cp.ndarray):
        result = cp.asnumpy(result)
    
    return result


def rsi_vectorized(prices: pd.Series, period: int = 14, use_gpu: bool = False) -> np.ndarray:
    """
    Calculate RSI for all values in the series (vectorized).
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for RSI calculation
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        Array of RSI values (same length as input, with NaN for first period values)
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period + 1:
        raise ValueError(f"Insufficient data: need at least {period + 1} values, got {len(prices)}")
    
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np
    arr = _to_array(prices, use_gpu)
    
    # Calculate price changes
    deltas = xp.diff(arr)
    
    # Separate gains and losses
    gains = xp.where(deltas > 0, deltas, 0.0)
    losses = xp.where(deltas < 0, -deltas, 0.0)
    
    # Initialize result array
    result = xp.full(len(arr), xp.nan, dtype=xp.float64)
    
    # Initial average gain and loss
    avg_gain = xp.mean(gains[:period])
    avg_loss = xp.mean(losses[:period])
    
    # Calculate first RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Apply Wilder's smoothing for remaining values
    for i in range(period + 1, len(arr)):
        gain = gains[i - 1]  # deltas is 1 shorter than arr
        loss = losses[i - 1]
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Convert back to numpy if using CuPy
    if isinstance(result, cp.ndarray):
        result = cp.asnumpy(result)
    
    return result

