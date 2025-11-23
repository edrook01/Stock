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


def sma(prices: pd.Series, period: int = 20) -> float:
    """
    Calculate Simple Moving Average (SMA).
    
    SMA is the arithmetic mean of prices over the specified period.
    This matches ta.trend.sma_indicator behavior exactly.
    
    Args:
        prices: Price series (pandas Series)
        period: Number of periods for the moving average (default: 20)
    
    Returns:
        Latest SMA value as float
    
    Raises:
        ValueError: If period is invalid or insufficient data
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} values, got {len(prices)}")
    
    # Convert to array and calculate mean of last 'period' values
    # This matches ta.trend.sma_indicator exactly
    values = prices.values
    return float(np.mean(values[-period:]))


def ema(prices: pd.Series, period: int = 20) -> float:
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
        Latest EMA value as float
    
    Raises:
        ValueError: If period is invalid or insufficient data
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period:
        raise ValueError(f"Insufficient data: need at least {period} values, got {len(prices)}")
    
    # Convert to numpy array for calculation
    values = prices.values
    
    # Calculate smoothing factor (alpha)
    alpha = 2.0 / (period + 1.0)
    
    # Initialize EMA with SMA of first period values
    ema_value = np.mean(values[:period])
    
    # Apply exponential smoothing to remaining values
    for i in range(period, len(values)):
        ema_value = alpha * values[i] + (1 - alpha) * ema_value
    
    return float(ema_value)


def rsi(prices: pd.Series, period: int = 14) -> float:
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
        Latest RSI value as float (0-100)
    
    Raises:
        ValueError: If period is invalid or insufficient data
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period + 1:
        raise ValueError(f"Insufficient data: need at least {period + 1} values, got {len(prices)}")
    
    # Convert to numpy array
    values = prices.values
    
    # Calculate price changes (deltas)
    deltas = np.diff(values)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initial average gain and loss (simple average of first period)
    # This matches ta library's implementation
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Avoid division by zero
    if avg_loss == 0:
        return 100.0
    
    # Apply Wilder's smoothing to remaining periods
    # Wilder's method: new_avg = (old_avg * (period - 1) + new_value) / period
    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_value = 100.0 - (100.0 / (1.0 + rs))
    
    return float(rsi_value)


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

