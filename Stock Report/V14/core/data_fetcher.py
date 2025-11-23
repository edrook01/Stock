"""
Async Data Fetcher with SQLite Cache
Provides portable, async price fetcher for Yahoo Finance with caching.

Uses ONLY relative paths via portable_paths.get_path().
Cache stored as SQLite DB: memory/cache.db
Daily data cached for 24 hours max.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Optional dependencies - handle gracefully if not installed
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Handle both relative and absolute imports for portability
try:
    from .portable_paths import get_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path


# Cache duration: 24 hours
CACHE_DURATION_HOURS = 24


def _get_cache_db_path() -> Path:
    """Get the path to the cache database."""
    memory_dir = get_path('memory')
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / 'cache.db'


def _init_cache_db() -> None:
    """Initialize the cache database if it doesn't exist."""
    db_path = _get_cache_db_path()
    
    if db_path.exists():
        return
    
    # Create database and table
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            cache_key TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            interval TEXT NOT NULL,
            data_json TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    # Create index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_interval 
        ON price_cache(ticker, interval)
    """)
    
    conn.commit()
    conn.close()


def _get_cache_key(ticker: str, interval: str) -> str:
    """Generate a cache key from ticker and interval."""
    return f"{ticker}_{interval}".upper()


def _is_cache_valid(timestamp_str: str) -> bool:
    """
    Check if cached data is still valid (within 24 hours).
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        True if cache is valid, False if expired
    """
    try:
        cached_time = datetime.fromisoformat(timestamp_str)
        age = datetime.now() - cached_time
        return age < timedelta(hours=CACHE_DURATION_HOURS)
    except (ValueError, TypeError):
        return False


async def _fetch_from_yahoo_finance(ticker: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch price data from Yahoo Finance API using aiohttp.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        interval: Time interval ('1d', '1h', '5m', etc.)
        
    Returns:
        pandas DataFrame with OHLCV data, or None if fetch fails
    """
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp is required for fetching price data. Install with: pip install aiohttp")
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for fetching price data. Install with: pip install pandas")
    # Map intervals to Yahoo Finance API format
    interval_map = {
        "1m": "1m", "5m": "5m", "10m": "5m", "15m": "15m",
        "30m": "30m", "1h": "1h", "4h": "1h", "1d": "1d",
        "1mo": "1mo", "1month": "1mo", "3mo": "3mo", "3month": "3mo",
        "1y": "1y", "1year": "1y", "1yr": "1y"
    }
    
    yf_interval = interval_map.get(interval.lower(), "1d")
    
    # Map intervals to range (time period) for Yahoo Finance API
    range_map = {
        "1m": "5d", "5m": "5d", "15m": "5d", "30m": "1mo",
        "1h": "1mo", "4h": "3mo", "1d": "1y",
        "1mo": "2y", "1month": "2y", "3mo": "2y", "3month": "2y",
        "1y": "5y", "1year": "5y", "1yr": "5y"
    }
    
    range_value = range_map.get(interval.lower(), "1y")
    
    # Build Yahoo Finance API URL
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "interval": yf_interval,
        "range": range_value
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                # Parse Yahoo Finance API response
                result = data.get("chart", {}).get("result", [])
                if not result:
                    return None
                
                chart_data = result[0]
                
                # Extract timestamps and indicators
                timestamps = chart_data.get("timestamp", [])
                indicators = chart_data.get("indicators", {}).get("quote", [])
                
                if not timestamps or not indicators:
                    return None
                
                quote = indicators[0]
                
                # Extract OHLCV data
                opens = quote.get("open", [])
                highs = quote.get("high", [])
                lows = quote.get("low", [])
                closes = quote.get("close", [])
                volumes = quote.get("volume", [])
                
                # Build DataFrame
                df_data = {
                    "Open": opens,
                    "High": highs,
                    "Low": lows,
                    "Close": closes,
                    "Volume": volumes
                }
                
                # Create DataFrame
                df = pd.DataFrame(df_data)
                
                # Convert timestamps to datetime index
                if timestamps:
                    df.index = pd.to_datetime(timestamps, unit='s')
                    df.index.name = 'Date'
                
                # Remove rows with all NaN values
                df = df.dropna(how='all')
                
                # Return None if DataFrame is empty
                if df.empty:
                    return None
                
                return df
                
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as e:
        # Silent failure - return None on error
        return None


def _save_to_cache(ticker: str, interval: str, df: pd.DataFrame) -> None:
    """
    Save DataFrame to SQLite cache as JSON blob.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval
        df: pandas DataFrame to cache
    """
    if df is None or df.empty:
        return
    
    try:
        cache_key = _get_cache_key(ticker, interval)
        timestamp = datetime.now().isoformat()
        
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Reset index to include Date as a column for JSON serialization
        # If index is named 'Date', reset_index() will preserve that name
        if df_copy.index.name == 'Date' or isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
            # Ensure the date column is named 'Date' (may be named 'index' if no name)
            if 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'Date'}, inplace=True)
        
        # Convert DataFrame to JSON
        # Use records orientation for portability
        data_json = df_copy.to_json(orient='records', date_format='iso')
        
        db_path = _get_cache_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Insert or replace cache entry
        cursor.execute("""
            INSERT OR REPLACE INTO price_cache 
            (cache_key, ticker, interval, data_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_key, ticker.upper(), interval, data_json, timestamp))
        
        conn.commit()
        conn.close()
        
    except Exception:
        # Silent failure on cache save errors
        pass


def _load_from_cache(ticker: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from SQLite cache if valid.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval
        
    Returns:
        pandas DataFrame if cache is valid, None otherwise
    """
    try:
        cache_key = _get_cache_key(ticker, interval)
        db_path = _get_cache_db_path()
        
        if not db_path.exists():
            return None
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data_json, timestamp
            FROM price_cache
            WHERE cache_key = ?
        """, (cache_key,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return None
        
        data_json, timestamp_str = result
        
        # Check if cache is still valid
        if not _is_cache_valid(timestamp_str):
            return None
        
        # Parse JSON and reconstruct DataFrame
        data_records = json.loads(data_json)
        df = pd.DataFrame(data_records)
        
        # Convert Date column to datetime index if present
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.index.name = 'Date'
        
        if df.empty:
            return None
        
        return df
        
    except Exception:
        # Return None on any error (invalid cache, parsing error, etc.)
        return None


async def fetch_prices(ticker: str, interval: str, use_multiple_providers: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch price data with caching and multiple provider support.
    
    Flow:
    1) Try cache first
    2) Fetch remote if expired/not found (try multiple providers with retry)
    3) Save new data to cache
    4) Return pandas DataFrame
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        interval: Time interval ('1d', '1h', '5m', etc.)
        use_multiple_providers: Whether to try multiple providers with fallback
        
    Returns:
        pandas DataFrame with OHLCV data, or None if fetch fails
        
    Example:
        >>> df = await fetch_prices('AAPL', '1d')
        >>> print(df.head())
    """
    # Initialize cache database if needed
    _init_cache_db()
    
    # Try cache first
    cached_df = _load_from_cache(ticker, interval)
    if cached_df is not None:
        return cached_df
    
    # Fetch from remote if cache miss or expired
    if use_multiple_providers:
        # Try multiple providers with retry
        try:
            from .data_providers import fetch_from_multiple_providers
            from .retry_handler import retry_with_backoff
            
            df = await retry_with_backoff(
                fetch_from_multiple_providers,
                max_retries=2,
                base_delay=1.0,
                max_delay=10.0,
                exceptions=(Exception,),
                ticker=ticker,
                interval=interval
            )
        except Exception:
            # Fallback to Yahoo Finance only
            df = await _fetch_from_yahoo_finance(ticker, interval)
    else:
        # Use Yahoo Finance only (original behavior)
        df = await _fetch_from_yahoo_finance(ticker, interval)
    
    # Save to cache if fetch was successful
    if df is not None and not df.empty:
        _save_to_cache(ticker, interval, df)
    
    return df

