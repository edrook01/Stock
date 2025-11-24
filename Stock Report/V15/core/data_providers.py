"""
Data Provider Abstraction
Abstract interface for multiple data providers with fallback support.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path
import json

# Handle pandas import with error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Handle aiohttp import with error handling
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .portable_paths import get_data_path


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    async def fetch_prices(
        self,
        ticker: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price data for a ticker and interval.
        
        Args:
            ticker: Stock ticker symbol
            interval: Time interval (1m, 5m, 1d, etc.)
            
        Returns:
            DataFrame with OHLCV data, or None if fetch fails
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is available (API key, connectivity, etc.).
        
        Returns:
            True if provider can be used
        """
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        Get provider priority (lower number = higher priority).
        
        Returns:
            Priority number
        """
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider (primary, no API key required)."""
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        self.name = "yahoo_finance"
        self.priority = 1  # Highest priority
    
    async def fetch_prices(
        self,
        ticker: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch prices from Yahoo Finance."""
        # Import here to avoid circular dependency
        from .data_fetcher import fetch_prices as yf_fetch_prices
        return await yf_fetch_prices(ticker, interval)
    
    def get_name(self) -> str:
        return self.name
    
    def is_available(self) -> bool:
        # Yahoo Finance is always available (no API key needed)
        return True
    
    def get_priority(self) -> int:
        return self.priority


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider (requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage provider.
        
        Args:
            api_key: Alpha Vantage API key (optional, can be loaded from config)
        """
        self.name = "alpha_vantage"
        self.priority = 2
        self.api_key = api_key or self._load_api_key()
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from config."""
        try:
            config_file = get_data_path() / 'config_v15.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get("data_providers", {}).get("alpha_vantage_api_key")
        except Exception:
            pass
        return None
    
    async def fetch_prices(
        self,
        ticker: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch prices from Alpha Vantage."""
        if not self.api_key:
            return None
        
        # Map intervals to Alpha Vantage function
        interval_map = {
            "1d": "TIME_SERIES_DAILY",
            "1w": "TIME_SERIES_WEEKLY",
            "1m": "TIME_SERIES_INTRADAY",  # Requires interval parameter
        }
        
        function = interval_map.get(interval.lower(), "TIME_SERIES_DAILY")
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": ticker,
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        # Add interval for intraday
        if interval.lower() in ["1m", "5m", "15m", "30m", "1h"]:
            params["interval"] = interval.lower()
            params["function"] = "TIME_SERIES_INTRADAY"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API limit message
                        if "Note" in data or "Error Message" in data:
                            return None
                        
                        # Parse Alpha Vantage response
                        time_series_key = None
                        for key in data.keys():
                            if "Time Series" in key:
                                time_series_key = key
                                break
                        
                        if not time_series_key:
                            return None
                        
                        time_series = data[time_series_key]
                        
                        # Convert to DataFrame
                        records = []
                        for timestamp, values in time_series.items():
                            records.append({
                                "Date": pd.to_datetime(timestamp),
                                "Open": float(values["1. open"]),
                                "High": float(values["2. high"]),
                                "Low": float(values["3. low"]),
                                "Close": float(values["4. close"]),
                                "Volume": int(values["5. volume"])
                            })
                        
                        df = pd.DataFrame(records)
                        df.set_index("Date", inplace=True)
                        df.sort_index(inplace=True)
                        
                        return df
        
        except Exception:
            return None
    
    def get_name(self) -> str:
        return self.name
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def get_priority(self) -> int:
        return self.priority


class PolygonProvider(DataProvider):
    """Polygon.io data provider (requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon provider.
        
        Args:
            api_key: Polygon API key (optional, can be loaded from config)
        """
        self.name = "polygon"
        self.priority = 3
        self.api_key = api_key or self._load_api_key()
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from config."""
        try:
            config_file = get_data_path() / 'config_v15.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get("data_providers", {}).get("polygon_api_key")
        except Exception:
            pass
        return None
    
    async def fetch_prices(
        self,
        ticker: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch prices from Polygon.io."""
        if not self.api_key:
            return None
        
        # Map intervals to Polygon multiplier/timespan
        interval_map = {
            "1m": ("1", "minute"),
            "5m": ("5", "minute"),
            "15m": ("15", "minute"),
            "1h": ("1", "hour"),
            "1d": ("1", "day"),
            "1w": ("1", "week"),
        }
        
        multiplier, timespan = interval_map.get(interval.lower(), ("1", "day"))
        
        # Get date range (last 30 days for intraday, 1 year for daily)
        from datetime import datetime, timedelta
        if interval.lower() in ["1m", "5m", "15m", "1h"]:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
        else:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=365)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") != "OK" or "results" not in data:
                            return None
                        
                        results = data["results"]
                        
                        # Convert to DataFrame
                        records = []
                        for bar in results:
                            timestamp_ms = bar["t"]
                            timestamp = pd.to_datetime(timestamp_ms, unit='ms')
                            records.append({
                                "Date": timestamp,
                                "Open": bar["o"],
                                "High": bar["h"],
                                "Low": bar["l"],
                                "Close": bar["c"],
                                "Volume": bar["v"]
                            })
                        
                        df = pd.DataFrame(records)
                        df.set_index("Date", inplace=True)
                        df.sort_index(inplace=True)
                        
                        return df
        
        except Exception:
            return None
    
    def get_name(self) -> str:
        return self.name
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def get_priority(self) -> int:
        return self.priority


def get_available_providers() -> List[DataProvider]:
    """
    Get list of available data providers sorted by priority.
    
    Returns:
        List of available DataProvider instances
    """
    providers = [
        YahooFinanceProvider(),
        AlphaVantageProvider(),
        PolygonProvider()
    ]
    
    # Filter to only available providers and sort by priority
    available = [p for p in providers if p.is_available()]
    available.sort(key=lambda p: p.get_priority())
    
    return available


async def fetch_from_multiple_providers(
    ticker: str,
    interval: str,
    providers: Optional[List[DataProvider]] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch data from multiple providers, using first successful response.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval
        providers: Optional list of providers (defaults to all available)
        
    Returns:
        DataFrame with price data, or None if all providers fail
    """
    if providers is None:
        providers = get_available_providers()
    
    if not providers:
        return None
    
    # Try providers in parallel, return first successful
    tasks = [provider.fetch_prices(ticker, interval) for provider in providers]
    
    # Use asyncio.wait to get first completed successful result
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Check first completed task
    for task in done:
        try:
            result = await task
            if result is not None and not result.empty:
                # Cancel remaining tasks
                for p in pending:
                    p.cancel()
                return result
        except Exception:
            continue
    
    # If first didn't work, wait for all and try next
    if pending:
        for task in pending:
            try:
                result = await task
                if result is not None and not result.empty:
                    return result
            except Exception:
                continue
    
    return None

