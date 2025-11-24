"""
Ticker Validation System
Validates ticker symbols using multiple data sources with caching and batch support.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

from .portable_paths import get_path, get_data_path


class TickerValidator:
    """Validates ticker symbols using multiple data sources."""
    
    def __init__(self):
        """Initialize ticker validator."""
        self.cache_duration = timedelta(hours=24)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
    
    async def validate_ticker(
        self,
        ticker: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a single ticker symbol.
        
        Args:
            ticker: Ticker symbol to validate
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with validation result:
            {
                "valid": bool,
                "name": str,
                "exchange": str,
                "type": str,
                "status": str,
                "source": str,
                "cached": bool
            }
        """
        ticker = ticker.upper().strip()
        
        # Check cache first
        if use_cache and ticker in self.cache:
            cached_result = self.cache[ticker]
            cached_time = datetime.fromisoformat(cached_result.get("timestamp", ""))
            if datetime.now() - cached_time < self.cache_duration:
                cached_result["cached"] = True
                return cached_result
        
        # Try Yahoo Finance first (most reliable, no API key needed)
        result = await self._validate_via_yahoo_finance(ticker)
        
        if result["valid"]:
            result["source"] = "yahoo_finance"
            result["cached"] = False
            result["timestamp"] = datetime.now().isoformat()
            self.cache[ticker] = result
            self._save_cache()
            return result
        
        # If Yahoo Finance fails, try other sources (if configured)
        # For now, return invalid result
        return {
            "valid": False,
            "name": "",
            "exchange": "",
            "type": "",
            "status": "not_found",
            "source": "yahoo_finance",
            "cached": False,
            "timestamp": datetime.now().isoformat()
        }
    
    async def batch_validate_tickers(
        self,
        tickers: List[str],
        max_concurrent: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols to validate
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping ticker -> validation result
        """
        # Remove duplicates and normalize
        unique_tickers = list(set(t.upper().strip() for t in tickers))
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_limit(ticker: str):
            async with semaphore:
                return ticker, await self.validate_ticker(ticker)
        
        # Run validations in parallel
        tasks = [validate_with_limit(ticker) for ticker in unique_tickers]
        results = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        return {ticker: result for ticker, result in results}
    
    async def _validate_via_yahoo_finance(self, ticker: str) -> Dict[str, Any]:
        """
        Validate ticker using Yahoo Finance API.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Validation result dictionary
        """
        url = f"https://query1.finance.yahoo.com/v8/finance/quoteSummary/{ticker}"
        params = {
            "modules": "assetProfile,summaryProfile"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse response
                        result = data.get("quoteSummary", {}).get("result", [])
                        if result and len(result) > 0:
                            quote_data = result[0]
                            summary = quote_data.get("summaryProfile", {})
                            asset = quote_data.get("assetProfile", {})
                            
                            return {
                                "valid": True,
                                "name": summary.get("longName", ticker),
                                "exchange": summary.get("exchange", ""),
                                "type": asset.get("sector", ""),
                                "status": "active",
                                "currency": summary.get("currency", "USD"),
                                "market_cap": summary.get("marketCap", 0)
                            }
                    
                    # If 404 or other error, ticker likely invalid
                    return {
                        "valid": False,
                        "name": "",
                        "exchange": "",
                        "type": "",
                        "status": "not_found"
                    }
        
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
            # Network error or parsing error - return unknown status
            return {
                "valid": False,
                "name": "",
                "exchange": "",
                "type": "",
                "status": "error"
            }
    
    async def fetch_ticker_metadata_batch(
        self,
        tickers: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch metadata for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker -> metadata
        """
        # Use batch validation which includes metadata
        validation_results = await self.batch_validate_tickers(tickers)
        
        # Extract metadata from validation results
        metadata = {}
        for ticker, result in validation_results.items():
            if result["valid"]:
                metadata[ticker] = {
                    "name": result.get("name", ""),
                    "exchange": result.get("exchange", ""),
                    "type": result.get("type", ""),
                    "currency": result.get("currency", "USD"),
                    "market_cap": result.get("market_cap", 0),
                    "status": result.get("status", "active")
                }
        
        return metadata
    
    def _load_cache(self) -> None:
        """Load validation cache from disk."""
        try:
            cache_file = get_path('memory') / 'ticker_validation_cache.json'
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception:
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save validation cache to disk."""
        try:
            cache_file = get_path('memory') / 'ticker_validation_cache.json'
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self.cache = {}
        self._save_cache()


# Global validator instance
_validator_instance: Optional[TickerValidator] = None


def get_ticker_validator() -> TickerValidator:
    """Get global ticker validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TickerValidator()
    return _validator_instance


# Convenience functions
async def validate_ticker(ticker: str) -> Dict[str, Any]:
    """Validate a single ticker (convenience function)."""
    validator = get_ticker_validator()
    return await validator.validate_ticker(ticker)


async def batch_validate_tickers(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Validate multiple tickers (convenience function)."""
    validator = get_ticker_validator()
    return await validator.batch_validate_tickers(tickers)

