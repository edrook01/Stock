"""
Trading212 Ticker Universe Loader.

Provides a single source of truth for the complete Trading212 market universe
as stored in ``data/tickers.txt``. The loader normalizes the symbols, removes
duplicates, and returns a stable ordering so all subsystems share the exact
same list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

try:
    from .portable_paths import get_data_path
except (ImportError, ValueError):
    from core.portable_paths import get_data_path  # type: ignore


TICKER_FILE_NAME = "tickers.txt"
_FALLBACK_TICKERS = ["AAPL", "MSFT", "TSLA"]


def _normalize_tickers(raw_lines: Iterable[str]) -> List[str]:
    """Normalize raw ticker lines."""
    normalized: List[str] = []
    seen = set()
    for line in raw_lines:
        ticker = line.strip().upper()
        if not ticker or ticker.startswith("#"):
            continue
        if ticker in seen:
            continue
        normalized.append(ticker)
        seen.add(ticker)
    return normalized


@lru_cache(maxsize=1)
def _load_trading212_tickers() -> tuple[str, ...]:
    """Load and cache the Trading212 ticker universe."""
    try:
        data_path = get_data_path()
        ticker_file: Path = data_path / TICKER_FILE_NAME
        if not ticker_file.exists():
            return tuple(_FALLBACK_TICKERS)
        with open(ticker_file, "r", encoding="utf-8") as fh:
            tickers = _normalize_tickers(fh.readlines())
        return tuple(tickers or _FALLBACK_TICKERS)
    except Exception:
        return tuple(_FALLBACK_TICKERS)


def get_trading212_tickers() -> List[str]:
    """Return a copy of the Trading212 ticker universe."""
    return list(_load_trading212_tickers())


def reload_trading212_tickers() -> List[str]:
    """Clear cache and reload Trading212 ticker universe from disk."""
    _load_trading212_tickers.cache_clear()
    return get_trading212_tickers()


__all__ = ["get_trading212_tickers", "reload_trading212_tickers"]


