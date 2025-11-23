"""
Sentiment Override Logic
Blocks trades or adjusts strategy based on sentiment and news events.
"""

from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta

# Handle both relative and absolute imports for portability
try:
    from .news_monitor import get_news_monitor
    from .analyzer import get_sentiment_analyzer
except ImportError:
    # Fallback for direct execution
    from sentiment.news_monitor import get_news_monitor
    from sentiment.analyzer import get_sentiment_analyzer


class SentimentOverride:
    """Implements sentiment-based override logic."""
    
    def __init__(self, override_threshold: float = 0.7):
        """
        Initialize sentiment override.
        
        Args:
            override_threshold: Sentiment threshold for override (0-1)
        """
        self.override_threshold = override_threshold
        self.news_monitor = get_news_monitor()
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.blocked_tickers: Dict[str, datetime] = {}
        self.protective_mode_active = False
    
    def should_block_trade(
        self,
        ticker: str,
        sentiment_score: Optional[float] = None,
        check_news: bool = True
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be blocked based on sentiment/news.
        
        Args:
            ticker: Stock ticker symbol
            sentiment_score: Sentiment score (-1 to +1), optional
            check_news: Whether to check news calendar
            
        Returns:
            Tuple of (should_block, reason)
        """
        # Check if ticker is currently blocked
        if ticker.upper() in self.blocked_tickers:
            block_until = self.blocked_tickers[ticker.upper()]
            if datetime.now() < block_until:
                return (True, f"Ticker blocked until {block_until.isoformat()}")
            else:
                # Block expired, remove
                del self.blocked_tickers[ticker.upper()]
        
        # Check for scheduled events
        if check_news:
            events = self.news_monitor.check_scheduled_events(ticker, hours_ahead=2)
            if events:
                event_types = [e.get("type") for e in events]
                if "earnings" in event_types:
                    return (True, "Earnings event scheduled within 2 hours")
                elif any(et in ["fda", "merger", "major_announcement"] for et in event_types):
                    return (True, f"Major event scheduled: {', '.join(event_types)}")
        
        # Check sentiment if provided
        if sentiment_score is not None:
            if abs(sentiment_score) > self.override_threshold:
                if sentiment_score < -self.override_threshold:
                    return (True, f"Very negative sentiment: {sentiment_score:.2f}")
                # Very positive sentiment might also block if it's a spike (potential overbought)
        
        # Check protective mode
        if self.protective_mode_active:
            major_events = self.news_monitor.get_major_events()
            if major_events:
                return (True, "Protective mode active due to major market events")
        
        return (False, "OK")
    
    def check_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Check sentiment for a ticker (wrapper for should_block_trade for backward compatibility).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with 'blocked' (bool) and 'reason' (str) keys
        """
        should_block, reason = self.should_block_trade(ticker)
        return {"blocked": should_block, "reason": reason}
    
    def block_ticker(self, ticker: str, until: Optional[datetime] = None) -> None:
        """
        Block a ticker from trading.
        
        Args:
            ticker: Stock ticker symbol
            until: Block until this time (defaults to 24 hours from now)
        """
        if until is None:
            until = datetime.now() + timedelta(hours=24)
        
        self.blocked_tickers[ticker.upper()] = until
    
    def enable_protective_mode(self) -> None:
        """Enable protective mode (blocks all trades during major events)."""
        self.protective_mode_active = True
    
    def disable_protective_mode(self) -> None:
        """Disable protective mode."""
        self.protective_mode_active = False
    
    def adjust_confidence(
        self,
        base_confidence: float,
        sentiment_score: float,
        ticker: str
    ) -> float:
        """
        Adjust model confidence based on sentiment.
        
        Args:
            base_confidence: Base confidence from model (0-1)
            sentiment_score: Sentiment score (-1 to +1)
            ticker: Stock ticker symbol
            
        Returns:
            Adjusted confidence (0-1)
        """
        # If sentiment contradicts technical signal, reduce confidence
        # Positive sentiment + positive prediction = boost confidence
        # Negative sentiment + positive prediction = reduce confidence
        
        # Simple adjustment: if sentiment is very negative, reduce confidence
        if sentiment_score < -0.5:
            adjusted = base_confidence * 0.7  # Reduce by 30%
        elif sentiment_score < -0.2:
            adjusted = base_confidence * 0.85  # Reduce by 15%
        elif sentiment_score > 0.5:
            adjusted = min(1.0, base_confidence * 1.1)  # Boost by 10%
        else:
            adjusted = base_confidence
        
        return max(0.0, min(1.0, adjusted))
    
    def get_override_status(self) -> Dict:
        """
        Get current override status.
        
        Returns:
            Dictionary with override status
        """
        return {
            "protective_mode": self.protective_mode_active,
            "blocked_tickers": {
                ticker: until.isoformat()
                for ticker, until in self.blocked_tickers.items()
            },
            "override_threshold": self.override_threshold
        }


# Global sentiment override instance
_sentiment_override: Optional[SentimentOverride] = None


def get_sentiment_override() -> SentimentOverride:
    """Get global sentiment override instance."""
    global _sentiment_override
    if _sentiment_override is None:
        _sentiment_override = SentimentOverride()
    return _sentiment_override

