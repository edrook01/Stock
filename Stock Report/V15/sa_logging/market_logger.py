"""
Market Data Logger
Logs market events, sentiment changes, news impact, and market conditions.
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import json
import csv

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path


class MarketLogger:
    """Comprehensive market data logger."""
    
    def __init__(self):
        """Initialize market logger."""
        self.history_dir = get_path('history')
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.history_dir / 'market_data.csv'
        self.json_file = self.history_dir / 'market_data.json'
        
        self._initialize_csv()
        self._market_logs: List[Dict] = []
        self._load_market_logs()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'EventType', 'Ticker', 'MarketIndex',
                    'SentimentScore', 'SentimentChange', 'SentimentSource',
                    'NewsCount', 'NewsHeadlines', 'NewsImpact',
                    'Volume', 'VolumeChange', 'VolumeChangePct',
                    'PriceChange', 'PriceChangePct', 'Volatility',
                    'MarketCondition', 'MarketTrend', 'Sector',
                    'KeyLevels', 'SupportLevels', 'ResistanceLevels',
                    'RSI', 'MACD', 'MovingAverages', 'Indicators',
                    'TradeID', 'PredictionID', 'Notes'
                ])
    
    def log_market_event(
        self,
        event_type: str,
        ticker: Optional[str] = None,
        market_index: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        sentiment_change: Optional[float] = None,
        sentiment_source: Optional[str] = None,
        news_count: Optional[int] = None,
        news_headlines: Optional[List[str]] = None,
        news_impact: Optional[str] = None,
        volume: Optional[float] = None,
        volume_change: Optional[float] = None,
        volume_change_pct: Optional[float] = None,
        price_change: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volatility: Optional[float] = None,
        market_condition: Optional[str] = None,
        market_trend: Optional[str] = None,
        sector: Optional[str] = None,
        key_levels: Optional[Dict] = None,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
        rsi: Optional[float] = None,
        macd: Optional[Dict] = None,
        moving_averages: Optional[Dict] = None,
        indicators: Optional[Dict] = None,
        trade_id: Optional[str] = None,
        prediction_id: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """
        Log a market event.
        
        Args:
            event_type: Type of event ('sentiment_change', 'news_impact', 'volume_spike', 
                       'price_breakout', 'support_resistance', 'market_open', 'market_close', etc.)
            ticker: Stock ticker symbol (optional)
            market_index: Market index (e.g., 'SPY', 'NASDAQ') (optional)
            sentiment_score: Sentiment score (optional)
            sentiment_change: Change in sentiment (optional)
            sentiment_source: Source of sentiment data (optional)
            news_count: Number of news items (optional)
            news_headlines: List of news headlines (optional)
            news_impact: Impact assessment ('positive', 'negative', 'neutral') (optional)
            volume: Trading volume (optional)
            volume_change: Change in volume (optional)
            volume_change_pct: Percentage change in volume (optional)
            price_change: Price change (optional)
            price_change_pct: Percentage price change (optional)
            volatility: Volatility measure (optional)
            market_condition: Market condition ('bull', 'bear', 'neutral', 'volatile') (optional)
            market_trend: Market trend ('up', 'down', 'sideways') (optional)
            sector: Stock sector (optional)
            key_levels: Dictionary of key price levels (optional)
            support_levels: List of support levels (optional)
            resistance_levels: List of resistance levels (optional)
            rsi: RSI indicator value (optional)
            macd: MACD indicator values (optional)
            moving_averages: Dictionary of moving average values (optional)
            indicators: Dictionary of other technical indicators (optional)
            trade_id: Associated trade ID (optional)
            prediction_id: Associated prediction ID (optional)
            notes: Additional notes
            
        Returns:
            Event ID (timestamp-based)
        """
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now()
        
        event_data = {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "ticker": ticker,
            "market_index": market_index,
            "sentiment_score": sentiment_score,
            "sentiment_change": sentiment_change,
            "sentiment_source": sentiment_source,
            "news_count": news_count,
            "news_headlines": news_headlines or [],
            "news_impact": news_impact,
            "volume": volume,
            "volume_change": volume_change,
            "volume_change_pct": volume_change_pct,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "volatility": volatility,
            "market_condition": market_condition,
            "market_trend": market_trend,
            "sector": sector,
            "key_levels": key_levels or {},
            "support_levels": support_levels or [],
            "resistance_levels": resistance_levels or [],
            "rsi": rsi,
            "macd": macd or {},
            "moving_averages": moving_averages or {},
            "indicators": indicators or {},
            "trade_id": trade_id,
            "prediction_id": prediction_id,
            "notes": notes
        }
        
        self._market_logs.append(event_data)
        self._save_market_logs()
        
        # Append to CSV
        self._append_csv_row(event_data)
        
        return event_id
    
    def log_sentiment_change(
        self,
        ticker: str,
        sentiment_score: float,
        sentiment_change: float,
        sentiment_source: str = "analyzer",
        notes: str = ""
    ) -> str:
        """Log a sentiment change event."""
        return self.log_market_event(
            event_type="sentiment_change",
            ticker=ticker,
            sentiment_score=sentiment_score,
            sentiment_change=sentiment_change,
            sentiment_source=sentiment_source,
            notes=notes
        )
    
    def log_news_impact(
        self,
        ticker: str,
        news_headlines: List[str],
        news_impact: str,
        sentiment_score: Optional[float] = None,
        notes: str = ""
    ) -> str:
        """Log a news impact event."""
        return self.log_market_event(
            event_type="news_impact",
            ticker=ticker,
            news_headlines=news_headlines,
            news_count=len(news_headlines),
            news_impact=news_impact,
            sentiment_score=sentiment_score,
            notes=notes
        )
    
    def log_volume_spike(
        self,
        ticker: str,
        volume: float,
        volume_change: float,
        volume_change_pct: float,
        price_change: Optional[float] = None,
        notes: str = ""
    ) -> str:
        """Log a volume spike event."""
        return self.log_market_event(
            event_type="volume_spike",
            ticker=ticker,
            volume=volume,
            volume_change=volume_change,
            volume_change_pct=volume_change_pct,
            price_change=price_change,
            notes=notes
        )
    
    def log_technical_analysis(
        self,
        ticker: str,
        indicators: Dict,
        rsi: Optional[float] = None,
        macd: Optional[Dict] = None,
        moving_averages: Optional[Dict] = None,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
        notes: str = ""
    ) -> str:
        """Log technical analysis data."""
        return self.log_market_event(
            event_type="technical_analysis",
            ticker=ticker,
            indicators=indicators,
            rsi=rsi,
            macd=macd,
            moving_averages=moving_averages,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            notes=notes
        )
    
    def _append_csv_row(self, event_data: Dict) -> None:
        """Append an event row to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Convert complex fields to JSON strings for CSV
                news_headlines_json = json.dumps(event_data.get("news_headlines", []))
                key_levels_json = json.dumps(event_data.get("key_levels", {}))
                support_levels_json = json.dumps(event_data.get("support_levels", []))
                resistance_levels_json = json.dumps(event_data.get("resistance_levels", []))
                macd_json = json.dumps(event_data.get("macd", {}))
                moving_averages_json = json.dumps(event_data.get("moving_averages", {}))
                indicators_json = json.dumps(event_data.get("indicators", {}))
                
                writer.writerow([
                    event_data.get("timestamp", ""),
                    event_data.get("event_type", ""),
                    event_data.get("ticker", ""),
                    event_data.get("market_index", ""),
                    event_data.get("sentiment_score", ""),
                    event_data.get("sentiment_change", ""),
                    event_data.get("sentiment_source", ""),
                    event_data.get("news_count", ""),
                    news_headlines_json,
                    event_data.get("news_impact", ""),
                    event_data.get("volume", ""),
                    event_data.get("volume_change", ""),
                    event_data.get("volume_change_pct", ""),
                    event_data.get("price_change", ""),
                    event_data.get("price_change_pct", ""),
                    event_data.get("volatility", ""),
                    event_data.get("market_condition", ""),
                    event_data.get("market_trend", ""),
                    event_data.get("sector", ""),
                    key_levels_json,
                    support_levels_json,
                    resistance_levels_json,
                    event_data.get("rsi", ""),
                    macd_json,
                    moving_averages_json,
                    indicators_json,
                    event_data.get("trade_id", ""),
                    event_data.get("prediction_id", ""),
                    event_data.get("notes", "")
                ])
        except Exception:
            # Silent failure on CSV write errors
            pass
    
    def _save_market_logs(self) -> None:
        """Save market logs to JSON file."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self._market_logs, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_market_logs(self) -> None:
        """Load market logs from JSON file."""
        try:
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self._market_logs = json.load(f)
        except Exception:
            # Silent failure on load errors
            self._market_logs = []
    
    def get_market_logs(
        self,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get logged market events, optionally filtered.
        
        Args:
            ticker: Filter by ticker (optional)
            event_type: Filter by event type (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of market event dictionaries
        """
        results = self._market_logs.copy()
        
        if ticker:
            results = [log for log in results if log.get("ticker") == ticker]
        
        if event_type:
            results = [log for log in results if log.get("event_type") == event_type]
        
        if start_date:
            results = [
                log for log in results
                if datetime.fromisoformat(log.get("timestamp", "")) >= start_date
            ]
        
        if end_date:
            results = [
                log for log in results
                if datetime.fromisoformat(log.get("timestamp", "")) <= end_date
            ]
        
        return results
    
    def get_sentiment_history(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get sentiment history for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            List of sentiment events
        """
        return self.get_market_logs(
            ticker=ticker,
            event_type="sentiment_change",
            start_date=start_date,
            end_date=end_date
        )


# Global market logger instance
_market_logger: Optional[MarketLogger] = None


def get_market_logger() -> MarketLogger:
    """Get global market logger instance."""
    global _market_logger
    if _market_logger is None:
        _market_logger = MarketLogger()
    return _market_logger

