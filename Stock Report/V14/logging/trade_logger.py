"""
Comprehensive Trade Logging
Logs all trade events in CSV and JSON formats.
"""

from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import csv
import json

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path


class TradeLogger:
    """Comprehensive trade logger."""
    
    def __init__(self):
        """Initialize trade logger."""
        self.history_dir = get_path('history')
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.history_dir / 'trades.csv'
        self.json_file = self.history_dir / 'trades.json'
        
        self._initialize_csv()
        self._trades: List[Dict] = []
        self._load_trades()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'DateTimeOpen', 'Ticker', 'Side', 'Size', 'EntryPrice',
                    'StopPrice', 'TargetPrice', 'DateTimeClose', 'ClosePrice',
                    'P/L', 'Result', 'Confidence', 'Timeframe', 
                    'PredictionID', 'SentimentScore', 'RSI', 'MACD',
                    'Volume', 'Volatility', 'SupportLevel', 'ResistanceLevel',
                    'Indicators', 'MarketCondition', 'NewsCount', 'Notes'
                ])
    
    def log_trade_entry(
        self,
        ticker: str,
        side: str,
        size: float,
        entry_price: float,
        stop_price: float,
        target_price: Optional[float],
        confidence: float,
        timeframe: str,
        prediction_id: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        rsi: Optional[float] = None,
        macd: Optional[Dict] = None,
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        indicators: Optional[Dict] = None,
        market_condition: Optional[str] = None,
        news_count: Optional[int] = None,
        notes: str = ""
    ) -> str:
        """
        Log a trade entry.
        
        Args:
            ticker: Stock ticker symbol
            side: Trade side ("LONG" or "SHORT")
            size: Position size
            entry_price: Entry price
            stop_price: Stop-loss price
            target_price: Take-profit price (optional)
            confidence: Model confidence (0-1)
            timeframe: Prediction timeframe
            prediction_id: Associated prediction ID (optional)
            sentiment_score: Sentiment score at entry (optional)
            rsi: RSI indicator value at entry (optional)
            macd: MACD indicator values at entry (optional)
            volume: Trading volume at entry (optional)
            volatility: Volatility measure at entry (optional)
            support_level: Support level (optional)
            resistance_level: Resistance level (optional)
            indicators: Dictionary of technical indicators (optional)
            market_condition: Market condition at entry (optional)
            news_count: Number of recent news items (optional)
            notes: Additional notes
            
        Returns:
            Trade ID (timestamp-based)
        """
        trade_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        entry_time = datetime.now()
        
        trade_data = {
            "trade_id": trade_id,
            "entry_time": entry_time.isoformat(),
            "ticker": ticker,
            "side": side.upper(),
            "size": size,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "confidence": confidence,
            "timeframe": timeframe,
            "prediction_id": prediction_id,
            "sentiment_score": sentiment_score,
            "rsi": rsi,
            "macd": macd,
            "volume": volume,
            "volatility": volatility,
            "support_level": support_level,
            "resistance_level": resistance_level,
            "indicators": indicators or {},
            "market_condition": market_condition,
            "news_count": news_count,
            "notes": notes,
            "exit_time": None,
            "close_price": None,
            "pnl": None,
            "result": None,
            "exit_reason": None
        }
        
        self._trades.append(trade_data)
        self._save_trades()
        
        return trade_id
    
    def log_trade_exit(
        self,
        trade_id: str,
        close_price: float,
        exit_reason: str,
        pnl: float,
        pnl_percentage: float,
        notes: str = ""
    ) -> bool:
        """
        Log a trade exit.
        
        Args:
            trade_id: Trade identifier
            close_price: Exit price
            exit_reason: Reason for exit ("TP", "SL", "Manual", "Missed")
            pnl: Profit/loss amount
            pnl_percentage: Profit/loss percentage
            notes: Additional notes
            
        Returns:
            True if trade found and updated, False otherwise
        """
        # Find trade by ID
        trade = None
        for t in self._trades:
            if t.get("trade_id") == trade_id:
                trade = t
                break
        
        if not trade:
            return False
        
        exit_time = datetime.now()
        result = "Win" if pnl > 0 else "Loss" if pnl < 0 else "Breakeven"
        
        # Update trade data
        trade["exit_time"] = exit_time.isoformat()
        trade["close_price"] = close_price
        trade["exit_reason"] = exit_reason
        trade["pnl"] = pnl
        trade["pnl_percentage"] = pnl_percentage
        trade["result"] = result
        if notes:
            trade["notes"] = f"{trade.get('notes', '')}; {notes}".strip('; ')
        
        # Write to CSV
        self._append_csv_row(trade)
        
        # Save JSON
        self._save_trades()
        
        return True
    
    def log_stop_update(
        self,
        trade_id: str,
        new_stop_price: float,
        reason: str = "Trailing stop"
    ) -> bool:
        """
        Log a stop-loss update (e.g., trailing stop).
        
        Args:
            trade_id: Trade identifier
            new_stop_price: New stop-loss price
            reason: Reason for update
            
        Returns:
            True if trade found and updated, False otherwise
        """
        trade = None
        for t in self._trades:
            if t.get("trade_id") == trade_id:
                trade = t
                break
        
        if not trade:
            return False
        
        trade["stop_price"] = new_stop_price
        if "stop_updates" not in trade:
            trade["stop_updates"] = []
        
        trade["stop_updates"].append({
            "timestamp": datetime.now().isoformat(),
            "new_stop": new_stop_price,
            "reason": reason
        })
        
        self._save_trades()
        return True
    
    def _append_csv_row(self, trade: Dict) -> None:
        """Append a trade row to CSV file."""
        try:
            import json
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Convert complex fields to JSON strings for CSV
                macd_json = json.dumps(trade.get("macd", {})) if trade.get("macd") else ""
                indicators_json = json.dumps(trade.get("indicators", {})) if trade.get("indicators") else ""
                
                writer.writerow([
                    trade.get("entry_time", ""),
                    trade.get("ticker", ""),
                    trade.get("side", ""),
                    trade.get("size", ""),
                    trade.get("entry_price", ""),
                    trade.get("stop_price", ""),
                    trade.get("target_price", ""),
                    trade.get("exit_time", ""),
                    trade.get("close_price", ""),
                    f"${trade.get('pnl', 0):.2f} ({trade.get('pnl_percentage', 0):.2f}%)" if trade.get("pnl") is not None else "",
                    trade.get("result", ""),
                    trade.get("confidence", ""),
                    trade.get("timeframe", ""),
                    trade.get("prediction_id", ""),
                    trade.get("sentiment_score", ""),
                    trade.get("rsi", ""),
                    macd_json,
                    trade.get("volume", ""),
                    trade.get("volatility", ""),
                    trade.get("support_level", ""),
                    trade.get("resistance_level", ""),
                    indicators_json,
                    trade.get("market_condition", ""),
                    trade.get("news_count", ""),
                    trade.get("notes", "")
                ])
        except Exception:
            # Silent failure on CSV write errors
            pass
    
    def _save_trades(self) -> None:
        """Save trades to JSON file."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self._trades, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_trades(self) -> None:
        """Load trades from JSON file."""
        try:
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self._trades = json.load(f)
        except Exception:
            # Silent failure on load errors
            self._trades = []
    
    def get_trades(self, ticker: Optional[str] = None) -> List[Dict]:
        """
        Get logged trades, optionally filtered.
        
        Args:
            ticker: Filter by ticker (optional)
            
        Returns:
            List of trade dictionaries
        """
        if ticker:
            return [t for t in self._trades if t.get("ticker") == ticker]
        return self._trades.copy()


# Global trade logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get global trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger

