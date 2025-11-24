"""
Trade Outcome Tracking
Tracks all executed trades and their outcomes for learning.

VERSION PORTABILITY:
===================
This module is fully self-contained and version-portable. To use it in a new version:

1. Copy the entire 'learning/' folder (or just 'trade_tracker.py') to the new version
2. Ensure the new version has a 'history/' directory (will be created automatically)
3. That's it! The module automatically detects the project root by looking for
   common directory markers (data/, core/, history/, learning/)

The module will:
- Automatically find the project root regardless of version number (V15, V15, etc.)
- Work with or without portable_paths.py (falls back to self-contained resolution)
- Create necessary directories automatically
- Store trade outcomes in history/trade_outcomes.json relative to project root

No configuration or path changes needed - just move the file/folder and it works!
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json
import sys
import time

if TYPE_CHECKING:  # Avoid runtime import cycles
    from learning.prediction_storage import PredictionRecord

# Version-portable path resolution
# Automatically finds project root regardless of version number or location
DEBUG_LOG_PATH = Path(r"c:\Users\edwar\Documents\GitHub\.cursor\debug.log")


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Optional[Dict] = None) -> None:
    """Append a single NDJSON instrumentation log entry."""
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _find_project_root() -> Path:
    """
    Find the project root by looking for common directory markers.
    Works regardless of version number (V15, V15, etc.) or absolute path.
    
    Returns:
        Path to project root directory
    """
    # Start from this file's location
    current = Path(__file__).resolve()
    
    # Common markers that indicate project root
    # These directories exist in all versions
    root_markers = [
        'data',      # data/ folder with config and tickers
        'core',      # core/ folder with portable_paths
        'history',   # history/ folder for trade outcomes
        'learning',  # learning/ folder (we're in it)
    ]
    
    # Traverse up the directory tree
    for parent in [current] + list(current.parents):
        # Check if this directory contains multiple root markers
        marker_count = sum(1 for marker in root_markers if (parent / marker).exists())
        
        # If we find at least 2 markers, we're likely at the project root
        # (we exclude 'learning' since we're in it, so need 2+ others)
        if marker_count >= 2:
            return parent
    
    # Fallback: if we're in learning/, go up one level to project root
    if current.parent.name == 'learning':
        return current.parent.parent
    
    # Last resort: go up one level from file location
    # (assumes file is in a subdirectory of project root)
    return current.parent


def _get_history_path() -> Path:
    """
    Get the history directory path, creating it if necessary.
    Version-portable: works regardless of version number.
    
    Returns:
        Path to history directory
    """
    project_root = _find_project_root()
    history_dir = project_root / 'history'
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


# Try to use portable_paths if available (for consistency with other modules)
# But fall back to our own path resolution if not available
try:
    from ..core.portable_paths import get_path
    _USE_PORTABLE_PATHS = True
except (ImportError, ValueError):
    _USE_PORTABLE_PATHS = False
    # Try absolute import as fallback
    try:
        project_root = _find_project_root()
        if (project_root / 'core' / 'portable_paths.py').exists():
            sys.path.insert(0, str(project_root))
            from core.portable_paths import get_path
            _USE_PORTABLE_PATHS = True
    except (ImportError, ValueError):
        pass


def _get_path(type: str) -> Path:
    """
    Get path for a given type, using portable_paths if available,
    otherwise using our own resolution.
    
    Args:
        type: Path type (e.g., 'history', 'data', 'root')
    
    Returns:
        Path object
    """
    if _USE_PORTABLE_PATHS:
        try:
            return get_path(type)
        except (ValueError, AttributeError):
            # Fall back to our own resolution
            pass
    
    # Self-contained path resolution
    project_root = _find_project_root()
    
    if type == 'history':
        return _get_history_path()
    elif type == 'root':
        return project_root
    elif type == 'data':
        return project_root / 'data'
    elif type == 'logs':
        return project_root / 'logs'
    else:
        # For other types, try to construct path relative to root
        return project_root / type


class TradeOutcome:
    """Represents a completed trade outcome."""
    
    def __init__(
        self,
        trade_id: str,
        ticker: str,
        direction: str,
        entry_time: datetime,
        entry_price: float,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        position_size: float,
        stop_price: float,
        target_price: Optional[float],
        confidence: float,
        timeframe: str,
        predicted_outcome: Optional[float] = None,
        actual_outcome: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_percentage: Optional[float] = None,
        prediction_id: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        market_condition: Optional[str] = None,
        indicators: Optional[Dict] = None
    ):
        """
        Initialize trade outcome.
        
        Args:
            trade_id: Unique trade identifier
            ticker: Stock ticker symbol
            direction: Trade direction ("LONG" or "SHORT")
            entry_time: Entry timestamp
            entry_price: Entry price
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit ("TP", "SL", "Manual", "Missed")
            position_size: Position size (number of units)
            stop_price: Stop-loss price
            target_price: Take-profit price (optional)
            confidence: Model confidence at entry (0-1)
            timeframe: Prediction timeframe
            predicted_outcome: Predicted price movement (optional)
            actual_outcome: Actual price movement (optional)
            pnl: Profit/loss amount (optional)
            pnl_percentage: Profit/loss percentage (optional)
            prediction_id: Associated prediction ID (optional)
            sentiment_score: Sentiment score at entry (optional)
            market_condition: Market condition at entry (optional)
            indicators: Technical indicators at entry (optional)
        """
        self.trade_id = trade_id
        self.ticker = ticker
        self.direction = direction.upper()
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.position_size = position_size
        self.stop_price = stop_price
        self.target_price = target_price
        self.confidence = confidence
        self.timeframe = timeframe
        self.predicted_outcome = predicted_outcome
        self.actual_outcome = actual_outcome
        self.pnl = pnl
        self.pnl_percentage = pnl_percentage
        self.prediction_id = prediction_id
        self.sentiment_score = sentiment_score
        self.market_condition = market_condition
        self.indicators = indicators or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "position_size": self.position_size,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
            "prediction_id": self.prediction_id,
            "sentiment_score": self.sentiment_score,
            "market_condition": self.market_condition,
            "indicators": self.indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeOutcome':
        """Create from dictionary."""
        return cls(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            direction=data["direction"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            exit_time=datetime.fromisoformat(data["exit_time"]),
            exit_price=data["exit_price"],
            exit_reason=data["exit_reason"],
            position_size=data["position_size"],
            stop_price=data["stop_price"],
            target_price=data.get("target_price"),
            confidence=data["confidence"],
            timeframe=data["timeframe"],
            predicted_outcome=data.get("predicted_outcome"),
            actual_outcome=data.get("actual_outcome"),
            pnl=data.get("pnl"),
            pnl_percentage=data.get("pnl_percentage"),
            prediction_id=data.get("prediction_id"),
            sentiment_score=data.get("sentiment_score"),
            market_condition=data.get("market_condition"),
            indicators=data.get("indicators", {})
        )

    @classmethod
    def from_prediction(cls, prediction: "PredictionRecord") -> Optional["TradeOutcome"]:
        """
        Create a synthetic TradeOutcome from an evaluated PredictionRecord so
        elapsed predictions can be reused as learning samples.
        """
        actual_close = prediction.actual_close or prediction.actual_price
        if actual_close is None:
            return None

        metadata = prediction.metadata or {}
        base_price = metadata.get("base_price")
        movement_pct = metadata.get("movement_pct")

        if base_price is None and movement_pct is not None:
            try:
                base_price = prediction.predicted_price / (1 + movement_pct / 100.0)
            except ZeroDivisionError:
                base_price = prediction.predicted_price
        if base_price is None or base_price <= 0:
            base_price = prediction.predicted_price

        if movement_pct is None and base_price:
            movement_pct = ((prediction.predicted_price - base_price) / base_price) * 100.0
        movement_pct = movement_pct or 0.0

        direction = "LONG" if movement_pct >= 0 else "SHORT"
        entry_time = prediction.timestamp
        exit_time = prediction.updated_at or prediction.timestamp

        raw_pct = ((actual_close - base_price) / base_price) * 100.0 if base_price else 0.0
        actual_outcome = raw_pct if direction == "LONG" else -raw_pct
        pnl_amount = (actual_close - base_price) if direction == "LONG" else (base_price - actual_close)
        pnl_percentage = (pnl_amount / base_price) * 100.0 if base_price else 0.0

        return cls(
            trade_id=f"prediction-{prediction.prediction_id}",
            ticker=prediction.ticker,
            direction=direction,
            entry_time=entry_time,
            entry_price=base_price,
            exit_time=exit_time,
            exit_price=actual_close,
            exit_reason="ELAPSED_PREDICTION",
            position_size=1.0,
            stop_price=prediction.predicted_range_low or base_price,
            target_price=prediction.predicted_range_high or base_price,
            confidence=prediction.confidence,
            timeframe=prediction.interval,
            predicted_outcome=movement_pct,
            actual_outcome=actual_outcome,
            pnl=pnl_amount,
            pnl_percentage=pnl_percentage,
            prediction_id=prediction.prediction_id,
            indicators=metadata.get("indicators", {}),
        )


class TradeTracker:
    """Tracks all trade outcomes."""
    
    def __init__(self):
        """Initialize trade tracker."""
        self.outcomes: List[TradeOutcome] = []
        self._load_outcomes()
    
    def add_outcome(self, outcome: TradeOutcome) -> None:
        """Add a trade outcome."""
        self.outcomes.append(outcome)
        self._save_outcomes()
    
    def get_outcomes(self, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> List[TradeOutcome]:
        """
        Get trade outcomes, optionally filtered.
        
        Args:
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)
            
        Returns:
            List of trade outcomes
        """
        results = self.outcomes
        
        if ticker:
            results = [o for o in results if o.ticker == ticker]
        
        if timeframe:
            results = [o for o in results if o.timeframe == timeframe]
        
        return results
    
    def get_statistics(self, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> Dict:
        """
        Get comprehensive statistics on trade outcomes (Core Analysis).
        
        Args:
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)
        
        Returns:
            Dictionary with comprehensive statistics
        """
        # Get filtered outcomes
        outcomes = self.get_outcomes(ticker=ticker, timeframe=timeframe)
        
        if not outcomes:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_percentage": 0.0,
                "recovery_factor": 0.0,
                "sharpe_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "risk_reward_ratio": 0.0,
                "expectancy": 0.0,
                "avg_confidence": 0.0,
                "avg_confidence_wins": 0.0,
                "avg_confidence_losses": 0.0,
                "prediction_accuracy": 0.0,
                "avg_prediction_error": 0.0,
                "by_ticker": {},
                "by_timeframe": {},
                "by_exit_reason": {},
                "by_direction": {}
            }
        
        # Filter outcomes with valid PnL
        valid_outcomes = [o for o in outcomes if o.pnl is not None]
        
        if not valid_outcomes:
            return {
                "total_trades": len(outcomes),
                "completed_trades": 0,
                "open_trades": len(outcomes),
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_confidence": sum(o.confidence for o in outcomes) / len(outcomes) if outcomes else 0.0
            }
        
        wins = [o for o in valid_outcomes if o.pnl and o.pnl > 0]
        losses = [o for o in valid_outcomes if o.pnl and o.pnl < 0]
        breakeven = [o for o in valid_outcomes if o.pnl == 0]
        
        # Basic counts
        total_trades = len(outcomes)
        completed_trades = len(valid_outcomes)
        open_trades = total_trades - completed_trades
        
        # PnL calculations
        total_pnl = sum(o.pnl for o in valid_outcomes)
        total_profit = sum(o.pnl for o in wins) if wins else 0.0
        total_loss = abs(sum(o.pnl for o in losses)) if losses else 0.0
        avg_pnl = total_pnl / completed_trades if completed_trades > 0 else 0.0
        
        # Win/loss statistics
        win_rate = len(wins) / completed_trades if completed_trades > 0 else 0.0
        avg_win = total_profit / len(wins) if wins else 0.0
        avg_loss = total_loss / len(losses) if losses else 0.0
        largest_win = max((o.pnl for o in wins), default=0.0) if wins else 0.0
        largest_loss = min((o.pnl for o in losses), default=0.0) if losses else 0.0
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        
        # Risk-reward ratio
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        # Expectancy: (Win% * Avg Win) - (Loss% * Avg Loss)
        win_percentage = len(wins) / completed_trades if completed_trades > 0 else 0.0
        loss_percentage = len(losses) / completed_trades if completed_trades > 0 else 0.0
        expectancy = (win_percentage * avg_win) - (loss_percentage * avg_loss)
        
        # Drawdown calculation
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        max_drawdown_percentage = 0.0
        
        # Sort by entry time for chronological analysis
        sorted_outcomes = sorted(valid_outcomes, key=lambda o: o.entry_time)
        
        for outcome in sorted_outcomes:
            cumulative_pnl += outcome.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                if peak != 0:
                    max_drawdown_percentage = (drawdown / peak) * 100
        
        # Recovery factor: Total PnL / Max Drawdown
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0.0
        
        # Sharpe ratio approximation (using returns)
        if completed_trades > 1:
            returns = [o.pnl_percentage if o.pnl_percentage else (o.pnl / o.entry_price * 100) for o in valid_outcomes]
            avg_return = sum(returns) / len(returns) if returns else 0.0
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns) if returns else 0.0
            std_dev = variance ** 0.5 if variance > 0 else 0.0
            sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0.0  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Confidence statistics
        avg_confidence = sum(o.confidence for o in outcomes) / len(outcomes) if outcomes else 0.0
        avg_confidence_wins = sum(o.confidence for o in wins) / len(wins) if wins else 0.0
        avg_confidence_losses = sum(o.confidence for o in losses) / len(losses) if losses else 0.0
        
        # Prediction accuracy
        predictions_with_actual = [
            o for o in valid_outcomes 
            if o.predicted_outcome is not None and o.actual_outcome is not None
        ]
        
        if predictions_with_actual:
            correct_predictions = sum(
                1 for o in predictions_with_actual
                if (o.predicted_outcome > 0 and o.actual_outcome > 0) or 
                   (o.predicted_outcome < 0 and o.actual_outcome < 0)
            )
            prediction_accuracy = correct_predictions / len(predictions_with_actual)
            avg_prediction_error = sum(
                abs(o.predicted_outcome - o.actual_outcome) 
                for o in predictions_with_actual
            ) / len(predictions_with_actual)
        else:
            prediction_accuracy = 0.0
            avg_prediction_error = 0.0
        
        # Breakdown by ticker
        by_ticker = {}
        for outcome in valid_outcomes:
            tick = outcome.ticker
            if tick not in by_ticker:
                by_ticker[tick] = {
                    "total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                    "win_rate": 0.0, "avg_pnl": 0.0
                }
            by_ticker[tick]["total"] += 1
            if outcome.pnl > 0:
                by_ticker[tick]["wins"] += 1
            elif outcome.pnl < 0:
                by_ticker[tick]["losses"] += 1
            by_ticker[tick]["total_pnl"] += outcome.pnl
        
        for tick in by_ticker:
            stats = by_ticker[tick]
            stats["win_rate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_pnl"] = stats["total_pnl"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Breakdown by timeframe
        by_timeframe = {}
        for outcome in valid_outcomes:
            tf = outcome.timeframe
            if tf not in by_timeframe:
                by_timeframe[tf] = {
                    "total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                    "win_rate": 0.0, "avg_pnl": 0.0
                }
            by_timeframe[tf]["total"] += 1
            if outcome.pnl > 0:
                by_timeframe[tf]["wins"] += 1
            elif outcome.pnl < 0:
                by_timeframe[tf]["losses"] += 1
            by_timeframe[tf]["total_pnl"] += outcome.pnl
        
        for tf in by_timeframe:
            stats = by_timeframe[tf]
            stats["win_rate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_pnl"] = stats["total_pnl"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Breakdown by exit reason
        by_exit_reason = {}
        for outcome in valid_outcomes:
            reason = outcome.exit_reason
            if reason not in by_exit_reason:
                by_exit_reason[reason] = {
                    "total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                    "win_rate": 0.0, "avg_pnl": 0.0
                }
            by_exit_reason[reason]["total"] += 1
            if outcome.pnl > 0:
                by_exit_reason[reason]["wins"] += 1
            elif outcome.pnl < 0:
                by_exit_reason[reason]["losses"] += 1
            by_exit_reason[reason]["total_pnl"] += outcome.pnl
        
        for reason in by_exit_reason:
            stats = by_exit_reason[reason]
            stats["win_rate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_pnl"] = stats["total_pnl"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Breakdown by direction
        by_direction = {}
        for outcome in valid_outcomes:
            direction = outcome.direction
            if direction not in by_direction:
                by_direction[direction] = {
                    "total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                    "win_rate": 0.0, "avg_pnl": 0.0
                }
            by_direction[direction]["total"] += 1
            if outcome.pnl > 0:
                by_direction[direction]["wins"] += 1
            elif outcome.pnl < 0:
                by_direction[direction]["losses"] += 1
            by_direction[direction]["total_pnl"] += outcome.pnl
        
        for direction in by_direction:
            stats = by_direction[direction]
            stats["win_rate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_pnl"] = stats["total_pnl"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            # Basic metrics
            "total_trades": total_trades,
            "completed_trades": completed_trades,
            "open_trades": open_trades,
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate": win_rate,
            
            # PnL metrics
            "total_pnl": total_pnl,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            
            # Risk metrics
            "profit_factor": profit_factor,
            "risk_reward_ratio": risk_reward_ratio,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "max_drawdown_percentage": max_drawdown_percentage,
            "recovery_factor": recovery_factor,
            "sharpe_ratio": sharpe_ratio,
            
            # Confidence metrics
            "avg_confidence": avg_confidence,
            "avg_confidence_wins": avg_confidence_wins,
            "avg_confidence_losses": avg_confidence_losses,
            
            # Prediction metrics
            "prediction_accuracy": prediction_accuracy,
            "avg_prediction_error": avg_prediction_error,
            "predictions_with_actual": len(predictions_with_actual),
            
            # Breakdowns
            "by_ticker": by_ticker,
            "by_timeframe": by_timeframe,
            "by_exit_reason": by_exit_reason,
            "by_direction": by_direction
        }
    
    def generate_analysis_report(self, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> str:
        """
        Generate a comprehensive text analysis report.
        
        Args:
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)
            
        Returns:
            Formatted analysis report string
        """
        stats = self.get_statistics(ticker=ticker, timeframe=timeframe)
        
        if stats["total_trades"] == 0:
            return f"""
Core Analysis Report
{'=' * 60}
No trades found{f" for {ticker}" if ticker else ""}{f" on {timeframe}" if timeframe else ""}.

"""
        
        report = f"""
Core Analysis Report
{'=' * 60}
{'Filtered by: ' + (f'Ticker: {ticker}, ' if ticker else '') + (f'Timeframe: {timeframe}' if timeframe else 'All Trades')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC METRICS
{'-' * 60}
Total Trades: {stats['total_trades']}
  - Completed: {stats['completed_trades']}
  - Open: {stats['open_trades']}
  
Win Rate: {stats['win_rate']:.2%} ({stats['wins']} wins / {stats['losses']} losses)
Breakeven Trades: {stats.get('breakeven', 0)}

PROFIT & LOSS
{'-' * 60}
Total P/L: ${stats['total_pnl']:,.2f}
Total Profit: ${stats['total_profit']:,.2f}
Total Loss: ${stats['total_loss']:,.2f}
Average P/L per Trade: ${stats['avg_pnl']:,.2f}

Average Win: ${stats['avg_win']:,.2f}
Average Loss: ${stats['avg_loss']:,.2f}
Largest Win: ${stats['largest_win']:,.2f}
Largest Loss: ${stats['largest_loss']:,.2f}

RISK METRICS
{'-' * 60}
Profit Factor: {stats['profit_factor']:.2f}
Risk-Reward Ratio: {stats['risk_reward_ratio']:.2f}
Expectancy: ${stats['expectancy']:,.2f}
Max Drawdown: ${stats['max_drawdown']:,.2f} ({stats['max_drawdown_percentage']:.2f}%)
Recovery Factor: {stats['recovery_factor']:.2f}
Sharpe Ratio: {stats['sharpe_ratio']:.2f}

CONFIDENCE METRICS
{'-' * 60}
Average Confidence: {stats['avg_confidence']:.3f}
Average Confidence (Wins): {stats['avg_confidence_wins']:.3f}
Average Confidence (Losses): {stats['avg_confidence_losses']:.3f}

PREDICTION METRICS
{'-' * 60}
Prediction Accuracy: {stats['prediction_accuracy']:.2%}
Average Prediction Error: {stats['avg_prediction_error']:.4f}
Predictions with Actual: {stats['predictions_with_actual']}
"""
        
        # Breakdown by ticker
        if stats['by_ticker']:
            report += "\nBREAKDOWN BY TICKER\n"
            report += "-" * 60 + "\n"
            sorted_tickers = sorted(
                stats['by_ticker'].items(),
                key=lambda x: x[1]['total_pnl'],
                reverse=True
            )
            for tick, tick_stats in sorted_tickers[:10]:  # Top 10
                report += f"{tick}:\n"
                report += f"  Trades: {tick_stats['total']} | Win Rate: {tick_stats['win_rate']:.2%}\n"
                report += f"  Total P/L: ${tick_stats['total_pnl']:,.2f} | Avg P/L: ${tick_stats['avg_pnl']:,.2f}\n"
        
        # Breakdown by timeframe
        if stats['by_timeframe']:
            report += "\nBREAKDOWN BY TIMEFRAME\n"
            report += "-" * 60 + "\n"
            sorted_timeframes = sorted(
                stats['by_timeframe'].items(),
                key=lambda x: x[1]['total_pnl'],
                reverse=True
            )
            for tf, tf_stats in sorted_timeframes:
                report += f"{tf}:\n"
                report += f"  Trades: {tf_stats['total']} | Win Rate: {tf_stats['win_rate']:.2%}\n"
                report += f"  Total P/L: ${tf_stats['total_pnl']:,.2f} | Avg P/L: ${tf_stats['avg_pnl']:,.2f}\n"
        
        # Breakdown by exit reason
        if stats['by_exit_reason']:
            report += "\nBREAKDOWN BY EXIT REASON\n"
            report += "-" * 60 + "\n"
            for reason, reason_stats in sorted(stats['by_exit_reason'].items()):
                report += f"{reason}:\n"
                report += f"  Trades: {reason_stats['total']} | Win Rate: {reason_stats['win_rate']:.2%}\n"
                report += f"  Total P/L: ${reason_stats['total_pnl']:,.2f} | Avg P/L: ${reason_stats['avg_pnl']:,.2f}\n"
        
        # Breakdown by direction
        if stats['by_direction']:
            report += "\nBREAKDOWN BY DIRECTION\n"
            report += "-" * 60 + "\n"
            for direction, dir_stats in sorted(stats['by_direction'].items()):
                report += f"{direction}:\n"
                report += f"  Trades: {dir_stats['total']} | Win Rate: {dir_stats['win_rate']:.2%}\n"
                report += f"  Total P/L: ${dir_stats['total_pnl']:,.2f} | Avg P/L: ${dir_stats['avg_pnl']:,.2f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
    
    def get_time_based_analysis(self, period_days: int = 30, ticker: Optional[str] = None) -> Dict:
        """
        Get performance analysis over time periods.
        
        Args:
            period_days: Number of days per period
            ticker: Filter by ticker (optional)
            
        Returns:
            Dictionary with time-based analysis
        """
        from datetime import timedelta
        
        outcomes = self.get_outcomes(ticker=ticker)
        if not outcomes:
            return {
                "periods": [],
                "period_days": period_days
            }
        
        # Get all entry times
        entry_times = [o.entry_time for o in outcomes if o.entry_time]
        if not entry_times:
            return {
                "periods": [],
                "period_days": period_days
            }
        
        # Find date range
        min_date = min(entry_times).date()
        max_date = max(entry_times).date()
        
        periods = []
        current_date = min_date
        
        while current_date <= max_date:
            period_end = current_date + timedelta(days=period_days)
            
            # Get outcomes in this period
            period_outcomes = [
                o for o in outcomes
                if o.entry_time and current_date <= o.entry_time.date() < period_end
            ]
            
            if period_outcomes:
                # Calculate stats for this period
                period_valid = [o for o in period_outcomes if o.pnl is not None]
                period_wins = [o for o in period_valid if o.pnl and o.pnl > 0]
                period_losses = [o for o in period_valid if o.pnl and o.pnl < 0]
                
                period_pnl = sum(o.pnl for o in period_valid if o.pnl)
                period_win_rate = len(period_wins) / len(period_valid) if period_valid else 0.0
                
                periods.append({
                    "period_start": current_date.isoformat(),
                    "period_end": period_end.isoformat(),
                    "total_trades": len(period_outcomes),
                    "completed_trades": len(period_valid),
                    "wins": len(period_wins),
                    "losses": len(period_losses),
                    "win_rate": period_win_rate,
                    "total_pnl": period_pnl,
                    "avg_pnl": period_pnl / len(period_valid) if period_valid else 0.0
                })
            
            current_date = period_end
        
        return {
            "periods": periods,
            "period_days": period_days,
            "date_range": {
                "start": min_date.isoformat(),
                "end": max_date.isoformat()
            },
            "total_periods": len(periods)
        }
    
    def _save_outcomes(self) -> None:
        """Save outcomes to file."""
        try:
            history_dir = _get_path('history')
            outcomes_file = history_dir / 'trade_outcomes.json'
            #region agent log
            _agent_debug_log(
                "H3",
                "learning/trade_tracker.py:785",
                "Saving trade outcomes",
                {"outcome_count": len(self.outcomes)}
            )
            #endregion
            outcomes_data = [outcome.to_dict() for outcome in self.outcomes]
            
            with open(outcomes_file, 'w') as f:
                json.dump(outcomes_data, f, indent=2)
        except Exception:
            # Silent failure on save errors
            _agent_debug_log(
                "H3",
                "learning/trade_tracker.py:822",
                "Error saving trade outcomes",
                {}
            )
            pass
    
    def _load_outcomes(self) -> None:
        """Load outcomes from file."""
        try:
            history_dir = _get_path('history')
            outcomes_file = history_dir / 'trade_outcomes.json'
            #region agent log
            _agent_debug_log(
                "H3",
                "learning/trade_tracker.py:800",
                "Loading trade outcomes",
                {"exists": outcomes_file.exists(), "path": str(outcomes_file)}
            )
            #endregion
            
            if not outcomes_file.exists():
                return
            
            with open(outcomes_file, 'r') as f:
                outcomes_data = json.load(f)
            
            self.outcomes = [TradeOutcome.from_dict(data) for data in outcomes_data]
            #region agent log
            _agent_debug_log(
                "H3",
                "learning/trade_tracker.py:808",
                "Loaded trade outcomes",
                {"loaded_count": len(self.outcomes)}
            )
            #endregion
        except Exception:
            # Silent failure on load errors
            _agent_debug_log(
                "H3",
                "learning/trade_tracker.py:855",
                "Error loading trade outcomes",
                {}
            )
            self.outcomes = []


# Global trade tracker instance
_trade_tracker: Optional[TradeTracker] = None


def get_trade_tracker() -> TradeTracker:
    """Get global trade tracker instance."""
    global _trade_tracker
    #region agent log
    _agent_debug_log(
        "H3",
        "learning/trade_tracker.py:823",
        "Accessing trade tracker singleton",
        {"tracker_initialized": _trade_tracker is not None}
    )
    #endregion
    if _trade_tracker is None:
        _trade_tracker = TradeTracker()
    return _trade_tracker

