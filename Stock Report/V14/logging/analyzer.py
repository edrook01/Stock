"""
Trade Log Analysis Tools
Provides utilities to analyze trade logs and calculate performance metrics.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

# Handle both relative and absolute imports for portability
try:
    from .trade_logger import get_trade_logger
except ImportError:
    # Fallback for direct execution - try multiple import strategies
    try:
        from v14_logging.trade_logger import get_trade_logger
    except ImportError:
        try:
            import sys
            import importlib.util
            from pathlib import Path
            # Try to load directly
            v14_root = Path(__file__).parent.parent
            trade_logger_spec = importlib.util.spec_from_file_location(
                "trade_logger", v14_root / "logging" / "trade_logger.py"
            )
            trade_logger_module = importlib.util.module_from_spec(trade_logger_spec)
            trade_logger_spec.loader.exec_module(trade_logger_module)
            get_trade_logger = trade_logger_module.get_trade_logger
        except Exception:
            raise ImportError("Could not import trade_logger")


def calculate_performance_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate performance metrics from trade logs.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0
        }
    
    # Filter completed trades
    completed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]
    
    if not completed:
        return {
            "total_trades": len(trades),
            "open_trades": len([t for t in trades if not t.get("exit_time")]),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0
        }
    
    wins = [t for t in completed if t.get("pnl", 0) > 0]
    losses = [t for t in completed if t.get("pnl", 0) < 0]
    
    total_profit = sum(t.get("pnl", 0) for t in wins)
    total_loss = abs(sum(t.get("pnl", 0) for t in losses))
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
    
    # Calculate drawdown
    cumulative_pnl = 0.0
    peak = 0.0
    max_drawdown = 0.0
    
    for trade in sorted(completed, key=lambda t: t.get("entry_time", "")):
        cumulative_pnl += trade.get("pnl", 0)
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        drawdown = peak - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    total_pnl = sum(t.get("pnl", 0) for t in completed)
    avg_pnl = total_pnl / len(completed) if completed else 0.0
    
    return {
        "total_trades": len(trades),
        "completed_trades": len(completed),
        "open_trades": len(trades) - len(completed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(completed) if completed else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "total_profit": total_profit,
        "total_loss": total_loss
    }


def compare_predicted_vs_actual(trades: List[Dict]) -> Dict:
    """
    Compare predicted vs actual outcomes.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with comparison metrics
    """
    completed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]
    
    if not completed:
        return {
            "total_comparisons": 0,
            "accuracy": 0.0,
            "avg_prediction_error": 0.0
        }
    
    correct_predictions = 0
    total_error = 0.0
    
    for trade in completed:
        predicted = trade.get("predicted_outcome")
        actual = trade.get("actual_outcome")
        
        if predicted is not None and actual is not None:
            # Check if prediction direction was correct
            if (predicted > 0 and actual > 0) or (predicted < 0 and actual < 0):
                correct_predictions += 1
            
            # Calculate error
            error = abs(predicted - actual)
            total_error += error
    
    return {
        "total_comparisons": len([t for t in completed if t.get("predicted_outcome") is not None]),
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / len(completed) if completed else 0.0,
        "avg_prediction_error": total_error / len(completed) if completed else 0.0
    }


def identify_patterns(trades: List[Dict]) -> Dict:
    """
    Identify patterns in successful/failed trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with identified patterns
    """
    completed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]
    
    if not completed:
        return {}
    
    wins = [t for t in completed if t.get("pnl", 0) > 0]
    losses = [t for t in completed if t.get("pnl", 0) < 0]
    
    # Analyze by confidence
    high_confidence_wins = [t for t in wins if t.get("confidence", 0) >= 0.8]
    high_confidence_losses = [t for t in losses if t.get("confidence", 0) >= 0.8]
    
    # Analyze by timeframe
    timeframe_stats = {}
    for trade in completed:
        tf = trade.get("timeframe", "unknown")
        if tf not in timeframe_stats:
            timeframe_stats[tf] = {"wins": 0, "losses": 0}
        if trade.get("pnl", 0) > 0:
            timeframe_stats[tf]["wins"] += 1
        else:
            timeframe_stats[tf]["losses"] += 1
    
    return {
        "high_confidence_win_rate": len(high_confidence_wins) / (len(high_confidence_wins) + len(high_confidence_losses)) if (high_confidence_wins or high_confidence_losses) else 0.0,
        "timeframe_stats": timeframe_stats,
        "avg_confidence_wins": sum(t.get("confidence", 0) for t in wins) / len(wins) if wins else 0.0,
        "avg_confidence_losses": sum(t.get("confidence", 0) for t in losses) / len(losses) if losses else 0.0
    }


def generate_performance_report(ticker: Optional[str] = None) -> str:
    """
    Generate a text performance report.
    
    Args:
        ticker: Filter by ticker (optional)
        
    Returns:
        Formatted performance report string
    """
    logger = get_trade_logger()
    trades = logger.get_trades(ticker=ticker)
    
    metrics = calculate_performance_metrics(trades)
    comparison = compare_predicted_vs_actual(trades)
    patterns = identify_patterns(trades)
    
    report = f"""
Performance Report
{'=' * 50}
Total Trades: {metrics['total_trades']}
Completed: {metrics['completed_trades']}
Open: {metrics['open_trades']}

Win Rate: {metrics['win_rate']:.2%}
Profit Factor: {metrics['profit_factor']:.2f}
Max Drawdown: ${metrics['max_drawdown']:.2f}

Total P/L: ${metrics['total_pnl']:.2f}
Average P/L: ${metrics['avg_pnl']:.2f}

Prediction Accuracy: {comparison['accuracy']:.2%}
Average Prediction Error: {comparison['avg_prediction_error']:.4f}

High Confidence Win Rate: {patterns.get('high_confidence_win_rate', 0):.2%}
Average Confidence (Wins): {patterns.get('avg_confidence_wins', 0):.3f}
Average Confidence (Losses): {patterns.get('avg_confidence_losses', 0):.3f}
"""
    
    return report

