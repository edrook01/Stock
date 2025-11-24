"""
Performance Evaluation
Evaluates simulation and trading performance.
"""

from typing import Dict, List
from ..learning.trade_tracker import get_trade_tracker
from ..sa_logging.analyzer import calculate_performance_metrics, compare_predicted_vs_actual, identify_patterns
from ..sa_logging.trade_logger import get_trade_logger


class PerformanceEvaluator:
    """Evaluates trading performance."""
    
    def __init__(self):
        """Initialize performance evaluator."""
        self.trade_tracker = get_trade_tracker()
        self.trade_logger = get_trade_logger()
    
    def evaluate_simulation(self, simulator) -> Dict:
        """
        Evaluate simulation performance.
        
        Args:
            simulator: TradingSimulatorV15 instance
            
        Returns:
            Dictionary with evaluation results
        """
        stats = simulator.get_statistics()
        trades = self.trade_logger.get_trades()
        
        metrics = calculate_performance_metrics(trades)
        comparison = compare_predicted_vs_actual(trades)
        patterns = identify_patterns(trades)
        
        return {
            "simulation_stats": stats,
            "performance_metrics": metrics,
            "prediction_accuracy": comparison,
            "patterns": patterns
        }
    
    def generate_report(self, simulator) -> str:
        """
        Generate performance report.
        
        Args:
            simulator: TradingSimulatorV15 instance
            
        Returns:
            Formatted report string
        """
        evaluation = self.evaluate_simulation(simulator)
        
        stats = evaluation["simulation_stats"]
        metrics = evaluation["performance_metrics"]
        comparison = evaluation["prediction_accuracy"]
        
        report = f"""
Performance Evaluation Report
{'=' * 70}

Simulation Statistics:
  Initial Balance: ${stats['initial_balance']:,.2f}
  Current Balance: ${stats['current_balance']:,.2f}
  Total P/L: ${stats['total_pnl']:,.2f}
  Total Trades: {stats['total_trades']}
  Win Rate: {stats['win_rate']:.2%}
  Open Positions: {stats['open_positions']}
  Current Exposure: {stats['exposure_pct']:.2f}%

Performance Metrics:
  Win Rate: {metrics['win_rate']:.2%}
  Profit Factor: {metrics['profit_factor']:.2f}
  Max Drawdown: ${metrics['max_drawdown']:,.2f}
  Total P/L: ${metrics['total_pnl']:,.2f}
  Average P/L: ${metrics['avg_pnl']:,.2f}

Prediction Accuracy:
  Accuracy: {comparison['accuracy']:.2%}
  Average Error: {comparison['avg_prediction_error']:.4f}
"""
        
        return report

