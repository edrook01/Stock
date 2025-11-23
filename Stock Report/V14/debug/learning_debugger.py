"""
Learning System Debugger
Debug utilities for adaptive learning and feedback loops.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

from ..learning.trade_tracker import get_trade_tracker, TradeOutcome
from ..learning.feedback_loop import get_feedback_loop
from ..learning.prediction_monitor import get_prediction_monitor
from ..learning.failure_tracker import get_failure_tracker
from ..learning.diagnostic import get_diagnostic
from ..learning.model_updater import get_model_updater


class LearningDebugger:
    """Debug learning system components."""
    
    def __init__(self):
        """Initialize learning debugger."""
        pass
    
    def debug_trade_tracking(
        self,
        test_outcomes: List[TradeOutcome]
    ) -> Dict[str, Any]:
        """
        Debug trade tracking system.
        
        Args:
            test_outcomes: List of test trade outcomes
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_trade_tracking",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "test_outcomes": len(test_outcomes)
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        tracker = get_trade_tracker()
        
        # Step 1: Add test outcomes
        for i, outcome in enumerate(test_outcomes):
            try:
                tracker.add_outcome(outcome)
                debug_info["steps"].append({
                    "step": i + 1,
                    "action": f"Add outcome {i+1}",
                    "result": f"Ticker: {outcome.ticker}, P/L: ${outcome.pnl or 0:.2f}"
                })
            except Exception as e:
                debug_info["errors"].append(f"Failed to add outcome {i+1}: {str(e)}")
        
        # Step 2: Get statistics
        stats = tracker.get_statistics()
        debug_info["steps"].append({
            "step": len(debug_info["steps"]) + 1,
            "action": "Get statistics",
            "result": f"Total: {stats['total_trades']}, Win rate: {stats['win_rate']:.2%}"
        })
        
        # Step 3: Test filtering
        if test_outcomes:
            ticker = test_outcomes[0].ticker
            filtered = tracker.get_outcomes(ticker=ticker)
            debug_info["steps"].append({
                "step": len(debug_info["steps"]) + 1,
                "action": f"Filter by ticker: {ticker}",
                "result": f"Found {len(filtered)} outcomes"
            })
        
        debug_info["output"] = {
            "statistics": stats,
            "total_outcomes": len(tracker.outcomes)
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_feedback_loop(
        self,
        test_outcome: TradeOutcome
    ) -> Dict[str, Any]:
        """
        Debug feedback loop processing.
        
        Args:
            test_outcome: Test trade outcome
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_feedback_loop",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        feedback_loop = get_feedback_loop()
        
        # Step 1: Process outcome
        try:
            adjustments = feedback_loop.process_trade_outcome(test_outcome)
            debug_info["steps"].append({
                "step": 1,
                "action": "Process trade outcome",
                "result": f"Adjustments: {len(adjustments)}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Processing error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Get confidence adjustment
        try:
            pattern_features = {}  # Would be actual features in real use
            adj = feedback_loop.get_confidence_adjustment(pattern_features)
            debug_info["steps"].append({
                "step": 2,
                "action": "Get confidence adjustment",
                "result": f"Adjustment: {adj:.4f}"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Confidence adjustment error: {str(e)}")
        
        # Step 3: Get pattern success rate
        try:
            success_rate = feedback_loop.get_pattern_success_rate("test_pattern")
            debug_info["steps"].append({
                "step": 3,
                "action": "Get pattern success rate",
                "result": f"Success rate: {success_rate:.2%}"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Pattern success rate error: {str(e)}")
        
        debug_info["output"] = {
            "adjustments": adjustments,
            "learning_history_count": len(feedback_loop.learning_history)
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_prediction_monitoring(
        self,
        open_trades: List[Dict]
    ) -> Dict[str, Any]:
        """
        Debug prediction monitoring.
        
        Args:
            open_trades: List of open trade dictionaries
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_prediction_monitoring",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "open_trades": len(open_trades)
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        monitor = get_prediction_monitor()
        
        # Step 1: Check for missed predictions
        try:
            missed = monitor.check_missed_predictions(open_trades)
            debug_info["steps"].append({
                "step": 1,
                "action": "Check missed predictions",
                "result": f"Found {len(missed)} missed predictions"
            })
        except Exception as e:
            debug_info["errors"].append(f"Monitoring error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Get missed predictions list
        missed_list = monitor.get_missed_predictions()
        debug_info["steps"].append({
            "step": 2,
            "action": "Get missed predictions list",
            "result": f"Total missed: {len(missed_list)}"
        })
        
        debug_info["output"] = {
            "missed_predictions": missed,
            "total_missed": len(missed_list)
        }
        debug_info["success"] = True
        
        return debug_info
    
    def debug_failure_tracking(
        self,
        test_failures: List[Dict]
    ) -> Dict[str, Any]:
        """
        Debug failure tracking system.
        
        Args:
            test_failures: List of test failure dictionaries
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_failure_tracking",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "test_failures": len(test_failures)
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        failure_tracker = get_failure_tracker()
        
        # Step 1: Check each failure
        for i, failure_data in enumerate(test_failures):
            try:
                failure = failure_tracker.check_trade_failure(
                    trade_id=failure_data.get("trade_id", f"test_{i}"),
                    entry_price=failure_data.get("entry_price", 100.0),
                    exit_price=failure_data.get("exit_price", 95.0),
                    position_size=failure_data.get("position_size", 10.0),
                    direction=failure_data.get("direction", "LONG"),
                    planned_stop_price=failure_data.get("planned_stop", 98.0)
                )
                
                if failure:
                    debug_info["steps"].append({
                        "step": i + 1,
                        "action": f"Check failure {i+1}",
                        "result": f"FLAGGED: {failure.get('failure_type')}"
                    })
                else:
                    debug_info["steps"].append({
                        "step": i + 1,
                        "action": f"Check failure {i+1}",
                        "result": "Not flagged (within threshold)"
                    })
            except Exception as e:
                debug_info["errors"].append(f"Failure check {i+1} error: {str(e)}")
        
        # Step 2: Get failure statistics
        stats = failure_tracker.get_failure_statistics()
        debug_info["steps"].append({
            "step": len(debug_info["steps"]) + 1,
            "action": "Get failure statistics",
            "result": f"Total failures: {stats['total_failures']}"
        })
        
        debug_info["output"] = {
            "failure_statistics": stats,
            "total_failures_tracked": len(failure_tracker.failed_trades)
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_model_updates(self) -> Dict[str, Any]:
        """
        Debug model update system.
        
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_model_updates",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        model_updater = get_model_updater()
        
        # Step 1: Check if should retrain
        should_retrain = model_updater.should_retrain()
        debug_info["steps"].append({
            "step": 1,
            "action": "Check if should retrain",
            "result": "YES" if should_retrain else "NO"
        })
        
        # Step 2: Get training data availability
        training_data = model_updater.get_training_data(min_trades=50)
        debug_info["steps"].append({
            "step": 2,
            "action": "Check training data",
            "result": "Available" if training_data else "Insufficient data"
        })
        
        # Step 3: Get model version history
        version_history = model_updater.get_model_version_history()
        debug_info["steps"].append({
            "step": 3,
            "action": "Get version history",
            "result": f"{len(version_history)} versions"
        })
        
        # Step 4: Check rollback capability
        can_rollback = model_updater.can_rollback()
        debug_info["steps"].append({
            "step": 4,
            "action": "Check rollback capability",
            "result": "Available" if can_rollback else "Not available"
        })
        
        debug_info["output"] = {
            "should_retrain": should_retrain,
            "has_training_data": training_data is not None,
            "version_count": len(version_history),
            "can_rollback": can_rollback,
            "last_retrain": model_updater.last_retrain_date.isoformat() if model_updater.last_retrain_date else None
        }
        debug_info["success"] = True
        
        return debug_info
    
    def debug_diagnostic(
        self,
        test_failure: Dict
    ) -> Dict[str, Any]:
        """
        Debug diagnostic analysis.
        
        Args:
            test_failure: Test failure dictionary
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_diagnostic",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        diagnostic = get_diagnostic()
        
        # Step 1: Diagnose failure
        import asyncio
        try:
            diagnosis = asyncio.run(diagnostic.diagnose_failure(test_failure))
            debug_info["steps"].append({
                "step": 1,
                "action": "Diagnose failure",
                "result": f"Severity: {diagnosis.get('severity')}, Causes: {len(diagnosis.get('causes', []))}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Diagnosis error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Generate report
        try:
            report = diagnostic.generate_diagnostic_report([diagnosis])
            debug_info["steps"].append({
                "step": 2,
                "action": "Generate diagnostic report",
                "result": f"Report length: {len(report)} characters"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Report generation error: {str(e)}")
        
        debug_info["output"] = {
            "diagnosis": diagnosis,
            "report_generated": "report" in locals()
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info


# Global learning debugger instance
_learning_debugger: Optional[LearningDebugger] = None


def get_learning_debugger() -> LearningDebugger:
    """Get global learning debugger instance."""
    global _learning_debugger
    if _learning_debugger is None:
        _learning_debugger = LearningDebugger()
    return _learning_debugger

