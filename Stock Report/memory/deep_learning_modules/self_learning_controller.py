#!/usr/bin/env python3
"""
Self-Learning Controller - Self-Contained Module
Orchestrates continuous learning loop.
"""

import os
import sys
import time
import threading
import datetime
from typing import Dict, List, Optional, Any

try:
    from safety_controller import get_safety_controller
    from unified_deep_engine import get_unified_engine
    from deep_learning_display import display_predictions_table
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


class SelfLearningController:
    """Orchestrates continuous learning and prediction updates."""
    
    def __init__(self, use_gpu: bool = True):
        self.active = False
        self.use_gpu = use_gpu
        self.predictions = {}  # {ticker: {interval: prediction_dict}}
        self.learning_thread = None
        self.safety_controller = None
        self.unified_engine = None
        
        if MODULES_AVAILABLE:
            try:
                self.safety_controller = get_safety_controller()
                self.unified_engine = get_unified_engine(use_gpu=use_gpu)
            except Exception:
                pass
    
    def start_continuous_learning(self):
        """Start continuous learning loop."""
        if self.active:
            return
        
        self.active = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
    
    def stop_continuous_learning(self):
        """Stop continuous learning loop."""
        self.active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
    
    def _learning_loop(self):
        """Main continuous learning loop - only runs when app is idle."""
        operation_id = "learning_loop"
        
        while self.active:
            try:
                # Check if app is idle before running learning tasks
                # Access the idle check function from main module
                is_idle = True
                try:
                    import sys
                    # Access __main__ module which contains the idle check functions
                    main_module = sys.modules.get('__main__')
                    if main_module and hasattr(main_module, '_is_app_idle'):
                        is_idle = main_module._is_app_idle()
                except (AttributeError, KeyError, Exception):
                    # If we can't check (module not loaded or function missing), assume idle and continue
                    # This allows learning to work even if idle detection isn't available
                    pass
                
                # If app is busy, wait longer before checking again
                if not is_idle:
                    time.sleep(2.0)
                    continue
                
                if self.safety_controller:
                    self.safety_controller.start_operation(operation_id)
                    if not self.safety_controller.check_operation(operation_id):
                        break
                    if self.safety_controller.is_kill_requested():
                        break
                
                # Only run learning tasks if app is idle
                # Check for expired predictions
                self._update_expired_predictions()
                
                # Check for missing predictions
                self._fill_missing_predictions()
                
                # Run learning cycle
                self._run_learning_cycle()
                
                # Small sleep to prevent CPU spinning (0.1s)
                time.sleep(0.1)
                
                if self.safety_controller:
                    self.safety_controller.end_operation(operation_id)
            
            except Exception as e:
                # Log error but continue
                pass
    
    def _update_expired_predictions(self):
        """Update predictions that have elapsed."""
        # Implementation would check prediction timestamps
        pass
    
    def _fill_missing_predictions(self):
        """Fill missing predictions for known tickers."""
        # Implementation would check for gaps
        pass
    
    def _run_learning_cycle(self):
        """Run one learning cycle."""
        # Implementation would train models, update strategies
        pass
    
    def add_prediction(self, ticker: str, interval: str, prediction: Dict):
        """Add a new prediction."""
        if ticker not in self.predictions:
            self.predictions[ticker] = {}
        self.predictions[ticker][interval] = prediction
    
    def get_predictions(self, ticker: str) -> List[Dict]:
        """Get all predictions for a ticker."""
        if ticker not in self.predictions:
            return []
        
        predictions = []
        for interval, pred in self.predictions[ticker].items():
            pred['interval'] = interval
            pred['ticker'] = ticker
            predictions.append(pred)
        
        return predictions


# Global self-learning controller instance
_self_learning_controller = None


def get_self_learning_controller(use_gpu: bool = True) -> SelfLearningController:
    """Get or create global self-learning controller instance."""
    global _self_learning_controller
    if _self_learning_controller is None:
        _self_learning_controller = SelfLearningController(use_gpu=use_gpu)
    return _self_learning_controller
