"""
Continuous Learning Service
Background service that periodically retrains models based on new trade data.
"""

import threading
import time
import asyncio
from typing import Optional, Dict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

from .model_updater import get_model_updater

# Import with fallback for direct execution
try:
    from ..model.trainer import get_model_trainer
    from ..core.portable_paths import get_data_path
except (ImportError, ValueError):
    # Fallback for direct execution or when relative imports fail
    v14_root = Path(__file__).parent.parent
    sys.path.insert(0, str(v14_root))
    from model.trainer import get_model_trainer
    from core.portable_paths import get_data_path

import json

logger = logging.getLogger(__name__)


class ContinuousLearningService:
    """Background service for continuous model learning."""
    
    def __init__(self, check_interval_hours: float = 6.0):
        """
        Initialize continuous learning service.
        
        Args:
            check_interval_hours: Hours between retraining checks (default: 6)
        """
        self.check_interval_hours = check_interval_hours
        self.check_interval_seconds = check_interval_hours * 3600
        self.model_updater = get_model_updater()
        self.model_trainer = get_model_trainer()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_check: Optional[datetime] = None
        self.last_training_result: Optional[Dict] = None
        self._load_state()
    
    def start(self) -> bool:
        """
        Start the continuous learning service.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Continuous learning service is already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Continuous learning service started")
        self._save_state()
        return True
    
    def stop(self) -> bool:
        """
        Stop the continuous learning service.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Continuous learning service is not running")
            return False
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)  # Wait up to 5 seconds
        logger.info("Continuous learning service stopped")
        self._save_state()
        return True
    
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.running
    
    def trigger_retrain(self) -> Dict:
        """
        Manually trigger a retraining check and training if needed.
        
        Returns:
            Dictionary with retraining result
        """
        return self._check_and_retrain()
    
    def _run_loop(self) -> None:
        """Main service loop (runs in background thread)."""
        logger.info("Continuous learning service loop started")
        
        while self.running:
            try:
                # Check if retraining is needed
                self._check_and_retrain()
                
                # Update last check time
                self.last_check = datetime.now()
                self._save_state()
                
                # Sleep until next check
                time.sleep(self.check_interval_seconds)
            
            except Exception as e:
                logger.error(f"Error in continuous learning service loop: {e}", exc_info=True)
                # Continue running even if there's an error
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_and_retrain(self) -> Dict:
        """
        Check if retraining is needed and perform retraining.
        
        Returns:
            Dictionary with retraining result
        """
        try:
            # Check if retraining is due
            if not self.model_updater.should_retrain():
                return {
                    "retrained": False,
                    "reason": "Retraining not due yet",
                    "last_retrain": self.model_updater.last_retrain_date.isoformat() if self.model_updater.last_retrain_date else None
                }
            
            # Get training data
            training_data = self.model_updater.get_training_data(min_trades=50)
            if training_data is None:
                return {
                    "retrained": False,
                    "reason": "Insufficient training data (need at least 50 trades)",
                    "available_trades": len(self.model_updater.trade_tracker.get_outcomes())
                }
            
            logger.info(f"Starting model retraining with {training_data['total_trades']} trades")
            
            # Train all models (async operation in sync context)
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                training_results = loop.run_until_complete(
                    self.model_trainer.train_all_models()
                )
                
                # Record retraining
                model_version = f"v{len(self.model_updater.model_versions) + 1}"
                performance_metrics = {
                    "training_samples": training_data["total_trades"],
                    "timeframes_trained": len([r for r in training_results.values() if "error" not in r]),
                    "errors": [r.get("error") for r in training_results.values() if "error" in r]
                }
                
                self.model_updater.record_retrain(
                    model_version=model_version,
                    performance_metrics=performance_metrics,
                    training_data_size=training_data["total_trades"]
                )
                
                result = {
                    "retrained": True,
                    "model_version": model_version,
                    "training_samples": training_data["total_trades"],
                    "timeframes_trained": len([r for r in training_results.values() if "error" not in r]),
                    "training_results": training_results,
                    "retrained_at": datetime.now().isoformat()
                }
                
                self.last_training_result = result
                logger.info(f"Model retraining completed: {model_version}")
                
                return result
            
            finally:
                loop.close()
        
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
            return {
                "retrained": False,
                "error": str(e),
                "retrained_at": datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict:
        """
        Get service status.
        
        Returns:
            Dictionary with service status
        """
        return {
            "running": self.running,
            "check_interval_hours": self.check_interval_hours,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "should_retrain": self.model_updater.should_retrain(),
            "last_retrain": self.model_updater.last_retrain_date.isoformat() if self.model_updater.last_retrain_date else None,
            "last_training_result": self.last_training_result,
            "available_trades": len(self.model_updater.trade_tracker.get_outcomes())
        }
    
    def _save_state(self) -> None:
        """Save service state to file."""
        try:
            state_file = get_data_path() / 'continuous_learning_state.json'
            state = {
                "running": self.running,
                "check_interval_hours": self.check_interval_hours,
                "last_check": self.last_check.isoformat() if self.last_check else None
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
    
    def _load_state(self) -> None:
        """Load service state from file."""
        try:
            state_file = get_data_path() / 'continuous_learning_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.check_interval_hours = state.get("check_interval_hours", 6.0)
                    self.check_interval_seconds = self.check_interval_hours * 3600
                    if state.get("last_check"):
                        self.last_check = datetime.fromisoformat(state["last_check"])
        except Exception:
            pass


# Global continuous learning service instance
_continuous_service: Optional[ContinuousLearningService] = None


def get_continuous_learning_service() -> ContinuousLearningService:
    """Get global continuous learning service instance."""
    global _continuous_service
    if _continuous_service is None:
        _continuous_service = ContinuousLearningService()
    return _continuous_service

