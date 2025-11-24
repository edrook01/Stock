"""
Continuous Model Updates
Periodically retrains models using recent trade data and supports incremental learning.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

from .trade_tracker import get_trade_tracker

# Import portable_paths with fallback for direct execution
try:
    from ..core.portable_paths import get_path
except (ImportError, ValueError):
    # Fallback for direct execution or when relative imports fail
    v14_root = Path(__file__).parent.parent
    sys.path.insert(0, str(v14_root))
    from core.portable_paths import get_path


class ModelUpdater:
    """Manages continuous model updates and retraining."""
    
    def __init__(self, retrain_interval_days: int = 7):
        """
        Initialize model updater.
        
        Args:
            retrain_interval_days: Days between retraining (default: 7)
        """
        self.retrain_interval_days = retrain_interval_days
        self.trade_tracker = get_trade_tracker()
        self.model_versions: List[Dict] = []
        self.last_retrain_date: Optional[datetime] = None
        self._load_model_history()
    
    def should_retrain(self) -> bool:
        """
        Check if model should be retrained.
        
        Returns:
            True if retraining is due, False otherwise
        """
        if self.last_retrain_date is None:
            return True  # Never retrained
        
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        return days_since_retrain >= self.retrain_interval_days
    
    def get_training_data(self, min_trades: int = 50) -> Optional[Dict]:
        """
        Get training data from recent trades.
        
        Args:
            min_trades: Minimum number of trades required
            
        Returns:
            Dictionary with training data, or None if insufficient data
        """
        outcomes = self.trade_tracker.get_outcomes()
        
        if len(outcomes) < min_trades:
            return None
        
        # Get recent outcomes (last 6 months or all if less)
        cutoff_date = datetime.now() - timedelta(days=180)
        recent_outcomes = [
            o for o in outcomes
            if o.entry_time >= cutoff_date
        ]
        
        if len(recent_outcomes) < min_trades:
            recent_outcomes = outcomes[-min_trades:]  # Use most recent N trades
        
        # Prepare training data
        training_data = {
            "outcomes": [o.to_dict() for o in recent_outcomes],
            "total_trades": len(recent_outcomes),
            "date_range": {
                "start": min(o.entry_time for o in recent_outcomes).isoformat(),
                "end": max(o.exit_time for o in recent_outcomes).isoformat()
            }
        }
        
        return training_data
    
    def record_retrain(
        self,
        model_version: str,
        performance_metrics: Dict,
        training_data_size: int
    ) -> None:
        """
        Record a model retraining event.
        
        Args:
            model_version: Model version identifier
            performance_metrics: Performance metrics before/after retraining
            training_data_size: Number of trades used for training
        """
        retrain_record = {
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "training_data_size": training_data_size,
            "performance_metrics": performance_metrics,
            "retrain_interval_days": self.retrain_interval_days
        }
        
        self.model_versions.append(retrain_record)
        self.last_retrain_date = datetime.now()
        self._save_model_history()
    
    def get_model_version_history(self) -> List[Dict]:
        """
        Get history of model versions.
        
        Returns:
            List of model version dictionaries
        """
        return self.model_versions.copy()
    
    def get_latest_model_version(self) -> Optional[Dict]:
        """
        Get the latest model version.
        
        Returns:
            Latest model version dictionary, or None if no versions
        """
        if not self.model_versions:
            return None
        return self.model_versions[-1]
    
    def can_rollback(self) -> bool:
        """
        Check if model can be rolled back.
        
        Returns:
            True if previous version exists, False otherwise
        """
        return len(self.model_versions) > 1
    
    def rollback_to_version(self, version_index: int) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            version_index: Index of version to rollback to (0 = oldest, -1 = previous)
            
        Returns:
            True if rollback successful, False otherwise
        """
        if not self.can_rollback():
            return False
        
        if version_index < 0:
            version_index = len(self.model_versions) + version_index
        
        if version_index < 0 or version_index >= len(self.model_versions):
            return False
        
        # Remove versions after the rollback point
        self.model_versions = self.model_versions[:version_index + 1]
        self._save_model_history()
        
        return True
    
    def _save_model_history(self) -> None:
        """Save model version history to file."""
        try:
            memory_dir = get_path('memory')
            memory_dir.mkdir(parents=True, exist_ok=True)
            
            history_file = memory_dir / 'model_version_history.json'
            
            history_data = {
                "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                "retrain_interval_days": self.retrain_interval_days,
                "model_versions": self.model_versions
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_model_history(self) -> None:
        """Load model version history from file."""
        try:
            memory_dir = get_path('memory')
            history_file = memory_dir / 'model_version_history.json'
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            self.last_retrain_date = datetime.fromisoformat(history_data["last_retrain_date"]) if history_data.get("last_retrain_date") else None
            self.retrain_interval_days = history_data.get("retrain_interval_days", 7)
            self.model_versions = history_data.get("model_versions", [])
        except Exception:
            # Silent failure on load errors
            self.model_versions = []
            self.last_retrain_date = None


# Global model updater instance
_model_updater: Optional[ModelUpdater] = None


def get_model_updater() -> ModelUpdater:
    """Get global model updater instance."""
    global _model_updater
    if _model_updater is None:
        _model_updater = ModelUpdater()
    return _model_updater

