"""
Feedback Loop Implementation
Updates model confidence calibration and adjusts feature weights based on trade outcomes.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import sys

from .trade_tracker import get_trade_tracker, TradeOutcome

# Import portable_paths with fallback for direct execution
try:
    from ..core.portable_paths import get_path
except (ImportError, ValueError):
    # Fallback for direct execution or when relative imports fail
    v14_root = Path(__file__).parent.parent
    sys.path.insert(0, str(v14_root))
    from core.portable_paths import get_path


class FeedbackLoop:
    """Implements feedback loop for adaptive learning."""
    
    def __init__(self):
        """Initialize feedback loop."""
        self.trade_tracker = get_trade_tracker()
        self.learning_history: List[Dict] = []
        self._load_learning_history()
    
    def process_trade_outcome(self, outcome: TradeOutcome) -> Dict:
        """
        Process a trade outcome and generate learning adjustments.
        
        Args:
            outcome: TradeOutcome instance
            
        Returns:
            Dictionary with learning adjustments
        """
        # Calculate prediction error
        prediction_error = None
        if outcome.predicted_outcome is not None and outcome.actual_outcome is not None:
            prediction_error = abs(outcome.predicted_outcome - outcome.actual_outcome)
        
        # Determine if prediction was correct
        prediction_correct = False
        if outcome.predicted_outcome is not None and outcome.actual_outcome is not None:
            # Check if direction was correct
            if (outcome.predicted_outcome > 0 and outcome.actual_outcome > 0) or \
               (outcome.predicted_outcome < 0 and outcome.actual_outcome < 0):
                prediction_correct = True
        
        # Generate adjustments
        adjustments = {
            "trade_id": outcome.trade_id,
            "timestamp": datetime.now().isoformat(),
            "prediction_error": prediction_error,
            "prediction_correct": prediction_correct,
            "confidence_adjustment": None,
            "feature_weight_adjustments": {},
            "pattern_notes": []
        }
        
        # Adjust confidence calibration
        if outcome.confidence >= 0.8 and not prediction_correct:
            # High confidence but wrong - reduce confidence for similar patterns
            adjustments["confidence_adjustment"] = -0.1
            adjustments["pattern_notes"].append("High confidence failure - reduce confidence for similar patterns")
        elif outcome.confidence < 0.65 and prediction_correct:
            # Low confidence but correct - increase confidence for similar patterns
            adjustments["confidence_adjustment"] = 0.05
            adjustments["pattern_notes"].append("Low confidence success - increase confidence for similar patterns")
        
        # Adjust feature weights for failed patterns
        if not prediction_correct and outcome.exit_reason == "SL":
            # Stop-loss hit - pattern didn't work
            adjustments["feature_weight_adjustments"]["pattern_failed"] = -0.1
            adjustments["pattern_notes"].append("Pattern failed - stop-loss hit")
        
        # Record learning adjustment
        self.learning_history.append(adjustments)
        self._save_learning_history()
        
        return adjustments
    
    def get_confidence_adjustment(self, pattern_features: Dict) -> float:
        """
        Get confidence adjustment for a pattern based on historical outcomes.
        
        Args:
            pattern_features: Dictionary of pattern features
            
        Returns:
            Confidence adjustment factor (-1.0 to 1.0)
        """
        # Simple implementation: check recent outcomes for similar patterns
        # In full implementation, this would use ML to match patterns
        
        recent_adjustments = [
            adj for adj in self.learning_history[-50:]  # Last 50 adjustments
            if adj.get("confidence_adjustment") is not None
        ]
        
        if not recent_adjustments:
            return 0.0
        
        # Average recent adjustments
        avg_adjustment = sum(adj["confidence_adjustment"] for adj in recent_adjustments) / len(recent_adjustments)
        
        # Clamp to reasonable range
        return max(-0.2, min(0.2, avg_adjustment))
    
    def get_pattern_success_rate(self, pattern_type: str) -> float:
        """
        Get success rate for a pattern type.
        
        Args:
            pattern_type: Pattern type identifier
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        outcomes = self.trade_tracker.get_outcomes()
        
        pattern_outcomes = [
            o for o in outcomes
            if hasattr(o, 'pattern_type') and o.pattern_type == pattern_type
        ]
        
        if not pattern_outcomes:
            return 0.5  # Default neutral
        
        wins = [o for o in pattern_outcomes if o.pnl and o.pnl > 0]
        return len(wins) / len(pattern_outcomes) if pattern_outcomes else 0.5
    
    def _save_learning_history(self) -> None:
        """Save learning history to file."""
        try:
            memory_dir = get_path('memory')
            memory_dir.mkdir(parents=True, exist_ok=True)
            
            history_file = memory_dir / 'learning_history.json'
            
            with open(history_file, 'w') as f:
                json.dump(self.learning_history, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_learning_history(self) -> None:
        """Load learning history from file."""
        try:
            memory_dir = get_path('memory')
            history_file = memory_dir / 'learning_history.json'
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r') as f:
                self.learning_history = json.load(f)
        except Exception:
            # Silent failure on load errors
            self.learning_history = []


# Global feedback loop instance
_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get global feedback loop instance."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop

