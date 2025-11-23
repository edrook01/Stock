"""
Confidence Calibration
Calibrates confidence scores to match actual accuracy.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

from ..core.portable_paths import get_path
from ..learning.trade_tracker import get_trade_tracker


class ConfidenceCalibrator:
    """Calibrates model confidence scores."""
    
    def __init__(self):
        """Initialize confidence calibrator."""
        self.trade_tracker = get_trade_tracker()
        self.calibration_data: Dict[str, List[Dict]] = {}
        self.calibration_factors: Dict[str, float] = {}
        self._load_calibration()
    
    def update_calibration(self, timeframe: str) -> Dict:
        """
        Update confidence calibration for a timeframe.
        
        Args:
            timeframe: Prediction timeframe
            
        Returns:
            Dictionary with calibration results
        """
        # Get trade outcomes for this timeframe
        outcomes = self.trade_tracker.get_outcomes()
        timeframe_outcomes = [o for o in outcomes if o.timeframe == timeframe]
        
        if len(timeframe_outcomes) < 20:  # Need minimum samples
            return {
                "timeframe": timeframe,
                "error": "Insufficient data for calibration",
                "samples": len(timeframe_outcomes)
            }
        
        # Group outcomes by confidence bins
        confidence_bins = {
            "high": [],      # >= 0.8
            "medium": [],    # 0.6-0.8
            "low": []        # < 0.6
        }
        
        for outcome in timeframe_outcomes:
            conf = outcome.confidence
            if conf >= 0.8:
                confidence_bins["high"].append(outcome)
            elif conf >= 0.6:
                confidence_bins["medium"].append(outcome)
            else:
                confidence_bins["low"].append(outcome)
        
        # Calculate accuracy for each bin
        calibration_results = {}
        
        for bin_name, bin_outcomes in confidence_bins.items():
            if not bin_outcomes:
                continue
            
            # Calculate win rate
            wins = [o for o in bin_outcomes if o.pnl and o.pnl > 0]
            win_rate = len(wins) / len(bin_outcomes) if bin_outcomes else 0.0
            
            # Expected win rate based on confidence
            if bin_name == "high":
                expected_win_rate = 0.8
            elif bin_name == "medium":
                expected_win_rate = 0.7
            else:
                expected_win_rate = 0.6
            
            # Calculate calibration factor
            if expected_win_rate > 0:
                calibration_factor = win_rate / expected_win_rate
            else:
                calibration_factor = 1.0
            
            calibration_results[bin_name] = {
                "win_rate": win_rate,
                "expected_win_rate": expected_win_rate,
                "calibration_factor": calibration_factor,
                "samples": len(bin_outcomes)
            }
        
        # Store calibration data
        self.calibration_data[timeframe] = calibration_results
        
        # Calculate overall calibration factor (weighted average)
        total_samples = sum(len(bin_outcomes) for bin_outcomes in confidence_bins.values())
        if total_samples > 0:
            overall_factor = sum(
                results["calibration_factor"] * len(confidence_bins[bin_name])
                for bin_name, results in calibration_results.items()
            ) / total_samples
        else:
            overall_factor = 1.0
        
        self.calibration_factors[timeframe] = overall_factor
        
        # Save calibration
        self._save_calibration()
        
        return {
            "timeframe": timeframe,
            "calibration_factors": calibration_results,
            "overall_factor": overall_factor,
            "total_samples": total_samples,
            "calibrated_at": datetime.now().isoformat()
        }
    
    def get_calibration_factor(self, timeframe: str) -> float:
        """
        Get calibration factor for a timeframe.
        
        Args:
            timeframe: Prediction timeframe
            
        Returns:
            Calibration factor (default: 1.0)
        """
        return self.calibration_factors.get(timeframe, 1.0)
    
    def _save_calibration(self) -> None:
        """Save calibration data to file."""
        try:
            memory_dir = get_path('memory')
            memory_dir.mkdir(parents=True, exist_ok=True)
            
            calibration_file = memory_dir / 'confidence_calibration.json'
            
            data = {
                "calibration_data": self.calibration_data,
                "calibration_factors": self.calibration_factors,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_calibration(self) -> None:
        """Load calibration data from file."""
        try:
            memory_dir = get_path('memory')
            calibration_file = memory_dir / 'confidence_calibration.json'
            
            if not calibration_file.exists():
                return
            
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            
            self.calibration_data = data.get("calibration_data", {})
            self.calibration_factors = data.get("calibration_factors", {})
        except Exception:
            # Silent failure on load errors
            self.calibration_data = {}
            self.calibration_factors = {}


# Global confidence calibrator instance
_confidence_calibrator: Optional[ConfidenceCalibrator] = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get global confidence calibrator instance."""
    global _confidence_calibrator
    if _confidence_calibrator is None:
        _confidence_calibrator = ConfidenceCalibrator()
    return _confidence_calibrator

