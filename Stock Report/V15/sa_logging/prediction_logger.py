"""
Prediction Data Logger
Logs all predictions made by models with full context for analysis.
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import json
import csv

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path


class PredictionLogger:
    """Comprehensive prediction logger."""
    
    def __init__(self):
        """Initialize prediction logger."""
        self.history_dir = get_path('history')
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.history_dir / 'predictions.csv'
        self.json_file = self.history_dir / 'predictions.json'
        
        self._initialize_csv()
        self._predictions: List[Dict] = []
        self._load_predictions()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Ticker', 'Timeframe', 'ModelType', 'PredictionType',
                    'PredictedPrice', 'PredictedChange', 'PredictedChangePct', 
                    'CurrentPrice', 'Confidence', 'ModelVersion', 'FeaturesUsed',
                    'InputIndicators', 'MarketConditions', 'SentimentScore',
                    'ActualPrice', 'ActualChange', 'ActualChangePct', 'Outcome',
                    'Error', 'ErrorPct', 'TradeID', 'Notes'
                ])
    
    def log_prediction(
        self,
        ticker: str,
        timeframe: str,
        model_type: str,
        prediction_type: str,
        predicted_price: Optional[float],
        predicted_change: Optional[float],
        predicted_change_pct: Optional[float],
        current_price: float,
        confidence: float,
        model_version: Optional[str] = None,
        features_used: Optional[Dict] = None,
        input_indicators: Optional[Dict] = None,
        market_conditions: Optional[Dict] = None,
        sentiment_score: Optional[float] = None,
        notes: str = ""
    ) -> str:
        """
        Log a prediction.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe
            model_type: Type of model used (e.g., 'unified', 'random_forest')
            prediction_type: Type of prediction (e.g., 'price', 'direction', 'movement')
            predicted_price: Predicted price (optional)
            predicted_change: Predicted price change (optional)
            predicted_change_pct: Predicted percentage change (optional)
            current_price: Current market price
            confidence: Model confidence (0-1)
            model_version: Model version identifier (optional)
            features_used: Dictionary of features used (optional)
            input_indicators: Dictionary of technical indicators (optional)
            market_conditions: Dictionary of market conditions (optional)
            sentiment_score: Sentiment score (optional)
            notes: Additional notes
            
        Returns:
            Prediction ID (timestamp-based)
        """
        prediction_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now()
        
        prediction_data = {
            "prediction_id": prediction_id,
            "timestamp": timestamp.isoformat(),
            "ticker": ticker,
            "timeframe": timeframe,
            "model_type": model_type,
            "prediction_type": prediction_type,
            "predicted_price": predicted_price,
            "predicted_change": predicted_change,
            "predicted_change_pct": predicted_change_pct,
            "current_price": current_price,
            "confidence": confidence,
            "model_version": model_version,
            "features_used": features_used or {},
            "input_indicators": input_indicators or {},
            "market_conditions": market_conditions or {},
            "sentiment_score": sentiment_score,
            "actual_price": None,
            "actual_change": None,
            "actual_change_pct": None,
            "outcome": None,
            "error": None,
            "error_pct": None,
            "trade_id": None,
            "notes": notes
        }
        
        self._predictions.append(prediction_data)
        self._save_predictions()
        
        # Append to CSV
        self._append_csv_row(prediction_data)
        
        return prediction_id
    
    def log_prediction_outcome(
        self,
        prediction_id: str,
        actual_price: float,
        actual_change: Optional[float] = None,
        actual_change_pct: Optional[float] = None,
        trade_id: Optional[str] = None,
        notes: str = ""
    ) -> bool:
        """
        Log the outcome of a prediction.
        
        Args:
            prediction_id: Prediction identifier
            actual_price: Actual price at target time
            actual_change: Actual price change (optional)
            actual_change_pct: Actual percentage change (optional)
            trade_id: Associated trade ID (optional)
            notes: Additional notes
            
        Returns:
            True if prediction found and updated, False otherwise
        """
        prediction = None
        for p in self._predictions:
            if p.get("prediction_id") == prediction_id:
                prediction = p
                break
        
        if not prediction:
            return False
        
        # Calculate outcome metrics
        predicted_price = prediction.get("predicted_price")
        current_price = prediction.get("current_price")
        
        if predicted_price and current_price:
            predicted_change = predicted_price - current_price
            actual_change_calc = actual_price - current_price
            
            error = abs(predicted_price - actual_price)
            error_pct = (error / current_price * 100) if current_price > 0 else 0
            
            # Determine outcome (correct direction if same sign)
            if predicted_change and actual_change_calc:
                if (predicted_change > 0 and actual_change_calc > 0) or \
                   (predicted_change < 0 and actual_change_calc < 0):
                    outcome = "Correct"
                else:
                    outcome = "Incorrect"
            else:
                outcome = "Partial"
        else:
            error = None
            error_pct = None
            outcome = "Unknown"
        
        # Update prediction data
        prediction["actual_price"] = actual_price
        prediction["actual_change"] = actual_change
        prediction["actual_change_pct"] = actual_change_pct
        prediction["outcome"] = outcome
        prediction["error"] = error
        prediction["error_pct"] = error_pct
        prediction["trade_id"] = trade_id
        if notes:
            prediction["notes"] = f"{prediction.get('notes', '')}; {notes}".strip('; ')
        
        # Re-save with updated data
        self._save_predictions()
        
        # Update CSV (would need to rewrite or use a different approach)
        # For now, we'll update JSON which is more flexible
        
        return True
    
    def _append_csv_row(self, prediction: Dict) -> None:
        """Append a prediction row to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Convert complex fields to JSON strings for CSV
                features_json = json.dumps(prediction.get("features_used", {}))
                indicators_json = json.dumps(prediction.get("input_indicators", {}))
                conditions_json = json.dumps(prediction.get("market_conditions", {}))
                
                writer.writerow([
                    prediction.get("timestamp", ""),
                    prediction.get("ticker", ""),
                    prediction.get("timeframe", ""),
                    prediction.get("model_type", ""),
                    prediction.get("prediction_type", ""),
                    prediction.get("predicted_price", ""),
                    prediction.get("predicted_change", ""),
                    prediction.get("predicted_change_pct", ""),
                    prediction.get("current_price", ""),
                    prediction.get("confidence", ""),
                    prediction.get("model_version", ""),
                    features_json,
                    indicators_json,
                    conditions_json,
                    prediction.get("sentiment_score", ""),
                    prediction.get("actual_price", ""),
                    prediction.get("actual_change", ""),
                    prediction.get("actual_change_pct", ""),
                    prediction.get("outcome", ""),
                    prediction.get("error", ""),
                    prediction.get("error_pct", ""),
                    prediction.get("trade_id", ""),
                    prediction.get("notes", "")
                ])
        except Exception:
            # Silent failure on CSV write errors
            pass
    
    def _save_predictions(self) -> None:
        """Save predictions to JSON file."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self._predictions, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_predictions(self) -> None:
        """Load predictions from JSON file."""
        try:
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self._predictions = json.load(f)
        except Exception:
            # Silent failure on load errors
            self._predictions = []
    
    def get_predictions(
        self,
        ticker: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_type: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> List[Dict]:
        """
        Get logged predictions, optionally filtered.
        
        Args:
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)
            model_type: Filter by model type (optional)
            outcome: Filter by outcome ("Correct", "Incorrect", "Partial", "Unknown") (optional)
            
        Returns:
            List of prediction dictionaries
        """
        results = self._predictions.copy()
        
        if ticker:
            results = [p for p in results if p.get("ticker") == ticker]
        
        if timeframe:
            results = [p for p in results if p.get("timeframe") == timeframe]
        
        if model_type:
            results = [p for p in results if p.get("model_type") == model_type]
        
        if outcome:
            results = [p for p in results if p.get("outcome") == outcome]
        
        return results
    
    def get_prediction_statistics(
        self,
        ticker: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Dict:
        """
        Get statistics on predictions.
        
        Args:
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)
            model_type: Filter by model type (optional)
            
        Returns:
            Dictionary with prediction statistics
        """
        predictions = self.get_predictions(ticker=ticker, timeframe=timeframe, model_type=model_type)
        
        if not predictions:
            return {
                "total_predictions": 0,
                "completed_predictions": 0,
                "accuracy": 0.0,
                "avg_error": 0.0,
                "avg_error_pct": 0.0
            }
        
        completed = [p for p in predictions if p.get("outcome") is not None]
        
        if not completed:
            return {
                "total_predictions": len(predictions),
                "completed_predictions": 0,
                "accuracy": 0.0,
                "avg_error": 0.0,
                "avg_error_pct": 0.0
            }
        
        correct = len([p for p in completed if p.get("outcome") == "Correct"])
        errors = [p.get("error") for p in completed if p.get("error") is not None]
        error_pcts = [p.get("error_pct") for p in completed if p.get("error_pct") is not None]
        
        return {
            "total_predictions": len(predictions),
            "completed_predictions": len(completed),
            "accuracy": correct / len(completed) if completed else 0.0,
            "correct_predictions": correct,
            "incorrect_predictions": len(completed) - correct,
            "avg_error": sum(errors) / len(errors) if errors else 0.0,
            "avg_error_pct": sum(error_pcts) / len(error_pcts) if error_pcts else 0.0,
            "min_error": min(errors) if errors else 0.0,
            "max_error": max(errors) if errors else 0.0
        }


# Global prediction logger instance
_prediction_logger: Optional[PredictionLogger] = None


def get_prediction_logger() -> PredictionLogger:
    """Get global prediction logger instance."""
    global _prediction_logger
    if _prediction_logger is None:
        _prediction_logger = PredictionLogger()
    return _prediction_logger

