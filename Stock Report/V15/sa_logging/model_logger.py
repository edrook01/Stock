"""
Model Performance Logger
Logs model training, evaluation, and performance metrics.
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


class ModelLogger:
    """Comprehensive model performance logger."""
    
    def __init__(self):
        """Initialize model logger."""
        self.history_dir = get_path('history')
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.history_dir / 'model_performance.csv'
        self.json_file = self.history_dir / 'model_performance.json'
        
        self._initialize_csv()
        self._performance_logs: List[Dict] = []
        self._load_performance_logs()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'EventType', 'ModelType', 'ModelVersion', 'Timeframe',
                    'TrainingSamples', 'ValidationSamples', 'TestSamples',
                    'TrainLoss', 'ValidationLoss', 'TestLoss',
                    'TrainAccuracy', 'ValidationAccuracy', 'TestAccuracy',
                    'TrainMSE', 'ValidationMSE', 'TestMSE',
                    'TrainMAE', 'ValidationMAE', 'TestMAE',
                    'TrainR2', 'ValidationR2', 'TestR2',
                    'FeatureImportance', 'Hyperparameters', 'TrainingTime',
                    'Epochs', 'LearningRate', 'BatchSize', 'Notes'
                ])
    
    def log_training_event(
        self,
        event_type: str,
        model_type: str,
        model_version: Optional[str] = None,
        timeframe: Optional[str] = None,
        training_samples: Optional[int] = None,
        validation_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        train_loss: Optional[float] = None,
        validation_loss: Optional[float] = None,
        test_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
        test_accuracy: Optional[float] = None,
        train_mse: Optional[float] = None,
        validation_mse: Optional[float] = None,
        test_mse: Optional[float] = None,
        train_mae: Optional[float] = None,
        validation_mae: Optional[float] = None,
        test_mae: Optional[float] = None,
        train_r2: Optional[float] = None,
        validation_r2: Optional[float] = None,
        test_r2: Optional[float] = None,
        feature_importance: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
        training_time: Optional[float] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        notes: str = ""
    ) -> str:
        """
        Log a model training or evaluation event.
        
        Args:
            event_type: Type of event ('training_start', 'training_end', 'evaluation', 'retraining', etc.)
            model_type: Type of model (e.g., 'unified', 'random_forest', 'xgboost')
            model_version: Model version identifier (optional)
            timeframe: Prediction timeframe (optional)
            training_samples: Number of training samples (optional)
            validation_samples: Number of validation samples (optional)
            test_samples: Number of test samples (optional)
            train_loss: Training loss (optional)
            validation_loss: Validation loss (optional)
            test_loss: Test loss (optional)
            train_accuracy: Training accuracy (optional)
            validation_accuracy: Validation accuracy (optional)
            test_accuracy: Test accuracy (optional)
            train_mse: Training MSE (optional)
            validation_mse: Validation MSE (optional)
            test_mse: Test MSE (optional)
            train_mae: Training MAE (optional)
            validation_mae: Validation MAE (optional)
            test_mae: Test MAE (optional)
            train_r2: Training R² score (optional)
            validation_r2: Validation R² score (optional)
            test_r2: Test R² score (optional)
            feature_importance: Dictionary of feature importance scores (optional)
            hyperparameters: Dictionary of hyperparameters (optional)
            training_time: Training time in seconds (optional)
            epochs: Number of training epochs (optional)
            learning_rate: Learning rate used (optional)
            batch_size: Batch size used (optional)
            notes: Additional notes
            
        Returns:
            Log entry ID (timestamp-based)
        """
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now()
        
        log_data = {
            "log_id": log_id,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "model_type": model_type,
            "model_version": model_version,
            "timeframe": timeframe,
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "test_mse": test_mse,
            "train_mae": train_mae,
            "validation_mae": validation_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "validation_r2": validation_r2,
            "test_r2": test_r2,
            "feature_importance": feature_importance or {},
            "hyperparameters": hyperparameters or {},
            "training_time": training_time,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "notes": notes
        }
        
        self._performance_logs.append(log_data)
        self._save_performance_logs()
        
        # Append to CSV
        self._append_csv_row(log_data)
        
        return log_id
    
    def log_evaluation(
        self,
        model_type: str,
        metrics: Dict[str, Any],
        model_version: Optional[str] = None,
        timeframe: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """
        Log model evaluation metrics.
        
        Args:
            model_type: Type of model
            metrics: Dictionary of evaluation metrics
            model_version: Model version identifier (optional)
            timeframe: Prediction timeframe (optional)
            notes: Additional notes
            
        Returns:
            Log entry ID
        """
        return self.log_training_event(
            event_type="evaluation",
            model_type=model_type,
            model_version=model_version,
            timeframe=timeframe,
            train_loss=metrics.get("train_loss"),
            validation_loss=metrics.get("validation_loss"),
            test_loss=metrics.get("test_loss"),
            train_accuracy=metrics.get("train_accuracy"),
            validation_accuracy=metrics.get("validation_accuracy"),
            test_accuracy=metrics.get("test_accuracy"),
            train_mse=metrics.get("train_mse"),
            validation_mse=metrics.get("validation_mse"),
            test_mse=metrics.get("test_mse"),
            train_mae=metrics.get("train_mae"),
            validation_mae=metrics.get("validation_mae"),
            test_mae=metrics.get("test_mae"),
            train_r2=metrics.get("train_r2"),
            validation_r2=metrics.get("validation_r2"),
            test_r2=metrics.get("test_r2"),
            feature_importance=metrics.get("feature_importance"),
            notes=notes
        )
    
    def _append_csv_row(self, log_data: Dict) -> None:
        """Append a log row to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Convert complex fields to JSON strings for CSV
                feature_importance_json = json.dumps(log_data.get("feature_importance", {}))
                hyperparameters_json = json.dumps(log_data.get("hyperparameters", {}))
                
                writer.writerow([
                    log_data.get("timestamp", ""),
                    log_data.get("event_type", ""),
                    log_data.get("model_type", ""),
                    log_data.get("model_version", ""),
                    log_data.get("timeframe", ""),
                    log_data.get("training_samples", ""),
                    log_data.get("validation_samples", ""),
                    log_data.get("test_samples", ""),
                    log_data.get("train_loss", ""),
                    log_data.get("validation_loss", ""),
                    log_data.get("test_loss", ""),
                    log_data.get("train_accuracy", ""),
                    log_data.get("validation_accuracy", ""),
                    log_data.get("test_accuracy", ""),
                    log_data.get("train_mse", ""),
                    log_data.get("validation_mse", ""),
                    log_data.get("test_mse", ""),
                    log_data.get("train_mae", ""),
                    log_data.get("validation_mae", ""),
                    log_data.get("test_mae", ""),
                    log_data.get("train_r2", ""),
                    log_data.get("validation_r2", ""),
                    log_data.get("test_r2", ""),
                    feature_importance_json,
                    hyperparameters_json,
                    log_data.get("training_time", ""),
                    log_data.get("epochs", ""),
                    log_data.get("learning_rate", ""),
                    log_data.get("batch_size", ""),
                    log_data.get("notes", "")
                ])
        except Exception:
            # Silent failure on CSV write errors
            pass
    
    def _save_performance_logs(self) -> None:
        """Save performance logs to JSON file."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self._performance_logs, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_performance_logs(self) -> None:
        """Load performance logs from JSON file."""
        try:
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self._performance_logs = json.load(f)
        except Exception:
            # Silent failure on load errors
            self._performance_logs = []
    
    def get_performance_logs(
        self,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get logged performance data, optionally filtered.
        
        Args:
            model_type: Filter by model type (optional)
            timeframe: Filter by timeframe (optional)
            event_type: Filter by event type (optional)
            
        Returns:
            List of performance log dictionaries
        """
        results = self._performance_logs.copy()
        
        if model_type:
            results = [log for log in results if log.get("model_type") == model_type]
        
        if timeframe:
            results = [log for log in results if log.get("timeframe") == timeframe]
        
        if event_type:
            results = [log for log in results if log.get("event_type") == event_type]
        
        return results
    
    def get_model_statistics(
        self,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict:
        """
        Get statistics on model performance.
        
        Args:
            model_type: Filter by model type (optional)
            timeframe: Filter by timeframe (optional)
            
        Returns:
            Dictionary with model statistics
        """
        logs = self.get_performance_logs(model_type=model_type, timeframe=timeframe)
        evaluations = [log for log in logs if log.get("event_type") == "evaluation"]
        
        if not evaluations:
            return {
                "total_events": len(logs),
                "evaluations": 0,
                "avg_accuracy": 0.0,
                "avg_loss": 0.0,
                "best_accuracy": 0.0,
                "best_loss": float('inf')
            }
        
        accuracies = [log.get("test_accuracy") for log in evaluations if log.get("test_accuracy") is not None]
        losses = [log.get("test_loss") for log in evaluations if log.get("test_loss") is not None]
        
        return {
            "total_events": len(logs),
            "evaluations": len(evaluations),
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "best_loss": min(losses) if losses else 0.0,
            "worst_accuracy": min(accuracies) if accuracies else 0.0,
            "worst_loss": max(losses) if losses else 0.0
        }


# Global model logger instance
_model_logger: Optional[ModelLogger] = None


def get_model_logger() -> ModelLogger:
    """Get global model logger instance."""
    global _model_logger
    if _model_logger is None:
        _model_logger = ModelLogger()
    return _model_logger

