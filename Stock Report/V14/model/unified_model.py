"""
Unified ML Model Architecture
Ensemble model for both investment and CFD strategies using V13 engine outputs as features.
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path
import pickle
import json

# Handle both relative and absolute imports for portability
try:
    from .feature_extractor import FeatureExtractor
    from ..core.portable_paths import get_path
    from ..core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES
except ImportError:
    # Fallback for direct execution
    from model.feature_extractor import FeatureExtractor
    from core.portable_paths import get_path
    from core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    GradientBoostingRegressor = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


class UnifiedModel:
    """
    Unified ML model that uses V13 engine outputs as features.
    Supports both investment and CFD timeframes.
    """
    
    def __init__(self, timeframe: str = "1d"):
        """
        Initialize unified model for a specific timeframe.
        
        Args:
            timeframe: Prediction timeframe (1m, 5m, 1d, etc.)
        """
        self.timeframe = timeframe
        self.feature_extractor = FeatureExtractor()
        self.models: Dict[str, any] = {}
        self.is_trained = False
        self.confidence_calibration: Dict[str, float] = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize ensemble models."""
        if SKLEARN_AVAILABLE:
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
    
    async def predict(
        self,
        ticker: str,
        df: Optional = None,
        v13_engine_outputs: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate prediction using unified model.
        
        Args:
            ticker: Stock ticker symbol
            df: Price DataFrame (optional)
            v13_engine_outputs: V13 engine outputs as features (optional)
            
        Returns:
            Dictionary with:
            - prediction: float - Predicted price movement percentage
            - confidence: float - Confidence score (0-1)
            - range_low: float - Lower bound
            - range_high: float - Upper bound
            - timeframe: str - Prediction timeframe
        """
        # Extract features
        features = await self.feature_extractor.extract_features(
            ticker=ticker,
            interval=self.timeframe,
            df=df
        )
        
        if not features:
            return self._default_prediction()
        
        # Normalize features
        feature_vector = self.feature_extractor.normalize_features(features)
        
        # If not trained, return default prediction
        if not self.is_trained:
            return self._default_prediction()
        
        # Get predictions from all models
        predictions = []
        for model_name, model in self.models.items():
            try:
                pred = model.predict(feature_vector.reshape(1, -1))[0]
                predictions.append(float(pred))
            except Exception:
                continue
        
        if not predictions:
            return self._default_prediction()
        
        # Ensemble prediction (average)
        ensemble_prediction = np.mean(predictions)
        
        # Calculate confidence (based on model agreement)
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            # Lower std dev = higher agreement = higher confidence
            confidence = max(0.0, min(1.0, 1.0 - (std_dev / abs(ensemble_prediction + 0.01))))
        else:
            confidence = 0.5
        
        # Apply confidence calibration
        calibrated_confidence = self._calibrate_confidence(confidence)
        
        # Calculate range
        range_low = ensemble_prediction - abs(ensemble_prediction) * 0.2
        range_high = ensemble_prediction + abs(ensemble_prediction) * 0.2
        
        return {
            "prediction": float(ensemble_prediction),
            "confidence": float(calibrated_confidence),
            "range_low": float(range_low),
            "range_high": float(range_high),
            "timeframe": self.timeframe,
            "ticker": ticker,
            "model_agreement": float(1.0 - (np.std(predictions) / abs(ensemble_prediction + 0.01))) if len(predictions) > 1 else 0.5
        }
    
    def _default_prediction(self) -> Dict:
        """Return default prediction when model not trained."""
        return {
            "prediction": 0.0,
            "confidence": 0.5,
            "range_low": -1.0,
            "range_high": 1.0,
            "timeframe": self.timeframe,
            "ticker": "",
            "model_agreement": 0.0
        }
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Calibrate confidence score based on historical accuracy.
        
        Args:
            raw_confidence: Raw confidence from model
            
        Returns:
            Calibrated confidence (0-1)
        """
        # Simple calibration - in full implementation, use historical data
        calibration_factor = self.confidence_calibration.get(self.timeframe, 1.0)
        calibrated = raw_confidence * calibration_factor
        return max(0.0, min(1.0, calibrated))
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the unified model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training metrics
        """
        if not SKLEARN_AVAILABLE and not XGBOOST_AVAILABLE:
            return {"error": "No ML libraries available"}
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        metrics = {}
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                metrics[model_name] = {
                    "train_score": float(train_score),
                    "val_score": float(val_score)
                }
            except Exception as e:
                metrics[model_name] = {"error": str(e)}
        
        self.is_trained = True
        
        return metrics
    
    def save(self, model_name: str = "default") -> bool:
        """
        Save model to disk.
        
        Args:
            model_name: Name for the model
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            weights_dir = get_path('model_weights')
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = weights_dir / f"unified_model_{self.timeframe}_{model_name}.pkl"
            
            model_data = {
                "timeframe": self.timeframe,
                "models": {},
                "is_trained": self.is_trained,
                "confidence_calibration": self.confidence_calibration
            }
            
            # Save each model
            for model_name_key, model in self.models.items():
                try:
                    # Use pickle for sklearn models
                    model_bytes = pickle.dumps(model)
                    model_data["models"][model_name_key] = model_bytes.hex()
                except Exception:
                    continue
            
            # Save metadata
            metadata_file = weights_dir / f"unified_model_{self.timeframe}_{model_name}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    "timeframe": self.timeframe,
                    "is_trained": self.is_trained,
                    "confidence_calibration": self.confidence_calibration,
                    "model_names": list(self.models.keys())
                }, f, indent=2)
            
            # Save models
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception:
            return False
    
    def load(self, model_name: str = "default") -> bool:
        """
        Load model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            weights_dir = get_path('model_weights')
            model_file = weights_dir / f"unified_model_{self.timeframe}_{model_name}.pkl"
            
            if not model_file.exists():
                return False
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load models
            for model_name_key, model_hex in model_data.get("models", {}).items():
                try:
                    model_bytes = bytes.fromhex(model_hex)
                    model = pickle.loads(model_bytes)
                    self.models[model_name_key] = model
                except Exception:
                    continue
            
            self.is_trained = model_data.get("is_trained", False)
            self.confidence_calibration = model_data.get("confidence_calibration", {})
            
            return True
        except Exception:
            return False


# Global model instances by timeframe
_model_instances: Dict[str, UnifiedModel] = {}


def get_model(timeframe: str) -> UnifiedModel:
    """
    Get unified model instance for a timeframe.
    
    Args:
        timeframe: Prediction timeframe
        
    Returns:
        UnifiedModel instance
    """
    if timeframe not in _model_instances:
        _model_instances[timeframe] = UnifiedModel(timeframe=timeframe)
        # Try to load existing model
        _model_instances[timeframe].load()
    
    return _model_instances[timeframe]

