"""
ML Model Debugger
Debug utilities for unified ML model.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ..model.unified_model import get_model
from ..model.feature_extractor import FeatureExtractor
from ..model.confidence_calibrator import get_confidence_calibrator


class ModelDebugger:
    """Debug ML model."""
    
    def __init__(self):
        """Initialize model debugger."""
        self.feature_extractor = FeatureExtractor()
    
    async def debug_feature_extraction(
        self,
        ticker: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Debug feature extraction."""
        debug_info = {
            "test": "debug_feature_extraction",
            "timestamp": datetime.now().isoformat(),
            "input": {"ticker": ticker},
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            features = await self.feature_extractor.extract_features(ticker, "1d", df)
            debug_info["steps"].append({
                "step": 1,
                "action": "Extract features",
                "result": f"Extracted {len(features)} features"
            })
            
            feature_vector = self.feature_extractor.normalize_features(features)
            debug_info["steps"].append({
                "step": 2,
                "action": "Normalize features",
                "result": f"Vector shape: {feature_vector.shape}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Feature extraction error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        debug_info["output"] = {
            "feature_count": len(features),
            "vector_shape": list(feature_vector.shape)
        }
        debug_info["success"] = True
        return debug_info
    
    async def debug_model_prediction(
        self,
        ticker: str,
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """Debug model prediction."""
        debug_info = {
            "test": "debug_model_prediction",
            "timestamp": datetime.now().isoformat(),
            "input": {"ticker": ticker, "timeframe": timeframe},
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            model = get_model(timeframe)
            debug_info["steps"].append({
                "step": 1,
                "action": "Get model instance",
                "result": f"Trained: {model.is_trained}"
            })
            
            if model.is_trained:
                prediction = await model.predict(ticker)
                debug_info["steps"].append({
                    "step": 2,
                    "action": "Generate prediction",
                    "result": f"Prediction: {prediction.get('prediction', 0):.2f}%"
                })
            else:
                debug_info["warnings"].append("Model not trained - using default")
        except Exception as e:
            debug_info["errors"].append(f"Prediction error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        debug_info["success"] = True
        return debug_info
    
    def debug_confidence_calibration(self) -> Dict[str, Any]:
        """Debug confidence calibration."""
        debug_info = {
            "test": "debug_confidence_calibration",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        calibrator = get_confidence_calibrator()
        
        try:
            calibration_data = calibrator.get_calibration_data()
            debug_info["steps"].append({
                "step": 1,
                "action": "Get calibration data",
                "result": f"Data points: {len(calibration_data)}"
            })
            
            accuracy = calibrator.get_calibration_accuracy()
            debug_info["steps"].append({
                "step": 2,
                "action": "Get calibration accuracy",
                "result": f"Accuracy: {accuracy:.2%}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Calibration error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        debug_info["success"] = True
        return debug_info


def get_model_debugger() -> ModelDebugger:
    """Get global model debugger instance."""
    global _model_debugger
    if _model_debugger is None:
        _model_debugger = ModelDebugger()
    return _model_debugger

