"""
Prediction Debugger
Debug utilities for unified model predictions and feature extraction.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

# Handle pandas and numpy imports with error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..model.unified_model import get_model
from ..model.feature_extractor import FeatureExtractor
from ..core.data_fetcher import fetch_prices
import asyncio


class PredictionDebugger:
    """Debug prediction generation and feature extraction."""
    
    def __init__(self):
        """Initialize prediction debugger."""
        self.feature_extractor = FeatureExtractor()
    
    async def debug_prediction(
        self,
        ticker: str,
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Debug a prediction for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "ticker": ticker,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }
        
        # Step 1: Fetch data
        debug_info["steps"].append({"step": "1", "action": "Fetch price data"})
        try:
            df = await fetch_prices(ticker, timeframe)
            if df is None or df.empty:
                debug_info["error"] = "Failed to fetch price data"
                return debug_info
            debug_info["data_points"] = len(df)
            debug_info["data_columns"] = list(df.columns)
            debug_info["latest_price"] = float(df['Close'].iloc[-1])
        except Exception as e:
            debug_info["error"] = f"Data fetch error: {str(e)}"
            return debug_info
        
        # Step 2: Extract features
        debug_info["steps"].append({"step": "2", "action": "Extract features"})
        try:
            features = await self.feature_extractor.extract_features(
                ticker=ticker,
                interval=timeframe,
                df=df
            )
            debug_info["features"] = features
            debug_info["feature_count"] = len(features)
        except Exception as e:
            debug_info["error"] = f"Feature extraction error: {str(e)}"
            return debug_info
        
        # Step 3: Normalize features
        debug_info["steps"].append({"step": "3", "action": "Normalize features"})
        try:
            feature_vector = self.feature_extractor.normalize_features(features)
            debug_info["feature_vector_shape"] = feature_vector.shape
            debug_info["feature_vector_stats"] = {
                "min": float(np.min(feature_vector)),
                "max": float(np.max(feature_vector)),
                "mean": float(np.mean(feature_vector)),
                "std": float(np.std(feature_vector)),
                "has_nan": bool(np.isnan(feature_vector).any()),
                "has_inf": bool(np.isinf(feature_vector).any())
            }
        except Exception as e:
            debug_info["error"] = f"Feature normalization error: {str(e)}"
            return debug_info
        
        # Step 4: Get model prediction
        debug_info["steps"].append({"step": "4", "action": "Generate prediction"})
        try:
            model = get_model(timeframe)
            debug_info["model_trained"] = model.is_trained
            debug_info["model_libraries"] = list(model.models.keys())
            
            if model.is_trained:
                prediction = await model.predict(ticker, df=df)
                debug_info["prediction"] = prediction
            else:
                debug_info["warning"] = "Model not trained - using default prediction"
                debug_info["prediction"] = model._default_prediction()
        except Exception as e:
            debug_info["error"] = f"Prediction error: {str(e)}"
            return debug_info
        
        debug_info["success"] = True
        return debug_info
    
    def debug_feature_extraction(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Debug feature extraction process.
        
        Args:
            df: Price DataFrame
            
        Returns:
            Dictionary with feature extraction debug info
        """
        debug_info = {
            "input_data_shape": df.shape,
            "input_columns": list(df.columns),
            "features": {}
        }
        
        # Test each feature extraction method
        try:
            price_features = self.feature_extractor._extract_price_features(df)
            debug_info["features"]["price"] = price_features
        except Exception as e:
            debug_info["features"]["price"] = {"error": str(e)}
        
        try:
            technical_features = self.feature_extractor._extract_technical_features(df)
            debug_info["features"]["technical"] = technical_features
        except Exception as e:
            debug_info["features"]["technical"] = {"error": str(e)}
        
        try:
            volume_features = self.feature_extractor._extract_volume_features(df)
            debug_info["features"]["volume"] = volume_features
        except Exception as e:
            debug_info["features"]["volume"] = {"error": str(e)}
        
        try:
            volatility_features = self.feature_extractor._extract_volatility_features(df)
            debug_info["features"]["volatility"] = volatility_features
        except Exception as e:
            debug_info["features"]["volatility"] = {"error": str(e)}
        
        try:
            momentum_features = self.feature_extractor._extract_momentum_features(df)
            debug_info["features"]["momentum"] = momentum_features
        except Exception as e:
            debug_info["features"]["momentum"] = {"error": str(e)}
        
        try:
            pattern_features = self.feature_extractor._extract_pattern_features(df)
            debug_info["features"]["pattern"] = pattern_features
        except Exception as e:
            debug_info["features"]["pattern"] = {"error": str(e)}
        
        return debug_info
    
    def compare_features(
        self,
        features1: Dict,
        features2: Dict
    ) -> Dict[str, Any]:
        """
        Compare two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "features1_keys": set(features1.keys()),
            "features2_keys": set(features2.keys()),
            "common_keys": set(features1.keys()) & set(features2.keys()),
            "only_in_1": set(features1.keys()) - set(features2.keys()),
            "only_in_2": set(features2.keys()) - set(features1.keys()),
            "differences": {}
        }
        
        for key in comparison["common_keys"]:
            val1 = features1.get(key)
            val2 = features2.get(key)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                comparison["differences"][key] = {
                    "value1": val1,
                    "value2": val2,
                    "difference": diff,
                    "percent_diff": (diff / abs(val1 + 0.0001)) * 100
                }
        
        return comparison


# Global debugger instance
_prediction_debugger: Optional[PredictionDebugger] = None


def get_prediction_debugger() -> PredictionDebugger:
    """Get global prediction debugger instance."""
    global _prediction_debugger
    if _prediction_debugger is None:
        _prediction_debugger = PredictionDebugger()
    return _prediction_debugger

