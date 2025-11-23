"""
Model Training Infrastructure
Training pipeline for unified ML models.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .unified_model import UnifiedModel, get_model
from .feature_extractor import FeatureExtractor
from ..core.portable_paths import get_path
from ..learning.trade_tracker import get_trade_tracker


class ModelTrainer:
    """Trains unified ML models using historical trade data."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.feature_extractor = FeatureExtractor()
        self.trade_tracker = get_trade_tracker()
    
    async def prepare_training_data(
        self,
        timeframe: str,
        min_samples: int = 50
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare training data from historical trades.
        
        Args:
            timeframe: Prediction timeframe
            min_samples: Minimum number of samples required
            
        Returns:
            Tuple of (X, y) arrays, or None if insufficient data
        """
        from ..core.data_fetcher import fetch_prices
        import asyncio
        
        # Get trade outcomes for this timeframe
        outcomes = self.trade_tracker.get_outcomes()
        timeframe_outcomes = [o for o in outcomes if o.timeframe == timeframe]
        
        if len(timeframe_outcomes) < min_samples:
            return None
        
        # Prepare feature vectors and targets
        X_list = []
        y_list = []
        
        # Process each trade outcome
        for outcome in timeframe_outcomes:
            try:
                # Fetch historical price data at entry time
                # Get data up to entry time (need to fetch more and slice)
                df = await fetch_prices(outcome.ticker, timeframe)
                
                if df is None or df.empty:
                    continue
                
                # Find the entry time in the dataframe
                entry_idx = None
                for idx, timestamp in enumerate(df.index):
                    if timestamp >= outcome.entry_time:
                        entry_idx = idx
                        break
                
                if entry_idx is None or entry_idx < 50:  # Need enough history
                    continue
                
                # Get data up to entry point (for feature extraction)
                df_at_entry = df.iloc[:entry_idx + 1]
                
                if len(df_at_entry) < 50:  # Need minimum data for indicators
                    continue
                
                # Extract features using feature_extractor
                features = await self.feature_extractor.extract_features(
                    outcome.ticker,
                    timeframe,
                    df=df_at_entry
                )
                
                if not features:
                    continue
                
                # Convert features dict to array (flatten)
                feature_vector = self._features_to_array(features)
                
                if feature_vector is None:
                    continue
                
                # Calculate actual outcome (price movement percentage)
                # Use exit price vs entry price
                if outcome.actual_outcome is not None:
                    actual_movement = outcome.actual_outcome
                elif outcome.exit_price and outcome.entry_price:
                    # Calculate percentage change
                    if outcome.direction == "LONG":
                        actual_movement = ((outcome.exit_price - outcome.entry_price) / outcome.entry_price) * 100
                    else:  # SHORT
                        actual_movement = ((outcome.entry_price - outcome.exit_price) / outcome.entry_price) * 100
                else:
                    continue  # Skip if no outcome data
                
                X_list.append(feature_vector)
                y_list.append(actual_movement)
                
            except Exception as e:
                # Skip this trade if there's an error
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error preparing training data for trade {outcome.trade_id}: {e}")
                continue
        
        if len(X_list) < min_samples:
            return None
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        return (X, y)
    
    def _features_to_array(self, features: Dict) -> Optional[np.ndarray]:
        """
        Convert features dictionary to numpy array.
        
        Args:
            features: Features dictionary
            
        Returns:
            Numpy array of features, or None if conversion fails
        """
        try:
            feature_list = []
            
            # Price action features
            price_features = features.get("price_action", {})
            feature_list.extend([
                price_features.get("price_change_pct", 0.0),
                price_features.get("high_low_range", 0.0),
                price_features.get("body_size", 0.0),
                price_features.get("upper_shadow", 0.0),
                price_features.get("lower_shadow", 0.0)
            ])
            
            # Technical indicators
            indicators = features.get("indicators", {})
            feature_list.extend([
                indicators.get("rsi", 50.0),
                indicators.get("sma_20", 0.0),
                indicators.get("sma_50", 0.0),
                indicators.get("ema_20", 0.0),
                indicators.get("ema_50", 0.0)
            ])
            
            # Volume features
            volume_features = features.get("volume", {})
            feature_list.extend([
                volume_features.get("volume_ratio", 1.0),
                volume_features.get("volume_trend", 0.0)
            ])
            
            # Volatility features
            volatility_features = features.get("volatility", {})
            feature_list.extend([
                volatility_features.get("atr", 0.0),
                volatility_features.get("atr_pct", 0.0)
            ])
            
            # Momentum features
            momentum_features = features.get("momentum", {})
            feature_list.extend([
                momentum_features.get("momentum_score", 0.0),
                momentum_features.get("trend_strength", 0.0)
            ])
            
            return np.array(feature_list)
        
        except Exception:
            return None
    
    async def train_model(
        self,
        timeframe: str,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train model for a timeframe.
        
        Args:
            timeframe: Prediction timeframe
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        model = get_model(timeframe)
        
        # Prepare training data (async)
        training_data = await self.prepare_training_data(timeframe)
        if training_data is None:
            return {
                "error": "Insufficient training data",
                "timeframe": timeframe,
                "message": "Need at least 50 completed trades with outcome data"
            }
        
        X, y = training_data
        
        # Train model
        metrics = model.train(X, y, validation_split=validation_split)
        
        # Save trained model
        model.save()
        
        return {
            "timeframe": timeframe,
            "training_samples": len(X),
            "metrics": metrics,
            "trained_at": datetime.now().isoformat()
        }
    
    async def train_all_models(self) -> Dict:
        """
        Train models for all timeframes.
        
        Returns:
            Dictionary with training results for each timeframe
        """
        from ..core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES
        
        results = {}
        
        all_timeframes = CFD_TIMEFRAMES + INVESTMENT_TIMEFRAMES
        
        for timeframe in all_timeframes:
            try:
                result = await self.train_model(timeframe)
                results[timeframe] = result
            except Exception as e:
                results[timeframe] = {
                    "error": str(e),
                    "timeframe": timeframe
                }
        
        return results


# Global model trainer instance
_model_trainer: Optional[ModelTrainer] = None


def get_model_trainer() -> ModelTrainer:
    """Get global model trainer instance."""
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = ModelTrainer()
    return _model_trainer

