#!/usr/bin/env python3
"""
Unified Deep Learning Engine - Self-Contained Module
Integrates all components and wraps existing engines.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import datetime

# Import other modules (will be available at runtime)
try:
    from deep_learning_models import get_model_selector, prepare_features, predict_with_model
    from rl_agent import get_rl_agent
    from genetic_optimizer import get_genetic_optimizer
    from meta_learner import get_meta_learner
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


class UnifiedDeepEngine:
    """Unified engine that integrates all deep learning components."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.model_selector = None
        self.rl_agent = None
        self.genetic_optimizer = None
        self.meta_learner = None
        
        if MODULES_AVAILABLE:
            try:
                self.model_selector = get_model_selector()
                self.rl_agent = get_rl_agent(use_gpu=use_gpu)
                self.genetic_optimizer = get_genetic_optimizer()
                self.meta_learner = get_meta_learner()
            except Exception:
                pass
    
    def predict(self, df, engine_results: List, current_price: float) -> Dict[str, Any]:
        """Make unified prediction using all components."""
        if not MODULES_AVAILABLE or not self.model_selector:
            return self._fallback_prediction(engine_results, current_price)
        
        try:
            # Prepare features
            features = prepare_features(df, use_gpu=self.use_gpu)
            
            # Select and create model
            data_shape = features.shape if features is not None else (20, 10)
            model_type = self.model_selector.select_best_model(data_shape)
            model = self.model_selector.create_model(model_type, input_size=features.shape[-1] if features is not None else 10, use_gpu=self.use_gpu)
            
            # Get predictions from model
            if features is not None:
                high_pred, low_pred = predict_with_model(model, features)
            else:
                high_pred, low_pred = current_price * 1.02, current_price * 0.98
            
            # Get RL-optimized weights
            state = self._create_state(engine_results, df)
            rl_weights = self.rl_agent.get_engine_weights(state) if self.rl_agent else (0.33, 0.33, 0.34)
            
            # Adapt weights using meta-learner
            if self.meta_learner:
                rl_weights = self.meta_learner.adapt_strategy(df, rl_weights)
            
            # Combine engine results with RL weights
            weighted_prediction = self._weighted_combination(engine_results, rl_weights)
            
            # Blend model prediction with weighted engine prediction
            final_high = 0.6 * high_pred + 0.4 * weighted_prediction
            final_low = 0.6 * low_pred + 0.4 * weighted_prediction
            
            # Ensure high >= low
            if final_high < final_low:
                final_high, final_low = final_low, final_high
            
            # Calculate confidence
            confidence = self._calculate_confidence(engine_results, model_type)
            
            return {
                'high_prediction': float(final_high),
                'low_prediction': float(final_low),
                'confidence': float(confidence),
                'model_type': model_type,
                'weights': rl_weights,
                'explanation': f"Unified prediction using {model_type} model with RL-optimized weights"
            }
        
        except Exception as e:
            return self._fallback_prediction(engine_results, current_price)
    
    def _create_state(self, engine_results: List, df) -> np.ndarray:
        """Create state vector for RL agent."""
        try:
            state_features = []
            
            # Engine predictions
            for result in engine_results:
                state_features.append(result.prediction / 100.0)  # Normalize
                state_features.append(result.confidence / 10.0)  # Normalize
            
            # Market features
            if hasattr(df, 'iloc') and len(df) > 0:
                latest = df.iloc[-1]
                if 'RSI14' in df.columns:
                    state_features.append(latest['RSI14'] / 100.0)
                if 'MACD' in df.columns:
                    state_features.append(latest['MACD'] / 100.0 if pd.notna(latest['MACD']) else 0.0)
            
            # Pad or trim to fixed size
            target_size = 50
            while len(state_features) < target_size:
                state_features.append(0.0)
            state_features = state_features[:target_size]
            
            return np.array(state_features, dtype=np.float32)
        except Exception:
            return np.zeros(50, dtype=np.float32)
    
    def _weighted_combination(self, engine_results: List, weights: Tuple[float, float, float]) -> float:
        """Combine engine results using weights."""
        if not engine_results:
            return 0.0
        
        # Map engines to weights
        engine_map = {'Statistical': weights[0], 'Technical': weights[1], 'ML': weights[2]}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in engine_results:
            weight = engine_map.get(result.engine, 0.33)
            weighted_sum += result.prediction * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, engine_results: List, model_type: str) -> float:
        """Calculate overall confidence."""
        if not engine_results:
            return 5.0
        
        # Average engine confidence
        avg_confidence = sum(r.confidence for r in engine_results) / len(engine_results)
        
        # Boost for model type performance
        model_boost = 0.5 if model_type == 'transformer' else 0.3
        
        confidence = min(10.0, avg_confidence + model_boost)
        return confidence
    
    def _fallback_prediction(self, engine_results: List, current_price: float) -> Dict[str, Any]:
        """Fallback prediction when modules unavailable."""
        if engine_results:
            avg_pred = sum(r.prediction for r in engine_results) / len(engine_results)
            avg_conf = sum(r.confidence for r in engine_results) / len(engine_results)
            high = current_price * (1 + abs(avg_pred) / 100.0)
            low = current_price * (1 - abs(avg_pred) / 100.0)
        else:
            high = current_price * 1.02
            low = current_price * 0.98
            avg_conf = 5.0
        
        return {
            'high_prediction': float(high),
            'low_prediction': float(low),
            'confidence': float(avg_conf),
            'model_type': 'fallback',
            'weights': (0.33, 0.33, 0.34),
            'explanation': 'Fallback prediction using engine averages'
        }
    
    def update_learning(self, prediction_result: Dict, actual_high: float, actual_low: float):
        """Update learning components with actual results."""
        if not MODULES_AVAILABLE:
            return
        
        try:
            # Calculate accuracy
            high_error = abs(prediction_result['high_prediction'] - actual_high) / actual_high
            low_error = abs(prediction_result['low_prediction'] - actual_low) / actual_low
            accuracy = max(0, 10 - (high_error + low_error) * 50)
            
            # Update RL agent
            if self.rl_agent:
                # Create reward
                reward = self.rl_agent.calculate_reward(accuracy)
                # Note: Full RL update requires state/action tracking
            
            # Update meta-learner
            if self.meta_learner:
                regime = self.meta_learner.regime_detector.detect_regime(
                    np.array([]), np.array([])  # Would need actual data
                )
                self.meta_learner.learn_strategy_for_regime(
                    regime,
                    prediction_result['weights'],
                    accuracy
                )
        except Exception:
            pass


# Global unified engine instance
_unified_engine = None


def get_unified_engine(use_gpu: bool = True) -> UnifiedDeepEngine:
    """Get or create global unified engine instance."""
    global _unified_engine
    if _unified_engine is None:
        _unified_engine = UnifiedDeepEngine(use_gpu=use_gpu)
    return _unified_engine
