#!/usr/bin/env python3
"""
Meta-Learning System - Self-Contained Module
Market regime detection and strategy adaptation.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None


class MarketRegimeDetector:
    """Detects market regimes: trending, mean-reverting, volatile."""
    
    def __init__(self):
        self.regime_history = []
    
    def detect_regime(self, returns: np.ndarray, prices: np.ndarray) -> str:
        """Detect current market regime."""
        if len(returns) < 20:
            return "unknown"
        
        # Calculate metrics
        volatility = np.std(returns)
        trend_strength = abs(np.mean(returns))
        autocorr = safe_corrcoef(returns[:-1].values, returns[1:].values) if len(returns) > 1 else 0
        
        # Regime classification
        if volatility > 0.03 and trend_strength > 0.01:
            return "volatile_trending"
        elif volatility > 0.03:
            return "volatile"
        elif trend_strength > 0.01 and autocorr > 0.3:
            return "trending"
        elif autocorr < -0.2:
            return "mean_reverting"
        else:
            return "neutral"
    
    def get_regime_features(self, df) -> Dict[str, float]:
        """Extract features for regime detection."""
        if pd is None or df is None or len(df) < 20:
            return {}
        
        try:
            returns = df['Close'].pct_change().dropna()
            return {
                'volatility': float(returns.std()),
                'trend_strength': float(abs(returns.mean())),
                'autocorr': safe_autocorr(returns, lag=1) if len(returns) > 1 else 0.0,
                'skewness': float(returns.skew()) if len(returns) > 2 else 0.0,
                'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0.0
            }
        except Exception:
            return {}


class MetaLearner:
    """Meta-learning system for strategy adaptation."""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.strategy_performance = defaultdict(list)  # {regime: [(strategy, accuracy)]}
        self.best_strategies = {}  # {regime: best_strategy}
    
    def learn_strategy_for_regime(self, regime: str, strategy: Tuple[float, float, float], accuracy: float):
        """Learn which strategy works best for each regime."""
        self.strategy_performance[regime].append((strategy, accuracy))
        # Keep only last 100 entries per regime
        if len(self.strategy_performance[regime]) > 100:
            self.strategy_performance[regime] = self.strategy_performance[regime][-100:]
        
        # Update best strategy for regime
        if regime not in self.best_strategies:
            self.best_strategies[regime] = strategy
        else:
            # Compare with current best
            current_best_avg = np.mean([acc for _, acc in self.strategy_performance[regime] 
                                      if _ == self.best_strategies[regime]])
            new_avg = np.mean([acc for _, acc in self.strategy_performance[regime] 
                              if _ == strategy])
            if new_avg > current_best_avg:
                self.best_strategies[regime] = strategy
    
    def get_optimal_strategy(self, regime: str) -> Tuple[float, float, float]:
        """Get optimal strategy for current regime."""
        if regime in self.best_strategies:
            return self.best_strategies[regime]
        # Default balanced strategy
        return (0.33, 0.33, 0.34)
    
    def adapt_strategy(self, df, current_strategy: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Adapt strategy based on detected regime."""
        regime = self.regime_detector.detect_regime(
            df['Close'].pct_change().dropna().values if pd is not None and 'Close' in df.columns else np.array([]),
            df['Close'].values if pd is not None and 'Close' in df.columns else np.array([])
        )
        
        optimal = self.get_optimal_strategy(regime)
        # Blend current and optimal (70% optimal, 30% current)
        adapted = tuple(0.7 * np.array(optimal) + 0.3 * np.array(current_strategy))
        # Normalize
        adapted = tuple(adapted / np.sum(adapted))
        return adapted
    
    def save(self, filepath: str):
        """Save meta-learner state."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'strategy_performance': dict(self.strategy_performance),
                    'best_strategies': self.best_strategies
                }, f)
        except Exception:
            pass
    
    def load(self, filepath: str):
        """Load meta-learner state."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.strategy_performance = defaultdict(list, data.get('strategy_performance', {}))
                    self.best_strategies = data.get('best_strategies', {})
        except Exception:
            pass


# Global meta-learner instance
_meta_learner = None


def get_meta_learner() -> MetaLearner:
    """Get or create global meta-learner instance."""
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
        # Try to load saved state
        learner_path = os.path.join(os.path.dirname(__file__), "..", "memory", "meta_learner.pkl")
        _meta_learner.load(learner_path)
    return _meta_learner
