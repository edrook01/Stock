"""
Feature Engineering from V13 Engines
Extracts features from V13 statistical/technical engines as input for unified ML model.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Handle both relative and absolute imports for portability
try:
    from ..core.indicators import rsi, sma, ema
    from ..core.data_fetcher import fetch_prices
except ImportError:
    # Fallback for direct execution
    from core.indicators import rsi, sma, ema
    from core.data_fetcher import fetch_prices
import asyncio


class FeatureExtractor:
    """Extracts features from V13 engines for ML model input."""
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    async def extract_features(
        self,
        ticker: str,
        interval: str,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Extract features from V13 engines and price data.
        
        Args:
            ticker: Stock ticker symbol
            interval: Time interval
            df: Price DataFrame (optional, will fetch if not provided)
            
        Returns:
            Dictionary of extracted features
        """
        # Fetch data if not provided
        if df is None or df.empty:
            df = await fetch_prices(ticker, interval)
            if df is None or df.empty:
                return {}
        
        features = {}
        
        # Price action features
        features.update(self._extract_price_features(df))
        
        # Technical indicator features
        features.update(self._extract_technical_features(df))
        
        # Volume features
        features.update(self._extract_volume_features(df))
        
        # Volatility features
        features.update(self._extract_volatility_features(df))
        
        # Momentum features
        features.update(self._extract_momentum_features(df))
        
        # Pattern features (basic)
        features.update(self._extract_pattern_features(df))
        
        return features
    
    def _extract_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract price action features."""
        if len(df) < 2:
            return {}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        features = {
            'price_current': float(close.iloc[-1]),
            'price_change_1': float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(df) >= 2 else 0.0,
            'price_change_5': float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(df) >= 6 else 0.0,
            'price_change_20': float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(df) >= 21 else 0.0,
            'high_low_range': float((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] * 100) if len(df) > 0 else 0.0,
            'high_low_range_5': float((high.iloc[-5:].max() - low.iloc[-5:].min()) / close.iloc[-1] * 100) if len(df) >= 5 else 0.0,
        }
        
        return features
    
    def _extract_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features."""
        if len(df) < 20:
            return {}
        
        close = df['Close']
        
        features = {}
        
        try:
            features['rsi_14'] = rsi(close, period=14)
        except Exception:
            features['rsi_14'] = 50.0
        
        try:
            features['sma_20'] = sma(close, period=20)
            features['sma_50'] = sma(close, period=50) if len(df) >= 50 else features['sma_20']
        except Exception:
            features['sma_20'] = float(close.iloc[-1])
            features['sma_50'] = float(close.iloc[-1])
        
        try:
            features['ema_20'] = ema(close, period=20)
            features['ema_50'] = ema(close, period=50) if len(df) >= 50 else features['ema_20']
        except Exception:
            features['ema_20'] = float(close.iloc[-1])
            features['ema_50'] = float(close.iloc[-1])
        
        # Price vs moving averages
        current_price = float(close.iloc[-1])
        features['price_vs_sma20'] = (current_price / features['sma_20'] - 1) * 100
        features['price_vs_sma50'] = (current_price / features['sma_50'] - 1) * 100
        features['price_vs_ema20'] = (current_price / features['ema_20'] - 1) * 100
        features['price_vs_ema50'] = (current_price / features['ema_50'] - 1) * 100
        
        # Moving average crossovers
        features['sma_cross'] = 1.0 if features['sma_20'] > features['sma_50'] else -1.0
        features['ema_cross'] = 1.0 if features['ema_20'] > features['ema_50'] else -1.0
        
        return features
    
    def _extract_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volume features."""
        if 'Volume' not in df.columns or len(df) < 5:
            return {}
        
        volume = df['Volume']
        
        features = {
            'volume_current': float(volume.iloc[-1]),
            'volume_avg_5': float(volume.iloc[-5:].mean()),
            'volume_avg_20': float(volume.iloc[-20:].mean()) if len(df) >= 20 else float(volume.iloc[-5:].mean()),
            'volume_ratio': float(volume.iloc[-1] / volume.iloc[-5:].mean()) if len(df) >= 5 else 1.0,
        }
        
        return features
    
    def _extract_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility features."""
        if len(df) < 14:
            return {}
        
        close = df['Close']
        
        # Calculate returns
        returns = close.pct_change().dropna()
        
        if len(returns) < 14:
            return {}
        
        features = {
            'volatility_14': float(returns.iloc[-14:].std() * 100),
            'volatility_20': float(returns.iloc[-20:].std() * 100) if len(returns) >= 20 else float(returns.iloc[-14:].std() * 100),
        }
        
        return features
    
    def _extract_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum features."""
        if len(df) < 10:
            return {}
        
        close = df['Close']
        
        features = {
            'momentum_5': float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(df) >= 6 else 0.0,
            'momentum_10': float((close.iloc[-1] / close.iloc[-11] - 1) * 100) if len(df) >= 11 else 0.0,
        }
        
        return features
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract basic pattern features."""
        if len(df) < 3:
            return {}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Simple pattern detection
        features = {
            'higher_high': 1.0 if len(df) >= 2 and high.iloc[-1] > high.iloc[-2] else 0.0,
            'lower_low': 1.0 if len(df) >= 2 and low.iloc[-1] < low.iloc[-2] else 0.0,
            'three_up': 1.0 if len(df) >= 3 and all(close.iloc[-i] > close.iloc[-i-1] for i in range(1, min(4, len(df)))) else 0.0,
            'three_down': 1.0 if len(df) >= 3 and all(close.iloc[-i] < close.iloc[-i-1] for i in range(1, min(4, len(df)))) else 0.0,
        }
        
        return features
    
    def normalize_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Normalize features into a numpy array for model input.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Normalized feature vector as numpy array
        """
        # Define feature order (important for model consistency)
        feature_order = [
            'price_current', 'price_change_1', 'price_change_5', 'price_change_20',
            'high_low_range', 'high_low_range_5',
            'rsi_14', 'sma_20', 'sma_50', 'ema_20', 'ema_50',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20', 'price_vs_ema50',
            'sma_cross', 'ema_cross',
            'volume_current', 'volume_avg_5', 'volume_avg_20', 'volume_ratio',
            'volatility_14', 'volatility_20',
            'momentum_5', 'momentum_10',
            'higher_high', 'lower_low', 'three_up', 'three_down'
        ]
        
        # Extract values in order, fill missing with 0
        feature_vector = []
        for feat_name in feature_order:
            value = features.get(feat_name, 0.0)
            if isinstance(value, (int, float)):
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float32)

