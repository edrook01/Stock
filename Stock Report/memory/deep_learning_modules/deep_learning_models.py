#!/usr/bin/env python3
"""
Deep Learning Models - Self-Contained Module
LSTM, Transformer, and Hybrid model implementations with auto-selection.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

# Try to import deep learning frameworks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    # Check for CUDA/GPU support
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None
    # Fallback definitions
    class nn:
        class Module:
            pass
        class LSTM:
            pass
        class Linear:
            pass
        class Transformer:
            pass
        class Conv1d:
            pass

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import pandas as pd
except ImportError:
    pd = None

# Model storage
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "memory", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """LSTM model for time series prediction with GPU support."""
    
    def __init__(self, input_size: int = 50, hidden_size: int = 128, num_layers: int = 2, output_size: int = 2, use_gpu: bool = True):
        if TORCH_AVAILABLE:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.use_gpu = use_gpu and CUDA_AVAILABLE
            if self.use_gpu:
                self.to(DEVICE)
        else:
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.model = None
            self.use_gpu = False
    
    def forward(self, x):
        """Forward pass."""
        if not TORCH_AVAILABLE:
            return np.array([[0.0, 0.0]])  # Fallback
        
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output
    
    def predict(self, x):
        """Make prediction with GPU support."""
        if not TORCH_AVAILABLE or self.model is None:
            return np.array([0.0, 0.0])  # Fallback: [high, low]
        
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            if self.use_gpu:
                x = x.to(DEVICE)
            pred = self.forward(x)
            if self.use_gpu:
                pred = pred.cpu()
            return pred.numpy()[0]


class TransformerModel(nn.Module if TORCH_AVAILABLE else object):
    """Transformer model for capturing long-range dependencies with GPU support."""
    
    def __init__(self, input_size: int = 50, d_model: int = 128, nhead: int = 8, num_layers: int = 4, output_size: int = 2, use_gpu: bool = True):
        if TORCH_AVAILABLE:
            super().__init__()
            self.input_projection = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, output_size)
            self.use_gpu = use_gpu and CUDA_AVAILABLE
            if self.use_gpu:
                self.to(DEVICE)
        else:
            self.model = None
            self.use_gpu = False
    
    def forward(self, x):
        """Forward pass."""
        if not TORCH_AVAILABLE:
            return np.array([[0.0, 0.0]])  # Fallback
        
        x = self.input_projection(x)
        x = self.transformer(x)
        # Take the last output
        last_output = x[:, -1, :]
        output = self.fc(last_output)
        return output
    
    def predict(self, x):
        """Make prediction."""
        if not TORCH_AVAILABLE or self.model is None:
            return np.array([0.0, 0.0])  # Fallback
        
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            pred = self.forward(x)
            return pred.numpy()[0]


class HybridModel(nn.Module if TORCH_AVAILABLE else object):
    """Hybrid model combining LSTM and CNN with GPU support."""
    
    def __init__(self, input_size: int = 50, lstm_hidden: int = 64, cnn_channels: int = 32, output_size: int = 2, use_gpu: bool = True):
        if TORCH_AVAILABLE:
            super().__init__()
            self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
            self.conv1d = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
            self.fc = nn.Linear(lstm_hidden + cnn_channels, output_size)
            self.use_gpu = use_gpu and CUDA_AVAILABLE
            if self.use_gpu:
                self.to(DEVICE)
        else:
            self.model = None
            self.use_gpu = False
    
    def forward(self, x):
        """Forward pass."""
        if not TORCH_AVAILABLE:
            return np.array([[0.0, 0.0]])  # Fallback
        
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        
        # CNN branch
        x_permuted = x.permute(0, 2, 1)  # (batch, features, time)
        cnn_out = self.conv1d(x_permuted)
        cnn_last = cnn_out[:, :, -1]  # Take last time step
        
        # Combine
        combined = torch.cat([lstm_last, cnn_last], dim=1)
        output = self.fc(combined)
        return output
    
    def predict(self, x):
        """Make prediction."""
        if not TORCH_AVAILABLE or self.model is None:
            return np.array([0.0, 0.0])  # Fallback
        
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            pred = self.forward(x)
            return pred.numpy()[0]


class ModelSelector:
    """Auto-selects best model architecture based on performance."""
    
    def __init__(self):
        self.models = {
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            'hybrid': HybridModel
        }
        self.performance_history = {}  # {model_name: [accuracy_scores]}
        self.current_model = None
        self.current_model_type = None
    
    def select_best_model(self, data_shape: Tuple, historical_performance: Optional[Dict] = None) -> str:
        """Select best model based on data and historical performance."""
        if historical_performance:
            # Use historical performance
            best_model = max(historical_performance.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
            return best_model[0] if best_model[1] else 'lstm'
        
        # Default selection based on data characteristics
        seq_length, feature_count = data_shape[0], data_shape[1] if len(data_shape) > 1 else data_shape[0]
        
        if seq_length > 100 and feature_count > 20:
            return 'transformer'  # Long sequences, many features
        elif feature_count > 30:
            return 'hybrid'  # Many features benefit from CNN
        else:
            return 'lstm'  # Default
    
    def create_model(self, model_type: str, input_size: int = 50, use_gpu: bool = True, **kwargs):
        """Create a model instance with GPU support."""
        if model_type not in self.models:
            model_type = 'lstm'  # Fallback
        
        model_class = self.models[model_type]
        model = model_class(input_size=input_size, use_gpu=use_gpu, **kwargs)
        self.current_model = model
        self.current_model_type = model_type
        return model
    
    def update_performance(self, model_type: str, accuracy: float):
        """Update performance history for a model."""
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []
        self.performance_history[model_type].append(accuracy)
        # Keep only last 100 scores
        if len(self.performance_history[model_type]) > 100:
            self.performance_history[model_type] = self.performance_history[model_type][-100:]
    
    def get_best_model_type(self) -> str:
        """Get the best performing model type."""
        if not self.performance_history:
            return 'lstm'
        
        avg_performances = {
            model_type: np.mean(scores) if scores else 0
            for model_type, scores in self.performance_history.items()
        }
        return max(avg_performances.items(), key=lambda x: x[1])[0]


def prepare_features(df, use_gpu: bool = False) -> Optional[np.ndarray]:
    """Prepare features from dataframe for model input with optional GPU acceleration."""
    if pd is None or df is None or len(df) < 20:
        return None
    
    try:
        feature_cols = [
            'SMA20', 'SMA50', 'RSI14', 'MACD', 'ATR14', 'OBV',
            'Dist_SMA20', 'Volatility20', 'BB_WIDTH',
            'Kurtosis20', 'Autocorr1', 'Hurst', 'TrendStrength',
            'Sharpe20', 'VaR95', 'MaxDrawdown20',
            'VolumeRatio', 'PriceVolumeDivergence', 'OrderFlowImbalance',
            'SpreadProxy', 'BodyRatio',
            'ROC10', 'ROC20', 'MomentumDivergence',
            'DistanceToResistance', 'DistanceToSupport',
            'ParkinsonVol', 'GKVol'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < 5:
            return None
        
        feature_df = df[available_cols].dropna()
        if len(feature_df) < 20:
            return None
        
        # Normalize features - use GPU if available
        features = feature_df.values
        
        if use_gpu and CUPY_AVAILABLE:
            # Use CuPy for GPU-accelerated computation
            features_gpu = cp.array(features)
            mean = cp.mean(features_gpu, axis=0)
            std = cp.std(features_gpu, axis=0) + 1e-8
            features = cp.asnumpy((features_gpu - mean) / std)
        else:
            # CPU computation
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            features = (features - mean) / std
        
        # Create sequences (use last 20 timesteps)
        seq_length = min(20, len(features))
        sequence = features[-seq_length:]
        
        # Pad if necessary
        if len(sequence) < 20:
            padding = np.zeros((20 - len(sequence), features.shape[1]))
            sequence = np.vstack([padding, sequence])
        
        return sequence.reshape(1, 20, -1)  # (batch, time, features)
    
    except Exception as e:
        return None


def predict_with_model(model, features: np.ndarray) -> Tuple[float, float]:
    """Make prediction using model. Returns (high_prediction, low_prediction)."""
    try:
        if model is None or features is None:
            return 0.0, 0.0
        
        pred = model.predict(features)
        if isinstance(pred, np.ndarray) and len(pred) >= 2:
            high_pred = float(pred[0])
            low_pred = float(pred[1])
            # Ensure high >= low
            if high_pred < low_pred:
                high_pred, low_pred = low_pred, high_pred
            return high_pred, low_pred
        else:
            return 0.0, 0.0
    except Exception:
        return 0.0, 0.0


# Global model selector instance
_model_selector = ModelSelector()


def get_model_selector() -> ModelSelector:
    """Get global model selector instance."""
    return _model_selector
