#!/usr/bin/env python3
"""
Self-Evaluator - Self-Contained Module
Evaluates AI system as a whole.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any


class SelfEvaluator:
    """Evaluates overall model accuracy."""
    
    def __init__(self):
        self.prediction_history = []
        self.accuracy_history = []
    
    def evaluate_model(self, predictions: List[Dict], actuals: Dict) -> Dict[str, float]:
        """Evaluate overall model accuracy."""
        if not predictions:
            return {'overall_accuracy': 0.0, 'model_health': 0.0}
        
        accuracies = []
        for pred in predictions:
            ticker = pred.get('ticker')
            interval = pred.get('interval')
            key = f"{ticker}_{interval}"
            
            if key in actuals:
                actual = actuals[key]
                high_pred = pred.get('high_prediction', 0)
                low_pred = pred.get('low_prediction', 0)
                
                # Calculate accuracy
                high_error = abs(high_pred - actual.get('high', 0)) / actual.get('high', 1)
                low_error = abs(low_pred - actual.get('low', 0)) / actual.get('low', 1)
                accuracy = max(0, 10 - (high_error + low_error) * 50)
                accuracies.append(accuracy)
        
        overall_accuracy = np.mean(accuracies) if accuracies else 0.0
        model_health = overall_accuracy / 10.0  # Normalize to 0-1
        
        return {
            'overall_accuracy': float(overall_accuracy),
            'model_health': float(model_health),
            'total_predictions': len(predictions),
            'evaluated_predictions': len(accuracies)
        }
