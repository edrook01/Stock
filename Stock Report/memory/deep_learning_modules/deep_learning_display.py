#!/usr/bin/env python3
"""
Deep Learning Display - Self-Contained Module
Console table display for predictions with progress bars.
"""

import os
import sys
import datetime
from typing import Dict, List, Optional, Tuple, Any


def format_time_remaining(elapsed_time: datetime.timedelta, interval: str) -> str:
    """Format time remaining until prediction elapses."""
    interval_durations = {
        '1m': datetime.timedelta(minutes=1),
        '5m': datetime.timedelta(minutes=5),
        '10m': datetime.timedelta(minutes=10),
        '1d': datetime.timedelta(days=1),
        '1mo': datetime.timedelta(days=30),
        '1y': datetime.timedelta(days=365)
    }
    
    duration = interval_durations.get(interval, datetime.timedelta(days=1))
    
    if elapsed_time >= duration:
        # Elapsed - return date
        elapsed_date = datetime.datetime.now() - (elapsed_time - duration)
        return elapsed_date.strftime("%a %Y-%m-%d")
    else:
        # Time remaining
        remaining = duration - elapsed_time
        if remaining.days > 0:
            return f"{remaining.days}d {remaining.seconds//3600}h remaining"
        elif remaining.seconds >= 3600:
            hours = remaining.seconds // 3600
            minutes = (remaining.seconds % 3600) // 60
            return f"{hours}h {minutes}m remaining"
        else:
            minutes = remaining.seconds // 60
            return f"{minutes}m remaining"


def create_progress_bar(current: float, target: float, width: int = 30, color: str = "green") -> str:
    """Create ASCII progress bar."""
    if target == 0:
        return " " * width
    
    progress = min(1.0, abs(current - target) / abs(target)) if target != 0 else 0.0
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - filled)
    
    # Color codes (simplified - actual colors handled by main UI)
    color_codes = {
        "green": "",
        "yellow": "",
        "red": ""
    }
    
    return f"[{bar}] {progress*100:.1f}%"


def create_time_progress_bar(elapsed: datetime.timedelta, duration: datetime.timedelta, width: int = 30) -> str:
    """Create red progress bar for time elapsed."""
    if duration.total_seconds() == 0:
        return " " * width
    
    progress = min(1.0, elapsed.total_seconds() / duration.total_seconds())
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - filled)
    
    return f"[{bar}] {progress*100:.1f}%"


def display_predictions_table(predictions: List[Dict], ticker: str) -> str:
    """Display predictions in formatted table."""
    if not predictions:
        return "No predictions available"
    
    # Table header
    lines = []
    lines.append(f"Predictions for {ticker}")
    lines.append("=" * 120)
    lines.append(f"{'Interval':<10} {'Current':<10} {'High Target':<12} {'Low Target':<12} {'High Acc':<8} {'Low Acc':<8} {'Time Status':<20}")
    lines.append("-" * 120)
    
    interval_durations = {
        '1m': datetime.timedelta(minutes=1),
        '5m': datetime.timedelta(minutes=5),
        '10m': datetime.timedelta(minutes=10),
        '1d': datetime.timedelta(days=1),
        '1mo': datetime.timedelta(days=30),
        '1y': datetime.timedelta(days=365)
    }
    
    for pred in predictions:
        interval = pred.get('interval', '1d')
        current_price = pred.get('current_price', 0.0)
        high_pred = pred.get('high_prediction', 0.0)
        low_pred = pred.get('low_prediction', 0.0)
        high_acc = pred.get('high_accuracy', 0.0)
        low_acc = pred.get('low_accuracy', 0.0)
        timestamp = pred.get('timestamp', datetime.datetime.now())
        
        elapsed = datetime.datetime.now() - timestamp
        time_status = format_time_remaining(elapsed, interval)
        
        # Row
        lines.append(f"{interval:<10} ${current_price:<9.2f} ${high_pred:<11.2f} ${low_pred:<11.2f} {high_acc:<7.1f} {low_acc:<7.1f} {time_status:<20}")
        
        # Progress bars
        # Price progress bar (green/yellow/red based on direction)
        price_progress = create_progress_bar(current_price, high_pred, width=40)
        lines.append(f"  Price Progress: {price_progress}")
        
        # Time progress bar (red)
        duration = interval_durations.get(interval, datetime.timedelta(days=1))
        time_progress = create_time_progress_bar(elapsed, duration, width=40)
        lines.append(f"  Time Progress:  {time_progress}")
        lines.append("")
    
    lines.append("=" * 120)
    
    return "\n".join(lines)
