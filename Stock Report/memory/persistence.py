"""
Persistence module for saving and loading models and history.
Uses pickle for models and JSON for history.
All paths use portable_paths for portability.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

# Try to import portable_paths, fallback to simple path utility
try:
    from V13.core.portable_paths import get_path, get_memory_path, get_history_path
except ImportError:
    # Fallback: create simple path utility based on project root
    def _get_project_root() -> Path:
        """Get project root by traversing up from this file."""
        current_file = Path(__file__).resolve()
        # Go up from memory/persistence.py to project root
        return current_file.parent.parent
    
    _PROJECT_ROOT = _get_project_root()
    
    def get_path(type: str) -> Path:
        """Simple path getter for fallback."""
        path_map = {
            'memory': 'memory',
            'history': 'history',
            'model': 'model',
        }
        if type in path_map:
            return _PROJECT_ROOT / path_map[type]
        return _PROJECT_ROOT / type
    
    def get_memory_path() -> Path:
        """Get memory directory path."""
        return get_path('memory')
    
    def get_history_path() -> Path:
        """Get history directory path."""
        return get_path('history')


def save_model(model_dict: Dict[str, Any]) -> None:
    """
    Save a model dictionary using pickle.
    
    Args:
        model_dict: Dictionary containing model data to save
    """
    memory_dir = get_memory_path()
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = memory_dir / 'model.pkl'
    
    with open(model_file, 'wb') as f:
        pickle.dump(model_dict, f)


def load_model() -> Dict[str, Any]:
    """
    Load a model dictionary from pickle file.
    
    Returns:
        Dictionary containing model data, or empty dict if file doesn't exist
    """
    memory_dir = get_memory_path()
    model_file = memory_dir / 'model.pkl'
    
    if not model_file.exists():
        return {}
    
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError):
        return {}


def save_history(history_list: List[Any]) -> None:
    """
    Save a history list using JSON.
    
    Args:
        history_list: List containing history data to save
    """
    history_dir = get_history_path()
    history_dir.mkdir(parents=True, exist_ok=True)
    
    history_file = history_dir / 'history.json'
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_list, f, indent=2, ensure_ascii=False)


def load_history() -> List[Any]:
    """
    Load a history list from JSON file.
    
    Returns:
        List containing history data, or empty list if file doesn't exist
    """
    history_dir = get_history_path()
    history_file = history_dir / 'history.json'
    
    if not history_file.exists():
        return []
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

