"""
Portable Paths Module for V15
Provides portable path management using Path(__file__) to determine project root.
Never uses absolute OS paths - always relative to project root.
"""

from pathlib import Path
from typing import Dict, Optional


def _get_project_root() -> Path:
    """
    Get the project root (V15) by traversing up from this file's location.
    
    Returns:
        Path: Absolute path to the project root (V15 directory)
    """
    # #region agent log
    import json
    from datetime import datetime
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V15\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"core/portable_paths.py:_get_project_root","message":"Function entry","data":{"__file__":str(__file__)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    # Get the directory containing this file (V15/core/)
    current_file = Path(__file__).resolve()
    # Go up one level to get V15/
    project_root = current_file.parent.parent
    
    # #region agent log
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V15\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"core/portable_paths.py:_get_project_root","message":"Function exit","data":{"current_file":str(current_file),"project_root":str(project_root),"exists":project_root.exists()},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    return project_root


# Cache the project root to avoid repeated calculations
_PROJECT_ROOT: Optional[Path] = None


def _get_cached_root() -> Path:
    """Get cached project root or calculate and cache it."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = _get_project_root()
    return _PROJECT_ROOT


# Path type mappings for V15
_PATH_TYPES: Dict[str, str] = {
    'root': '',
    'data': 'data',
    'logs': 'logs',
    'memory': 'memory',
    'history': 'history',
    'model': 'model',
    'core': 'core',
    'micro': 'micro',
    'trading': 'trading',
    'ui': 'ui',
    'browser': 'browser',
    'risk': 'risk',
    'sentiment': 'sentiment',
    'learning': 'learning',
    'logging': 'logging',
    'config': 'data',  # config.json is in data/
    'tickers': 'data',  # tickers.txt is in data/
    'deep_learning': 'memory/deep_learning_modules',
    'models': 'model/models',
    'model_weights': 'model/weights',  # V15 unified model weights
    'memory_models': 'memory/memory/models',
    'strategy_modules': 'memory/strategy_modules',
    'cache': 'cache',
}


def get_path(type: str) -> Path:
    """
    Get absolute path for a given type.
    
    Args:
        type: Path type identifier (e.g., 'data', 'logs', 'memory', 'root', etc.)
    
    Returns:
        Path: Absolute path to the requested directory or file location
    
    Raises:
        ValueError: If the path type is not recognized
    
    Examples:
        >>> get_path('data')  # Returns absolute path to V15/data/
        >>> get_path('logs')  # Returns absolute path to V15/logs/
        >>> get_path('root')  # Returns absolute path to V15/
    """
    # #region agent log
    import json
    from datetime import datetime
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V15\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"core/portable_paths.py:get_path","message":"Function entry","data":{"type":type},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    if type not in _PATH_TYPES:
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V15\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"core/portable_paths.py:get_path","message":"Invalid path type","data":{"type":type,"available":list(_PATH_TYPES.keys())},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
        raise ValueError(
            f"Unknown path type: '{type}'. "
            f"Available types: {', '.join(sorted(_PATH_TYPES.keys()))}"
        )
    
    project_root = _get_cached_root()
    relative_path = _PATH_TYPES[type]
    
    if relative_path == '':
        # Return project root
        result = project_root
    else:
        # Return path relative to project root
        result = project_root / relative_path
    
    # #region agent log
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V15\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"core/portable_paths.py:get_path","message":"Function exit","data":{"type":type,"result":str(result),"exists":result.exists() if hasattr(result,'exists') else None},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    return result


def initialize_structure() -> None:
    """
    Create all necessary directory structure if it doesn't exist.
    Creates all directories defined in _PATH_TYPES.
    
    This function is idempotent - safe to call multiple times.
    """
    project_root = _get_cached_root()
    
    # Create all directories
    directories_to_create = set()
    
    # Add all path types as directories
    for path_type, relative_path in _PATH_TYPES.items():
        if relative_path:  # Skip empty string (root)
            directories_to_create.add(relative_path)
    
    # Also add nested directories that might be needed
    nested_dirs = [
        'memory/deep_learning_modules',
        'memory/memory',
        'memory/memory/models',
        'memory/strategy_modules',
        'model/models',
        'model/weights',  # V15 unified model weights
    ]
    
    directories_to_create.update(nested_dirs)
    
    # Create all directories
    for relative_path in sorted(directories_to_create):
        full_path = project_root / relative_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure project root exists (should always exist, but just in case)
    project_root.mkdir(parents=True, exist_ok=True)


# Convenience functions for common paths
def get_data_path() -> Path:
    """Get absolute path to data directory."""
    return get_path('data')


def get_logs_path() -> Path:
    """Get absolute path to logs directory."""
    return get_path('logs')


def get_memory_path() -> Path:
    """Get absolute path to memory directory."""
    return get_path('memory')


def get_history_path() -> Path:
    """Get absolute path to history directory."""
    return get_path('history')


def get_model_path() -> Path:
    """Get absolute path to model directory."""
    return get_path('model')


def get_root_path() -> Path:
    """Get absolute path to project root (V15)."""
    return get_path('root')

