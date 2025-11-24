"""
Error Logger
Centralized error logging to logs/error.log file.
Handles both soft errors (warnings, recoverable) and hard errors (exceptions, fatal).
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import traceback
import sys
import threading

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path
    _USE_PORTABLE_PATHS = True
except (ImportError, ValueError):
    # Fallback for direct execution
    try:
        from core.portable_paths import get_path
        _USE_PORTABLE_PATHS = True
    except (ImportError, ValueError):
        _USE_PORTABLE_PATHS = False

# Thread-safe file writing lock
_write_lock = threading.Lock()


def _get_logs_path() -> Path:
    """
    Get the logs directory path.
    Resolves to Stock Report/logs/ (parent of V15).
    
    Returns:
        Path to logs directory
    """
    # Try to use portable_paths if available
    if _USE_PORTABLE_PATHS:
        try:
            # portable_paths.get_path('logs') returns V15/logs/, but we need Stock Report/logs/
            # So we'll calculate it manually but use portable_paths for consistency
            V15_ROOT = Path(__file__).parent.parent
            stock_report_root = V15_ROOT.parent
            logs_dir = stock_report_root / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            return logs_dir
        except (ValueError, AttributeError):
            # Fall back to manual calculation
            pass
    
    # Manual path calculation (fallback)
    # Get V15 root
    V15_ROOT = Path(__file__).parent.parent
    
    # Go up to Stock Report/ and then to logs/
    # V15 is in Stock Report/V15/, so logs/ is in Stock Report/logs/
    stock_report_root = V15_ROOT.parent
    logs_dir = stock_report_root / 'logs'
    
    # Create if doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return logs_dir


def _format_error_entry(
    level: str,
    message: str,
    error: Optional[Exception] = None,
    component: Optional[str] = None,
    function: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    is_hard_error: bool = False
) -> str:
    """
    Format an error log entry.
    
    Args:
        level: Error level (ERROR, WARNING, CRITICAL)
        message: Error message
        error: Exception object (optional)
        component: Component name (optional)
        function: Function name (optional)
        context: Additional context (optional)
        is_hard_error: Whether this is a hard (fatal) error
        
    Returns:
        Formatted log entry string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build entry
    parts = [f"[{timestamp}] {level}"]
    
    if component:
        parts.append(f"[{component}]")
    if function:
        parts.append(f"[{function}]")
    
    parts.append(f": {message}")
    
    if error:
        error_type = type(error).__name__
        error_msg = str(error)
        parts.append(f" | Exception: {error_type}: {error_msg}")
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        parts.append(f" | Context: {context_str}")
    
    if is_hard_error:
        parts.append(" | [HARD ERROR - FATAL]")
    
    entry = "".join(parts)
    
    # Add stack trace if error exists
    if error:
        stack_trace = traceback.format_exc()
        if stack_trace and stack_trace.strip() != "NoneType: None":
            entry += f"\n{stack_trace}"
    
    return entry


def log_error(
    message: str,
    error: Optional[Exception] = None,
    component: Optional[str] = None,
    function: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    is_hard_error: bool = False
) -> None:
    """
    Log an error to logs/error.log.
    Thread-safe and handles write failures gracefully.
    
    Args:
        message: Error message
        error: Exception object (optional)
        component: Component name (optional)
        function: Function name (optional)
        context: Additional context dictionary (optional)
        is_hard_error: Whether this is a hard (fatal) error
    """
    try:
        logs_dir = _get_logs_path()
        error_log_file = logs_dir / 'error.log'
        
        level = "CRITICAL" if is_hard_error else "ERROR"
        entry = _format_error_entry(
            level=level,
            message=message,
            error=error,
            component=component,
            function=function,
            context=context,
            is_hard_error=is_hard_error
        )
        
        # Thread-safe write
        with _write_lock:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(entry + "\n")
                f.flush()  # Ensure immediate write
    except Exception:
        # Silent failure - don't break execution if logging fails
        # Fallback to stderr
        try:
            print(f"[ERROR LOGGER FAILED] {message}", file=sys.stderr)
            if error:
                print(f"Exception: {error}", file=sys.stderr)
        except Exception:
            pass


def log_warning(
    message: str,
    component: Optional[str] = None,
    function: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a warning to logs/error.log.
    
    Args:
        message: Warning message
        component: Component name (optional)
        function: Function name (optional)
        context: Additional context dictionary (optional)
    """
    try:
        logs_dir = _get_logs_path()
        error_log_file = logs_dir / 'error.log'
        
        entry = _format_error_entry(
            level="WARNING",
            message=message,
            component=component,
            function=function,
            context=context,
            is_hard_error=False
        )
        
        # Thread-safe write
        with _write_lock:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(entry + "\n")
                f.flush()
    except Exception:
        # Silent failure
        pass


def log_info(
    message: str,
    component: Optional[str] = None,
    function: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an info message to logs/app.log.
    
    Args:
        message: Info message
        component: Component name (optional)
        function: Function name (optional)
        context: Additional context dictionary (optional)
    """
    try:
        logs_dir = _get_logs_path()
        app_log_file = logs_dir / 'app.log'
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        parts = [f"[{timestamp}]"]
        if component:
            parts.append(f"[{component}]")
        if function:
            parts.append(f"[{function}]")
        parts.append(f": {message}")
        
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f" | Context: {context_str}")
        
        entry = "".join(parts)
        
        # Thread-safe write
        with _write_lock:
            with open(app_log_file, 'a', encoding='utf-8') as f:
                f.write(entry + "\n")
                f.flush()
    except Exception:
        # Silent failure
        pass


# Convenience function for exception handling
def log_exception(
    message: str,
    error: Exception,
    component: Optional[str] = None,
    function: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    is_hard_error: bool = False
) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        message: Error message
        error: Exception object
        component: Component name (optional)
        function: Function name (optional)
        context: Additional context dictionary (optional)
        is_hard_error: Whether this is a hard (fatal) error
    """
    log_error(
        message=message,
        error=error,
        component=component,
        function=function,
        context=context,
        is_hard_error=is_hard_error
    )

