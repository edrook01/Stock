"""
System Event Logger
Logs system events, errors, warnings, and operational data.
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import json
import csv
import traceback

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path


class SystemLogger:
    """Comprehensive system event logger."""
    
    def __init__(self):
        """Initialize system logger."""
        self.history_dir = get_path('history')
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.history_dir / 'system_events.csv'
        self.json_file = self.history_dir / 'system_events.json'
        
        self._initialize_csv()
        self._system_logs: List[Dict] = []
        self._load_system_logs()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Level', 'Category', 'Component', 'Function',
                    'EventType', 'Message', 'ErrorType', 'ErrorMessage',
                    'StackTrace', 'Context', 'Duration', 'Status', 'Notes'
                ])
    
    def log_event(
        self,
        level: str,
        category: str,
        component: str,
        function: str,
        event_type: str,
        message: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[Dict] = None,
        duration: Optional[float] = None,
        status: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """
        Log a system event.
        
        Args:
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            category: Event category ('data_fetch', 'model_training', 'prediction', 
                     'trade_execution', 'system', 'cache', 'network', etc.)
            component: Component name (e.g., 'data_fetcher', 'unified_model')
            function: Function name where event occurred
            event_type: Type of event ('start', 'end', 'success', 'failure', 'retry', etc.)
            message: Event message
            error_type: Error type (if error) (optional)
            error_message: Error message (if error) (optional)
            stack_trace: Stack trace (if error) (optional)
            context: Additional context dictionary (optional)
            duration: Duration in seconds (optional)
            status: Status ('success', 'failure', 'partial', 'timeout') (optional)
            notes: Additional notes
            
        Returns:
            Log entry ID (timestamp-based)
        """
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now()
        
        log_data = {
            "log_id": log_id,
            "timestamp": timestamp.isoformat(),
            "level": level.upper(),
            "category": category,
            "component": component,
            "function": function,
            "event_type": event_type,
            "message": message,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "context": context or {},
            "duration": duration,
            "status": status,
            "notes": notes
        }
        
        self._system_logs.append(log_data)
        self._save_system_logs()
        
        # Append to CSV
        self._append_csv_row(log_data)
        
        return log_id
    
    def log_info(
        self,
        category: str,
        component: str,
        function: str,
        message: str,
        context: Optional[Dict] = None,
        notes: str = ""
    ) -> str:
        """Log an info-level event."""
        return self.log_event(
            level="INFO",
            category=category,
            component=component,
            function=function,
            event_type="info",
            message=message,
            context=context,
            notes=notes
        )
    
    def log_warning(
        self,
        category: str,
        component: str,
        function: str,
        message: str,
        context: Optional[Dict] = None,
        notes: str = ""
    ) -> str:
        """Log a warning-level event."""
        return self.log_event(
            level="WARNING",
            category=category,
            component=component,
            function=function,
            event_type="warning",
            message=message,
            context=context,
            notes=notes
        )
    
    def log_error(
        self,
        category: str,
        component: str,
        function: str,
        message: str,
        error: Exception,
        context: Optional[Dict] = None,
        notes: str = ""
    ) -> str:
        """Log an error-level event."""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        return self.log_event(
            level="ERROR",
            category=category,
            component=component,
            function=function,
            event_type="error",
            message=message,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context,
            notes=notes
        )
    
    def log_function_call(
        self,
        category: str,
        component: str,
        function: str,
        message: str,
        duration: Optional[float] = None,
        status: Optional[str] = None,
        context: Optional[Dict] = None,
        notes: str = ""
    ) -> str:
        """Log a function call event."""
        event_type = "function_call"
        if duration:
            event_type = "function_end" if status else "function_start"
        
        return self.log_event(
            level="DEBUG",
            category=category,
            component=component,
            function=function,
            event_type=event_type,
            message=message,
            context=context,
            duration=duration,
            status=status,
            notes=notes
        )
    
    def _append_csv_row(self, log_data: Dict) -> None:
        """Append a log row to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Convert complex fields to JSON strings for CSV
                context_json = json.dumps(log_data.get("context", {}))
                
                writer.writerow([
                    log_data.get("timestamp", ""),
                    log_data.get("level", ""),
                    log_data.get("category", ""),
                    log_data.get("component", ""),
                    log_data.get("function", ""),
                    log_data.get("event_type", ""),
                    log_data.get("message", ""),
                    log_data.get("error_type", ""),
                    log_data.get("error_message", ""),
                    log_data.get("stack_trace", ""),
                    context_json,
                    log_data.get("duration", ""),
                    log_data.get("status", ""),
                    log_data.get("notes", "")
                ])
        except Exception:
            # Silent failure on CSV write errors
            pass
    
    def _save_system_logs(self) -> None:
        """Save system logs to JSON file."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self._system_logs, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_system_logs(self) -> None:
        """Load system logs from JSON file."""
        try:
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self._system_logs = json.load(f)
        except Exception:
            # Silent failure on load errors
            self._system_logs = []
    
    def get_system_logs(
        self,
        level: Optional[str] = None,
        category: Optional[str] = None,
        component: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get logged system events, optionally filtered.
        
        Args:
            level: Filter by log level (optional)
            category: Filter by category (optional)
            component: Filter by component (optional)
            event_type: Filter by event type (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of system event dictionaries
        """
        results = self._system_logs.copy()
        
        if level:
            results = [log for log in results if log.get("level") == level.upper()]
        
        if category:
            results = [log for log in results if log.get("category") == category]
        
        if component:
            results = [log for log in results if log.get("component") == component]
        
        if event_type:
            results = [log for log in results if log.get("event_type") == event_type]
        
        if start_date:
            results = [
                log for log in results
                if datetime.fromisoformat(log.get("timestamp", "")) >= start_date
            ]
        
        if end_date:
            results = [
                log for log in results
                if datetime.fromisoformat(log.get("timestamp", "")) <= end_date
            ]
        
        return results
    
    def get_error_summary(
        self,
        component: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get summary of errors.
        
        Args:
            component: Filter by component (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            Dictionary with error summary
        """
        errors = self.get_system_logs(
            level="ERROR",
            component=component,
            start_date=start_date,
            end_date=end_date
        )
        
        if not errors:
            return {
                "total_errors": 0,
                "error_types": {},
                "components": {}
            }
        
        error_types = {}
        components = {}
        
        for error in errors:
            error_type = error.get("error_type", "Unknown")
            component_name = error.get("component", "Unknown")
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            components[component_name] = components.get(component_name, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_types": error_types,
            "components": components,
            "errors": errors[-10:]  # Last 10 errors
        }


# Global system logger instance
_system_logger: Optional[SystemLogger] = None


def get_system_logger() -> SystemLogger:
    """Get global system logger instance."""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    return _system_logger

