"""
Error Handling & Recovery
Handles browser automation errors and implements recovery strategies.
"""

from typing import Optional, Callable, Any
import time
from datetime import datetime


class BrowserErrorHandler:
    """Handles errors in browser automation."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_log: list = []
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result, or None if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self._log_error(e, attempt + 1)
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Last attempt failed
                    break
        
        return None
    
    def handle_element_not_found(
        self,
        browser,
        selector: str,
        action: str = "click"
    ) -> bool:
        """
        Handle element not found error.
        
        Args:
            browser: BrowserAutomation instance
            selector: Element selector
            action: Action attempted ("click", "type", etc.)
            
        Returns:
            True if handled/recovered, False otherwise
        """
        # Try refreshing page
        try:
            if browser.library_used == "undetected_chromedriver":
                browser.driver.refresh()
            elif browser.library_used == "playwright":
                browser.page.reload()
            
            time.sleep(2)  # Wait for page to load
            return True
        except Exception:
            return False
    
    def handle_session_timeout(self, browser, executor) -> bool:
        """
        Handle session timeout by re-logging in.
        
        Args:
            browser: BrowserAutomation instance
            executor: TradeExecutor instance
            
        Returns:
            True if re-logged in successfully, False otherwise
        """
        try:
            # Close and reinitialize browser
            browser.close()
            if not browser.initialize():
                return False
            
            # Try to log in again
            return executor.login()
        except Exception:
            return False
    
    def _log_error(self, error: Exception, attempt: int) -> None:
        """Log error for debugging."""
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempt": attempt
        })
    
    def get_error_log(self) -> list:
        """Get error log."""
        return self.error_log.copy()
    
    def clear_error_log(self) -> None:
        """Clear error log."""
        self.error_log = []

