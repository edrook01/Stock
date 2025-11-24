"""
Browser Automation Debugger
Debug utilities for browser automation and Trading212 interactions.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from ..browser.automation import BrowserAutomation, load_trading212_credentials
from ..browser.human_behavior import HumanBehavior
from ..browser.trade_executor import TradeExecutor
from ..browser.error_handler import BrowserErrorHandler


class BrowserDebugger:
    """Debug browser automation."""
    
    def __init__(self):
        """Initialize browser debugger."""
        self.human_behavior = HumanBehavior()
        self.error_handler = BrowserErrorHandler()
    
    def debug_browser_init(
        self,
        headless: bool = False,
        use_playwright: bool = False
    ) -> Dict[str, Any]:
        """
        Debug browser initialization.
        
        Args:
            headless: Run in headless mode
            use_playwright: Force Playwright
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_browser_init",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "headless": headless,
                "use_playwright": use_playwright
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": [],
            "performance": {}
        }
        
        start_time = time.time()
        
        # Step 1: Create browser instance
        try:
            browser = BrowserAutomation(headless=headless, use_playwright=use_playwright)
            debug_info["steps"].append({
                "step": 1,
                "action": "Create BrowserAutomation instance",
                "result": "OK"
            })
        except Exception as e:
            debug_info["errors"].append(f"Failed to create browser: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Initialize browser
        step_start = time.time()
        try:
            initialized = browser.initialize()
            step_duration = (time.time() - step_start) * 1000
            
            debug_info["steps"].append({
                "step": 2,
                "action": "Initialize browser",
                "result": "SUCCESS" if initialized else "FAILED",
                "duration_ms": step_duration
            })
            
            if not initialized:
                debug_info["errors"].append("Browser initialization failed")
                debug_info["success"] = False
                return debug_info
        except Exception as e:
            debug_info["errors"].append(f"Initialization error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 3: Check library used
        debug_info["steps"].append({
            "step": 3,
            "action": "Check library used",
            "result": browser.library_used or "None"
        })
        
        # Step 4: Test navigation
        step_start = time.time()
        try:
            nav_success = browser.navigate("https://www.google.com")
            step_duration = (time.time() - step_start) * 1000
            
            debug_info["steps"].append({
                "step": 4,
                "action": "Test navigation",
                "result": "SUCCESS" if nav_success else "FAILED",
                "duration_ms": step_duration
            })
            
            if nav_success:
                current_url = browser.get_current_url()
                debug_info["steps"].append({
                    "step": 5,
                    "action": "Get current URL",
                    "result": current_url or "None"
                })
        except Exception as e:
            debug_info["warnings"].append(f"Navigation test failed: {str(e)}")
        
        # Step 5: Close browser
        try:
            browser.close()
            debug_info["steps"].append({
                "step": len(debug_info["steps"]) + 1,
                "action": "Close browser",
                "result": "OK"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Close error: {str(e)}")
        
        total_duration = (time.time() - start_time) * 1000
        debug_info["performance"] = {
            "total_duration_ms": total_duration,
            "slowest_step": max([s.get("duration_ms", 0) for s in debug_info["steps"]], default=0)
        }
        
        debug_info["output"] = {
            "library_used": browser.library_used,
            "initialized": browser.is_initialized,
            "is_ready": browser.is_ready()
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_human_behavior(self) -> Dict[str, Any]:
        """
        Debug human-like behavior simulation.
        
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_human_behavior",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Test random delays
        delays = []
        for i in range(5):
            start = time.time()
            self.human_behavior.random_delay(0.1, 0.2)
            delay = (time.time() - start) * 1000
            delays.append(delay)
        
        debug_info["steps"].append({
            "step": 1,
            "action": "Test random delays",
            "result": f"Average: {sum(delays)/len(delays):.2f}ms"
        })
        
        # Test typing delays
        typing_delays = []
        for i in range(5):
            start = time.time()
            self.human_behavior.typing_delay()
            delay = (time.time() - start) * 1000
            typing_delays.append(delay)
        
        debug_info["steps"].append({
            "step": 2,
            "action": "Test typing delays",
            "result": f"Average: {sum(typing_delays)/len(typing_delays):.2f}ms"
        })
        
        # Test Bézier curve
        start_point = (0, 0)
        end_point = (100, 100)
        curve_points = self.human_behavior.bezier_curve(start_point, end_point)
        
        debug_info["steps"].append({
            "step": 3,
            "action": "Test Bézier curve generation",
            "result": f"Generated {len(curve_points)} points"
        })
        
        # Test mouse jitter
        jittered = self.human_behavior.add_mouse_jitter((50, 50))
        debug_info["steps"].append({
            "step": 4,
            "action": "Test mouse jitter",
            "result": f"Jittered point: {jittered}"
        })
        
        # Test variable speed
        speed_points = self.human_behavior.variable_speed_movement(curve_points[:10])
        debug_info["steps"].append({
            "step": 5,
            "action": "Test variable speed movement",
            "result": f"Processed {len(speed_points)} points"
        })
        
        debug_info["output"] = {
            "delay_stats": {
                "random_delays": {
                    "min": min(delays),
                    "max": max(delays),
                    "avg": sum(delays) / len(delays)
                },
                "typing_delays": {
                    "min": min(typing_delays),
                    "max": max(typing_delays),
                    "avg": sum(typing_delays) / len(typing_delays)
                }
            },
            "curve_points": len(curve_points),
            "speed_points": len(speed_points)
        }
        debug_info["success"] = True
        
        return debug_info
    
    def debug_login_flow(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Debug login flow (without actually logging in).
        
        Args:
            username: Trading212 username (optional)
            password: Trading212 password (optional)
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_login_flow",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Check credentials
        creds = load_trading212_credentials()
        if not creds and (not username or not password):
            debug_info["warnings"].append("No credentials found in config or provided")
            debug_info["output"] = {"has_credentials": False}
            debug_info["success"] = False
            return debug_info
        
        if creds:
            debug_info["steps"].append({
                "step": 1,
                "action": "Load credentials from config",
                "result": "Credentials found"
            })
        else:
            debug_info["steps"].append({
                "step": 1,
                "action": "Use provided credentials",
                "result": "Credentials provided"
            })
        
        # Step 2: Initialize browser
        try:
            browser = BrowserAutomation()
            if not browser.initialize():
                debug_info["errors"].append("Browser initialization failed")
                debug_info["success"] = False
                return debug_info
            
            debug_info["steps"].append({
                "step": 2,
                "action": "Initialize browser",
                "result": "OK"
            })
        except Exception as e:
            debug_info["errors"].append(f"Browser init error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 3: Navigate to login page
        try:
            nav_success = browser.navigate("https://www.trading212.com/en/login")
            debug_info["steps"].append({
                "step": 3,
                "action": "Navigate to login page",
                "result": "SUCCESS" if nav_success else "FAILED"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Navigation error: {str(e)}")
        
        # Step 4: Create executor
        try:
            executor = TradeExecutor(browser)
            debug_info["steps"].append({
                "step": 4,
                "action": "Create trade executor",
                "result": "OK"
            })
        except Exception as e:
            debug_info["errors"].append(f"Executor creation error: {str(e)}")
            debug_info["success"] = False
            browser.close()
            return debug_info
        
        # Note: Actual login would require UI element selectors
        # This is a structure test
        debug_info["warnings"].append("Actual login requires UI element selectors (not implemented)")
        
        browser.close()
        
        debug_info["output"] = {
            "has_credentials": creds is not None or (username and password),
            "browser_initialized": True,
            "executor_created": True
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_error_recovery(
        self,
        test_scenarios: List[str]
    ) -> Dict[str, Any]:
        """
        Debug error recovery mechanisms.
        
        Args:
            test_scenarios: List of error scenarios to test
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_error_recovery",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "scenarios": test_scenarios
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Test retry with backoff
        def failing_function():
            raise ValueError("Test error")
        
        result = self.error_handler.retry_with_backoff(failing_function)
        debug_info["steps"].append({
            "step": 1,
            "action": "Test retry with backoff",
            "result": "Failed as expected" if result is None else "Unexpected success"
        })
        
        # Test error logging
        error_log = self.error_handler.get_error_log()
        debug_info["steps"].append({
            "step": 2,
            "action": "Check error log",
            "result": f"{len(error_log)} errors logged"
        })
        
        debug_info["output"] = {
            "error_log_count": len(error_log),
            "retry_behavior": "Working" if len(error_log) > 0 else "Not working"
        }
        debug_info["success"] = True
        
        return debug_info


# Global browser debugger instance
_browser_debugger: Optional[BrowserDebugger] = None


def get_browser_debugger() -> BrowserDebugger:
    """Get global browser debugger instance."""
    global _browser_debugger
    if _browser_debugger is None:
        _browser_debugger = BrowserDebugger()
    return _browser_debugger

