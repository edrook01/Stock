"""
Browser Window Manager
Manages CFD browser window state, visualization, and monitoring.
"""

from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
import time
import base64

from .automation import BrowserAutomation
from ..core.portable_paths import get_path


class WindowManager:
    """Manages browser window state and provides monitoring capabilities."""
    
    def __init__(self, browser: BrowserAutomation):
        """
        Initialize window manager.
        
        Args:
            browser: BrowserAutomation instance
        """
        self.browser = browser
        self.window_state_file = get_path('memory') / 'browser_window_state.json'
        self.window_state: Dict[str, Any] = {
            "window_id": None,
            "url": None,
            "title": None,
            "is_logged_in": False,
            "last_activity": None,
            "position_count": 0,
            "window_size": None,
            "window_position": None
        }
        self._load_state()
    
    def initialize_window(
        self,
        width: int = 1920,
        height: int = 1080,
        position_x: int = 0,
        position_y: int = 0
    ) -> bool:
        """
        Initialize browser window with specific dimensions and position.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            position_x: Window X position
            position_y: Window Y position
            
        Returns:
            True if successful, False otherwise
        """
        if not self.browser.is_ready():
            if not self.browser.initialize():
                return False
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                self.browser.driver.set_window_size(width, height)
                self.browser.driver.set_window_position(position_x, position_y)
                window_id = self.browser.driver.current_window_handle
                
            elif self.browser.library_used == "playwright":
                # Playwright sets viewport, not window size
                self.browser.page.set_viewport_size({"width": width, "height": height})
                window_id = "playwright_window"
            
            self.window_state.update({
                "window_id": window_id,
                "window_size": {"width": width, "height": height},
                "window_position": {"x": position_x, "y": position_y},
                "last_activity": datetime.now().isoformat()
            })
            
            self._save_state()
            return True
        
        except Exception:
            return False
    
    def maximize_window(self) -> bool:
        """Maximize browser window."""
        if not self.browser.is_ready():
            return False
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                self.browser.driver.maximize_window()
                window_id = self.browser.driver.current_window_handle
                
                # Get actual window size after maximizing
                size = self.browser.driver.get_window_size()
                self.window_state.update({
                    "window_id": window_id,
                    "window_size": {"width": size["width"], "height": size["height"]},
                    "window_position": {"x": 0, "y": 0},
                    "maximized": True
                })
            elif self.browser.library_used == "playwright":
                # Playwright doesn't have maximize, use fullscreen
                self.browser.page.set_viewport_size({"width": 1920, "height": 1080})
                self.window_state.update({
                    "window_size": {"width": 1920, "height": 1080},
                    "maximized": True
                })
            
            self._save_state()
            return True
        
        except Exception:
            return False
    
    def get_window_info(self) -> Dict[str, Any]:
        """
        Get current window information.
        
        Returns:
            Dictionary with window state information
        """
        if not self.browser.is_ready():
            return self.window_state
        
        try:
            current_url = self.browser.get_current_url()
            window_title = None
            
            if self.browser.library_used == "undetected-chromedriver":
                window_title = self.browser.driver.title
                try:
                    size = self.browser.driver.get_window_size()
                    position = self.browser.driver.get_window_position()
                    self.window_state["window_size"] = {"width": size["width"], "height": size["height"]}
                    self.window_state["window_position"] = {"x": position["x"], "y": position["y"]}
                except:
                    pass
            elif self.browser.library_used == "playwright":
                window_title = self.browser.page.title()
                try:
                    viewport = self.browser.page.viewport_size
                    if viewport:
                        self.window_state["window_size"] = {"width": viewport["width"], "height": viewport["height"]}
                except:
                    pass
            
            self.window_state.update({
                "url": current_url,
                "title": window_title,
                "last_activity": datetime.now().isoformat()
            })
            
            self._save_state()
        
        except Exception:
            pass
        
        return self.window_state.copy()
    
    def take_screenshot(
        self,
        save_path: Optional[Path] = None,
        element_selector: Optional[str] = None
    ) -> Optional[Path]:
        """
        Take a screenshot of the browser window or specific element.
        
        Args:
            save_path: Path to save screenshot (defaults to memory/screenshots/)
            element_selector: Optional CSS/XPath selector for element screenshot
            
        Returns:
            Path to saved screenshot, or None if failed
        """
        if not self.browser.is_ready():
            return None
        
        try:
            if save_path is None:
                screenshot_dir = get_path('memory') / 'screenshots'
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = screenshot_dir / f"screenshot_{timestamp}.png"
            
            if self.browser.library_used == "undetected-chromedriver":
                if element_selector:
                    # Screenshot specific element
                    element = self.browser.find_element(element_selector, timeout=5.0)
                    if element:
                        element.screenshot(str(save_path))
                    else:
                        return None
                else:
                    # Full window screenshot
                    self.browser.driver.save_screenshot(str(save_path))
            
            elif self.browser.library_used == "playwright":
                if element_selector:
                    # Screenshot specific element
                    try:
                        self.browser.page.locator(element_selector).screenshot(path=str(save_path))
                    except:
                        return None
                else:
                    # Full page screenshot
                    self.browser.page.screenshot(path=str(save_path), full_page=True)
            
            return save_path
        
        except Exception:
            return None
    
    def take_screenshot_base64(self, element_selector: Optional[str] = None) -> Optional[str]:
        """
        Take screenshot and return as base64 string.
        
        Args:
            element_selector: Optional CSS/XPath selector for element screenshot
            
        Returns:
            Base64 encoded screenshot string, or None if failed
        """
        if not self.browser.is_ready():
            return None
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                if element_selector:
                    element = self.browser.find_element(element_selector, timeout=5.0)
                    if element:
                        screenshot_bytes = element.screenshot_as_png
                    else:
                        return None
                else:
                    screenshot_bytes = self.browser.driver.get_screenshot_as_png()
            
            elif self.browser.library_used == "playwright":
                if element_selector:
                    screenshot_bytes = self.browser.page.locator(element_selector).screenshot()
                else:
                    screenshot_bytes = self.browser.page.screenshot(full_page=True)
            
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        
        except Exception:
            return None
    
    def wait_for_element_and_verify(
        self,
        selector: str,
        expected_text: Optional[str] = None,
        timeout: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Wait for element and optionally verify its text.
        
        Args:
            selector: Element selector
            expected_text: Optional expected text to verify
            timeout: Maximum wait time
            
        Returns:
            Tuple of (found: bool, actual_text: Optional[str])
        """
        element = self.browser.find_element(selector, timeout=timeout, wait_visible=True)
        
        if not element:
            return (False, None)
        
        if expected_text is None:
            return (True, None)
        
        # Get element text
        try:
            if self.browser.library_used == "undetected-chromedriver":
                actual_text = element.text
            elif self.browser.library_used == "playwright":
                actual_text = element.text_content()
            else:
                actual_text = None
            
            matches = expected_text.lower() in actual_text.lower() if actual_text else False
            return (matches, actual_text)
        
        except Exception:
            return (False, None)
    
    def scroll_to_element(self, selector: str) -> bool:
        """
        Scroll to element to bring it into view.
        
        Args:
            selector: Element selector
            
        Returns:
            True if successful, False otherwise
        """
        element = self.browser.find_element(selector, timeout=5.0)
        if not element:
            return False
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            elif self.browser.library_used == "playwright":
                self.browser.page.locator(selector).scroll_into_view_if_needed()
            
            time.sleep(0.5)  # Brief pause after scrolling
            return True
        
        except Exception:
            return False
    
    def execute_javascript(self, script: str, *args) -> Any:
        """
        Execute JavaScript in the browser context.
        
        Args:
            script: JavaScript code to execute
            *args: Arguments to pass to script
            
        Returns:
            Script execution result
        """
        if not self.browser.is_ready():
            return None
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                return self.browser.driver.execute_script(script, *args)
            elif self.browser.library_used == "playwright":
                return self.browser.page.evaluate(script, *args)
        except Exception:
            return None
    
    def wait_for_url_change(
        self,
        from_url: Optional[str] = None,
        to_url_pattern: Optional[str] = None,
        timeout: float = 15.0
    ) -> bool:
        """
        Wait for URL to change.
        
        Args:
            from_url: Starting URL (defaults to current URL)
            to_url_pattern: Optional pattern to match in target URL
            timeout: Maximum wait time
            
        Returns:
            True if URL changed (and matches pattern if provided), False otherwise
        """
        if from_url is None:
            from_url = self.browser.get_current_url()
        
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            current_url = self.browser.get_current_url()
            
            if current_url != from_url:
                if to_url_pattern:
                    if to_url_pattern in current_url:
                        return True
                else:
                    return True
            
            time.sleep(0.2)
        
        return False
    
    def refresh_page(self) -> bool:
        """Refresh current page."""
        if not self.browser.is_ready():
            return False
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                self.browser.driver.refresh()
            elif self.browser.library_used == "playwright":
                self.browser.page.reload()
            
            self.browser.wait_for_page_load(timeout=10.0)
            self._update_last_activity()
            return True
        
        except Exception:
            return False
    
    def go_back(self) -> bool:
        """Navigate back in browser history."""
        if not self.browser.is_ready():
            return False
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                self.browser.driver.back()
            elif self.browser.library_used == "playwright":
                self.browser.page.go_back()
            
            self.browser.wait_for_page_load(timeout=10.0)
            self._update_last_activity()
            return True
        
        except Exception:
            return False
    
    def _update_last_activity(self) -> None:
        """Update last activity timestamp."""
        self.window_state["last_activity"] = datetime.now().isoformat()
        self._save_state()
    
    def _save_state(self) -> None:
        """Save window state to file."""
        try:
            self.window_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.window_state_file, 'w') as f:
                json.dump(self.window_state, f, indent=2)
        except Exception:
            pass
    
    def _load_state(self) -> None:
        """Load window state from file."""
        try:
            if self.window_state_file.exists():
                with open(self.window_state_file, 'r') as f:
                    self.window_state.update(json.load(f))
        except Exception:
            pass
    
    def update_login_state(self, is_logged_in: bool) -> None:
        """Update login state in window state."""
        self.window_state["is_logged_in"] = is_logged_in
        self._update_last_activity()
    
    def update_position_count(self, count: int) -> None:
        """Update position count in window state."""
        self.window_state["position_count"] = count
        self._update_last_activity()

