"""
Browser Automation Setup
Initializes browser with undetected-chromedriver (primary) or Playwright (fallback).
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
import time

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path, get_data_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path, get_data_path

# Try to import browser automation libraries
try:
    import undetected_chromedriver as uc
    UNDETECTED_CHROMEDRIVER_AVAILABLE = True
except ImportError:
    UNDETECTED_CHROMEDRIVER_AVAILABLE = False
    uc = None

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None
    Browser = None
    Page = None


class BrowserAutomation:
    """Manages browser automation for Trading212 with context manager support."""
    
    def __init__(self, headless: bool = False, use_playwright: bool = False):
        """
        Initialize browser automation.
        
        Args:
            headless: Run browser in headless mode
            use_playwright: Force use of Playwright instead of undetected-chromedriver
        """
        self.headless = headless
        self.use_playwright = use_playwright
        self.driver = None
        self.browser = None
        self.page = None
        self.library_used = None
        self.is_initialized = False
    
    def __enter__(self):
        """Context manager entry - automatically initialize browser."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize browser")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close browser."""
        self.close()
        return False  # Don't suppress exceptions
    
    def initialize(self) -> bool:
        """
        Initialize browser.
        
        Returns:
            True if initialized successfully, False otherwise
        """
        # Try undetected-chromedriver first (unless forced to use Playwright)
        if not self.use_playwright and UNDETECTED_CHROMEDRIVER_AVAILABLE:
            try:
                options = uc.ChromeOptions()
                if self.headless:
                    options.add_argument('--headless')
                
                # Add human-like options
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option('useAutomationExtension', False)
                
                self.driver = uc.Chrome(options=options, version_main=None)
                self.library_used = "undetected-chromedriver"
                self.is_initialized = True
                return True
            except Exception:
                # Fall back to Playwright
                pass
        
        # Try Playwright
        if PLAYWRIGHT_AVAILABLE:
            try:
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=self.headless)
                self.page = self.browser.new_page()
                self.library_used = "playwright"
                self.is_initialized = True
                return True
            except Exception:
                return False
        
        return False
    
    def navigate(self, url: str) -> bool:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if navigation successful, False otherwise
        """
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            if self.library_used == "undetected-chromedriver":
                self.driver.get(url)
                return True
            elif self.library_used == "playwright":
                self.page.goto(url)
                return True
        except Exception:
            return False
        
        return False
    
    def get_current_url(self) -> Optional[str]:
        """Get current browser URL."""
        if not self.is_initialized:
            return None
        
        try:
            if self.library_used == "undetected_chromedriver":
                return self.driver.current_url
            elif self.library_used == "playwright":
                return self.page.url
        except Exception:
            pass
        
        return None
    
    def close(self) -> None:
        """Close browser."""
        try:
            if self.library_used == "undetected_chromedriver" and self.driver:
                self.driver.quit()
            elif self.library_used == "playwright":
                if self.browser:
                    self.browser.close()
                if hasattr(self, 'playwright'):
                    self.playwright.stop()
        except Exception:
            pass
        
        self.is_initialized = False
        self.driver = None
        self.browser = None
        self.page = None
    
    def get_driver(self):
        """Get Selenium driver (for undetected-chromedriver)."""
        return self.driver
    
    def get_page(self):
        """Get Playwright page (for Playwright)."""
        return self.page
    
    def is_ready(self) -> bool:
        """Check if browser is ready."""
        return self.is_initialized and (self.driver is not None or self.page is not None)
    
    def wait_for_page_load(self, timeout: float = 10.0) -> bool:
        """
        Wait for page to finish loading.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if page loaded, False on timeout
        """
        if not self.is_initialized:
            return False
        
        try:
            if self.library_used == "undetected-chromedriver":
                # Wait for document ready state
                end_time = time.time() + timeout
                while time.time() < end_time:
                    ready_state = self.driver.execute_script("return document.readyState")
                    if ready_state == "complete":
                        return True
                    time.sleep(0.1)
                return False
            elif self.library_used == "playwright":
                # Playwright waits for load by default
                return True
        except Exception:
            return False
        
        return False
    
    def find_element(
        self,
        selectors: Union[str, List[str]],
        by: str = "css",
        timeout: float = 10.0,
        wait_visible: bool = True
    ) -> Optional[Any]:
        """
        Find element using multiple selector strategies.
        
        Args:
            selectors: Single selector string or list of selectors to try
            by: Selector type ("css", "xpath", "id", "class", "name")
            timeout: Maximum wait time in seconds
            wait_visible: Wait for element to be visible
            
        Returns:
            Element object if found, None otherwise
        """
        if not self.is_initialized:
            return None
        
        # Convert single selector to list
        if isinstance(selectors, str):
            selectors = [selectors]
        
        # Wait for page load first
        self.wait_for_page_load(timeout)
        
        for selector in selectors:
            try:
                element = None
                
                # Auto-detect selector type if not explicitly specified
                selector_by = by
                if selector.startswith("//") or selector.startswith(".//") or selector.startswith("("):
                    selector_by = "xpath"
                elif selector.startswith((".", "#")) or "[" in selector and not selector.startswith("//"):
                    selector_by = "css"
                
                if self.library_used == "undetected-chromedriver":
                    from selenium.webdriver.common.by import By
                    from selenium.webdriver.support.ui import WebDriverWait
                    from selenium.webdriver.support import expected_conditions as EC
                    
                    # Determine By strategy
                    if selector_by == "xpath" or selector.startswith("//") or selector.startswith(".//") or selector.startswith("("):
                        by_strategy = By.XPATH
                    elif selector_by == "css" or not selector.startswith("//"):
                        by_strategy = By.CSS_SELECTOR
                    elif selector_by == "id":
                        by_strategy = By.ID
                    elif selector_by == "class":
                        by_strategy = By.CLASS_NAME
                    elif selector_by == "name":
                        by_strategy = By.NAME
                    else:
                        # Default to CSS selector
                        by_strategy = By.CSS_SELECTOR
                    
                    wait = WebDriverWait(self.driver, timeout)
                    
                    try:
                        if wait_visible:
                            element = wait.until(EC.presence_of_element_located((by_strategy, selector)))
                            element = wait.until(EC.visibility_of_element_located((by_strategy, selector)))
                        else:
                            element = wait.until(EC.presence_of_element_located((by_strategy, selector)))
                    except Exception:
                        # Try next selector
                        continue
                    
                elif self.library_used == "playwright":
                    try:
                        # Playwright needs to know if it's xpath or css
                        if selector_by == "xpath" or selector.startswith("//"):
                            # Use XPath for Playwright
                            if wait_visible:
                                element = self.page.wait_for_selector(f"xpath={selector}", state="visible", timeout=int(timeout * 1000))
                            else:
                                element = self.page.wait_for_selector(f"xpath={selector}", state="attached", timeout=int(timeout * 1000))
                        else:
                            # CSS selector
                            if wait_visible:
                                element = self.page.wait_for_selector(selector, state="visible", timeout=int(timeout * 1000))
                            else:
                                element = self.page.wait_for_selector(selector, state="attached", timeout=int(timeout * 1000))
                    except Exception:
                        # Try next selector
                        continue
                
                if element:
                    return element
                    
            except Exception:
                # Try next selector
                continue
        
        return None
    
    def type_text(
        self,
        element: Any,
        text: str,
        clear_first: bool = True,
        human_like: bool = True
    ) -> bool:
        """
        Type text into element with human-like delays.
        
        Args:
            element: Element to type into
            text: Text to type
            clear_first: Clear field before typing
            human_like: Use human-like typing delays
            
        Returns:
            True if successful, False otherwise
        """
        if not element:
            return False
        
        try:
            if self.library_used == "undetected-chromedriver":
                if clear_first:
                    element.clear()
                
                if human_like:
                    # Type character by character with delays
                    for char in text:
                        element.send_keys(char)
                        time.sleep(0.05 + (ord(char) % 10) / 1000.0)  # Variable delay
                else:
                    element.send_keys(text)
                    
            elif self.library_used == "playwright":
                if clear_first:
                    element.fill("")
                    time.sleep(0.1)
                
                if human_like:
                    # Type with delays
                    element.type(text, delay=50 + (len(text) % 10))
                else:
                    element.fill(text)
            
            return True
        except Exception:
            return False
    
    def click_element(self, element: Any, human_like: bool = True) -> bool:
        """
        Click element with human-like behavior.
        
        Args:
            element: Element to click
            human_like: Add small delay before clicking
            
        Returns:
            True if successful, False otherwise
        """
        if not element:
            return False
        
        try:
            if human_like:
                time.sleep(0.1 + (hash(str(element)) % 10) / 100.0)  # Small random delay
            
            if self.library_used == "undetected-chromedriver":
                element.click()
            elif self.library_used == "playwright":
                element.click()
            
            return True
        except Exception:
            return False
    
    def wait_for_navigation(self, timeout: float = 15.0, expected_url_pattern: Optional[str] = None) -> bool:
        """
        Wait for page navigation after action.
        
        Args:
            timeout: Maximum wait time in seconds
            expected_url_pattern: Optional URL pattern to wait for
            
        Returns:
            True if navigation detected, False on timeout
        """
        if not self.is_initialized:
            return False
        
        start_url = self.get_current_url()
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            current_url = self.get_current_url()
            
            if current_url != start_url:
                # Check if matches expected pattern
                if expected_url_pattern:
                    if expected_url_pattern in current_url:
                        self.wait_for_page_load(5.0)
                        return True
                else:
                    # Any URL change is good
                    self.wait_for_page_load(5.0)
                    return True
            
            time.sleep(0.2)
        
        return False


def load_trading212_credentials() -> Optional[Dict[str, str]]:
    """
    Load Trading212 credentials from config.
    
    Returns:
        Dictionary with username and password, or None if not found
    """
    try:
        config_file = get_data_path() / 'config_v15.json'
        if not config_file.exists():
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        trading212_config = config.get("browser_automation", {}).get("trading212", {})
        
        username = trading212_config.get("username")
        password = trading212_config.get("password")
        
        if username and password:
            return {
                "username": username,
                "password": password
            }
    except Exception:
        pass
    
    return None

