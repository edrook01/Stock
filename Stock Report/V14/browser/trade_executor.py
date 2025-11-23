"""
Trade Execution Functions
Executes trades in Trading212 web interface with human-like behavior.
"""

from typing import Dict, Optional, Tuple, List
from datetime import datetime
import time

from .automation import BrowserAutomation, load_trading212_credentials
from .human_behavior import HumanBehavior
from .window_manager import WindowManager
from .position_monitor import PositionMonitor


class TradeExecutor:
    """Executes trades via browser automation."""
    
    def __init__(self, browser: BrowserAutomation):
        """
        Initialize trade executor.
        
        Args:
            browser: BrowserAutomation instance
        """
        self.browser = browser
        self.human_behavior = HumanBehavior()
        self.window_manager = WindowManager(browser)
        self.position_monitor = PositionMonitor(browser, self.window_manager)
        self.logged_in = False
    
    def login(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Log in to Trading212.
        
        Args:
            username: Trading212 username (optional, will load from config)
            password: Trading212 password (optional, will load from config)
            
        Returns:
            True if login successful, False otherwise
        """
        # Load credentials if not provided
        if not username or not password:
            creds = load_trading212_credentials()
            if not creds:
                return False
            username = creds["username"]
            password = creds["password"]
        
        # Navigate to login page
        if not self.browser.navigate("https://www.trading212.com/en/login"):
            return False
        
        # Wait for page to load
        if not self.browser.wait_for_page_load(timeout=15.0):
            return False
        
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Find username/email field - try multiple selectors for robustness
        username_selectors = [
            "input[name='email']",
            "input[type='email']",
            "input[name='username']",
            "input[type='text'][name='email']",
            "#email",
            "#username",
            "input.email",
            "input.username",
            "//input[@type='email']",
            "//input[@name='email']",
            "//input[@name='username']"
        ]
        
        username_field = self.browser.find_element(username_selectors, timeout=10.0, wait_visible=True)
        if not username_field:
            # Try to find by label text
            username_selectors_fallback = [
                "//label[contains(text(), 'Email')]/following::input[1]",
                "//label[contains(text(), 'email')]/following::input[1]",
                "//input[@aria-label='Email']",
                "//input[@aria-label='email']"
            ]
            username_field = self.browser.find_element(username_selectors_fallback, by="xpath", timeout=5.0, wait_visible=True)
        
        if not username_field:
            return False
        
        # Type username with human-like delays
        self.human_behavior.random_delay(0.3, 0.8)
        if not self.browser.type_text(username_field, username, clear_first=True, human_like=True):
            return False
        
        self.human_behavior.random_delay(0.5, 1.0)
        
        # Find password field - try multiple selectors
        password_selectors = [
            "input[name='password']",
            "input[type='password']",
            "#password",
            "input.password",
            "//input[@type='password']",
            "//input[@name='password']"
        ]
        
        password_field = self.browser.find_element(password_selectors, timeout=10.0, wait_visible=True)
        if not password_field:
            # Try to find by label text
            password_selectors_fallback = [
                "//label[contains(text(), 'Password')]/following::input[1]",
                "//label[contains(text(), 'password')]/following::input[1]",
                "//input[@aria-label='Password']",
                "//input[@aria-label='password']"
            ]
            password_field = self.browser.find_element(password_selectors_fallback, by="xpath", timeout=5.0, wait_visible=True)
        
        if not password_field:
            return False
        
        # Type password with human-like delays
        self.human_behavior.random_delay(0.3, 0.8)
        if not self.browser.type_text(password_field, password, clear_first=True, human_like=True):
            return False
        
        self.human_behavior.random_delay(0.5, 1.0)
        
        # Find login button - try multiple selectors
        login_button_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button[contains(@class, 'login')]",
            "button[contains(@class, 'submit')]",
            "//button[contains(text(), 'Log in')]",
            "//button[contains(text(), 'Login')]",
            "//button[contains(text(), 'Sign in')]",
            "//button[contains(text(), 'Submit')]",
            "//button[@type='submit']",
            "//input[@type='submit']"
        ]
        
        login_button = self.browser.find_element(login_button_selectors, timeout=10.0, wait_visible=True)
        if not login_button:
            return False
        
        # Click login button
        self.human_behavior.random_delay(0.3, 0.8)
        if not self.browser.click_element(login_button, human_like=True):
            return False
        
        # Wait for navigation after login
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Wait for navigation to complete (should leave login page)
        navigation_success = self.browser.wait_for_navigation(
            timeout=15.0,
            expected_url_pattern=None  # Any navigation away from login page is success
        )
        
        if not navigation_success:
            # Still check if URL changed or if we're on dashboard
            current_url = self.browser.get_current_url()
            if current_url and "login" not in current_url.lower():
                navigation_success = True
            else:
                # Check for error messages
                error_selectors = [
                    ".error",
                    ".error-message",
                    "[role='alert']",
                    "//div[contains(@class, 'error')]",
                    "//div[contains(text(), 'incorrect')]",
                    "//div[contains(text(), 'Invalid')]"
                ]
                error_element = self.browser.find_element(error_selectors, timeout=3.0, wait_visible=False)
                if error_element:
                    # Login failed - error message present
                    return False
        
        # Verify login success - check if we're on trading platform or dashboard
        time.sleep(1.0)  # Brief wait for page to settle
        current_url = self.browser.get_current_url()
        
        if not current_url:
            return False
        
        # Check if we're on a Trading212 platform page (not login)
        login_indicators = ["login", "sign-in", "auth"]
        if any(indicator in current_url.lower() for indicator in login_indicators):
            # Still on login page - check for 2FA or other requirements
            two_factor_selectors = [
                "input[name='code']",
                "input[type='text'][name='code']",
                "//input[contains(@placeholder, 'code')]",
                "//input[contains(@placeholder, 'Code')]",
                "//input[@name='code']"
            ]
            two_factor_field = self.browser.find_element(two_factor_selectors, timeout=3.0, wait_visible=False)
            if two_factor_field:
                # 2FA required - login partially successful but needs 2FA
                # For now, return False as we don't handle 2FA yet
                return False
            
            return False
        
        # Success indicators - check for trading platform elements
        platform_indicators = [
            "trading212.com/en/trading",
            "trading212.com/trading",
            "trading212.com/en/platform",
            "trading212.com/platform"
        ]
        
        login_successful = any(indicator in current_url.lower() for indicator in platform_indicators)
        
        if not login_successful:
            # Check for dashboard/account elements as alternative success indicator
            dashboard_selectors = [
                "[data-testid='dashboard']",
                "[class*='dashboard']",
                "[class*='trading-platform']",
                "//div[contains(@class, 'dashboard')]",
                "//div[contains(@class, 'trading-platform')]"
            ]
            dashboard_element = self.browser.find_element(dashboard_selectors, timeout=3.0, wait_visible=False)
            login_successful = dashboard_element is not None
        
        if login_successful:
            self.logged_in = True
            self.window_manager.update_login_state(True)
            return True
        
        return False
    
    def open_trade(
        self,
        ticker: str,
        side: str,
        size: float,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Open a trade in Trading212.
        
        Args:
            ticker: Stock ticker symbol
            side: Trade side ("BUY" or "SELL")
            size: Position size
            stop_price: Stop-loss price (optional)
            target_price: Take-profit price (optional)
            
        Returns:
            Dictionary with execution result
        """
        if not self.logged_in:
            if not self.login():
                return {
                    "success": False,
                    "error": "Not logged in and login failed"
                }
        
        # Navigate to CFD trading interface
        trading_url = "https://www.trading212.com/en/trading/platform/cfd"
        if not self.browser.navigate(trading_url):
            return {
                "success": False,
                "error": "Failed to navigate to trading interface"
            }
        
        if not self.browser.wait_for_page_load(timeout=15.0):
            return {
                "success": False,
                "error": "Page did not load properly"
            }
        
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Step 1: Search for ticker
        search_selectors = [
            "input[placeholder*='Search']",
            "input[type='search']",
            "input.search",
            "[data-testid='search-input']",
            "//input[contains(@placeholder, 'Search')]",
            "//input[contains(@placeholder, 'search')]"
        ]
        
        search_field = self.browser.find_element(search_selectors, timeout=10.0, wait_visible=True)
        if not search_field:
            return {
                "success": False,
                "error": "Could not find search field"
            }
        
        # Clear and type ticker
        self.human_behavior.random_delay(0.3, 0.8)
        if not self.browser.type_text(search_field, ticker.upper(), clear_first=True, human_like=True):
            return {
                "success": False,
                "error": "Failed to type ticker in search field"
            }
        
        self.human_behavior.random_delay(0.5, 1.0)
        
        # Step 2: Click on ticker result from dropdown
        ticker_result_selectors = [
            f"[data-symbol='{ticker.upper()}']",
            f"[data-ticker='{ticker.upper()}']",
            f"//div[contains(text(), '{ticker.upper()}')]",
            f"//span[contains(text(), '{ticker.upper()}')]",
            f"//a[contains(@href, '{ticker.upper()}')]",
            f".search-result[data-symbol='{ticker.upper()}']"
        ]
        
        ticker_result = self.browser.find_element(ticker_result_selectors, timeout=5.0, wait_visible=True)
        if not ticker_result:
            # Try clicking first result in dropdown
            first_result_selectors = [
                ".search-result:first-child",
                ".search-dropdown li:first-child",
                "//div[@class='search-result'][1]",
                "[class*='search-result']:first-of-type"
            ]
            ticker_result = self.browser.find_element(first_result_selectors, timeout=3.0, wait_visible=True)
        
        if not ticker_result:
            return {
                "success": False,
                "error": f"Could not find ticker {ticker} in search results"
            }
        
        self.human_behavior.random_delay(0.3, 0.8)
        if not self.browser.click_element(ticker_result, human_like=True):
            return {
                "success": False,
                "error": "Failed to click ticker result"
            }
        
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Step 3: Wait for trading panel to open and click Buy/Sell button
        side_upper = side.upper()
        if side_upper not in ["BUY", "SELL"]:
            return {
                "success": False,
                "error": f"Invalid trade side: {side}. Must be 'BUY' or 'SELL'"
            }
        
        buy_sell_selectors = [
            f"button[data-side='{side_upper.lower()}']",
            f"button.{side_upper.lower()}",
            f"//button[contains(text(), '{side_upper}')]",
            f"//button[contains(@class, '{side_upper.lower()}')]",
            f"[data-testid='{side_upper.lower()}-button']",
            f"button:contains('{side_upper}')"
        ]
        
        buy_sell_button = self.browser.find_element(buy_sell_selectors, timeout=10.0, wait_visible=True)
        if not buy_sell_button:
            return {
                "success": False,
                "error": f"Could not find {side_upper} button"
            }
        
        # Scroll to button if needed
        self.window_manager.scroll_to_element(buy_sell_selectors[0])
        self.human_behavior.random_delay(0.3, 0.8)
        
        if not self.browser.click_element(buy_sell_button, human_like=True):
            return {
                "success": False,
                "error": f"Failed to click {side_upper} button"
            }
        
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Step 4: Enter position size
        size_field_selectors = [
            "input[name='quantity']",
            "input[name='size']",
            "input[name='amount']",
            "input[type='number']",
            "[data-testid='quantity-input']",
            "//input[contains(@placeholder, 'quantity')]",
            "//input[contains(@placeholder, 'size')]"
        ]
        
        size_field = self.browser.find_element(size_field_selectors, timeout=10.0, wait_visible=True)
        if not size_field:
            return {
                "success": False,
                "error": "Could not find position size input field"
            }
        
        # Clear and enter size
        self.human_behavior.random_delay(0.3, 0.8)
        size_str = f"{size:.2f}" if size % 1 != 0 else f"{int(size)}"
        if not self.browser.type_text(size_field, size_str, clear_first=True, human_like=True):
            return {
                "success": False,
                "error": "Failed to enter position size"
            }
        
        self.human_behavior.random_delay(0.5, 1.0)
        
        # Step 5: Set stop-loss if provided
        if stop_price:
            stop_loss_selectors = [
                "input[name='stopLoss']",
                "input[name='stop-loss']",
                "[data-testid='stop-loss-input']",
                "//input[contains(@placeholder, 'Stop')]",
                "//input[contains(@placeholder, 'stop')]"
            ]
            
            stop_field = self.browser.find_element(stop_loss_selectors, timeout=5.0, wait_visible=True)
            if stop_field:
                self.human_behavior.random_delay(0.2, 0.5)
                stop_str = f"{stop_price:.2f}"
                self.browser.type_text(stop_field, stop_str, clear_first=True, human_like=True)
                self.human_behavior.random_delay(0.3, 0.8)
        
        # Step 6: Set take-profit if provided
        if target_price:
            take_profit_selectors = [
                "input[name='takeProfit']",
                "input[name='take-profit']",
                "[data-testid='take-profit-input']",
                "//input[contains(@placeholder, 'Take')]",
                "//input[contains(@placeholder, 'take')]"
            ]
            
            target_field = self.browser.find_element(take_profit_selectors, timeout=5.0, wait_visible=True)
            if target_field:
                self.human_behavior.random_delay(0.2, 0.5)
                target_str = f"{target_price:.2f}"
                self.browser.type_text(target_field, target_str, clear_first=True, human_like=True)
                self.human_behavior.random_delay(0.3, 0.8)
        
        # Step 7: Click confirm/execute button
        confirm_selectors = [
            "button[type='submit']",
            "button.confirm",
            "button.execute",
            "[data-testid='confirm-button']",
            "//button[contains(text(), 'Confirm')]",
            "//button[contains(text(), 'Execute')]",
            "//button[contains(text(), 'Open')]"
        ]
        
        confirm_button = self.browser.find_element(confirm_selectors, timeout=10.0, wait_visible=True)
        if not confirm_button:
            return {
                "success": False,
                "error": "Could not find confirm/execute button"
            }
        
        # Scroll to confirm button
        self.window_manager.scroll_to_element(confirm_selectors[0])
        self.human_behavior.random_delay(0.5, 1.0)
        
        if not self.browser.click_element(confirm_button, human_like=True):
            return {
                "success": False,
                "error": "Failed to click confirm button"
            }
        
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Step 8: Verify trade opened - check for success message or position in list
        success_indicators = [
            "//div[contains(text(), 'success')]",
            "//div[contains(text(), 'opened')]",
            "//div[contains(text(), 'executed')]",
            ".success-message",
            "[data-testid='trade-success']"
        ]
        
        # Wait a moment for confirmation
        self.human_behavior.random_delay(1.0, 2.0)
        
        success_message = self.browser.find_element(success_indicators, timeout=5.0, wait_visible=False)
        if success_message:
            # Trade confirmed
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                "success": True,
                "trade_id": trade_id,
                "ticker": ticker.upper(),
                "side": side_upper,
                "size": size,
                "stop_price": stop_price,
                "target_price": target_price,
                "executed_at": datetime.now().isoformat(),
                "confirmed": True
            }
        
        # If no success message, still return success (might be silent confirmation)
        # User should verify manually or check positions
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "success": True,
            "trade_id": trade_id,
            "ticker": ticker.upper(),
            "side": side_upper,
            "size": size,
            "stop_price": stop_price,
            "target_price": target_price,
            "executed_at": datetime.now().isoformat(),
            "confirmed": False,
            "warning": "Trade executed but confirmation message not detected. Please verify manually."
        }
    
    def close_trade(self, position_id: str) -> Dict[str, any]:
        """
        Close a trade in Trading212.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Dictionary with execution result
        """
        if not self.logged_in:
            return {
                "success": False,
                "error": "Not logged in"
            }
        
        # Navigate to positions/portfolio page
        positions_url = "https://www.trading212.com/en/trading/platform/cfd"
        if not self.browser.navigate(positions_url):
            return {
                "success": False,
                "error": "Failed to navigate to positions"
            }
        
        if not self.browser.wait_for_page_load(timeout=15.0):
            return {
                "success": False,
                "error": "Page did not load properly"
            }
        
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Step 1: Find position in positions list
        # Position can be identified by position_id, ticker, or other identifier
        # Try multiple strategies to find the position
        
        position_selectors = [
            f"[data-position-id='{position_id}']",
            f"[data-id='{position_id}']",
            f"//tr[contains(@data-position-id, '{position_id}')]",
            f"//div[contains(@data-position-id, '{position_id}')]",
            f".position-item[data-id='{position_id}']"
        ]
        
        position_element = self.browser.find_element(position_selectors, timeout=10.0, wait_visible=True)
        
        # If not found by ID, try finding by text/name
        if not position_element:
            # Scroll to positions section
            positions_section_selectors = [
                ".positions-list",
                "[data-testid='positions']",
                "//div[contains(@class, 'positions')]",
                "//section[contains(@class, 'positions')]"
            ]
            positions_section = self.browser.find_element(positions_section_selectors, timeout=5.0, wait_visible=False)
            if positions_section:
                self.window_manager.scroll_to_element(positions_section_selectors[0])
                self.human_behavior.random_delay(0.5, 1.0)
        
        # Step 2: Find and click close button for the position
        close_button_selectors = [
            f"button[data-position-id='{position_id}'][data-action='close']",
            f"[data-position-id='{position_id}'] button.close",
            f"[data-position-id='{position_id}'] button[aria-label*='Close']",
            f"//tr[contains(@data-position-id, '{position_id}')]//button[contains(text(), 'Close')]",
            f"//div[contains(@data-position-id, '{position_id}')]//button[contains(@class, 'close')]",
            f"//button[contains(@aria-label, 'Close') and contains(@data-position-id, '{position_id}')]"
        ]
        
        close_button = self.browser.find_element(close_button_selectors, timeout=10.0, wait_visible=True)
        
        if not close_button:
            # Try finding first close button if position ID not found
            # This is a fallback - user should verify correct position
            fallback_close_selectors = [
                "button.close:first-of-type",
                ".position-item:first-of-type button.close",
                "//button[contains(text(), 'Close')][1]",
                "[data-action='close']:first-of-type"
            ]
            close_button = self.browser.find_element(fallback_close_selectors, timeout=5.0, wait_visible=True)
        
        if not close_button:
            return {
                "success": False,
                "error": f"Could not find close button for position {position_id}"
            }
        
        # Scroll to close button
        self.window_manager.scroll_to_element(close_button_selectors[0] if close_button_selectors else None)
        self.human_behavior.random_delay(0.3, 0.8)
        
        if not self.browser.click_element(close_button, human_like=True):
            return {
                "success": False,
                "error": "Failed to click close button"
            }
        
        self.human_behavior.random_delay(0.5, 1.0)
        
        # Step 3: Confirm closure if confirmation dialog appears
        confirm_dialog_selectors = [
            ".modal button.confirm",
            "[data-testid='confirm-close']",
            "//button[contains(text(), 'Confirm')]",
            "//button[contains(text(), 'Close Position')]",
            "//button[contains(text(), 'Yes')]"
        ]
        
        confirm_button = self.browser.find_element(confirm_dialog_selectors, timeout=5.0, wait_visible=True)
        if confirm_button:
            self.human_behavior.random_delay(0.3, 0.8)
            self.browser.click_element(confirm_button, human_like=True)
            self.human_behavior.random_delay(1.0, 2.0)
        
        # Step 4: Verify position closed - check for success message or position removed
        self.human_behavior.random_delay(1.0, 2.0)
        
        # Check if position still exists (should not)
        position_still_exists = self.browser.find_element(
            [f"[data-position-id='{position_id}']"],
            timeout=3.0,
            wait_visible=False
        ) is not None
        
        if position_still_exists:
            # Position might still be closing or confirmation failed
            return {
                "success": False,
                "error": "Position still appears to be open. Closure may have failed.",
                "position_id": position_id,
                "closed_at": datetime.now().isoformat()
            }
        
        # Success - position closed
        return {
            "success": True,
            "position_id": position_id,
            "closed_at": datetime.now().isoformat(),
            "confirmed": True
        }
    
    def get_account_status(self) -> Dict[str, any]:
        """
        Get account status (equity, margin, positions).
        
        Returns:
            Dictionary with account status
        """
        if not self.logged_in:
            return {
                "error": "Not logged in"
            }
        
        # Use position monitor to get account status
        account_status = self.position_monitor.get_account_status()
        positions = self.position_monitor.get_open_positions()
        
        account_status["positions"] = positions
        account_status["timestamp"] = datetime.now().isoformat()
        
        return account_status
    
    def get_open_positions(self) -> List[Dict[str, any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.logged_in:
            return []
        
        return self.position_monitor.get_open_positions()
    
    def get_position_by_ticker(self, ticker: str) -> Optional[Dict[str, any]]:
        """
        Get position by ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Position dictionary if found, None otherwise
        """
        if not self.logged_in:
            return None
        
        return self.position_monitor.get_position_by_ticker(ticker)

