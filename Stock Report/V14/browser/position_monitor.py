"""
Position Monitoring
Monitors open positions from the Trading212 CFD browser interface.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import re
import time

from .automation import BrowserAutomation
from .window_manager import WindowManager


class PositionMonitor:
    """Monitors positions from Trading212 CFD interface."""
    
    def __init__(self, browser: BrowserAutomation, window_manager: WindowManager):
        """
        Initialize position monitor.
        
        Args:
            browser: BrowserAutomation instance
            window_manager: WindowManager instance
        """
        self.browser = browser
        self.window_manager = window_manager
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions from the Trading212 interface.
        
        Returns:
            List of position dictionaries with:
            {
                "position_id": str,
                "ticker": str,
                "side": str,  # "BUY" or "SELL"
                "size": float,
                "entry_price": float,
                "current_price": float,
                "unrealized_pnl": float,
                "unrealized_pnl_percent": float,
                "stop_loss": Optional[float],
                "take_profit": Optional[float]
            }
        """
        if not self.browser.is_ready():
            return []
        
        # Navigate to positions if not already there
        current_url = self.browser.get_current_url()
        if not current_url or "trading" not in current_url.lower():
            self.browser.navigate("https://www.trading212.com/en/trading/platform/cfd")
            self.browser.wait_for_page_load(timeout=10.0)
            time.sleep(1.0)
        
        positions = []
        
        # Try multiple strategies to find positions
        # Strategy 1: Find positions table/list
        positions_container_selectors = [
            ".positions-list",
            ".open-positions",
            "[data-testid='positions']",
            "//div[contains(@class, 'positions')]",
            "//table[contains(@class, 'positions')]"
        ]
        
        positions_container = self.browser.find_element(
            positions_container_selectors,
            timeout=5.0,
            wait_visible=False
        )
        
        if positions_container:
            # Extract positions from container using JavaScript
            try:
                if self.browser.library_used == "undetected-chromedriver":
                    positions_data = self.browser.driver.execute_script("""
                        var positions = [];
                        var rows = document.querySelectorAll('.position-item, tr[data-position-id], [data-position-id]');
                        for (var i = 0; i < rows.length; i++) {
                            var row = rows[i];
                            var positionId = row.getAttribute('data-position-id') || row.getAttribute('data-id') || 'pos_' + i;
                            var ticker = row.querySelector('[data-ticker]')?.getAttribute('data-ticker') || 
                                       row.querySelector('.ticker')?.textContent?.trim() || '';
                            var side = row.querySelector('[data-side]')?.getAttribute('data-side')?.toUpperCase() || 
                                     (row.querySelector('.side')?.textContent?.toUpperCase() || '');
                            var size = parseFloat(row.querySelector('[data-size]')?.getAttribute('data-size') || 
                                                 row.querySelector('.size')?.textContent?.replace(/[^0-9.]/g, '') || 0);
                            var entryPrice = parseFloat(row.querySelector('[data-entry-price]')?.getAttribute('data-entry-price') || 
                                                       row.querySelector('.entry-price')?.textContent?.replace(/[^0-9.]/g, '') || 0);
                            var currentPrice = parseFloat(row.querySelector('[data-current-price]')?.getAttribute('data-current-price') || 
                                                         row.querySelector('.current-price')?.textContent?.replace(/[^0-9.]/g, '') || 0);
                            var pnl = parseFloat(row.querySelector('.pnl')?.textContent?.replace(/[^0-9.-]/g, '') || 0);
                            var pnlPercent = parseFloat(row.querySelector('.pnl-percent')?.textContent?.replace(/[^0-9.-]/g, '') || 0);
                            
                            if (ticker) {
                                positions.push({
                                    position_id: positionId,
                                    ticker: ticker,
                                    side: side,
                                    size: size,
                                    entry_price: entryPrice,
                                    current_price: currentPrice,
                                    unrealized_pnl: pnl,
                                    unrealized_pnl_percent: pnlPercent
                                });
                            }
                        }
                        return positions;
                    """)
                elif self.browser.library_used == "playwright":
                    positions_data = self.browser.page.evaluate("""
                        () => {
                            var positions = [];
                            var rows = document.querySelectorAll('.position-item, tr[data-position-id], [data-position-id]');
                            for (var i = 0; i < rows.length; i++) {
                                var row = rows[i];
                                var positionId = row.getAttribute('data-position-id') || row.getAttribute('data-id') || 'pos_' + i;
                                var ticker = row.querySelector('[data-ticker]')?.getAttribute('data-ticker') || 
                                           row.querySelector('.ticker')?.textContent?.trim() || '';
                                var side = row.querySelector('[data-side]')?.getAttribute('data-side')?.toUpperCase() || 
                                         (row.querySelector('.side')?.textContent?.toUpperCase() || '');
                                var size = parseFloat(row.querySelector('[data-size]')?.getAttribute('data-size') || 
                                                     row.querySelector('.size')?.textContent?.replace(/[^0-9.]/g, '') || 0);
                                var entryPrice = parseFloat(row.querySelector('[data-entry-price]')?.getAttribute('data-entry-price') || 
                                                           row.querySelector('.entry-price')?.textContent?.replace(/[^0-9.]/g, '') || 0);
                                var currentPrice = parseFloat(row.querySelector('[data-current-price]')?.getAttribute('data-current-price') || 
                                                             row.querySelector('.current-price')?.textContent?.replace(/[^0-9.]/g, '') || 0);
                                var pnl = parseFloat(row.querySelector('.pnl')?.textContent?.replace(/[^0-9.-]/g, '') || 0);
                                var pnlPercent = parseFloat(row.querySelector('.pnl-percent')?.textContent?.replace(/[^0-9.-]/g, '') || 0);
                                
                                if (ticker) {
                                    positions.push({
                                        position_id: positionId,
                                        ticker: ticker,
                                        side: side,
                                        size: size,
                                        entry_price: entryPrice,
                                        current_price: currentPrice,
                                        unrealized_pnl: pnl,
                                        unrealized_pnl_percent: pnlPercent
                                    });
                                }
                            }
                            return positions;
                        }
                    """)
                
                if positions_data:
                    positions = positions_data
                
            except Exception:
                pass
        
        # Strategy 2: Try scraping from page HTML if JavaScript extraction failed
        if not positions:
            positions = self._scrape_positions_from_html()
        
        # Update window manager with position count
        if positions:
            self.window_manager.update_position_count(len(positions))
        
        return positions
    
    def _scrape_positions_from_html(self) -> List[Dict[str, Any]]:
        """
        Scrape positions from page HTML as fallback method.
        
        Returns:
            List of position dictionaries
        """
        positions = []
        
        try:
            if self.browser.library_used == "undetected-chromedriver":
                page_source = self.browser.driver.page_source
            elif self.browser.library_used == "playwright":
                page_source = self.browser.page.content()
            else:
                return positions
            
            # Simple regex-based extraction (fallback)
            # This is a basic implementation - real implementation would need
            # Trading212-specific HTML structure knowledge
            
            position_pattern = re.compile(
                r'data-position-id=["\']([^"\']+)["\'].*?data-ticker=["\']([^"\']+)["\']',
                re.DOTALL
            )
            
            matches = position_pattern.findall(page_source)
            for match in matches:
                position_id, ticker = match
                positions.append({
                    "position_id": position_id,
                    "ticker": ticker,
                    "side": "BUY",  # Default, would need to extract from HTML
                    "size": 0.0,
                    "entry_price": 0.0,
                    "current_price": 0.0,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_percent": 0.0
                })
        
        except Exception:
            pass
        
        return positions
    
    def get_account_status(self) -> Dict[str, Any]:
        """
        Get account status (equity, balance, margin) from Trading212 interface.
        
        Returns:
            Dictionary with account status:
            {
                "equity": float,
                "balance": float,
                "margin_used": float,
                "margin_available": float,
                "free_margin": float,
                "margin_level": float,
                "positions_count": int
            }
        """
        if not self.browser.is_ready():
            return {
                "equity": 0.0,
                "balance": 0.0,
                "margin_used": 0.0,
                "margin_available": 0.0,
                "free_margin": 0.0,
                "margin_level": 0.0,
                "positions_count": 0
            }
        
        account_status = {
            "equity": 0.0,
            "balance": 0.0,
            "margin_used": 0.0,
            "margin_available": 0.0,
            "free_margin": 0.0,
            "margin_level": 0.0,
            "positions_count": 0
        }
        
        try:
            # Try to extract account info using JavaScript
            if self.browser.library_used == "undetected-chromedriver":
                account_data = self.browser.driver.execute_script("""
                    var account = {};
                    
                    // Try to find equity element
                    var equityEl = document.querySelector('[data-account-equity], .equity, [data-testid="equity"]');
                    if (equityEl) {
                        account.equity = parseFloat(equityEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                    }
                    
                    // Try to find balance element
                    var balanceEl = document.querySelector('[data-account-balance], .balance, [data-testid="balance"]');
                    if (balanceEl) {
                        account.balance = parseFloat(balanceEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                    }
                    
                    // Try to find margin elements
                    var marginUsedEl = document.querySelector('[data-margin-used], .margin-used, [data-testid="margin-used"]');
                    if (marginUsedEl) {
                        account.margin_used = parseFloat(marginUsedEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                    }
                    
                    var marginAvailableEl = document.querySelector('[data-margin-available], .margin-available, [data-testid="margin-available"]');
                    if (marginAvailableEl) {
                        account.margin_available = parseFloat(marginAvailableEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                    }
                    
                    return account;
                """)
            elif self.browser.library_used == "playwright":
                account_data = self.browser.page.evaluate("""
                    () => {
                        var account = {};
                        
                        var equityEl = document.querySelector('[data-account-equity], .equity, [data-testid="equity"]');
                        if (equityEl) {
                            account.equity = parseFloat(equityEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                        }
                        
                        var balanceEl = document.querySelector('[data-account-balance], .balance, [data-testid="balance"]');
                        if (balanceEl) {
                            account.balance = parseFloat(balanceEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                        }
                        
                        var marginUsedEl = document.querySelector('[data-margin-used], .margin-used, [data-testid="margin-used"]');
                        if (marginUsedEl) {
                            account.margin_used = parseFloat(marginUsedEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                        }
                        
                        var marginAvailableEl = document.querySelector('[data-margin-available], .margin-available, [data-testid="margin-available"]');
                        if (marginAvailableEl) {
                            account.margin_available = parseFloat(marginAvailableEl.textContent.replace(/[^0-9.-]/g, '')) || 0;
                        }
                        
                        return account;
                    }
                """)
            
            if account_data:
                account_status.update(account_data)
            
            # Get position count
            positions = self.get_open_positions()
            account_status["positions_count"] = len(positions)
            
            # Calculate free margin
            if account_status["equity"] > 0:
                account_status["free_margin"] = account_status["equity"] - account_status["margin_used"]
            
            # Calculate margin level
            if account_status["margin_used"] > 0:
                account_status["margin_level"] = (account_status["equity"] / account_status["margin_used"]) * 100
        
        except Exception:
            pass
        
        return account_status
    
    def get_position_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific position by ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Position dictionary if found, None otherwise
        """
        positions = self.get_open_positions()
        ticker_upper = ticker.upper()
        
        for position in positions:
            if position.get("ticker", "").upper() == ticker_upper:
                return position
        
        return None

