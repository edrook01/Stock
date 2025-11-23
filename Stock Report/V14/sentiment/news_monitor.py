"""
News Monitoring
Monitors news feeds for trading instruments and maintains economic calendar.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

# Handle both relative and absolute imports for portability
try:
    from ..core.portable_paths import get_path
except ImportError:
    # Fallback for direct execution
    from core.portable_paths import get_path


class NewsMonitor:
    """Monitors news feeds for trading instruments."""
    
    def __init__(self):
        """Initialize news monitor."""
        self.economic_calendar: List[Dict] = []
        self.news_cache: Dict[str, List[Dict]] = {}
        self._load_economic_calendar()
    
    def check_earnings(self, ticker: str, date: Optional[datetime] = None) -> Optional[Dict]:
        """
        Check if ticker has earnings on a given date.
        
        Args:
            ticker: Stock ticker symbol
            date: Date to check (defaults to today)
            
        Returns:
            Earnings event dictionary, or None if no earnings
        """
        if date is None:
            date = datetime.now()
        
        # Check economic calendar
        for event in self.economic_calendar:
            if event.get("type") == "earnings" and event.get("ticker") == ticker.upper():
                event_date = datetime.fromisoformat(event.get("date", ""))
                if abs((event_date - date).days) <= 1:  # Within 1 day
                    return event
        
        return None
    
    def check_scheduled_events(
        self,
        ticker: str,
        hours_ahead: int = 24
    ) -> List[Dict]:
        """
        Check for scheduled events in the next N hours.
        
        Args:
            ticker: Stock ticker symbol
            hours_ahead: Hours to look ahead
            
        Returns:
            List of scheduled events
        """
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)
        
        events = []
        for event in self.economic_calendar:
            if event.get("ticker") == ticker.upper():
                event_date = datetime.fromisoformat(event.get("date", ""))
                if now <= event_date <= cutoff:
                    events.append(event)
        
        return events
    
    def add_economic_event(
        self,
        event_type: str,
        ticker: str,
        date: datetime,
        description: str = ""
    ) -> None:
        """
        Add an economic event to the calendar.
        
        Args:
            event_type: Event type ("earnings", "fda", "merger", etc.)
            ticker: Stock ticker symbol
            date: Event date
            description: Event description
        """
        event = {
            "type": event_type,
            "ticker": ticker.upper(),
            "date": date.isoformat(),
            "description": description,
            "added_at": datetime.now().isoformat()
        }
        
        self.economic_calendar.append(event)
        self._save_economic_calendar()
    
    def get_major_events(self, date: Optional[datetime] = None) -> List[Dict]:
        """
        Get major market events (Fed decisions, etc.).
        
        Args:
            date: Date to check (defaults to today)
            
        Returns:
            List of major events
        """
        if date is None:
            date = datetime.now()
        
        major_events = []
        for event in self.economic_calendar:
            if event.get("type") in ["fed_decision", "election", "major_announcement"]:
                event_date = datetime.fromisoformat(event.get("date", ""))
                if abs((event_date - date).days) <= 1:
                    major_events.append(event)
        
        return major_events
    
    def _save_economic_calendar(self) -> None:
        """Save economic calendar to file."""
        try:
            data_dir = get_path('data')
            data_dir.mkdir(parents=True, exist_ok=True)
            
            calendar_file = data_dir / 'economic_calendar.json'
            
            with open(calendar_file, 'w') as f:
                json.dump(self.economic_calendar, f, indent=2)
        except Exception:
            # Silent failure on save errors
            pass
    
    def _load_economic_calendar(self) -> None:
        """Load economic calendar from file."""
        try:
            data_dir = get_path('data')
            calendar_file = data_dir / 'economic_calendar.json'
            
            if not calendar_file.exists():
                return
            
            with open(calendar_file, 'r') as f:
                self.economic_calendar = json.load(f)
        except Exception:
            # Silent failure on load errors
            self.economic_calendar = []


# Global news monitor instance
_news_monitor: Optional[NewsMonitor] = None


def get_news_monitor() -> NewsMonitor:
    """Get global news monitor instance."""
    global _news_monitor
    if _news_monitor is None:
        _news_monitor = NewsMonitor()
    return _news_monitor

