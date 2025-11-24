"""
Sentiment System Debugger
Debug utilities for sentiment analysis and news monitoring.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from ..sentiment.news_monitor import get_news_monitor
from ..sentiment.analyzer import get_sentiment_analyzer
from ..sentiment.override import get_sentiment_override


class SentimentDebugger:
    """Debug sentiment system."""
    
    def __init__(self):
        """Initialize sentiment debugger."""
        pass
    
    def debug_news_monitoring(
        self,
        ticker: str
    ) -> Dict[str, Any]:
        """Debug news monitoring for a ticker."""
        debug_info = {
            "test": "debug_news_monitoring",
            "timestamp": datetime.now().isoformat(),
            "input": {"ticker": ticker},
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        monitor = get_news_monitor()
        
        try:
            news_items = monitor.get_recent_news(ticker, hours=24)
            debug_info["steps"].append({
                "step": 1,
                "action": f"Fetch news for {ticker}",
                "result": f"Found {len(news_items)} items"
            })
        except Exception as e:
            debug_info["errors"].append(f"News fetch error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        debug_info["output"] = {"news_count": len(news_items)}
        debug_info["success"] = True
        return debug_info
    
    def debug_sentiment_analysis(
        self,
        test_texts: List[str]
    ) -> Dict[str, Any]:
        """Debug sentiment analysis."""
        debug_info = {
            "test": "debug_sentiment_analysis",
            "timestamp": datetime.now().isoformat(),
            "input": {"test_texts": len(test_texts)},
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        analyzer = get_sentiment_analyzer()
        results = []
        
        for i, text in enumerate(test_texts):
            try:
                result = analyzer.analyze_text(text)
                results.append(result)
                debug_info["steps"].append({
                    "step": i + 1,
                    "action": f"Analyze text {i+1}",
                    "result": f"Score: {result.get('sentiment_score', 0):.2f}"
                })
            except Exception as e:
                debug_info["warnings"].append(f"Analysis {i+1} error: {str(e)}")
        
        debug_info["output"] = {"results": results}
        debug_info["success"] = True
        return debug_info
    
    def debug_override_logic(
        self,
        ticker: str
    ) -> Dict[str, Any]:
        """Debug sentiment override logic."""
        debug_info = {
            "test": "debug_override_logic",
            "timestamp": datetime.now().isoformat(),
            "input": {"ticker": ticker},
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        override = get_sentiment_override()
        
        try:
            should_block, reason = override.should_block_trade(ticker)
            debug_info["steps"].append({
                "step": 1,
                "action": "Check if should block trade",
                "result": "BLOCK" if should_block else "ALLOW"
            })
            
            status = override.get_override_status()
            debug_info["steps"].append({
                "step": 2,
                "action": "Get override status",
                "result": f"Protective mode: {status['protective_mode']}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Override check error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        debug_info["output"] = {
            "should_block": should_block,
            "reason": reason,
            "status": status
        }
        debug_info["success"] = True
        return debug_info


def get_sentiment_debugger() -> SentimentDebugger:
    """Get global sentiment debugger instance."""
    global _sentiment_debugger
    if _sentiment_debugger is None:
        _sentiment_debugger = SentimentDebugger()
    return _sentiment_debugger

