"""
Sentiment Analysis
Analyzes news sentiment using NLP and keyword-based scoring.
"""

from typing import Dict, List, Optional
from datetime import datetime
import re


class SentimentAnalyzer:
    """Analyzes sentiment from news and text."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        # Positive keywords
        self.positive_keywords = [
            "beat", "exceed", "growth", "profit", "gain", "rise", "surge",
            "upgrade", "bullish", "positive", "strong", "outperform"
        ]
        
        # Negative keywords
        self.negative_keywords = [
            "miss", "decline", "loss", "drop", "fall", "downgrade",
            "bearish", "negative", "weak", "underperform", "warning"
        ]
        
        # Major event keywords
        self.major_event_keywords = [
            "earnings", "fda", "merger", "acquisition", "lawsuit",
            "ceo", "cfo", "resignation", "bankruptcy"
        ]
    
    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment from text using word boundary matching to avoid false positives.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        text_lower = text.lower()
        
        # Count positive and negative keywords using word boundaries to avoid false positives
        # e.g., "miss" in "dismiss" or "drop" in "raindrop" won't match
        positive_count = 0
        negative_count = 0
        
        for keyword in self.positive_keywords:
            # Use word boundaries for exact word matching
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                positive_count += 1
        
        for keyword in self.negative_keywords:
            # Use word boundaries for exact word matching
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                negative_count += 1
        
        # Calculate sentiment score (-1 to +1)
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / max(total_keywords, 1)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Check for major events using word boundaries
        major_events = []
        for keyword in self.major_event_keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                major_events.append(keyword)
        
        # Calculate confidence based on keyword count and sentiment consistency
        total_keywords = positive_count + negative_count
        base_confidence = min(1.0, total_keywords / 5.0) if total_keywords > 0 else 0.0
        
        # Boost confidence if sentiment is consistent (all positive or all negative)
        if total_keywords > 0:
            sentiment_consistency = abs(sentiment_score)  # Higher if all positive or all negative
            consistency_bonus = sentiment_consistency * 0.2
            # Boost confidence if major events detected (higher importance)
            importance_bonus = min(0.3, len(major_events) * 0.15)
            confidence = min(1.0, base_confidence + consistency_bonus + importance_bonus)
        else:
            confidence = 0.0
        
        return {
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "positive_keywords": positive_count,
            "negative_keywords": negative_count,
            "major_events": major_events,
            "is_major_event": len(major_events) > 0
        }
    
    def analyze_news_list(self, news_items: List[Dict]) -> Dict[str, any]:
        """
        Analyze sentiment from a list of news items.
        
        Args:
            news_items: List of news dictionaries with 'title' and/or 'content'
            
        Returns:
            Aggregated sentiment analysis
        """
        if not news_items:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "news_count": 0
            }
        
        sentiments = []
        all_major_events = []
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            analysis = self.analyze_text(text)
            sentiments.append(analysis["sentiment_score"])
            
            if analysis.get("is_major_event"):
                all_major_events.extend(analysis.get("major_events", []))
        
        # Aggregate sentiment (weighted average)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Higher confidence if more news items
        confidence = min(1.0, len(news_items) / 10.0)
        
        return {
            "sentiment_score": avg_sentiment,
            "confidence": confidence,
            "news_count": len(news_items),
            "major_events": list(set(all_major_events)),
            "is_major_event": len(all_major_events) > 0
        }
    
    def get_sentiment_category(self, sentiment_score: float) -> str:
        """
        Categorize sentiment score.
        
        Args:
            sentiment_score: Sentiment score (-1 to +1)
            
        Returns:
            Sentiment category string
        """
        if sentiment_score >= 0.6:
            return "very_positive"
        elif sentiment_score >= 0.2:
            return "positive"
        elif sentiment_score >= -0.2:
            return "neutral"
        elif sentiment_score >= -0.6:
            return "negative"
        else:
            return "very_negative"


# Global sentiment analyzer instance
_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get global sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

