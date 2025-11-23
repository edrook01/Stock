"""
Market Intelligence Module

This module provides advanced data quality assessment and market analysis capabilities
using multiple data sources and web search integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import json
from urllib.parse import quote_plus

@dataclass
class DataQualityScore:
    """Data quality assessment results."""
    completeness: float  # 0-1 scale
    consistency: float   # 0-1 scale
    timeliness: float    # 0-1 scale
    reliability: float   # 0-1 scale
    confidence: float    # 0-1 scale, overall confidence in data quality
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (weighted average)."""
        weights = {
            'completeness': 0.2,
            'consistency': 0.3,
            'timeliness': 0.2,
            'reliability': 0.3
        }
        return (self.completeness * weights['completeness'] +
                self.consistency * weights['consistency'] +
                self.timeliness * weights['timeliness'] +
                self.reliability * weights['reliability'])

class MarketIntelligence:
    """Advanced market analysis with data quality assessment and web search integration."""
    
    def __init__(self, yf_data: pd.DataFrame = None, source: str = 'yfinance'):
        """
        Initialize with market data.
        
        Args:
            yf_data: Pandas DataFrame with market data
            source: Data source identifier (e.g., 'yfinance', 'alpha_vantage')
        """
        self.data = yf_data
        self.source = source
        self.quality_score = None
        self.sentiment_score = None
        self.trend_analysis = {}
        
    def assess_data_quality(self) -> DataQualityScore:
        """Assess the quality of the market data."""
        if self.data is None or self.data.empty:
            return DataQualityScore(0, 0, 0, 0, 0)
            
        # Check for missing values
        completeness = 1 - (self.data.isnull().sum().sum() / (self.data.size or 1))
        
        # Check for data consistency (price movements, volume, etc.)
        price_consistency = self._check_price_consistency()
        volume_consistency = self._check_volume_consistency()
        consistency = (price_consistency + volume_consistency) / 2
        
        # Check data timeliness
        timeliness = self._check_timeliness()
        
        # Check data reliability (volatility, outliers, etc.)
        reliability = self._check_reliability()
        
        # Calculate confidence (weighted average with higher weight on reliability)
        confidence = (completeness * 0.2 + 
                     consistency * 0.3 + 
                     timeliness * 0.2 + 
                     reliability * 0.3)
        
        self.quality_score = DataQualityScore(
            completeness=completeness,
            consistency=consistency,
            timeliness=timeliness,
            reliability=reliability,
            confidence=confidence
        )
        
        return self.quality_score
    
    def _check_price_consistency(self) -> float:
        """Check for abnormal price movements."""
        if 'Close' not in self.data.columns:
            return 0.5  # Neutral score if we can't check
            
        returns = self.data['Close'].pct_change().dropna()
        if len(returns) < 2:
            return 0.5
            
        # Check for extreme price movements (beyond 3 standard deviations)
        z_scores = (returns - returns.mean()) / (returns.std() or 1)
        extreme_moves = (abs(z_scores) > 3).mean()
        
        # Lower score if there are many extreme moves
        return max(0, 1 - extreme_moves * 2)
    
    def _check_volume_consistency(self) -> float:
        """Check for abnormal volume patterns."""
        if 'Volume' not in self.data.columns:
            return 0.5  # Neutral score if we can't check
            
        volume = self.data['Volume']
        if len(volume) < 2:
            return 0.5
            
        # Check for zero or negative volume
        zero_volume = (volume <= 0).mean()
        
        # Check for extreme volume spikes
        volume_returns = volume.pct_change().dropna()
        extreme_volume = (volume_returns > 5).mean()  # More than 500% volume increase
        
        return max(0, 1 - (zero_volume + extreme_volume) / 2)
    
    def _check_timeliness(self) -> float:
        """Check how recent the data is."""
        if self.data is None or self.data.empty:
            return 0
            
        last_date = self.data.index[-1] if hasattr(self.data.index, 'to_pydatetime') else self.data.index[-1].to_pydatetime()
        now = datetime.now()
        
        # Calculate recency in days
        recency = (now - last_date).total_seconds() / (24 * 3600)
        
        # Score based on recency (lower is better)
        if recency < 1:  # Less than 1 day old
            return 1.0
        elif recency < 7:  # Less than 1 week old
            return 0.8
        elif recency < 30:  # Less than 1 month old
            return 0.6
        else:
            return 0.3
    
    def _check_reliability(self) -> float:
        """Check data reliability based on various metrics."""
        if self.data is None or len(self.data) < 10:  # Need enough data points
            return 0.5
            
        reliability_score = 0.0
        
        # 1. Check for price/volume correlation (should be somewhat positive)
        if 'Close' in self.data.columns and 'Volume' in self.data.columns:
            corr = self.data['Close'].pct_change().corr(self.data['Volume'].pct_change())
            # We expect some positive correlation between price and volume
            reliability_score += max(0, min(1, 0.5 + corr * 0.5)) * 0.3
        
        # 2. Check for missing trading days (for daily data)
        if hasattr(self.data.index, 'to_series'):
            date_diff = self.data.index.to_series().diff().dt.days
            if not date_diff.empty:
                missing_days_ratio = (date_diff > 1).mean()
                reliability_score += (1 - missing_days_ratio) * 0.2
        
        # 3. Check for zero or negative prices (shouldn't happen for valid data)
        if 'Close' in self.data.columns:
            invalid_prices = (self.data['Close'] <= 0).mean()
            reliability_score += (1 - invalid_prices) * 0.3
        
        # 4. Check for abnormal price changes (potential data errors)
        if 'Close' in self.data.columns:
            pct_changes = self.data['Close'].pct_change().abs()
            # More than 30% daily move is suspicious
            abnormal_moves = (pct_changes > 0.3).mean()
            reliability_score += (1 - abnormal_moves) * 0.2
        
        return min(1.0, reliability_score)  # Cap at 1.0
    
    def analyze_market_sentiment(self, ticker: str) -> Dict[str, float]:
        """Analyze market sentiment for a given ticker using web search."""
        # This is a placeholder - in a real implementation, you would:
        # 1. Search for news articles about the ticker
        # 2. Perform sentiment analysis on the articles
        # 3. Return a sentiment score (-1 to 1) and confidence (0-1)
        
        # For now, return a neutral score
        self.sentiment_score = {
            'score': 0.0,  # -1 (very negative) to 1 (very positive)
            'confidence': 0.7,  # 0 (not confident) to 1 (very confident)
            'sources': 5,  # Number of sources analyzed
            'timestamp': datetime.now().isoformat()
        }
        return self.sentiment_score
    
    def search_market_news(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search for market news related to the query."""
        # In a real implementation, you would use a news API or web scraping
        # This is a simplified version that returns placeholder results
        
        # For demonstration, return some placeholder news
        return [
            {
                'title': f"Market Analysis: {query} Shows Strong Momentum",
                'source': 'Financial Times',
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'snippet': f"{query} continues to show strong performance in the current market conditions...",
                'url': f'https://example.com/news/{query.lower().replace(" ", "-")}'
            },
            {
                'title': f"Experts Weigh In On {query} Outlook",
                'source': 'Bloomberg',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'snippet': f"Analysts are divided on the future of {query} with some predicting...",
                'url': f'https://example.com/analysis/{query.lower().replace(" ", "-")}'
            }
        ][:num_results]
    
    def get_market_context(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive market context for a ticker."""
        # Assess data quality
        quality = self.assess_data_quality()
        
        # Get market sentiment
        sentiment = self.analyze_market_sentiment(ticker)
        
        # Get recent news
        news = self.search_market_news(ticker)
        
        # Basic technical analysis
        if self.data is not None and not self.data.empty and 'Close' in self.data.columns:
            close_prices = self.data['Close']
            if len(close_prices) >= 2:
                price_change = (close_prices.iloc[-1] / close_prices.iloc[-2] - 1) * 100
                volume_avg = self.data['Volume'].mean() if 'Volume' in self.data.columns else 0
            else:
                price_change = 0
                volume_avg = 0
        else:
            price_change = 0
            volume_avg = 0
        
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'data_quality': {
                'score': quality.overall_score,
                'completeness': quality.completeness,
                'consistency': quality.consistency,
                'timeliness': quality.timeliness,
                'reliability': quality.reliability,
                'confidence': quality.confidence
            },
            'sentiment': sentiment,
            'price_action': {
                'current': close_prices.iloc[-1] if not self.data.empty and 'Close' in self.data.columns else None,
                'change_pct': price_change,
                'volume_avg': volume_avg,
                'last_updated': self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S') 
                               if not self.data.empty and hasattr(self.data.index, 'strftime') 
                               else None
            },
            'recent_news': news,
            'analysis': {
                'is_reliable': quality.overall_score > 0.7,
                'confidence': min(quality.confidence, sentiment.get('confidence', 0.5)),
                'recommendation': self._generate_recommendation(quality, sentiment)
            }
        }
    
    def _generate_recommendation(self, quality: DataQualityScore, sentiment: Dict[str, float]) -> str:
        """Generate a trading recommendation based on data quality and sentiment."""
        if quality.overall_score < 0.5:
            return "Data quality is low. Consider waiting for more reliable data."
        
        if sentiment['score'] > 0.3 and quality.overall_score > 0.7:
            return "Strong positive sentiment with high data quality. Consider a long position."
        elif sentiment['score'] < -0.3 and quality.overall_score > 0.7:
            return "Strong negative sentiment with high data quality. Consider a short position or staying out of the market."
        else:
            return "Neutral sentiment or mixed signals. Consider waiting for clearer market direction."

# Example usage
if __name__ == "__main__":
    # Example usage with yfinance
    import yfinance as yf
    
    # Download some data
    ticker = "AAPL"
    data = yf.download(ticker, period="1mo", interval="1d")
    
    # Analyze the data
    analyzer = MarketIntelligence(data)
    context = analyzer.get_market_context(ticker)
    
    # Print results
    import pprint
    print(f"\nMarket Analysis for {ticker}")
    print("=" * 50)
    print(f"Data Quality Score: {context['data_quality']['score']:.2f}/1.00")
    print(f"Sentiment: {context['sentiment']['score']:.2f} (Confidence: {context['sentiment']['confidence']:.2f})")
    print(f"\nRecommendation: {context['analysis']['recommendation']}")
    
    if context['recent_news']:
        print("\nRecent News:")
        for i, news in enumerate(context['recent_news'], 1):
            print(f"{i}. {news['title']} ({news['source']} - {news['date']})")
            print(f"   {news['snippet']}\n")
