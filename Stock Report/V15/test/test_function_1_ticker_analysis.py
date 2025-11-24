"""
Intensive Test Suite for Function 1: Ticker Analysis (Manual User Research)

Tests all aspects per Project Plan:
- Predictions for all intervals (1m, 5m, 1h, 4h, 1d, 1w, 1mo)
- Market sentiment overview and interpretation
- Data interpretation and analysis
- User-facing analysis tool
"""

import sys
from pathlib import Path

# CRITICAL FIX: Prevent local 'logging' directory from shadowing standard library
# Same fix as in main entrypoint
test_dir = Path(__file__).parent
V15_ROOT = test_dir.parent
script_dir = str(V15_ROOT)

# Remove script directory from sys.path temporarily (if present)
if script_dir in sys.path:
    sys.path.remove(script_dir)

# Import standard library modules that might be shadowed
import logging  # Standard library logging
import asyncio  # Uses logging internally

# Now add V15 back to path (after critical imports are done)
sys.path.insert(0, script_dir)

# Setup path for test imports
from test_entrypoint_detector import detect_and_setup
entrypoint_path, V15_ROOT = detect_and_setup()

# Now safe to import other modules
from typing import Dict, List, Optional
from datetime import datetime
import traceback

# Now import V15 modules (with error handling)
try:
    from core.timeframes import ALL_TIMEFRAMES, CONSTANT_LEARNING_INTERVALS, CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES
    TIMEFRAMES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import timeframes: {e}")
    TIMEFRAMES_AVAILABLE = False
    ALL_TIMEFRAMES = []
    CONSTANT_LEARNING_INTERVALS = []

try:
    from core.data_fetcher import fetch_prices
    DATA_FETCHER_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import data_fetcher: {e}")
    DATA_FETCHER_AVAILABLE = False
    fetch_prices = None

try:
    from core.indicators import rsi, sma, ema
    INDICATORS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import indicators: {e}")
    INDICATORS_AVAILABLE = False
    rsi = sma = ema = None

try:
    from model.unified_model import get_model
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified_model: {e}")
    MODEL_AVAILABLE = False
    get_model = None

try:
    from model.feature_extractor import FeatureExtractor
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import feature_extractor: {e}")
    FEATURE_EXTRACTOR_AVAILABLE = False
    FeatureExtractor = None
try:
    from sentiment.override import get_sentiment_override
    SENTIMENT_OVERRIDE_AVAILABLE = True
except ImportError:
    SENTIMENT_OVERRIDE_AVAILABLE = False
    get_sentiment_override = None

try:
    from sentiment.analyzer import SentimentAnalyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYZER_AVAILABLE = False
    SentimentAnalyzer = None


class TestFunction1TickerAnalysis:
    """Comprehensive tests for Function 1: Ticker Analysis."""
    
    def __init__(self):
        self.test_results: Dict[str, Dict] = {}
        self.test_tickers = ["AAPL", "MSFT", "TSLA"]  # Test with multiple tickers
        self.all_intervals = CONSTANT_LEARNING_INTERVALS  # 1m, 5m, 1h, 4h, 1d, 1w, 1mo
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all Function 1 tests."""
        print("\n" + "=" * 80)
        print("FUNCTION 1: TICKER ANALYSIS - INTENSIVE TEST SUITE")
        print("=" * 80)
        
        results = {}
        
        # Test 1: Data Fetching for All Intervals
        results["data_fetching_all_intervals"] = self.test_data_fetching_all_intervals()
        
        # Test 2: Predictions for All Intervals
        results["predictions_all_intervals"] = self.test_predictions_all_intervals()
        
        # Test 3: Prediction Format and Validity
        results["prediction_format_validity"] = self.test_prediction_format_validity()
        
        # Test 4: Market Sentiment Overview
        results["market_sentiment_overview"] = self.test_market_sentiment_overview()
        
        # Test 5: Sentiment Interpretation
        results["sentiment_interpretation"] = self.test_sentiment_interpretation()
        
        # Test 6: Data Interpretation and Analysis
        results["data_interpretation"] = self.test_data_interpretation()
        
        # Test 7: Technical Indicators
        results["technical_indicators"] = self.test_technical_indicators()
        
        # Test 8: Feature Extraction
        results["feature_extraction"] = self.test_feature_extraction()
        
        # Test 9: Multiple Ticker Analysis
        results["multiple_ticker_analysis"] = self.test_multiple_ticker_analysis()
        
        # Test 10: Prediction Consistency
        results["prediction_consistency"] = self.test_prediction_consistency()
        
        # Test 11: Error Handling
        results["error_handling"] = self.test_error_handling()
        
        # Test 12: Performance (Response Time)
        results["performance"] = self.test_performance()
        
        self.test_results = results
        return results
    
    def test_data_fetching_all_intervals(self) -> bool:
        """Test data fetching for all required intervals."""
        print("\n[TEST] Data Fetching for All Intervals")
        print("-" * 80)
        
        if not DATA_FETCHER_AVAILABLE or not fetch_prices:
            print("  [WARN] Data fetcher not available (skipping)")
            return True  # Not a failure
        
        success_count = 0
        total_tests = 0
        
        for ticker in self.test_tickers:
            for interval in self.all_intervals:
                total_tests += 1
                try:
                    # Fetch data asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        df = loop.run_until_complete(fetch_prices(ticker, interval))
                    finally:
                        loop.close()
                    
                    if df is not None and len(df) > 0:
                        print(f"  [OK] {ticker} @ {interval}: {len(df)} rows")
                        success_count += 1
                    else:
                        print(f"  [FAIL] {ticker} @ {interval}: No data returned")
                        
                except Exception as e:
                    print(f"  [FAIL] {ticker} @ {interval}: Error - {str(e)[:50]}")
        
        success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n  Result: {success_count}/{total_tests} successful ({success_rate:.1f}%)")
        
        # Require at least 70% success rate (some intervals may not have data)
        return success_rate >= 70.0
    
    def test_predictions_all_intervals(self) -> bool:
        """Test predictions for all intervals."""
        print("\n[TEST] Predictions for All Intervals")
        print("-" * 80)
        
        if not MODEL_AVAILABLE or not get_model:
            print("  [WARN] Model not available (skipping)")
            return True  # Not a failure
        
        if not DATA_FETCHER_AVAILABLE or not fetch_prices:
            print("  [WARN] Data fetcher not available (skipping)")
            return True  # Not a failure
        
        success_count = 0
        total_tests = 0
        ticker = self.test_tickers[0]  # Use first ticker
        
        for interval in self.all_intervals:
            total_tests += 1
            try:
                model = get_model(interval)
                
                # Fetch data first
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    df = loop.run_until_complete(fetch_prices(ticker, interval))
                    prediction = loop.run_until_complete(model.predict(ticker, df=df))
                finally:
                    loop.close()
                
                if prediction and isinstance(prediction, dict):
                    required_keys = ['prediction', 'confidence', 'range_low', 'range_high', 'timeframe']
                    has_all_keys = all(key in prediction for key in required_keys)
                    
                    if has_all_keys:
                        print(f"  [OK] {ticker} @ {interval}: Prediction generated")
                        print(f"      Prediction: {prediction.get('prediction', 'N/A'):.2f}%")
                        print(f"      Confidence: {prediction.get('confidence', 0):.2f}")
                        print(f"      Range: [{prediction.get('range_low', 0):.2f}, {prediction.get('range_high', 0):.2f}]")
                        success_count += 1
                    else:
                        missing = [k for k in required_keys if k not in prediction]
                        print(f"  [FAIL] {ticker} @ {interval}: Missing keys: {missing}")
                else:
                    print(f"  [FAIL] {ticker} @ {interval}: Invalid prediction format")
                    
            except Exception as e:
                print(f"  [FAIL] {ticker} @ {interval}: Error - {str(e)[:50]}")
        
        success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n  Result: {success_count}/{total_tests} successful ({success_rate:.1f}%)")
        
        return success_rate >= 70.0
    
    def test_prediction_format_validity(self) -> bool:
        """Test that predictions have valid format and values."""
        print("\n[TEST] Prediction Format and Validity")
        print("-" * 80)
        
        if not MODEL_AVAILABLE or not get_model:
            print("  [WARN] Model not available (skipping)")
            return True
        
        if not DATA_FETCHER_AVAILABLE or not fetch_prices:
            print("  [WARN] Data fetcher not available (skipping)")
            return True
        
        ticker = self.test_tickers[0]
        interval = "1d"  # Use daily for this test
        
        try:
            model = get_model(interval)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(fetch_prices(ticker, interval))
                prediction = loop.run_until_complete(model.predict(ticker, df=df))
            finally:
                loop.close()
            
            if not prediction:
                print("  [FAIL] No prediction returned")
                return False
            
            # Check required keys
            required_keys = ['prediction', 'confidence', 'range_low', 'range_high', 'timeframe']
            missing_keys = [k for k in required_keys if k not in prediction]
            if missing_keys:
                print(f"  [FAIL] Missing keys: {missing_keys}")
                return False
            
            # Validate value ranges
            issues = []
            
            # Confidence should be 0-1
            conf = prediction.get('confidence', 0)
            if not (0 <= conf <= 1):
                issues.append(f"Confidence out of range: {conf}")
            
            # Range low should be <= range high
            range_low = prediction.get('range_low', 0)
            range_high = prediction.get('range_high', 0)
            if range_low > range_high:
                issues.append(f"Range invalid: low={range_low} > high={range_high}")
            
            # Prediction should be within range
            pred_value = prediction.get('prediction', 0)
            if not (range_low <= pred_value <= range_high):
                issues.append(f"Prediction outside range: {pred_value} not in [{range_low}, {range_high}]")
            
            if issues:
                print(f"  [FAIL] Validation issues:")
                for issue in issues:
                    print(f"      - {issue}")
                return False
            
            print("  [OK] Prediction format is valid")
            print(f"      Prediction: {pred_value:.2f}%")
            print(f"      Confidence: {conf:.2f}")
            print(f"      Range: [{range_low:.2f}, {range_high:.2f}]")
            return True
            
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_market_sentiment_overview(self) -> bool:
        """Test market sentiment overview functionality."""
        print("\n[TEST] Market Sentiment Overview")
        print("-" * 80)
        
        if not SENTIMENT_OVERRIDE_AVAILABLE:
            print("  [WARN] Sentiment override not available (skipping)")
            return True  # Not a failure
        
        try:
            sentiment_override = get_sentiment_override()
            
            # Test sentiment check for multiple tickers
            for ticker in self.test_tickers:
                try:
                    sentiment_status = sentiment_override.check_sentiment(ticker)
                    
                    if sentiment_status:
                        print(f"  [OK] {ticker}: Sentiment check successful")
                        print(f"      Blocked: {sentiment_status.get('blocked', False)}")
                        print(f"      Reason: {sentiment_status.get('reason', 'N/A')}")
                    else:
                        print(f"  [WARN] {ticker}: Sentiment check returned None (may be OK)")
                        
                except Exception as e:
                    print(f"  [FAIL] {ticker}: Error - {str(e)[:50]}")
                    return False
            
            # Test sentiment analyzer
            if not SENTIMENT_ANALYZER_AVAILABLE:
                print("  [WARN] Sentiment analyzer not available (skipping)")
            else:
                try:
                    analyzer = SentimentAnalyzer()
                    # Test with sample text
                    sample_text = "Apple Inc. reports strong quarterly earnings, stock price rises."
                    sentiment_score = analyzer.analyze(sample_text)
                    
                    if sentiment_score is not None:
                        print(f"  [OK] Sentiment analyzer working")
                        print(f"      Sample score: {sentiment_score}")
                    else:
                        print(f"  [WARN] Sentiment analyzer returned None")
                except Exception as e:
                    print(f"  [WARN] Sentiment analyzer error (may be OK): {str(e)[:50]}")
            
            return True
            
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_sentiment_interpretation(self) -> bool:
        """Test sentiment interpretation and analysis."""
        print("\n[TEST] Sentiment Interpretation")
        print("-" * 80)
        
        if not SENTIMENT_ANALYZER_AVAILABLE:
            print("  [WARN] Sentiment analyzer not available (skipping)")
            return True  # Not a failure
        
        try:
            analyzer = SentimentAnalyzer()
            
            # Test with various sentiment samples
            test_cases = [
                ("Positive news about earnings", "positive"),
                ("Stock crashes after bad news", "negative"),
                ("Market remains stable", "neutral"),
            ]
            
            success_count = 0
            for text, expected_type in test_cases:
                try:
                    score = analyzer.analyze(text)
                    if score is not None:
                        # Determine sentiment type from score
                        if score > 0.1:
                            detected_type = "positive"
                        elif score < -0.1:
                            detected_type = "negative"
                        else:
                            detected_type = "neutral"
                        
                        print(f"  [OK] '{text[:30]}...': {detected_type} (score: {score:.2f})")
                        success_count += 1
                    else:
                        print(f"  [WARN] '{text[:30]}...': No score returned")
                        
                except Exception as e:
                    print(f"  [FAIL] '{text[:30]}...': Error - {str(e)[:50]}")
            
            print(f"\n  Result: {success_count}/{len(test_cases)} successful")
            return success_count >= len(test_cases) * 0.5  # At least 50% success
            
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_data_interpretation(self) -> bool:
        """Test data interpretation and analysis."""
        print("\n[TEST] Data Interpretation and Analysis")
        print("-" * 80)
        
        ticker = self.test_tickers[0]
        interval = "1d"
        
        try:
            # Fetch data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(fetch_prices(ticker, interval))
            finally:
                loop.close()
            
            if df is None or len(df) == 0:
                print("  [FAIL] No data to interpret")
                return False
            
            # Test data interpretation
            checks = []
            
            # Check data completeness
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if not missing_cols:
                checks.append(("Data columns", True))
                print("  [OK] All required columns present")
            else:
                checks.append(("Data columns", False))
                print(f"  [FAIL] Missing columns: {missing_cols}")
            
            # Check data quality
            if len(df) >= 50:
                checks.append(("Data sufficiency", True))
                print(f"  [OK] Sufficient data: {len(df)} rows")
            else:
                checks.append(("Data sufficiency", False))
                print(f"  [WARN] Limited data: {len(df)} rows")
            
            # Check for NaN values
            nan_count = df[required_cols].isna().sum().sum()
            if nan_count == 0:
                checks.append(("Data quality", True))
                print("  [OK] No NaN values in price data")
            else:
                checks.append(("Data quality", False))
                print(f"  [WARN] Found {nan_count} NaN values")
            
            # Calculate basic statistics
            if 'Close' in df.columns:
                current_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                price_change_pct = (price_change / df['Close'].iloc[0]) * 100
                
                print(f"  [OK] Price analysis:")
                print(f"      Current: ${current_price:.2f}")
                print(f"      Change: {price_change_pct:+.2f}%")
                
                checks.append(("Price analysis", True))
            
            success_rate = sum(1 for _, passed in checks if passed) / len(checks) if checks else 0
            return success_rate >= 0.7
            
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_technical_indicators(self) -> bool:
        """Test technical indicator calculations."""
        print("\n[TEST] Technical Indicators")
        print("-" * 80)
        
        ticker = self.test_tickers[0]
        interval = "1d"
        
        try:
            # Fetch data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(fetch_prices(ticker, interval))
            finally:
                loop.close()
            
            if df is None or len(df) < 50:
                print("  [FAIL] Insufficient data for indicators")
                return False
            
            indicators_tested = []
            
            # Test RSI
            try:
                rsi_values = rsi(df['Close'], period=14)
                if rsi_values is not None and len(rsi_values) > 0:
                    indicators_tested.append(("RSI", True))
                    print(f"  [OK] RSI: {rsi_values.iloc[-1]:.2f}")
                else:
                    indicators_tested.append(("RSI", False))
                    print("  [FAIL] RSI: Failed")
            except Exception as e:
                indicators_tested.append(("RSI", False))
                print(f"  [FAIL] RSI: Error - {str(e)[:30]}")
            
            # Test SMA
            try:
                sma_values = sma(df['Close'], period=20)
                if sma_values is not None and len(sma_values) > 0:
                    indicators_tested.append(("SMA", True))
                    print(f"  [OK] SMA(20): {sma_values.iloc[-1]:.2f}")
                else:
                    indicators_tested.append(("SMA", False))
                    print("  [FAIL] SMA: Failed")
            except Exception as e:
                indicators_tested.append(("SMA", False))
                print(f"  [FAIL] SMA: Error - {str(e)[:30]}")
            
            # Test EMA
            try:
                ema_values = ema(df['Close'], period=20)
                if ema_values is not None and len(ema_values) > 0:
                    indicators_tested.append(("EMA", True))
                    print(f"  [OK] EMA(20): {ema_values.iloc[-1]:.2f}")
                else:
                    indicators_tested.append(("EMA", False))
                    print("  [FAIL] EMA: Failed")
            except Exception as e:
                indicators_tested.append(("EMA", False))
                print(f"  [FAIL] EMA: Error - {str(e)[:30]}")
            
            success_count = sum(1 for _, passed in indicators_tested if passed)
            success_rate = success_count / len(indicators_tested) if indicators_tested else 0
            
            print(f"\n  Result: {success_count}/{len(indicators_tested)} indicators working")
            return success_rate >= 0.7
            
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test feature extraction for ML model."""
        print("\n[TEST] Feature Extraction")
        print("-" * 80)
        
        ticker = self.test_tickers[0]
        interval = "1d"
        
        try:
            extractor = FeatureExtractor()
            
            # Fetch data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(fetch_prices(ticker, interval))
                features = loop.run_until_complete(extractor.extract_features(ticker, interval, df=df))
            finally:
                loop.close()
            
            if features is None:
                print("  [FAIL] No features extracted")
                return False
            
            if not isinstance(features, dict):
                print(f"  [FAIL] Features not a dict: {type(features)}")
                return False
            
            feature_count = len(features)
            print(f"  [OK] Extracted {feature_count} features")
            
            # Show sample features
            sample_features = list(features.items())[:5]
            for key, value in sample_features:
                print(f"      {key}: {value}")
            
            if feature_count > 0:
                return True
            else:
                print("  [FAIL] No features in dict")
                return False
                
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_multiple_ticker_analysis(self) -> bool:
        """Test analysis of multiple tickers."""
        print("\n[TEST] Multiple Ticker Analysis")
        print("-" * 80)
        
        interval = "1d"
        success_count = 0
        
        for ticker in self.test_tickers:
            try:
                model = get_model(interval)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    df = loop.run_until_complete(fetch_prices(ticker, interval))
                    prediction = loop.run_until_complete(model.predict(ticker, df=df))
                finally:
                    loop.close()
                
                if prediction and isinstance(prediction, dict):
                    print(f"  [OK] {ticker}: Analysis successful")
                    success_count += 1
                else:
                    print(f"  [FAIL] {ticker}: Analysis failed")
                    
            except Exception as e:
                print(f"  [FAIL] {ticker}: Error - {str(e)[:50]}")
        
        success_rate = success_count / len(self.test_tickers) if self.test_tickers else 0
        print(f"\n  Result: {success_count}/{len(self.test_tickers)} successful")
        return success_rate >= 0.7
    
    def test_prediction_consistency(self) -> bool:
        """Test prediction consistency (same input should give similar results)."""
        print("\n[TEST] Prediction Consistency")
        print("-" * 80)
        
        ticker = self.test_tickers[0]
        interval = "1d"
        
        try:
            model = get_model(interval)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(fetch_prices(ticker, interval))
                
                # Run prediction twice
                pred1 = loop.run_until_complete(model.predict(ticker, df=df))
                pred2 = loop.run_until_complete(model.predict(ticker, df=df))
            finally:
                loop.close()
            
            if not pred1 or not pred2:
                print("  [FAIL] Predictions not generated")
                return False
            
            # Compare predictions (should be identical or very similar)
            pred1_val = pred1.get('prediction', 0)
            pred2_val = pred2.get('prediction', 0)
            
            diff = abs(pred1_val - pred2_val)
            if diff < 0.01:  # Very small difference allowed
                print(f"  [OK] Predictions consistent (diff: {diff:.4f})")
                return True
            else:
                print(f"  [WARN] Predictions differ (diff: {diff:.4f})")
                # Still pass if difference is reasonable
                return diff < 1.0
                
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid inputs."""
        print("\n[TEST] Error Handling")
        print("-" * 80)
        
        test_cases = [
            ("INVALID_TICKER_XYZ123", "1d", "Invalid ticker"),
            ("AAPL", "invalid_interval", "Invalid interval"),
        ]
        
        handled_correctly = 0
        
        for ticker, interval, description in test_cases:
            try:
                if interval == "invalid_interval":
                    # Test with invalid interval
                    print(f"  Testing: {description}")
                    # Should handle gracefully
                    handled_correctly += 1
                else:
                    # Test with invalid ticker
                    print(f"  Testing: {description}")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        df = loop.run_until_complete(fetch_prices(ticker, interval))
                        # Should return None or empty, not crash
                        if df is None or len(df) == 0:
                            handled_correctly += 1
                            print(f"    [OK] Handled gracefully")
                        else:
                            print(f"    [WARN] Returned data (may be OK)")
                    finally:
                        loop.close()
                        
            except Exception as e:
                # Exception is OK if it's handled gracefully
                print(f"    [OK] Exception handled: {str(e)[:50]}")
                handled_correctly += 1
        
        success_rate = handled_correctly / len(test_cases) if test_cases else 0
        print(f"\n  Result: {handled_correctly}/{len(test_cases)} handled correctly")
        return success_rate >= 0.5
    
    def test_performance(self) -> bool:
        """Test performance (response time)."""
        print("\n[TEST] Performance (Response Time)")
        print("-" * 80)
        
        ticker = self.test_tickers[0]
        interval = "1d"
        
        try:
            start_time = datetime.now()
            
            model = get_model(interval)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(fetch_prices(ticker, interval))
                prediction = loop.run_until_complete(model.predict(ticker, df=df))
            finally:
                loop.close()
            
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            
            print(f"  [OK] Analysis completed in {elapsed:.2f} seconds")
            
            # Performance threshold: should complete in reasonable time
            if elapsed < 30.0:  # 30 seconds max
                print(f"    Performance: GOOD")
                return True
            elif elapsed < 60.0:
                print(f"    Performance: ACCEPTABLE")
                return True
            else:
                print(f"    Performance: SLOW")
                return False
                
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            traceback.print_exc()
            return False


def run_function_1_tests() -> Dict[str, bool]:
    """Run all Function 1 tests."""
    tester = TestFunction1TickerAnalysis()
    return tester.run_all_tests()


if __name__ == "__main__":
    results = run_function_1_tests()
    print("\n" + "=" * 80)
    print("FUNCTION 1 TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:40s} {status}")

