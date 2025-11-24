"""
V15 Comprehensive Test Suite
Tests all V15 systems ensuring full stability and portability.
"""

import asyncio
import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Tuple
from functools import wraps
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

# Handle pandas and numpy imports with error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    print("WARNING: pandas is not installed. Some tests may fail.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("WARNING: numpy is not installed. Some tests may fail.")

try:
    import pytest
except Exception:  # pragma: no cover - handled via fallback runner
    pytest = None  # type: ignore

# Add V15 to path
V15_ROOT = Path(__file__).parent
sys.path.insert(0, str(V15_ROOT))

# Import V15 modules
from core.portable_paths import get_path, initialize_structure
from core.data_fetcher import fetch_prices
from core.indicators import rsi, sma, ema
from core.timeframes import is_valid_timeframe, is_cfd_timeframe, CFD_TIMEFRAMES
from risk.volatility import calculate_atr
from risk.profiles import RiskProfile, get_profile_config
from risk.stop_loss import calculate_stop_loss_distance, calculate_stop_loss_price
from risk.position_sizing import calculate_position_size
from risk.exposure_tracker import ExposureTracker, Position as ExposurePosition
from model.feature_extractor import FeatureExtractor
from model.unified_model import UnifiedModel, get_model
from learning.trade_tracker import TradeTracker, TradeOutcome
from learning.feedback_loop import FeedbackLoop
from learning.prediction_monitor import PredictionMonitor
from sentiment.analyzer import SentimentAnalyzer
from sentiment.override import SentimentOverride
from sa_logging.trade_logger import TradeLogger


# ============================================================================
# Test Fixtures
# ============================================================================

def _create_sample_price_data():
    """Generate sample price data used by fixtures and fallback runner."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    base_price = 100.0
    prices = []
    for _ in range(100):
        change = np.random.normal(0, 2)
        base_price += change
        prices.append(base_price)

    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, 100),
        },
        index=dates,
    )


def _create_sample_config():
    """Generate sample configuration used by fixtures and fallback runner."""
    return {
        "version": "15.0",
        "risk_profile": "medium",
        "model": {"use_gpu": False, "confidence_threshold": 0.65},
    }


if pytest:

    @pytest.fixture
    def sample_price_data():
        """Create sample price DataFrame for testing."""
        return _create_sample_price_data()

    @pytest.fixture
    def sample_config():
        """Create sample configuration."""
        return _create_sample_config()

else:

    def sample_price_data():
        return _create_sample_price_data()

    def sample_config():
        return _create_sample_config()

def _asyncio_wrapper(func: Callable):
    """Run async test functions without requiring pytest-asyncio."""

    @wraps(func)
    def _sync_wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return _sync_wrapper


# Determine whether pytest-asyncio (or equivalent) is available
_async_plugin_available = False
if pytest:
    try:
        importlib.import_module("pytest_asyncio")
        _async_plugin_available = True
    except Exception:
        _async_plugin_available = False

if _async_plugin_available:
    mark_asyncio = pytest.mark.asyncio
else:

    def mark_asyncio(func: Callable):
        """Decorator that executes async tests via asyncio.run when pytest plugin is unavailable."""
        return _asyncio_wrapper(func)


# ============================================================================
# Core Module Tests
# ============================================================================

def test_portable_paths():
    """Test portable paths module."""
    initialize_structure()
    root = get_path('root')
    assert root.exists()
    assert root.name == 'V15'


def test_timeframes():
    """Test timeframe configuration."""
    assert is_valid_timeframe("1m")
    assert is_valid_timeframe("1d")
    assert not is_valid_timeframe("invalid")
    assert is_cfd_timeframe("1m")
    assert not is_cfd_timeframe("1d")


def test_indicators(sample_price_data):
    """Test technical indicators."""
    close = sample_price_data['Close']
    
    rsi_value = rsi(close, period=14).iloc[-1]
    assert 0 <= rsi_value <= 100
    
    sma_value = sma(close, period=20).iloc[-1]
    assert sma_value > 0
    
    ema_value = ema(close, period=20).iloc[-1]
    assert ema_value > 0


def test_atr_calculation(sample_price_data):
    """Test ATR calculation."""
    atr = calculate_atr(sample_price_data, period=14)
    assert atr > 0
    assert isinstance(atr, float)


# ============================================================================
# Risk Management Tests
# ============================================================================

def test_risk_profiles():
    """Test risk profile system."""
    profile = RiskProfile.MEDIUM
    config = get_profile_config(profile)
    assert config["equity_risk_per_trade_max"] == 1.0
    assert config["max_combined_exposure"] == 10.0


def test_stop_loss_calculation(sample_price_data):
    """Test stop-loss calculation."""
    stop_distance, atr = calculate_stop_loss_distance(
        df=sample_price_data,
        profile=RiskProfile.MEDIUM,
        confidence=0.7,
        asset_risk_category="medium"
    )
    assert stop_distance > 0
    
    stop_price = calculate_stop_loss_price(100.0, "LONG", stop_distance)
    assert stop_price < 100.0  # Stop below entry for LONG


def test_position_sizing():
    """Test position sizing calculation."""
    position_size, risk_amount = calculate_position_size(
        equity=10000.0,
        entry_price=100.0,
        stop_price=98.0,
        risk_percentage=1.0,
        direction="LONG"
    )
    assert position_size > 0
    assert risk_amount == 100.0  # 1% of 10000


def test_exposure_tracker():
    """Test exposure tracking."""
    tracker = ExposureTracker(equity=10000.0, profile=RiskProfile.MEDIUM)
    
    position = ExposurePosition(
        position_id="test1",
        ticker="AAPL",
        direction="LONG",
        entry_price=100.0,
        quantity=10.0,
        stop_price=98.0,
        current_price=100.0
    )
    
    tracker.add_position(position)
    exposure = tracker.get_total_exposure()
    assert exposure >= 0
    assert exposure <= 10.0  # Max 10%


# ============================================================================
# Model Tests
# ============================================================================

@mark_asyncio
async def test_feature_extractor(sample_price_data):
    """Test feature extraction."""
    extractor = FeatureExtractor()
    features = await extractor.extract_features(
        ticker="AAPL",
        interval="1d",
        df=sample_price_data
    )
    assert isinstance(features, dict)
    assert "rsi_14" in features
    assert "price_current" in features


def test_unified_model():
    """Test unified model."""
    model = UnifiedModel(timeframe="1d")
    assert model.timeframe == "1d"
    assert not model.is_trained  # Should not be trained initially


# ============================================================================
# Learning Tests
# ============================================================================

def test_trade_tracker():
    """Test trade tracking."""
    tracker = TradeTracker()
    
    outcome = TradeOutcome(
        trade_id="test1",
        ticker="AAPL",
        direction="LONG",
        entry_time=datetime.now(),
        entry_price=100.0,
        exit_time=datetime.now(),
        exit_price=105.0,
        exit_reason="TP",
        position_size=10.0,
        stop_price=98.0,
        target_price=105.0,
        confidence=0.8,
        timeframe="1d",
        pnl=50.0,
        pnl_percentage=5.0
    )
    
    tracker.add_outcome(outcome)
    outcomes = tracker.get_outcomes()
    assert len(outcomes) > 0


def test_feedback_loop():
    """Test feedback loop."""
    loop = FeedbackLoop()
    tracker = TradeTracker()
    
    outcome = TradeOutcome(
        trade_id="test1",
        ticker="AAPL",
        direction="LONG",
        entry_time=datetime.now(),
        entry_price=100.0,
        exit_time=datetime.now(),
        exit_price=105.0,
        exit_reason="TP",
        position_size=10.0,
        stop_price=98.0,
        target_price=105.0,
        confidence=0.8,
        timeframe="1d",
        pnl=50.0,
        pnl_percentage=5.0
    )
    
    adjustments = loop.process_trade_outcome(outcome)
    assert "confidence_adjustment" in adjustments


def test_prediction_monitor():
    """Test prediction monitoring."""
    monitor = PredictionMonitor()
    
    open_trades = [{
        "trade_id": "test1",
        "ticker": "AAPL",
        "entry_time": datetime.now() - timedelta(minutes=10),
        "timeframe": "5m",
        "target_price": 105.0,
        "stop_price": 98.0,
        "current_price": 100.0,
        "direction": "LONG"
    }]
    
    missed = monitor.check_missed_predictions(open_trades)
    # Should detect missed if timeframe expired
    assert isinstance(missed, list)


# ============================================================================
# Sentiment Tests
# ============================================================================

def test_sentiment_analyzer():
    """Test sentiment analysis."""
    analyzer = SentimentAnalyzer()
    
    result = analyzer.analyze_text("Stock beats earnings expectations, strong growth")
    assert result["sentiment_score"] > 0
    assert "sentiment_score" in result


def test_sentiment_override():
    """Test sentiment override."""
    override = SentimentOverride()
    
    should_block, reason = override.should_block_trade("AAPL")
    assert isinstance(should_block, bool)
    assert isinstance(reason, str)


# ============================================================================
# Logging Tests
# ============================================================================

def test_trade_logger():
    """Test trade logging."""
    logger = TradeLogger()
    
    trade_id = logger.log_trade_entry(
        ticker="AAPL",
        side="LONG",
        size=10.0,
        entry_price=100.0,
        stop_price=98.0,
        target_price=105.0,
        confidence=0.8,
        timeframe="1d"
    )
    
    assert trade_id is not None
    
    success = logger.log_trade_exit(
        trade_id=trade_id,
        close_price=105.0,
        exit_reason="TP",
        pnl=50.0,
        pnl_percentage=5.0
    )
    
    assert success


# ============================================================================
# Integration Tests
# ============================================================================

@mark_asyncio
async def test_full_workflow():
    """Test full workflow integration."""
    # This would test the complete flow:
    # 1. Generate prediction
    # 2. Calculate risk
    # 3. Check sentiment override
    # 4. Open position (simulation)
    # 5. Log trade
    # 6. Track outcome
    
    # Placeholder - would need full integration
    assert True


def _run_tests_without_pytest() -> bool:
    """Execute tests sequentially without relying on pytest."""
    manual_tests: List[Tuple[str, Callable[[], None]]] = [
        ("test_portable_paths", test_portable_paths),
        ("test_timeframes", test_timeframes),
        ("test_indicators", lambda: test_indicators(sample_price_data())),
        ("test_atr_calculation", lambda: test_atr_calculation(sample_price_data())),
        ("test_risk_profiles", test_risk_profiles),
        ("test_stop_loss_calculation", lambda: test_stop_loss_calculation(sample_price_data())),
        ("test_position_sizing", test_position_sizing),
        ("test_exposure_tracker", test_exposure_tracker),
        ("test_feature_extractor", lambda: asyncio.run(test_feature_extractor(sample_price_data()))),
        ("test_unified_model", test_unified_model),
        ("test_trade_tracker", test_trade_tracker),
        ("test_feedback_loop", test_feedback_loop),
        ("test_prediction_monitor", test_prediction_monitor),
        ("test_sentiment_analyzer", test_sentiment_analyzer),
        ("test_sentiment_override", test_sentiment_override),
        ("test_trade_logger", test_trade_logger),
        ("test_full_workflow", lambda: asyncio.run(test_full_workflow())),
    ]

    passed = 0
    failures: List[Tuple[str, str]] = []

    print("\n" + "=" * 70)
    print("RUNNING test_v15.py WITHOUT pytest")
    print("=" * 70)

    for name, test_callable in manual_tests:
        try:
            test_callable()
            print(f"[OK] {name}")
            passed += 1
        except AssertionError as exc:
            message = f"Assertion failed: {exc}"
            print(f"[FAIL] {name} - {message}")
            failures.append((name, message))
        except Exception as exc:  # Catch unexpected runtime errors
            message = f"Error: {exc}"
            print(f"[ERROR] {name} - {message}")
            failures.append((name, message))

    total = len(manual_tests)
    print("\n" + "=" * 70)
    print(f"Completed {total} tests -> Passed: {passed}, Failed: {len(failures)}")
    if failures:
        print("Failures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
    print("=" * 70 + "\n")

    return len(failures) == 0


if __name__ == "__main__":
    if pytest:
        raise SystemExit(pytest.main([__file__, "-v"]))
    success = _run_tests_without_pytest()
    sys.exit(0 if success else 1)

