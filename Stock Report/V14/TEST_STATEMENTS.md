# V14 Test Statements

## Test Coverage

### Core Module Tests
- ✅ Portable paths initialization
- ✅ Timeframe validation
- ✅ Technical indicators (RSI, SMA, EMA)
- ✅ ATR calculation

### Risk Management Tests
- ✅ Risk profile system
- ✅ Stop-loss calculation (ATR-based)
- ✅ Position sizing
- ✅ Exposure tracking

### Model Tests
- ✅ Feature extraction
- ✅ Unified model initialization
- ⏳ Model training (requires data)
- ⏳ Prediction generation (requires trained model)

### Learning Tests
- ✅ Trade tracking
- ✅ Feedback loop
- ✅ Prediction monitoring
- ⏳ Model updates (requires trade data)

### Sentiment Tests
- ✅ Sentiment analysis
- ✅ Sentiment override logic

### Logging Tests
- ✅ Trade logging (entry/exit)
- ✅ Log analysis

### Integration Tests
- ⏳ Full workflow (requires all components)
- ⏳ Browser automation (requires Chrome)
- ⏳ Simulation mode (requires data)

## Running Tests

```bash
# Run all tests
pytest V14/test_v14.py -v

# Run specific test category
pytest V14/test_v14.py::test_risk_management -v

# Run with coverage
pytest V14/test_v14.py --cov=V14 --cov-report=html
```

## Test Data Requirements

Some tests require:
- Historical price data (fetched automatically)
- Trade history (for learning tests)
- Trained models (for prediction tests)

## Performance Benchmarks

### Data Fetching
- Target: 5-10x faster than V13 (async)
- Test: Fetch 10 tickers concurrently

### Risk Calculations
- Target: Real-time (< 100ms)
- Test: Calculate stops for 100 positions

### Model Predictions
- Target: < 1 second per prediction
- Test: Generate predictions for all timeframes

## Edge Cases Tested

- Empty dataframes
- Invalid timeframes
- Zero equity
- Extreme confidence values
- Missing price data
- Network failures
- Invalid risk profiles

## Manual Testing Checklist

- [ ] First-run setup
- [ ] Risk profile selection
- [ ] Browser automation initialization
- [ ] Trade execution (simulation)
- [ ] Log viewing
- [ ] Performance report generation
- [ ] Model training
- [ ] Sentiment override activation

