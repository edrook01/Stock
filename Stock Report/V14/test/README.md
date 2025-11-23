# Intensive Test Suite for Stock Analyzer V14

## Overview

This comprehensive test suite tests all 3 core functions of Stock Analyzer V14 as defined in the Project Plan:

1. **Function 1: Ticker Analysis (Manual User Research)** - ✅ Tested
   - Predictions for all intervals (1m, 5m, 1h, 4h, 1d, 1w, 1mo)
   - Market sentiment overview and interpretation
   - Data interpretation and analysis
   - User-facing analysis tool

2. **Function 2: Autonomous Trading** - ⚠️ Disabled
   - Trading tests are disabled (no linked accounts)
   - As requested, trading functionality is not tested

3. **Function 3: Continuous Learning Model** - ✅ Tested
   - Autonomous constant learning system
   - Prediction evaluation and scoring (THE CORE FEATURE)
   - Interval-specific learning
   - Parameter optimization
   - Trade outcome integration

## Test Structure

```
test/
├── __init__.py                          # Test package initialization
├── test_entrypoint_detector.py          # Auto-detects main entrypoint
├── test_function_1_ticker_analysis.py   # Function 1 comprehensive tests
├── test_function_3_continuous_learning.py  # Function 3 comprehensive tests
├── test_runner.py                       # Main test orchestrator
└── README.md                            # This file
```

## Running the Tests

### Quick Start

From the V14 directory:

```bash
python test/test_runner.py
```

Or from the test directory:

```bash
cd test
python test_runner.py
```

### Individual Test Suites

You can also run individual test suites:

```bash
# Function 1 tests only
python test/test_function_1_ticker_analysis.py

# Function 3 tests only
python test/test_function_3_continuous_learning.py
```

## Test Coverage

### Function 1: Ticker Analysis Tests

1. **Data Fetching for All Intervals** - Tests data retrieval for all required intervals
2. **Predictions for All Intervals** - Tests prediction generation across all timeframes
3. **Prediction Format and Validity** - Validates prediction structure and values
4. **Market Sentiment Overview** - Tests sentiment checking functionality
5. **Sentiment Interpretation** - Tests sentiment analysis accuracy
6. **Data Interpretation** - Tests data quality and analysis
7. **Technical Indicators** - Tests RSI, SMA, EMA calculations
8. **Feature Extraction** - Tests ML feature extraction
9. **Multiple Ticker Analysis** - Tests batch analysis capability
10. **Prediction Consistency** - Tests prediction reproducibility
11. **Error Handling** - Tests graceful error handling
12. **Performance** - Tests response time

### Function 3: Continuous Learning Tests

1. **Prediction Storage System** - Tests storage and retrieval
2. **Prediction Creation** - Tests prediction creation for all intervals
3. **Prediction Evaluation** - Tests evaluation (THE CORE FEATURE)
4. **Accuracy Scoring** - Tests accuracy calculation
5. **Confidence Calibration** - Tests confidence calibration
6. **Interval-Specific Learning** - Tests isolated learning per interval
7. **Parameter Optimization** - Tests parameter update system
8. **Trade Outcome Integration** - Tests trade-based learning with higher weight
9. **Constant Learning Engine** - Tests engine initialization and control
10. **Learning Statistics** - Tests statistics tracking
11. **Settings Integration** - Tests settings menu integration
12. **Autonomous Operation** - Tests autonomous operation capability
13. **Expired Prediction Detection** - Tests expiration detection
14. **Parameter Update History** - Tests parameter history tracking

## Test Reports

Test reports are automatically saved to:
```
test/test_report_YYYYMMDD_HHMMSS.json
```

Reports include:
- Overall statistics (total, passed, failed, success rate)
- Function-by-function breakdown
- Detailed test results
- Execution time
- Recommendations

## Entrypoint Detection

The test suite automatically detects the main entrypoint (`Stock Analyzer V14.py`) by:
1. Searching for common entrypoint filenames
2. Verifying the file contains `main()` or `if __name__ == "__main__"`
3. Checking for Stock Analyzer references

## Requirements

The test suite requires:
- Python 3.7+
- All V14 dependencies (pandas, numpy, etc.)
- Network access for data fetching tests
- V14 project structure intact

## Notes

- Some tests may show warnings if data is unavailable (this is expected)
- Trading tests are intentionally disabled
- Tests use real ticker symbols (AAPL, MSFT, TSLA) for realistic testing
- Test data is created and cleaned up automatically
- Tests are designed to be non-destructive

## Troubleshooting

### Import Errors

If you see import errors, ensure:
1. You're running from the V14 directory or test directory
2. The V14 root is in Python's path
3. All dependencies are installed

### Data Fetching Failures

If data fetching tests fail:
- Check network connection
- Verify ticker symbols are valid
- Some intervals may not have data (this is OK)

### Prediction Evaluation Failures

If prediction evaluation fails:
- Models may need training first
- Some predictions may not have expired yet
- This is expected for new installations

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Use descriptive test names
3. Include error handling
4. Add to appropriate test suite file
5. Update this README if adding new test categories

