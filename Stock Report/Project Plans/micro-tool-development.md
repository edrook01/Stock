# V13 Micro Tool Development Plan

## Overview
Create next-generation micro tool for V13 with enhanced continuous learning mode, table output matching the main V13 application, and optimized performance. The micro tool will be lightweight, scriptable, and perfect for AI agent integration while maintaining feature parity with core analysis capabilities.

## Phase 1: Enhance Micro Tool with Full Learning Mode

### 1.1 Enhance Continuous Learning Controller
- **File**: `V13/micro/stock_predictor.py`
- **Enhancements**:
  - Full `SelfLearningController` matching V13 main implementation
  - Automatic prediction evaluation every 60 seconds
  - Prediction accuracy tracking and display
  - Learning feedback integration into ML engine
  - Learning summary display with metrics
  - Confidence history tracking
  - Self-improvement based on results
  - Continuous background learning loop (runs every 1 second)

### 1.2 Add Enhanced Table Output
- Replace simple `display_all_predictions()` with enhanced table format
- Add progress bars for price and time tracking
- Use UIComponents-style table formatting (separators, aligned rows)
- Include columns: Ticker, Interval, Current, Target, High, Low, Confidence, Time Status
- Match V13 main application table format exactly

### 1.3 Integrate Table Display Functions
- Import or replicate `deep_learning_display.py` table functions
- Add `format_time_remaining()` with proper formatting
- Add progress bar functions for visual feedback
- Create `display_predictions_table()` function matching V13 format

## Phase 2: Feature Parity with V13 Main

### 2.1 Core Prediction Engines
- Ensure all engines match V13 main:
  - Statistical engine
  - Technical engine
  - ML engine with learning feedback
  - Prediction consolidation
- Reuse core logic from V13 main (avoid code duplication)
- Import same modules that main program uses for calculations

### 2.2 Learning Mode Features
- Continuous background learning loop
- Automatic prediction evaluation
- Accuracy tracking per ticker
- Learning summary with metrics
- Confidence history tracking
- Self-improvement based on results

### 2.3 Data Management
- Match V13 data structure
- Prediction history format
- Training data caching
- Model persistence
- Share config files with main tool or have defaults that mirror main's settings

## Phase 3: Micro Tool Optimization (From User Plan Section 7)

### 3.1 Separate Micro Module
- **Location**: `V13/micro/`
- Trimmed-down version of main tool
- Core analysis functions that can be invoked programmatically
- Without full interactive UI and advanced learning modes

### 3.2 Minimal UI, Direct Functions
- Single Python script or package
- Expose functions like `analyze_ticker(ticker, interval) -> JSON`
- Return results in structured format (JSON/dict containing predictions, signals)
- Enable external AI agents or automation to easily use it

### 3.3 Command-Line Interface
- Implement CLI entry-point for micro tool
- Example: `python micro_analyzer.py --ticker AAPL --interval 1d`
- Output summary analysis in console or print JSON
- Support necessary options (which strategies to run, output format)
- Use Python's argparse library for parameter parsing

### 3.4 Reuse Core Logic
- Import same modules that main program uses for calculations
- Bypass interactive menus
- Improvements to core logic automatically apply to micro tool
- Avoid duplicating code

### 3.5 Speed and Stability
- Exclude advanced features like continuous learning or plotting (if not needed by AI agent)
- Load faster and run with less memory
- Handle analysis requests robustly
- Return errors to caller in controlled way (e.g., "Ticker not found" as JSON error message)

### 3.6 AI Agent Integration
- Functions return machine-readable results
- Simple outputs (avoid unnecessary formatting or color codes)
- Focus on raw data
- Enable integration with chatbots or larger automation pipelines
- Agent can query micro tool for latest prediction and make decisions

### 3.7 Micro vs Full Version Parity
- Document differences between micro tool and full version
- Micro tool focuses on analysis, omits learning mode and auto-trading (require state and interaction)
- Ensure analysis results from micro are identical to full tool given same inputs
- Load same ticker list and configurations
- Share config files with main tool or have own defaults

### 3.8 Packaging Consideration
- Structure code for future packaging as installable package or API (e.g., via pip)
- Clear function interfaces, minimal side effects
- Could be turned into library if needed
- For now, folder with script is sufficient

## Phase 4: Performance Optimization for Micro Tool

### 4.1 Result Caching
- Implement caching for prediction results
- Use LRU cache (in-memory) or persistent cache file
- Key by ticker and analysis parameters
- Reuse recent predictions if underlying data hasn't changed

### 4.2 Efficient Data Fetching
- Support async data fetching (if main tool implements it)
- Use cached data when available
- Minimize external API calls

### 4.3 Algorithmic Efficiency
- Benefit from V13 main optimizations (NumPy vectorization, Numba JIT)
- Use efficient libraries for computations
- Fast execution for scriptable use cases

## Phase 5: Testing and Debugging

### 5.1 Micro Tool Test Suite
- **File**: `V13/micro/test_micro.py`
- **Test Categories**:
  1. **Learning Mode Tests**
     - Test continuous learning loop startup/shutdown
     - Test prediction evaluation
     - Test accuracy tracking
     - Test learning summary generation
  
  2. **Table Display Tests**
     - Test table formatting
     - Test progress bars
     - Test time remaining calculations
     - Test prediction table display
  
  3. **Prediction Engine Tests**
     - Test statistical engine
     - Test technical engine
     - Test ML engine with learning feedback
     - Test prediction consolidation
  
  4. **CLI Interface Tests**
     - Test command-line argument parsing
     - Test JSON output format
     - Test error handling in CLI mode
     - Test direct function calls
  
  5. **Integration Tests**
     - Test full workflow: analyze → predict → evaluate → learn
     - Test learning loop stability
     - Test data persistence
     - Test error recovery
     - Test AI agent integration (mock agent calling functions)
  
  6. **Error Handling Tests**
     - Test invalid ticker handling
     - Test network failure handling
     - Test data quality issues
     - Test learning loop error recovery
     - Test JSON error message format

### 5.2 Generate Test Statements
- **File**: `V13/micro/TEST_STATEMENTS.md`
- Comprehensive test cases for micro tool
- Expected outputs
- Edge cases
- Performance benchmarks
- AI agent integration examples

### 5.3 Debug and Fix
- Run test suite
- Identify issues
- Fix bugs
- Verify stability
- Ensure micro tool matches main tool output for same inputs

## Phase 6: Documentation

### 6.1 Update Micro Tool README
- Document enhanced features
- Learning mode usage
- Table output format
- Integration examples
- CLI usage examples
- AI agent integration guide

### 6.2 Create Migration Guide
- Guide for upgrading from V12 micro tool
- Feature comparison
- Breaking changes (if any)
- Configuration migration steps

### 6.3 API Documentation
- Document all exposed functions
- Function signatures and return types
- Example usage for each function
- JSON output format specification

## Implementation Details

### Key Files to Modify/Create:
1. `V13/micro/stock_predictor.py` - Enhanced micro tool
2. `V13/micro/micro_analyzer.py` - CLI entry point (if separate from stock_predictor.py)
3. `V13/micro/test_micro.py` - Test suite
4. `V13/micro/TEST_STATEMENTS.md` - Test documentation
5. `V13/micro/README.md` - Updated documentation
6. `V13/micro/API.md` - API documentation for AI agent integration

### Table Output Format (matching V13 main):
```
Predictions for {ticker}
================================================================================
Interval    Current     Target      High        Low         Conf    Time Status
--------------------------------------------------------------------------------
1d          $150.00     $155.00     $158.00     $152.00     7.5     12h remaining
  Price Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 50.0%
  Time Progress:  [████████████████████████████████] 100.0%
================================================================================
```

### Learning Mode Features:
- Continuous background learning loop (runs every 1 second)
- Automatic prediction evaluation (every 60 seconds)
- Prediction accuracy tracking per ticker
- Learning summary display with table format
- Confidence history tracking
- Self-improvement based on prediction results
- Learning feedback integration into ML predictions

### CLI Usage Examples:
```bash
# Basic analysis
python micro_analyzer.py --ticker AAPL --interval 1d

# JSON output for AI agents
python micro_analyzer.py --ticker AAPL --interval 1d --format json

# Multiple tickers
python micro_analyzer.py --tickers AAPL MSFT GOOGL --interval 1d

# Specify strategies
python micro_analyzer.py --ticker AAPL --strategies statistical technical
```

### Function Interface Example:
```python
from micro.stock_predictor import analyze_ticker

# Direct function call
result = analyze_ticker("AAPL", interval="1d")
# Returns: {
#   "ticker": "AAPL",
#   "interval": "1d",
#   "prediction": 2.5,
#   "confidence": 7.5,
#   "range_low": 0.5,
#   "range_high": 4.5,
#   "engines": ["Statistical", "Technical", "ML"],
#   "timestamp": "2025-01-20T12:00:00"
# }
```

### Differences from V12 Micro Tool:
- Enhanced table output with progress bars
- Full continuous learning mode (not just basic)
- Automatic prediction evaluation
- Learning summary with metrics
- Better integration with V13 main application
- Improved error handling and recovery
- CLI interface for scriptable use
- JSON output format for AI agent integration
- Performance optimizations (caching, efficient algorithms)

### Differences from V13 Main Tool:
- No interactive menu system
- No graphing support (focused on data output)
- No auto-trading features (requires state management)
- Simplified UI (CLI only)
- Faster startup (excludes heavy features)
- Machine-readable output focus
- Designed for programmatic use
