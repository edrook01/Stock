# Stock Analyzer Development Plan

## Overview
This document tracks development plans for Stock Analyzer versions V13 and V14.

### V13 Stock Analyzer Development Plan

Create a complete V13 version of the Stock Analyzer with all V12 features except AI interface, plus major enhancements for efficiency, reliability, and scalability. This includes optimization improvements, UI redesign, graphing support, auto-trading foundation, and comprehensive testing infrastructure.

### V14 Stock Analyzer Development Plan

Stock Analyzer V14 is a comprehensive trading system integrating long-term investment analysis and short-term CFD trading. It combines a unified ML model with robust risk management, automated browser-based trade execution, and adaptive learning. Key features include:

- **Unified ML Model Architecture**: Uses V13 statistical/technical engine outputs as input features (not logic) for ensemble ML predictions
- **Volatility-Based Risk Management**: ATR-based stop-losses, trailing stops, and dynamic position sizing
- **Browser-Based CFD Trading**: Automated Trading212 trading via browser automation with human-like behavior
- **Adaptive Learning**: Continuous improvement from trade outcomes with feedback loops
- **Sentiment Override Layer**: News monitoring and sentiment-based trade blocking
- **Comprehensive Logging**: Detailed trade logs with performance analysis
- **Enhanced Simulation**: V13 simulator extended with all V14 risk management features

**Status**: V14 Core Architecture Complete (11/15 phases). See `V14/IMPLEMENTATION_STATUS.md` for details.

## Phase 1: Create V13 Directory Structure

### 1.1 Create V13 Folder Structure
- Create `V13/` folder with complete structure:
  - `V13/Stock Analyzer V13.py` - Main full-featured file
  - `V13/micro/` - Next-generation micro tool folder
  - `V13/core/` - Core functionality modules (data fetching, analysis engines)
  - `V13/data/` - Configuration files
  - `V13/memory/` - Deep learning modules and memory
  - `V13/logs/` - Application logs
  - `V13/model/` - Model storage
  - `V13/history/` - Prediction history
  - `V13/cache/` - Data cache for fetched market data

### 1.2 Copy Base Structure from V12
- Copy directory structure from `V12 (ai Removed)/`
- Copy configuration files (config.json, tickers.txt)
- Copy deep learning modules structure
- Ensure all paths are relative for portability

## Phase 2: Build Complete V13 Main File

### 2.1 Create Stock Analyzer V13.py
- **Source**: `V12 (ai Removed)/Stock Analyzer V12.py`
- **Removals**:
  - Remove all AI interface code
  - Remove AI training modules
  - Remove AI-specific imports and functions
- **Keep All Features**:
  - ✅ Market intelligence
  - ✅ Deep learning models
  - ✅ Self-learning controller
  - ✅ Enhanced table displays
  - ✅ Prediction engines (Statistical, Technical, ML)
  - ✅ Continuous learning mode
  - ✅ Batch analysis
  - ✅ Historical pattern analysis
  - ✅ Correlation analysis
  - ✅ UI components and formatting

### 2.2 Verify Feature Completeness
- Ensure all V12 features are present except AI interface
- Test all imports and dependencies
- Verify all modules load correctly

## Phase 3: Ticker List Optimization

### 3.1 Automatic Ticker Validation
- Implement batch-verify function for ticker symbols
- Use Yahoo Finance or Polygon API for validation
- Fetch metadata for multiple tickers in one request
- Check if each symbol is valid/listed
- Cross-check with multiple sources for reliability

### 3.2 Removal or Update of Invalid Tickers
- Flag and remove delisted tickers
- Maintain mapping of old symbols to new ones (for renamed tickers)
- Keep list of delisted companies
- Auto-update ticker list when symbols change

### 3.3 Batch Metadata Fetch
- Utilize endpoints that return info for multiple tickers
- Process results to filter out symbols with no data or deprecated symbols
- Improve efficiency with batch operations

### 3.4 Scheduled Refresh & Audits
- Add maintenance task (menu option or scheduled job)
- Periodically re-validate all stored tickers
- Output summary of findings (X tickers delisted, Y symbols updated)
- Prompt user or automatically apply updates

### 3.5 User Feedback and Logging
- Provide informative logs when tickers are cleaned
- Record all changes in ticker_audit.log
- Suggest alternatives for delisted tickers

### 3.6 Persisting Clean Ticker List
- Save cleaned ticker list to CSV or JSON in data folder
- Serve as master ticker list for analysis features
- Ensure no wasted effort on bad symbols

## Phase 4: Prediction Engine Optimization

### 4.1 Profiling and Bottleneck Identification
- Profile existing prediction algorithms
- Find slow sections (iterative calculations, repeated computations)
- Focus optimization efforts on hotspots

### 4.2 Result Caching
- Implement LRU cache (in-memory) for prediction results
- Use persistent cache file keyed by ticker and analysis parameters
- Reuse recent predictions if underlying data hasn't changed
- Avoid redundant calculations

### 4.3 Parallel Factor Calculations
- Run multiple strategy engines in parallel threads/processes
- Use concurrent.futures or multiprocessing
- Execute multiple strategies simultaneously for single ticker
- Exploit multiple CPU cores to reduce analysis time

### 4.4 Algorithmic Efficiency
- Replace Python loops with NumPy vectorized operations
- Use efficient libraries for heavy computations
- Consider Numba JIT compilation for critical functions
- Add @jit decorator with nopython=True for speed-ups

### 4.5 Optional GPU Acceleration
- Detect GPU availability
- Use CuPy for array computations (drop-in NumPy replacement)
- Use Numba's CUDA support for GPU operations
- Toggle feature based on user preference and availability

### 4.6 PyPy and Alternative Runtimes
- Ensure compatibility with PyPy
- Avoid CPython-specific hacks
- Make external libraries (Numba, CuPy) optional
- Allow program to run under PyPy for faster execution

### 4.7 Maintain Prediction Reliability
- Verify each strategy's output remains consistent with V12
- Use unit tests to compare outputs before/after changes
- Ensure dynamic weighting and ensembling logic produces same signals
- Analyze any deviations for numerical issues

## Phase 5: Data Fetching Acceleration

### 5.1 Asynchronous Downloads
- Replace sequential HTTP requests with asynchronous/multithreaded requests
- Use asyncio with aiohttp for concurrent requests
- Overlap network calls to reduce wait time
- Limit concurrent threads to avoid overloading APIs

### 5.2 Local Data Caching
- Introduce caching layer on disk (cache/ticker_interval.csv)
- Store fetched data with timestamp metadata
- Load from cache if recent (within freshness interval)
- Only hit API if cache missing or stale

### 5.3 Cache Invalidation Strategy
- Set maximum age for cached data (24h for daily, minutes for intraday)
- Update cache only when data is outdated
- Balance data freshness with API call frequency

### 5.4 Concurrent Provider Usage
- Configure multiple data providers (Yahoo Finance, Alpha Vantage, Polygon)
- Attempt parallel retrieval from multiple providers
- Use first successful response
- Include providers with generous free limits for backup

### 5.5 Robust Retry Logic
- Implement retries with exponential backoff
- Handle transient network errors
- Catch common failures (HTTP errors, JSON decode errors)
- Log warnings when sources fail
- Fallback to next provider automatically

### 5.6 Disk Space and Cache Maintenance
- Monitor cache folder size
- Provide mechanism to clear/prune old cache files
- Remove files not accessed in long time
- Implement FIFO policy to limit cache size

### 5.7 Data Integrity Checks
- Verify cached data isn't corrupted
- Perform sanity checks (non-zero rows, expected columns)
- Refetch if cached file is malformed
- Validate all downloads before feeding into analysis

## Phase 6: Menu Redesign

### 6.1 Main Menu Simplification
- Present high-level categories instead of long list:
  - 1. Core Analysis
  - 2. Learning & Training
  - 3. Data & Logs
  - 4. System & Maintenance
- Declutter initial view
- Guide users to right area for specific tasks

### 6.2 Submenu Organization
- **Core Analysis**: Analyze Single Ticker, Analyze Multiple Tickers (Batch), Compare Tickers
- **Learning & Training**: Start/Stop Continuous Training, Review Training Performance, Reset Learned Model
- **Data & Logs**: View Prediction History, View Training Log, View Error Log, Export Logs/History
- **System & Maintenance**: Ticker List Audit/Refresh, Update Data Providers/API Keys, Clear Cache, Check for Updates/Patchnotes

### 6.3 Keyboard Shortcuts
- Enable two-level shortcut commands (e.g., "1A" for Core Analysis > Analyze Single Ticker)
- Display hints in menu (e.g., "1. Core Analysis (e.g., type 1A for Single Ticker)")
- Speed up navigation for advanced users

### 6.4 UI Consistency
- Use consistent prefix and style (numbers for main menu, letters for sub-options)
- Ensure UIComponents handle coloring and formatting uniformly
- Maintain clear instructions at each step
- Always show how to go back and exit

### 6.5 Back Navigation
- Implement robust "Back to Main Menu" option in each submenu
- Provide graceful exit option
- Ensure user can return to main screen from any depth

### 6.6 Testing and Refinement
- Usability-test new menu structure
- Confirm logical grouping of functionalities
- Verify common tasks involve fewer keystrokes
- Refine category labels or groupings if needed

## Phase 7: Graphing Support

### 7.1 Integrate Charting Library
- Use Matplotlib with optional mplfinance extension
- Generate charts without external GUI dependency
- Handle case where library is not installed
- Add to dependencies or provide menu option to install

### 7.2 Price Trend Charts
- Plot price history for given ticker and interval
- Line chart of closing price over time
- Candlestick charts for detailed view
- Access via analysis result prompt or dedicated menu option

### 7.3 Prediction vs Actual Comparison
- Plot predicted values versus actual outcomes
- For each past prediction, plot forecasted percentage gain alongside actual price movement
- Visual performance analysis to see prediction accuracy over time

### 7.4 Prediction Confidence Over Time
- Graph showing model's confidence metric over time
- Timeline where each point is confidence level of prediction
- Highlight trends (increasing/decreasing confidence)
- Overlay actual performance to see if confidence correlates with accuracy

### 7.5 Interactive Elements
- Save charts as image files (history/ticker_chart.png)
- Open with default image viewer for convenience
- In notebook/GUI environment, show plot window directly
- Fallback to file saving for console-only use

### 7.6 Customization Options
- Allow user preferences: time range, chart type (line vs candlestick)
- Include volume subplot option
- Default to sensible choices with flexibility

### 7.7 Performance Considerations
- Implement downsampling for large datasets
- Auto-limit to last N data points for readability
- Warn or limit when plotting very large datasets
- Focus on conveying trends, not every tick

### 7.8 Documentation & Usage
- Update README with chart features
- Document where chart files are saved
- Ensure graceful failure (notify if insufficient data)
- Avoid crashes on chart generation errors

## Phase 8: CFD Auto-Trading Foundation

### 8.1 Modular Trading Interface
- Design abstract BrokerAPI class interface
- Methods: get_balance(), get_positions(), place_order(ticker, quantity, order_type)
- Allow implementation for any brokerage
- Core system interacts with any broker via appropriate module

### 8.2 Trading 212 Integration
- Implement connector module for Trading 212
- Use Trading 212 public API (beta) for CFD accounts
- REST endpoints for placing market orders, checking pending orders, account info
- Proper authentication mechanism
- Leverage demo environment for testing

### 8.3 Simulation Mode (Paper Trading)
- **Using Broker's Demo**: Toggle between demo and live API endpoints
- Demo mode uses demo.trading212.com (virtual trades)
- **Internal Simulator**: Intercept trade orders and record in simulated portfolio
- Update simulated positions with new market data
- Track P/L for each trade and overall portfolio metrics

### 8.4 Safety Layers and Risk Management
- **Confidence Threshold**: Only execute trades if prediction confidence ≥7/10 (configurable)
- **Max Drawdown Limit**: Monitor portfolio drawdown, halt trading if exceeds limit (e.g., 10%)
- **Position Sizing & Limits**: Risk 2% of equity per trade, restrict simultaneous positions
- **Order Types**: Start with basic market orders, extend to limit/stop orders as API allows

### 8.5 Logging and Alerts
- Log every trade decision with details (timestamp, instrument, action, size, price, reason)
- Log safety rule triggers (e.g., halting trades due to drawdown)
- Optionally alert user (email or console warning)

### 8.6 Future Broker Support
- Design system for easy addition of other brokers (MetaTrader, Alpaca, Interactive Brokers)
- Keep broker-specific code in separate modules
- Use config setting to choose broker
- Modular approach ensures scalability

### 8.7 Testing Auto-Trading Features
- Use paper trading mode extensively
- Test end-to-end flow (signal generation → order execution → P/L tracking)
- Simulate various scenarios (winning streak, losing streak, low-confidence signals)
- Verify safety checks work correctly
- Only attempt live trading after thorough testing with small positions

## Phase 9: Testing and Debugging

### 9.1 Create Test Suite
- **File**: `V13/test_v13.py`
- **Test Categories**:
  1. **Ticker Validation Tests**: Feed known valid/invalid tickers, assert correct identification
  2. **Prediction Engine Tests**: Run each strategy engine with dummy dataframe, verify format and bounds
  3. **Data Fetching Tests**: Test async fetching returns DataFrame, verify caching works
  4. **Menu Flow Tests**: Simulate inputs to ensure correct navigation
  5. **Auto-Trading Logic Tests**: Test confidence thresholds, drawdown limits, position sizing
  6. **Basic Functionality Tests**: Ticker analysis, prediction generation, data fetching, indicator computation
  7. **Learning Mode Tests**: Continuous learning loop, prediction evaluation, accuracy tracking
  8. **Table Display Tests**: Table formatting, progress bars, time remaining calculations
  9. **Integration Tests**: Full workflow, batch analysis, historical pattern analysis, correlation analysis
  10. **Error Handling Tests**: Invalid ticker handling, network failure handling, data quality issues

### 9.2 Generate Test Statements
- **File**: `V13/TEST_STATEMENTS.md`
- Comprehensive test cases for manual/automated testing
- Expected outputs
- Edge cases
- Performance benchmarks (e.g., data fetching speed V13 vs V12)
- Stress tests (e.g., running analysis on 100 tickers concurrently)
- Document how to run tests and interpret results

### 9.3 Debug and Fix
- Run full test suite
- Identify and fix issues
- Verify stability
- Ensure all optimizations maintain correctness

## Phase 10: Project Infrastructure and Portability

### 10.1 Structured File Organization
- **Core functionality**: `core/data_fetch.py`, `core/engines/dow_theory.py`, etc.
- **Micro tool**: `micro/` folder
- **Runtime data folders**: `memory/`, `model/`, `history/`, `logs/`, `cache/`
- Keep learned models and history in dedicated folders within project
- Ensure user can copy entire project directory to new machine
- Create directories at startup if they don't exist
- Use relative paths, not absolute

### 10.2 Unit Test Suite (test_v13.py)
- Focus on critical components
- Ticker validation tests
- Prediction engine tests
- Data fetching tests
- Menu flow tests
- Auto-trading logic tests
- Run tests to verify units work and catch regressions

### 10.3 Test Documentation (TEST_STATEMENTS.md)
- Record testing strategy and performance benchmarks
- List scenarios tested (including manual tests)
- Document before/after timing for optimizations
- Document stress-test results
- Describe how to run tests and interpret results

### 10.4 README.md Updates
- **Introduction & Features**: Overview, "What's New" section highlighting 8 key improvements
- **Installation**: Required Python version, dependencies (aiohttp, matplotlib, numba, etc.), API key setup
- **Usage Guide**: Menu navigation, micro tool usage, continuous training, sample commands/screenshots
- **Portability Note**: Emphasize user data stored in memory/, model/, history/ for portability
- **Troubleshooting & FAQ**: Common issues (API limits, GPU support installation, etc.)
- **Patch Notes**: Enumerate changes from V12 to V13 (new features, improvements, bug fixes)

### 10.5 Version Control and Collaboration
- Ensure project in version control (Git) with clear commit history
- Help manage development of V13 and beyond

### 10.6 Continuous Integration (Optional)
- Set up CI pipeline (e.g., GitHub Actions) to run test suite on each commit
- Ensure nothing breaks as new features are added
- Emphasize reliability and scalability

## Implementation Details

### Key Files to Create:
1. `V13/Stock Analyzer V13.py` - Main complete version
2. `V13/core/` - Core modules (data fetching, engines)
3. `V13/test_v13.py` - Test suite
4. `V13/TEST_STATEMENTS.md` - Test documentation
5. `V13/README.md` - V13 documentation
6. `V13/PATCHNOTES.md` - Version changes
7. `V13/patchnotes.txt` - Version changes (alternative format)

### Table Output Format (to match V12):
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
- Continuous background learning loop
- Automatic prediction evaluation
- Accuracy tracking per ticker
- Learning summary with metrics
- Confidence history tracking
- Self-improvement based on results

### Performance Goals:
- 5-10x faster data fetching (via async)
- Parallel strategy execution (exploit multi-core)
- GPU acceleration for large computations (optional)
- Cached predictions to avoid redundant calculations
- Efficient ticker validation with batch operations

