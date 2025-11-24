# Detailed Requirements Breakdown

## Overview
This document provides a detailed, actionable breakdown of every requirement from `Project Plans/Project Plan.md`, organized by priority and implementation complexity.

**Date**: 2024-01-XX
**Purpose**: Guide implementation of missing V13/V15 features

---

## Priority 1: Critical Missing Features (High Impact, High Priority)

### 1.1 Ticker Validation System
**Phase**: 3.1-3.6
**Priority**: CRITICAL
**Complexity**: Medium
**Estimated Effort**: 4-6 hours

#### Detailed Requirements:

**3.1 Automatic Ticker Validation**
- [ ] Create `core/ticker_validator.py`
- [ ] Function: `validate_ticker(ticker: str) -> Dict[str, Any]`
  - Returns: `{"valid": bool, "name": str, "exchange": str, "type": str, "status": str}`
- [ ] Function: `batch_validate_tickers(tickers: List[str]) -> Dict[str, Dict]`
  - Validate multiple tickers in parallel
  - Use asyncio with aiohttp for concurrent requests
  - Return dict mapping ticker -> validation result
- [ ] Support multiple sources:
  - Primary: Yahoo Finance API (`https://query1.finance.yahoo.com/v8/finance/quoteSummary/{ticker}`)
  - Fallback: Alpha Vantage (if API key available)
  - Fallback: Polygon.io (if API key available)
- [ ] Check if symbol is valid/listed
- [ ] Cross-check with multiple sources for reliability
- [ ] Cache validation results (24-hour TTL)

**3.2 Removal or Update of Invalid Tickers**
- [ ] Function: `flag_delisted_tickers(ticker_list: List[str]) -> Dict[str, str]`
  - Returns: `{"delisted": List[str], "renamed": Dict[str, str], "valid": List[str]}`
- [ ] Maintain mapping file: `data/ticker_mappings.json`
  - Format: `{"old_symbol": "new_symbol", "reason": "renamed|merged|split"}`
- [ ] Maintain delisted list: `data/delisted_tickers.json`
  - Format: `["TICKER1", "TICKER2", ...]`
- [ ] Auto-update ticker list when symbols change
- [ ] Function: `update_ticker_list(file_path: Path, remove_invalid: bool = True) -> Dict`

**3.3 Batch Metadata Fetch**
- [ ] Function: `fetch_ticker_metadata_batch(tickers: List[str]) -> Dict[str, Dict]`
  - Use Yahoo Finance batch endpoint if available
  - Or use concurrent async requests
  - Fetch: name, exchange, sector, industry, market cap, etc.
- [ ] Process results to filter out symbols with no data
- [ ] Return only valid tickers with metadata
- [ ] Handle rate limiting (max 10 concurrent requests)

**3.4 Scheduled Refresh & Audits**
- [ ] Create `core/ticker_auditor.py`
- [ ] Function: `audit_ticker_list(file_path: Path) -> Dict`
  - Returns: `{"total": int, "valid": int, "invalid": int, "delisted": int, "renamed": int, "report": str}`
- [ ] Add menu option: "System & Maintenance > Ticker List Audit/Refresh"
- [ ] Implement scheduled job capability (optional cron-like)
- [ ] Output summary: "X tickers delisted, Y symbols updated"
- [ ] Prompt user or automatically apply updates
- [ ] Function: `schedule_ticker_audit(interval_days: int = 30) -> None`

**3.5 User Feedback and Logging**
- [ ] Create log file: `logs/ticker_audit.log`
- [ ] Log format: `[TIMESTAMP] ACTION: TICKER - REASON`
- [ ] Examples:
  - `[2024-01-01 10:00:00] DELISTED: XYZ - No longer trading`
  - `[2024-01-01 10:00:01] RENAMED: ABC -> ABC1 - Symbol change`
- [ ] Function: `log_ticker_change(action: str, ticker: str, reason: str, new_ticker: str = None)`
- [ ] Suggest alternatives for delisted tickers:
  - Check for similar tickers
  - Suggest related companies in same sector
  - Function: `suggest_alternatives(ticker: str) -> List[str]`

**3.6 Persisting Clean Ticker List**
- [ ] Function: `save_clean_ticker_list(tickers: List[str], file_path: Path, format: str = "json")`
- [ ] Support formats: JSON, CSV, TXT
- [ ] JSON format: `{"tickers": ["AAPL", "MSFT", ...], "last_updated": "2024-01-01T10:00:00", "total": 100}`
- [ ] CSV format: `Ticker,Name,Exchange,Status`
- [ ] TXT format: One ticker per line
- [ ] Save to: `data/validated_tickers.json` (master list)
- [ ] Function: `load_validated_tickers(file_path: Path) -> List[str]`

#### Implementation Files:
- `core/ticker_validator.py` (new file)
- `core/ticker_auditor.py` (new file)
- `data/ticker_mappings.json` (new file)
- `data/delisted_tickers.json` (new file)
- `data/validated_tickers.json` (new file)
- Update `ui/menu_v15.py` to add audit menu option

---

### 1.2 Multiple Data Provider Support
**Phase**: 5.4-5.5
**Priority**: HIGH
**Complexity**: Medium
**Estimated Effort**: 3-4 hours

#### Detailed Requirements:

**5.4 Concurrent Provider Usage**
- [ ] Create `core/data_providers.py`
- [ ] Abstract base class: `DataProvider`
  ```python
  class DataProvider(ABC):
      @abstractmethod
      async def fetch_prices(self, ticker: str, interval: str) -> Optional[pd.DataFrame]
      @abstractmethod
      def get_name(self) -> str
      @abstractmethod
      def is_available(self) -> bool
  ```
- [ ] Implement providers:
  - `YahooFinanceProvider` (existing, refactor)
  - `AlphaVantageProvider` (new, requires API key)
  - `PolygonProvider` (new, requires API key)
- [ ] Function: `fetch_from_multiple_providers(ticker: str, interval: str, providers: List[DataProvider]) -> Optional[pd.DataFrame]`
  - Try providers in parallel
  - Use first successful response
  - Log which provider was used
- [ ] Configuration in `data/config_v15.json`:
  ```json
  "data_providers": {
    "primary": "yahoo_finance",
    "fallbacks": ["alpha_vantage", "polygon"],
    "alpha_vantage_api_key": "",
    "polygon_api_key": ""
  }
  ```
- [ ] Function: `get_available_providers() -> List[DataProvider]`
  - Check API keys
  - Test connectivity
  - Return only available providers

**5.5 Robust Retry Logic**
- [ ] Create `core/retry_handler.py`
- [ ] Function: `retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0) -> Any`
  - Exponential backoff: delay = min(base_delay * (2 ** attempt), max_delay)
  - Jitter: add random 0-1 second
  - Retry on: `aiohttp.ClientError`, `asyncio.TimeoutError`, `ConnectionError`
- [ ] Function: `fetch_with_retry(provider: DataProvider, ticker: str, interval: str) -> Optional[pd.DataFrame]`
  - Retry up to 3 times with exponential backoff
  - Log each retry attempt
  - Fallback to next provider if all retries fail
- [ ] Update `core/data_fetcher.py` to use retry handler
- [ ] Log warnings when sources fail:
  - `logs/data_fetch_errors.log`
  - Format: `[TIMESTAMP] PROVIDER: TICKER - ERROR - RETRY_COUNT`

#### Implementation Files:
- `core/data_providers.py` (new file)
- `core/retry_handler.py` (new file)
- Update `core/data_fetcher.py` to use providers
- Update `data/config_v15.json` with provider config

---

### 1.3 Cache Maintenance System
**Phase**: 5.6
**Priority**: MEDIUM
**Complexity**: Low
**Estimated Effort**: 2-3 hours

#### Detailed Requirements:

**5.6 Disk Space and Cache Maintenance**
- [ ] Create `core/cache_manager.py`
- [ ] Function: `get_cache_size() -> Dict[str, Any]`
  - Returns: `{"total_size_mb": float, "file_count": int, "oldest_file": str, "newest_file": str}`
- [ ] Function: `prune_cache(max_age_days: int = 30, max_size_mb: float = 1000.0) -> Dict`
  - Remove files older than max_age_days
  - If still over max_size_mb, remove oldest files (FIFO)
  - Returns: `{"removed_count": int, "freed_mb": float}`
- [ ] Function: `clear_cache(confirm: bool = False) -> bool`
  - Clear all cache files
  - Require confirmation
- [ ] Function: `get_cache_statistics() -> Dict`
  - Cache hit rate
  - Average age of cached files
  - Size distribution by ticker/interval
- [ ] Add menu option: "System & Maintenance > Cache Management"
  - Show cache statistics
  - Option to prune old cache
  - Option to clear all cache
- [ ] Automatic pruning:
  - Run on startup if cache > 2GB
  - Run weekly if configured
- [ ] Log cache operations: `logs/cache_operations.log`

#### Implementation Files:
- `core/cache_manager.py` (new file)
- Update `ui/menu_v15.py` to add cache management
- Update `core/setup.py` to check cache size on startup

---

## Priority 2: Enhanced Features (Medium Priority)

### 2.1 Advanced Charting with Matplotlib
**Phase**: 7.1-7.6
**Priority**: MEDIUM
**Complexity**: Medium-High
**Estimated Effort**: 6-8 hours

#### Detailed Requirements:

**7.1 Integrate Charting Library**
- [ ] Add to dependencies: `matplotlib`, `mplfinance` (optional)
- [ ] Create `ui/charts.py`
- [ ] Function: `check_charting_available() -> Dict[str, bool]`
  - Check if matplotlib installed
  - Check if mplfinance installed
  - Return availability status
- [ ] Graceful fallback to Streamlit charts if matplotlib not available
- [ ] Installation prompt if library missing

**7.2 Price Trend Charts**
- [ ] Function: `plot_price_trend(df: pd.DataFrame, ticker: str, chart_type: str = "line") -> Path`
  - Chart types: "line", "candlestick", "ohlc"
  - Save to: `history/charts/{ticker}_{timestamp}.png`
  - Return file path
- [ ] Function: `display_price_chart(df: pd.DataFrame, ticker: str, chart_type: str = "line")`
  - Display in Streamlit if available
  - Or save and show file path
- [ ] Candlestick charts using mplfinance:
  ```python
  import mplfinance as mpf
  mpf.plot(df, type='candle', style='yahoo', savefig='chart.png')
  ```

**7.3 Prediction vs Actual Comparison**
- [ ] Function: `plot_prediction_vs_actual(trades: List[Dict]) -> Path`
  - X-axis: Time
  - Y-axis: Percentage change
  - Plot predicted % vs actual %
  - Color code: green if accurate, red if inaccurate
  - Add confidence bands
- [ ] Function: `calculate_prediction_accuracy(trades: List[Dict]) -> Dict`
  - Mean absolute error
  - Mean squared error
  - Accuracy within confidence range
- [ ] Add to `ui/pages/trade_history.py`:
  - New section: "Prediction Accuracy Analysis"
  - Chart: Prediction vs Actual
  - Metrics: MAE, MSE, Accuracy %

**7.4 Prediction Confidence Over Time**
- [ ] Function: `plot_confidence_over_time(predictions: List[Dict]) -> Path`
  - X-axis: Time
  - Y-axis: Confidence (0-1)
  - Overlay: Actual win rate (if available)
  - Highlight trends
- [ ] Function: `analyze_confidence_trends(predictions: List[Dict]) -> Dict`
  - Increasing/decreasing trend
  - Correlation with accuracy
  - Confidence calibration status
- [ ] Add to `ui/pages/stock_analysis.py`:
  - Confidence history chart
  - Trend analysis

**7.5 Interactive Elements**
- [ ] Function: `save_chart(fig, filename: str, format: str = "png") -> Path`
  - Save as PNG, PDF, SVG
  - Return file path
- [ ] Function: `open_chart(file_path: Path) -> None`
  - Open with default image viewer (platform-specific)
- [ ] Streamlit integration:
  - Display matplotlib figures with `st.pyplot()`
  - Download button for saved charts

**7.6 Customization Options**
- [ ] Create chart configuration:
  ```python
  class ChartConfig:
      time_range: Optional[Tuple[datetime, datetime]]
      chart_type: str  # "line", "candlestick", "ohlc"
      show_volume: bool
      indicators: List[str]  # ["sma_20", "ema_50", "rsi"]
      style: str  # "yahoo", "classic", "night"
  ```
- [ ] Function: `create_custom_chart(df: pd.DataFrame, config: ChartConfig) -> Path`
- [ ] Add chart settings to `ui/pages/stock_analysis.py`:
  - Time range selector
  - Chart type selector
  - Show volume checkbox
  - Indicator overlays

#### Implementation Files:
- `ui/charts.py` (new file)
- Update `ui/pages/stock_analysis.py` with advanced charts
- Update `ui/pages/trade_history.py` with prediction analysis
- Update `requirements.txt` or `setup.py` with matplotlib dependency

---

### 2.2 Keyboard Shortcuts System
**Phase**: 6.3
**Priority**: LOW-MEDIUM
**Complexity**: Low-Medium
**Estimated Effort**: 3-4 hours

#### Detailed Requirements:

**6.3 Keyboard Shortcuts**
- [ ] Create `ui/keyboard_shortcuts.py`
- [ ] Function: `parse_shortcut(input_str: str) -> Tuple[str, Optional[str]]`
  - Parse "1A" -> ("1", "A")
  - Parse "1" -> ("1", None)
  - Handle invalid formats
- [ ] Shortcut mapping:
  ```python
  SHORTCUTS = {
      "1": "Core Analysis",
      "1A": "Core Analysis > Analyze Single Ticker",
      "1B": "Core Analysis > Analyze Multiple Tickers",
      "1C": "Core Analysis > Compare Tickers",
      "2": "Learning & Training",
      "2A": "Learning & Training > Start Continuous Training",
      # ... etc
  }
  ```
- [ ] Function: `display_shortcut_hints() -> None`
  - Show shortcuts in menu display
  - Format: "1. Core Analysis (type 1A for Single Ticker)"
- [ ] Update `ui/menu_v15.py`:
  - Accept two-level shortcuts
  - Display hints
  - Handle shortcut navigation
- [ ] Function: `get_shortcut_help() -> str`
  - Return formatted help text
  - Show all available shortcuts

#### Implementation Files:
- `ui/keyboard_shortcuts.py` (new file)
- Update `ui/menu_v15.py` to support shortcuts

---

### 2.3 Broker Abstraction Interface
**Phase**: 8.6
**Priority**: MEDIUM
**Complexity**: Medium
**Estimated Effort**: 4-5 hours

#### Detailed Requirements:

**8.6 Future Broker Support**
- [ ] Create `trading/broker_interface.py`
- [ ] Abstract base class:
  ```python
  class BrokerAPI(ABC):
      @abstractmethod
      async def get_balance(self) -> float
      @abstractmethod
      async def get_positions(self) -> List[Dict]
      @abstractmethod
      async def place_order(self, ticker: str, quantity: float, order_type: str) -> str
      @abstractmethod
      async def cancel_order(self, order_id: str) -> bool
      @abstractmethod
      async def get_account_status(self) -> Dict
  ```
- [ ] Implement `Trading212Broker` (wrap existing browser automation)
- [ ] Implement `AlpacaBroker` (future)
- [ ] Implement `InteractiveBrokersBroker` (future)
- [ ] Configuration:
  ```json
  "trading": {
    "broker": "trading212",
    "brokers": {
      "trading212": {...},
      "alpaca": {...}
    }
  }
  ```
- [ ] Function: `get_broker(broker_name: str) -> BrokerAPI`
  - Factory function
  - Return configured broker instance
- [ ] Update trading modules to use BrokerAPI interface

#### Implementation Files:
- `trading/broker_interface.py` (new file)
- `trading/brokers/trading212_broker.py` (new file, wraps browser automation)
- Update `browser/trade_executor.py` to implement BrokerAPI
- Update `data/config_v15.json` with broker config

---

## Priority 3: Menu Feature Completion

### 3.1 Missing V13 Menu Features
**Phase**: 6.2
**Priority**: LOW-MEDIUM
**Complexity**: Low-Medium
**Estimated Effort**: 4-6 hours

#### Detailed Requirements:

**Core Analysis Submenu:**
- [ ] "Analyze Single Ticker" - ✅ Exists (via V15 features)
- [ ] "Analyze Multiple Tickers (Batch)" - ⚠️ Needs implementation
  - Function: `batch_analyze_tickers(tickers: List[str], timeframe: str) -> Dict`
  - Process in parallel
  - Return results for all tickers
- [ ] "Compare Tickers" - ❌ Missing
  - Function: `compare_tickers(tickers: List[str], timeframe: str) -> Dict`
  - Side-by-side comparison
  - Correlation analysis
  - Relative performance

**Learning & Training Submenu:**
- [ ] "Start/Stop Continuous Training" - ⚠️ Partial (exists but needs enhancement)
- [ ] "Review Training Performance" - ✅ Exists (via trade log analysis)
- [ ] "Reset Learned Model" - ❌ Missing
  - Function: `reset_model(timeframe: str) -> bool`
  - Clear model weights
  - Reset confidence calibration

**Data & Logs Submenu:**
- [ ] "View Prediction History" - ✅ Exists (via trade history)
- [ ] "View Training Log" - ✅ Exists
- [ ] "View Error Log" - ✅ Exists
- [ ] "Export Logs/History" - ✅ Exists (via trade history export)

**System & Maintenance Submenu:**
- [ ] "Ticker List Audit/Refresh" - ❌ Missing (see 1.1)
- [ ] "Update Data Providers/API Keys" - ⚠️ Partial (needs UI)
  - Settings page has this but needs enhancement
- [ ] "Clear Cache" - ❌ Missing (see 1.3)
- [ ] "Check for Updates/Patchnotes" - ❌ Missing
  - Function: `check_for_updates() -> Dict`
  - Read `PATCHNOTES.md`
  - Compare versions
  - Display changelog

#### Implementation Files:
- `core/batch_analyzer.py` (new file)
- `core/ticker_comparator.py` (new file)
- `core/update_checker.py` (new file)
- Update `ui/menu_v15.py` with all submenu options

---

## Priority 4: V13 Main File (If Required)

### 4.1 Determine V13 vs V15 Relationship
**Phase**: 2.1-2.2
**Priority**: CLARIFICATION NEEDED
**Complexity**: Unknown
**Estimated Effort**: TBD

#### Questions to Answer:
1. Should V13 exist as separate version?
2. Or is V15 the evolution that includes all V13 features?
3. If V13 needed, what's the relationship to V15?

#### If V13 Required:
- [ ] Create `V13/Stock Analyzer V13.py`
- [ ] Copy from V12 (if available)
- [ ] Remove AI interface code
- [ ] Keep all V13 features
- [ ] Ensure compatibility with V15

---

## Implementation Priority Order

1. **Week 1**: Ticker Validation System (1.1)
2. **Week 1**: Multiple Data Provider Support (1.2)
3. **Week 1**: Cache Maintenance System (1.3)
4. **Week 2**: Advanced Charting (2.1)
5. **Week 2**: Missing Menu Features (3.1)
6. **Week 3**: Keyboard Shortcuts (2.2)
7. **Week 3**: Broker Abstraction (2.3)
8. **Week 4**: V13 Main File (if required) (4.1)

---

## Testing Requirements

For each feature, create:
- [ ] Unit tests in `test_v15.py`
- [ ] Integration tests
- [ ] Documentation updates
- [ ] Update `TEST_STATEMENTS.md`
- [ ] Update `README.md` if user-facing

---

## Configuration Updates Needed

Update `data/config_v15.json` with:
- [ ] Ticker validation settings
- [ ] Data provider configuration
- [ ] Cache management settings
- [ ] Chart preferences
- [ ] Broker selection

---

## Documentation Updates Needed

- [ ] Update `README.md` with new features
- [ ] Update `PATCHNOTES.md` with changes
- [ ] Create user guide for new features
- [ ] Update `IMPLEMENTATION_STATUS.md`

