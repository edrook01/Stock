# Data Logs Extension - Complete Documentation

## Overview

The data logging system has been significantly extended to capture comprehensive data across all aspects of the Stock Analyzer V15 system. This extension adds detailed logging for predictions, model performance, market events, system events, and enhanced trade context.

## New Logging Modules

### 1. Prediction Logger (`prediction_logger.py`)

Logs all predictions made by models with full context for analysis and learning.

**Features:**
- Logs every prediction with inputs, outputs, and context
- Tracks prediction outcomes and accuracy
- Stores model type, version, and confidence scores
- Captures features used, indicators, and market conditions
- Links predictions to trades

**Data Captured:**
- Prediction ID, timestamp, ticker, timeframe
- Model type, version, confidence
- Predicted price, change, percentage change
- Current price at prediction time
- Features used, technical indicators
- Market conditions, sentiment scores
- Actual outcomes (updated later)
- Error metrics and accuracy

**Files Generated:**
- `history/predictions.csv` - CSV format for analysis
- `history/predictions.json` - JSON format with full detail

**Usage:**
```python
from sa_logging import get_prediction_logger

logger = get_prediction_logger()
prediction_id = logger.log_prediction(
    ticker="AAPL",
    timeframe="1d",
    model_type="unified",
    prediction_type="price",
    predicted_price=150.50,
    predicted_change=2.50,
    predicted_change_pct=1.69,
    current_price=148.00,
    confidence=0.85,
    sentiment_score=0.75
)

# Later, log the outcome
logger.log_prediction_outcome(
    prediction_id=prediction_id,
    actual_price=151.00,
    trade_id=trade_id
)
```

### 2. Model Performance Logger (`model_logger.py`)

Logs model training, evaluation, and performance metrics.

**Features:**
- Tracks all training events and evaluations
- Stores comprehensive performance metrics
- Captures hyperparameters and feature importance
- Monitors model versioning and improvements
- Tracks training times and resource usage

**Data Captured:**
- Event type (training_start, training_end, evaluation, retraining)
- Model type, version, timeframe
- Training/validation/test sample counts
- Loss metrics (train, validation, test)
- Accuracy metrics (train, validation, test)
- MSE, MAE, R² scores
- Feature importance rankings
- Hyperparameters used
- Training duration and epochs
- Learning rate, batch size

**Files Generated:**
- `history/model_performance.csv` - CSV format for analysis
- `history/model_performance.json` - JSON format with full detail

**Usage:**
```python
from sa_logging import get_model_logger

logger = get_model_logger()
log_id = logger.log_training_event(
    event_type="training_end",
    model_type="unified",
    model_version="v1.2.3",
    timeframe="1d",
    train_accuracy=0.92,
    validation_accuracy=0.89,
    test_accuracy=0.87,
    training_time=120.5,
    epochs=50
)
```

### 3. Market Data Logger (`market_logger.py`)

Logs market events, sentiment changes, news impact, and market conditions.

**Features:**
- Tracks sentiment changes and news impacts
- Logs volume spikes and price movements
- Captures technical analysis data
- Monitors market conditions and trends
- Links events to trades and predictions

**Data Captured:**
- Event type (sentiment_change, news_impact, volume_spike, etc.)
- Ticker, market index
- Sentiment scores and changes
- News headlines, count, and impact assessment
- Volume and price changes
- Volatility measures
- Market conditions and trends
- Technical indicators (RSI, MACD, moving averages)
- Support and resistance levels
- Sector information

**Files Generated:**
- `history/market_data.csv` - CSV format for analysis
- `history/market_data.json` - JSON format with full detail

**Usage:**
```python
from sa_logging import get_market_logger

logger = get_market_logger()
event_id = logger.log_sentiment_change(
    ticker="AAPL",
    sentiment_score=0.75,
    sentiment_change=0.15,
    sentiment_source="analyzer"
)

# Or log news impact
event_id = logger.log_news_impact(
    ticker="AAPL",
    news_headlines=["Apple announces new product", "Strong earnings report"],
    news_impact="positive",
    sentiment_score=0.80
)
```

### 4. System Event Logger (`system_logger.py`)

Logs system events, errors, warnings, and operational data.

**Features:**
- Comprehensive system event tracking
- Error logging with stack traces
- Performance monitoring (function call durations)
- Component-level event tracking
- Context preservation for debugging

**Data Captured:**
- Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Category, component, function
- Event type, message
- Error type, message, stack trace
- Context dictionary
- Duration for performance tracking
- Status (success, failure, partial, timeout)

**Files Generated:**
- `history/system_events.csv` - CSV format for analysis
- `history/system_events.json` - JSON format with full detail

**Usage:**
```python
from sa_logging import get_system_logger

logger = get_system_logger()

# Log info
logger.log_info(
    category="data_fetch",
    component="data_fetcher",
    function="fetch_prices",
    message="Successfully fetched data for AAPL"
)

# Log errors
try:
    # Some operation
    pass
except Exception as e:
    logger.log_error(
        category="model_training",
        component="unified_model",
        function="train",
        message="Training failed",
        error=e
    )
```

## Enhanced Existing Loggers

### 5. Extended Trade Logger (`trade_logger.py`)

The existing trade logger has been extended with additional context fields.

**New Fields Added:**
- `prediction_id` - Links trade to prediction
- `sentiment_score` - Sentiment at entry
- `rsi` - RSI indicator value
- `macd` - MACD indicator values
- `volume` - Trading volume
- `volatility` - Volatility measure
- `support_level` - Support level
- `resistance_level` - Resistance level
- `indicators` - Dictionary of technical indicators
- `market_condition` - Market condition at entry
- `news_count` - Number of recent news items

**Enhanced CSV Headers:**
The CSV now includes all new fields for comprehensive analysis.

### 6. Enhanced Trade Tracker (`trade_tracker.py`)

The trade tracker now includes prediction and market context.

**New Fields Added:**
- `prediction_id` - Associated prediction ID
- `sentiment_score` - Sentiment score at entry
- `market_condition` - Market condition at entry
- `indicators` - Technical indicators at entry

## Data Storage

All logs are stored in the `history/` directory with the following structure:

```
history/
├── trades.csv              # Enhanced trade logs (CSV)
├── trades.json             # Enhanced trade logs (JSON)
├── trade_outcomes.json     # Trade outcomes with context
├── predictions.csv         # All predictions (CSV)
├── predictions.json        # All predictions (JSON)
├── model_performance.csv   # Model training/eval logs (CSV)
├── model_performance.json  # Model training/eval logs (JSON)
├── market_data.csv         # Market events (CSV)
├── market_data.json        # Market events (JSON)
├── system_events.csv       # System events (CSV)
└── system_events.json      # System events (JSON)
```

## Benefits

1. **Comprehensive Analysis**: All aspects of trading and predictions are now logged
2. **Learning Improvement**: Rich context enables better model training
3. **Debugging**: Detailed system logs help identify issues
4. **Performance Tracking**: Monitor model improvements over time
5. **Market Insights**: Track sentiment and news impacts on trading
6. **Trade Analysis**: Enhanced trade context for strategy refinement

## Integration Points

### Prediction Flow
1. Model makes prediction → `PredictionLogger.log_prediction()`
2. Trade is executed → `TradeLogger.log_trade_entry()` (with prediction_id)
3. Trade exits → `TradeLogger.log_trade_exit()`
4. Update prediction outcome → `PredictionLogger.log_prediction_outcome()`

### Training Flow
1. Training starts → `ModelLogger.log_training_event(event_type="training_start")`
2. Training completes → `ModelLogger.log_training_event(event_type="training_end")`
3. Model evaluated → `ModelLogger.log_evaluation()`

### Market Events
1. Sentiment changes → `MarketLogger.log_sentiment_change()`
2. News events → `MarketLogger.log_news_impact()`
3. Volume spikes → `MarketLogger.log_volume_spike()`
4. Technical analysis → `MarketLogger.log_technical_analysis()`

### System Monitoring
1. Function calls → `SystemLogger.log_function_call()`
2. Errors → `SystemLogger.log_error()`
3. Warnings → `SystemLogger.log_warning()`
4. Info events → `SystemLogger.log_info()`

## Querying and Analysis

All loggers provide query methods:

```python
# Get predictions for a ticker
predictions = prediction_logger.get_predictions(ticker="AAPL", timeframe="1d")

# Get model statistics
stats = model_logger.get_model_statistics(model_type="unified")

# Get sentiment history
sentiment_history = market_logger.get_sentiment_history(ticker="AAPL")

# Get error summary
error_summary = system_logger.get_error_summary(component="data_fetcher")
```

## Future Enhancements

Potential future enhancements:
- Real-time log streaming
- Log aggregation and summarization
- Automated log analysis reports
- Integration with visualization tools
- Log retention policies
- Performance optimization for large datasets

