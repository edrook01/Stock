# Stock Analyzer V15 - User Guide

## Overview

Stock Analyzer V15 is a comprehensive trading system integrating long-term investment analysis and short-term CFD trading. It combines a unified ML model with robust risk management, automated browser-based trade execution, and adaptive learning.

**Key Features:**
- **Unified ML Model**: Uses V13 engine outputs as features for ensemble predictions
- **Volatility-Based Risk Management**: ATR-based stop-losses and trailing stops
- **Browser Automation**: Automated Trading212 CFD trading with human-like behavior
- **Adaptive Learning**: Continuous improvement from trade outcomes
- **Sentiment Override**: News monitoring and sentiment-based trade blocking
- **Comprehensive Logging**: Detailed trade logs and performance analysis

## Installation & Dependencies

### System Requirements
- Python 3.8 or higher
- Windows, Linux, or macOS
- Google Chrome (for browser automation)
- Internet connection for data fetching

### Required Python Packages

```bash
pip install pandas numpy aiohttp scikit-learn xgboost
```

**Optional (for browser automation):**
```bash
pip install undetected-chromedriver  # Primary
# OR
pip install playwright  # Fallback
```

**Optional (for GPU acceleration):**
```bash
pip install cupy  # Requires CUDA-compatible GPU
```

### Quick Install

```bash
pip install pandas numpy aiohttp scikit-learn xgboost undetected-chromedriver
```

## Configuration

### First Run Setup

On first run, V15 will:
1. Create directory structure
2. Initialize default configuration (`data/config_v15.json`)
3. Prompt for Trading212 credentials (optional)

### Configuration File

Edit `data/config_v15.json` to customize:
- Risk profile (low, medium, high)
- Model parameters
- Browser automation settings
- Sentiment override thresholds
- Timeframe settings

## Usage

### Running V15

```bash
python "Stock Analyzer V15.py"
```

### Main Menu

1. **Core Analysis** - Analyze tickers, generate predictions
2. **Learning & Training** - Train models, view performance
3. **Data & Logs** - View trade logs, export data
4. **System & Maintenance** - Configuration, maintenance
5. **V15 Features** - V15-specific features:
   - 5A. Unified Model - Generate Prediction
   - 5B. Risk Profile Selection
   - 5C. Browser Automation Status
   - 5D. Sentiment Override Settings
   - 5E. Trade Log Analysis
   - 5F. Performance Report

### Risk Profiles

- **Low**: 0.5-1% equity risk, stable assets only, tight stops
- **Medium**: 1% equity risk, moderate assets, balanced approach
- **High**: 1-2% equity risk, all assets, wider stops

### Timeframes

**CFD Timeframes**: 1m, 5m, 10m, 15m, 1h
**Investment Timeframes**: 1d, 1w

## Portability

V15 is fully portable. Simply copy the entire `V15/` folder to a new machine and run. All data, models, and logs are stored within the V15 directory using relative paths.

**Portable Components:**
- Model weights: `model/weights/`
- Trade logs: `history/`
- Configuration: `data/config_v15.json`
- Learning history: `memory/`

## Browser Automation

### Setup

1. Install browser automation library:
   ```bash
   pip install undetected-chromedriver
   ```

2. Configure Trading212 credentials in `data/config_v15.json` or via menu

3. Initialize browser automation from menu (5C)

### Features

- Human-like behavior (randomized timing, mouse movements)
- Automatic login
- Trade execution (open/close)
- Account status monitoring
- Error handling and recovery

## Risk Management

### Position Sizing

Position size is calculated based on:
- Account equity
- Risk percentage (0.5-2% based on profile)
- Stop-loss distance (ATR-based)

### Exposure Limits

- Maximum 2% equity risk per trade
- Maximum 10% combined exposure across all positions

### Stop-Loss Management

- ATR-based stop distances
- Trailing stops that lock in profits
- Never move stops backward

## Adaptive Learning

V15 learns from trade outcomes:
- Tracks all trade results
- Adjusts confidence calibration
- Updates model weights
- Identifies failed patterns

## Sentiment Override

Monitors news and sentiment to:
- Block trades during major events (earnings, FDA decisions)
- Adjust confidence based on sentiment
- Enter protective mode during market volatility

## Logging

All trades are logged to:
- `history/trades.csv` - CSV format
- `history/trades.json` - JSON format

Logs include:
- Entry/exit details
- P/L and performance
- Confidence levels
- Risk metrics
- Prediction accuracy

## Troubleshooting

### Browser Automation Issues

- Ensure Chrome is installed
- Check Trading212 credentials
- Verify undetected-chromedriver is installed
- Try Playwright as fallback

### Model Not Trained

- Models need training data (historical trades)
- Use Learning & Training menu to train models
- Minimum 50 trades recommended per timeframe

### Import Errors

- Ensure all dependencies are installed
- Check Python version (3.8+)
- Verify V15 directory structure is intact

## Performance

### Optimization Features

- Async data fetching
- Cached predictions
- Parallel model execution
- GPU acceleration (optional)

### Benchmarks

- Data fetching: 5-10x faster than V13 (async)
- Prediction generation: Parallel execution
- Risk calculations: Real-time ATR updates

## Safety Features

- **Simulation Mode**: Test strategies without real money
- **Risk Limits**: Hard limits on position size and exposure
- **Sentiment Override**: Blocks trades during risky events
- **Failure Tracking**: Monitors and learns from failures
- **Comprehensive Logging**: Full audit trail

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review `IMPLEMENTATION_STATUS.md` for feature status
3. Check `TEST_STATEMENTS.md` for test coverage

## Version History

See `PATCHNOTES.md` for detailed version history and changes from V13.

