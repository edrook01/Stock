# Stock Analyzer V15 - Patch Notes

## Version 15.0 - Initial Release

### Major Features

#### 1. Unified ML Model Architecture
- **New**: Ensemble ML model using V13 engine outputs as features
- **New**: Feature extraction from statistical/technical engines
- **New**: Confidence calibration system
- **New**: Model training infrastructure
- **Files**: `model/unified_model.py`, `model/feature_extractor.py`, `model/trainer.py`, `model/confidence_calibrator.py`

#### 2. Volatility-Based Risk Management
- **New**: ATR (Average True Range) calculation
- **New**: Dynamic stop-loss based on ATR multipliers
- **New**: Trailing stops that lock in profits
- **New**: Risk profile system (Low/Medium/High)
- **Files**: `risk/volatility.py`, `risk/stop_loss.py`, `risk/trailing_stop.py`, `risk/profiles.py`

#### 3. Position Sizing & Equity Management
- **New**: Risk-based position sizing (0.5-2% equity risk)
- **New**: Combined exposure tracking (max 10%)
- **New**: Account equity monitoring with drawdown tracking
- **Files**: `risk/position_sizing.py`, `risk/exposure_tracker.py`, `risk/equity_monitor.py`

#### 4. Browser-Based CFD Trading Automation
- **New**: Trading212 browser automation
- **New**: Human-like behavior simulation
- **New**: Trade execution functions (open/close)
- **New**: Error handling and recovery
- **Files**: `browser/automation.py`, `browser/human_behavior.py`, `browser/trade_executor.py`, `browser/error_handler.py`

#### 5. Adaptive Learning & Feedback Loops
- **New**: Trade outcome tracking
- **New**: Feedback loop for confidence adjustment
- **New**: Continuous model updates
- **New**: Missed prediction monitoring
- **New**: Failure tracking and diagnostics
- **Files**: `learning/trade_tracker.py`, `learning/feedback_loop.py`, `learning/model_updater.py`, `learning/prediction_monitor.py`, `learning/failure_tracker.py`, `learning/diagnostic.py`

#### 6. Sentiment Detection Override Layer
- **New**: News monitoring and economic calendar
- **New**: Sentiment analysis (NLP/keyword-based)
- **New**: Trade blocking based on sentiment/events
- **New**: Confidence adjustment based on sentiment
- **Files**: `sentiment/news_monitor.py`, `sentiment/analyzer.py`, `sentiment/override.py`

#### 7. Comprehensive Trade Logging
- **New**: Detailed trade logging (CSV and JSON)
- **New**: Performance analysis tools
- **New**: Predicted vs actual comparison
- **New**: Pattern identification
- **Files**: `logging/trade_logger.py`, `logging/analyzer.py`

#### 8. Enhanced Simulation Mode
- **Enhanced**: V13 simulator extended with V15 features
- **New**: ATR-based stops in simulation
- **New**: Trailing stops in simulation
- **New**: Risk profile support
- **New**: Sentiment override simulation
- **Files**: `trading/simulator_v15.py`, `trading/performance_evaluator.py`

#### 9. Prediction Timeframes
- **New**: Timeframe configuration system
- **New**: CFD timeframes: 1m, 5m, 10m, 15m, 1h
- **New**: Investment timeframes: 1d, 1w
- **New**: Prediction scheduler
- **Files**: `core/timeframes.py`, `core/prediction_scheduler.py`

#### 10. Configuration & Portability
- **New**: V15-specific configuration
- **New**: First-run setup
- **New**: Portability verification
- **Files**: `data/config_v15.json`, `core/setup.py`, `core/portability_check.py`

### Improvements Over V13

1. **Risk Management**: Advanced ATR-based stops vs simple percentage stops
2. **Learning**: Adaptive learning from outcomes vs static models
3. **Automation**: Browser automation vs manual trading
4. **Sentiment**: News monitoring vs basic sentiment
5. **Logging**: Comprehensive logging vs basic logs
6. **Model**: Unified ML model vs separate engines

### Breaking Changes

- Configuration file changed from `config.json` to `config_v15.json`
- Model weights stored in `model/weights/` (new location)
- Trade logs format enhanced (backward compatible)

### Dependencies Added

- `scikit-learn` - ML models
- `xgboost` - Ensemble model
- `undetected-chromedriver` or `playwright` - Browser automation

### Migration from V13

1. Copy V13 data to V15 if needed
2. Run V15 first-time setup
3. Configure Trading212 credentials (if using browser automation)
4. Train models with historical data
5. Review and adjust risk profile settings

### Known Limitations

- Browser automation requires Trading212 web interface (desktop Chrome)
- Model training requires sufficient historical trade data
- Some features require additional API keys (news sources)

### Future Enhancements

- Mobile browser support
- Additional broker integrations
- Advanced sentiment analysis (finBERT)
- Real-time news feed integration
- Enhanced pattern recognition

