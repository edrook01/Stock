# Debug Agent Task Queue

## Current Assignment: Agent 8 - Debug Agent

### Tasks to Complete

#### Priority 1: Core Debug Utilities
- [x] `debug/prediction_debugger.py` - Debug predictions ✅
- [ ] `debug/risk_debugger.py` - Debug risk calculations ⏳ NEXT
- [ ] `debug/browser_debugger.py` - Debug browser automation
- [ ] `debug/learning_debugger.py` - Debug learning system
- [ ] `debug/integration_debugger.py` - Debug integration

#### Priority 2: Specialized Debuggers
- [ ] `debug/sentiment_debugger.py` - Debug sentiment system
- [ ] `debug/model_debugger.py` - Debug ML model

#### Priority 3: Debug Tools
- [ ] `debug/debug_runner.py` - Run all debuggers
- [ ] `debug/debug_report.py` - Generate debug reports
- [ ] `debug/performance_profiler.py` - Performance profiling

## Debug Methods to Implement

### Risk Debugger Methods:
- `debug_atr_calculation()` - Step-by-step ATR calculation
- `debug_stop_loss()` - Stop-loss calculation breakdown
- `debug_position_sizing()` - Position size calculation details
- `debug_exposure()` - Exposure tracking analysis
- `debug_risk_profile()` - Risk profile validation
- `debug_trailing_stop()` - Trailing stop behavior

### Browser Debugger Methods:
- `debug_browser_init()` - Browser initialization steps
- `debug_element_finding()` - Test element selectors
- `debug_human_behavior()` - Test human-like actions
- `debug_trade_flow()` - Complete trade execution flow
- `debug_error_recovery()` - Error handling tests
- `debug_login_flow()` - Login process debugging

### Learning Debugger Methods:
- `debug_trade_tracking()` - Trade outcome tracking
- `debug_feedback_loop()` - Feedback loop analysis
- `debug_model_updates()` - Model update process
- `debug_prediction_monitoring()` - Prediction monitoring
- `debug_failure_analysis()` - Failure tracking
- `debug_diagnostic()` - Diagnostic analysis

### Integration Debugger Methods:
- `debug_imports()` - Test all module imports
- `debug_data_flow()` - Trace data through system
- `debug_config_loading()` - Configuration loading
- `debug_portability()` - Portability checks
- `debug_module_communication()` - Inter-module communication
- `debug_menu_flow()` - Menu navigation testing

### Sentiment Debugger Methods:
- `debug_news_monitoring()` - News feed monitoring
- `debug_sentiment_analysis()` - Sentiment scoring
- `debug_override_logic()` - Override decision making
- `debug_event_detection()` - Event detection

### Model Debugger Methods:
- `debug_feature_extraction()` - Feature extraction process
- `debug_model_prediction()` - Model prediction breakdown
- `debug_confidence_calibration()` - Confidence calibration
- `debug_model_training()` - Training process

## Test Scenarios

### Risk Debugger Test Scenarios:
1. ATR calculation with insufficient data
2. Stop-loss with extreme volatility
3. Position sizing with zero equity
4. Exposure tracking with multiple positions
5. Risk profile switching
6. Trailing stop edge cases

### Browser Debugger Test Scenarios:
1. Browser initialization failure
2. Element not found errors
3. Session timeout recovery
4. Trade execution errors
5. Human behavior validation
6. Login failure recovery

### Learning Debugger Test Scenarios:
1. Trade tracking with missing data
2. Feedback loop with no outcomes
3. Model update with insufficient data
4. Prediction monitoring edge cases
5. Failure tracking threshold breaches
6. Diagnostic with incomplete data

## Expected Output Format

Each debugger should output:
```json
{
  "debugger": "risk_debugger",
  "test": "debug_atr_calculation",
  "timestamp": "2024-01-01T12:00:00",
  "input": {...},
  "steps": [
    {"step": 1, "action": "...", "result": "...", "duration_ms": 10},
    {"step": 2, "action": "...", "result": "...", "duration_ms": 5}
  ],
  "output": {...},
  "errors": [],
  "warnings": [],
  "performance": {
    "total_duration_ms": 15,
    "slowest_step": 1
  },
  "success": true
}
```

## Review Checklist

Before marking debugger complete:
- [ ] All methods implemented
- [ ] Error handling present
- [ ] Output format consistent
- [ ] Tests written
- [ ] Documentation complete
- [ ] No hardcoded values
- [ ] Uses relative paths
- [ ] Performance metrics included

