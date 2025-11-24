# Multi-Agent Development Coordination Guide

## Agent Setup Instructions

To use multiple agents for V15 development, you need to set up separate AI assistant instances. Here's how:

### Option 1: Multiple Cursor Sessions
1. Open multiple Cursor windows/tabs
2. Each window = one agent
3. Assign each agent a specific role (see below)
4. Use this coordination file to track work

### Option 2: Different AI Assistants
1. Use different AI assistants (Claude, GPT, etc.) in separate sessions
2. Share this coordination file with all agents
3. Each agent works on assigned modules

### Option 3: Sequential Agent Handoff
1. Complete work as one agent
2. Hand off to next agent role
3. Use task queue below

## Agent Role Definitions

### Agent 1: Model Agent
**Responsibilities:**
- `model/unified_model.py` - ML model architecture
- `model/feature_extractor.py` - Feature engineering
- `model/trainer.py` - Training pipeline
- `model/confidence_calibrator.py` - Confidence calibration
- `model/timeframe_predictor.py` - Timeframe-specific predictions

**Status**: ✅ COMPLETE
**Next Tasks**: Debug, optimize, enhance

### Agent 2: Browser Agent
**Responsibilities:**
- `browser/automation.py` - Browser setup
- `browser/human_behavior.py` - Human-like behavior
- `browser/trade_executor.py` - Trade execution
- `browser/error_handler.py` - Error handling
- Browser automation testing

**Status**: ✅ COMPLETE (structure done, needs Trading212 element selectors)
**Next Tasks**: Implement actual Trading212 UI element finding and interaction

### Agent 3: Risk Agent
**Responsibilities:**
- `risk/volatility.py` - ATR calculation
- `risk/stop_loss.py` - Stop-loss calculation
- `risk/trailing_stop.py` - Trailing stops
- `risk/profiles.py` - Risk profiles
- `risk/position_sizing.py` - Position sizing
- `risk/exposure_tracker.py` - Exposure tracking
- `risk/equity_monitor.py` - Equity monitoring

**Status**: ✅ COMPLETE
**Next Tasks**: Testing, edge case handling

### Agent 4: Learning Agent
**Responsibilities:**
- `learning/trade_tracker.py` - Trade tracking
- `learning/feedback_loop.py` - Feedback loops
- `learning/model_updater.py` - Model updates
- `learning/prediction_monitor.py` - Prediction monitoring
- `learning/failure_tracker.py` - Failure tracking
- `learning/diagnostic.py` - Diagnostics

**Status**: ✅ COMPLETE
**Next Tasks**: Testing, performance optimization

### Agent 5: Sentiment Agent
**Responsibilities:**
- `sentiment/news_monitor.py` - News monitoring
- `sentiment/analyzer.py` - Sentiment analysis
- `sentiment/override.py` - Override logic
- News API integration
- Economic calendar updates

**Status**: ✅ COMPLETE (basic structure)
**Next Tasks**: Real news API integration, enhanced sentiment analysis

### Agent 6: Integration Agent
**Responsibilities:**
- `Stock Analyzer V15.py` - Main entry point
- `ui/menu_v15.py` - Menu system
- Module integration
- Configuration management
- Portability verification

**Status**: ✅ COMPLETE
**Next Tasks**: Full V13 module integration, complete menu system

### Agent 7: Testing Agent
**Responsibilities:**
- `test_v15.py` - Test suite
- Integration tests
- Performance tests
- Edge case tests
- Debug utilities

**Status**: ✅ COMPLETE (basic tests)
**Next Tasks**: Comprehensive test coverage, debug tools

### Agent 8: Debug Agent (NEW)
**Responsibilities:**
- `debug/prediction_debugger.py` - Prediction debugging
- `debug/risk_debugger.py` - Risk calculation debugging
- `debug/browser_debugger.py` - Browser automation debugging
- `debug/learning_debugger.py` - Learning system debugging
- `debug/integration_debugger.py` - Integration debugging
- `debug/sentiment_debugger.py` - Sentiment system debugging
- `debug/model_debugger.py` - ML model debugging

**Status**: ✅ COMPLETE
**Completed Tasks**: All debug utilities created
**Files Created**: 8 debug modules + coordination files

## Task Queue System

### Current Task Assignments

**Agent 8 (Debug Agent) - ✅ COMPLETE:**
- [x] Create `debug/prediction_debugger.py` ✅
- [x] Create `debug/risk_debugger.py` ✅
- [x] Create `debug/browser_debugger.py` ✅
- [x] Create `debug/learning_debugger.py` ✅
- [x] Create `debug/integration_debugger.py` ✅
- [x] Create `debug/sentiment_debugger.py` ✅
- [x] Create `debug/model_debugger.py` ✅
- [x] Create `debug/debug_runner.py` - Run all debuggers ✅

**Agent 2 (Browser Agent) - IN PROGRESS:**
- [x] Implement Trading212 UI element selectors ✅ COMPLETE & APPROVED (See DEBUG_APPROVAL_TASK_01.md)
- [x] Test login flow ⏳ IN PROGRESS
- [ ] Test trade execution flow
- [ ] Test error recovery

**Agent 5 (Sentiment Agent) - PENDING:**
- [ ] Integrate real news API (Yahoo Finance, Alpha Vantage)
- [ ] Enhance sentiment analysis (finBERT option)
- [ ] Real-time news feed integration

**Agent 6 (Integration Agent) - PENDING:**
- [ ] Copy/extend V13 trading modules
- [ ] Copy/extend V13 UI modules (graphs, etc.)
- [ ] Complete menu system with all V13 features
- [ ] Integration testing

**UI Agent (Current) - ✅ COMPLETE:**
- [x] AUTO_FIX_REQUEST_20251124_132551 (logging import + Function 1/3 suites)

## Communication Protocol

### When Starting Work:
1. Check `AGENT_COORDINATION.md` for your assigned tasks
2. Mark task as "IN PROGRESS" in this file
3. Work on assigned module
4. Mark as "COMPLETE" when done

### When Requesting Review:
1. Create a review request in format:
   ```
   [REVIEW REQUEST]
   Agent: [Your Agent Number]
   Module: [module_name.py]
   Status: Ready for review
   Notes: [Any specific concerns]
   ```

### When Completing Work:
1. Update status in this file
2. Add completion notes
3. Notify next agent if handoff needed

## Module Dependencies

### Dependency Graph:
```
Core Modules (no dependencies)
    ↓
Model Modules (depends on Core)
    ↓
Risk Modules (depends on Core)
    ↓
Learning Modules (depends on Model, Risk, Core)
    ↓
Browser Modules (depends on Risk, Core)
    ↓
Sentiment Modules (depends on Core)
    ↓
Trading Modules (depends on Risk, Learning, Sentiment, Core)
    ↓
UI/Menu (depends on all)
    ↓
Main Entry Point (depends on all)
```

## Code Review Checklist

When reviewing another agent's code:
- [ ] Imports are correct
- [ ] Error handling present
- [ ] Docstrings complete
- [ ] Uses relative paths (portable)
- [ ] Follows V15 architecture patterns
- [ ] No hardcoded values
- [ ] Edge cases handled
- [ ] Performance considerations
- [ ] Security (no exposed credentials)

## Testing Protocol

1. **Unit Tests**: Agent writes unit tests for their module
2. **Integration Tests**: Testing Agent creates integration tests
3. **Review**: Reviewer checks test coverage
4. **Fix**: Builder fixes issues found in tests

## Current Status Summary

- ✅ **Model Agent**: Complete
- ✅ **Risk Agent**: Complete
- ✅ **Learning Agent**: Complete
- ✅ **Sentiment Agent**: Complete (basic)
- ✅ **Browser Agent**: Complete (structure, needs UI selectors)
- ✅ **Integration Agent**: Complete (basic, needs V13 integration)
- ✅ **Testing Agent**: Complete (basic tests)
- ✅ **Debug Agent**: Complete

## Next Multi-Agent Tasks

1. **Debug Agent** → Create all debug utilities
2. **Browser Agent** → Implement Trading212 UI interactions
3. **Sentiment Agent** → Real news API integration
4. **Integration Agent** → Complete V13 module integration
5. **Testing Agent** → Comprehensive test coverage

## How to Use This System

1. **Assign Agents**: Give each AI assistant a number (1-8)
2. **Share This File**: All agents reference this coordination file
3. **Work in Parallel**: Agents can work on different modules simultaneously
4. **Update Status**: Agents update their status in this file
5. **Request Reviews**: Use review request format above
6. **Hand Off**: When complete, notify next agent

## Notes

- All agents should use relative paths
- All agents should follow existing code patterns
- All agents should write tests for their modules
- All agents should update this file when completing work

