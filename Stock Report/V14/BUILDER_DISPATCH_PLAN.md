# Sequential Builder Dispatch Plan

## Overview
This document tracks the sequential commissioning of Builder Agent tasks, with Debug Agent approval gates between each item.

## Workflow
1. **Management Agent** identifies next pending task from `AGENT_COORDINATION.md`
2. **Management Agent** creates Builder brief with scope, files, success criteria
3. **Management Agent** updates `AGENT_COORDINATION.md` marking task as "IN PROGRESS"
4. **Builder Agent** completes task and reports completion
5. **Debug Agent** reviews, tests, and approves (or requests fixes)
6. **Management Agent** marks task as "COMPLETE" and moves to next item
7. Repeat until all tasks are complete

## Pending Tasks Queue

### Task 1: Browser Agent - Trading212 UI Element Selectors
**Status**: PENDING → IN PROGRESS
**Priority**: HIGH
**Brief**:
- Implement actual Trading212 UI element finding in `browser/trade_executor.py`
- Replace TODO placeholders in `login()` method with actual element selectors
- Implement element finding for login page (username field, password field, login button)
- Use robust selectors (CSS, XPath, ID) with fallback options
- Test with real Trading212 login page structure
- Handle dynamic content and page load delays

**Files to Modify**:
- `browser/trade_executor.py` - Replace login() TODO with actual implementation
- `browser/automation.py` - Add helper methods for element finding if needed

**Success Criteria**:
- Login function can locate and interact with all required elements
- Handles page load timing correctly
- Works with both undetected-chromedriver and Playwright backends
- Debug Agent approves after testing

---

### Task 2: Browser Agent - Test Login Flow
**Status**: PENDING
**Priority**: HIGH
**Dependencies**: Task 1 must be complete
**Brief**:
- Create comprehensive login flow tests
- Test successful login scenarios
- Test failure scenarios (wrong credentials, network errors)
- Test session persistence
- Verify error handling and recovery

**Success Criteria**:
- All login flow tests pass
- Debug Agent approves test coverage and implementation

---

### Task 3: Browser Agent - Trading212 Trade Execution Element Selectors
**Status**: PENDING
**Priority**: HIGH
**Dependencies**: Task 1, Task 2 must be complete
**Brief**:
- Implement actual Trading212 trade execution element finding
- Replace TODO placeholders in `open_trade()` method
- Implement selectors for: ticker search, buy/sell buttons, position size input, stop-loss input, take-profit input, confirm button
- Handle dynamic order form and validation
- Test with Trading212 CFD interface

**Files to Modify**:
- `browser/trade_executor.py` - Replace open_trade() TODO with actual implementation

**Success Criteria**:
- Can locate all trade execution elements
- Handles form validation and dynamic content
- Debug Agent approves after testing

---

### Task 4: Browser Agent - Test Trade Execution Flow
**Status**: PENDING
**Priority**: HIGH
**Dependencies**: Task 3 must be complete
**Brief**:
- Create comprehensive trade execution tests
- Test successful trade opening (buy/sell)
- Test with/without stop-loss and take-profit
- Test position size validation
- Test error scenarios (insufficient funds, invalid ticker, etc.)
- Use demo account for safety

**Success Criteria**:
- All trade execution tests pass
- Debug Agent approves test coverage

---

### Task 5: Browser Agent - Test Error Recovery
**Status**: PENDING
**Priority**: MEDIUM
**Dependencies**: Task 4 must be complete
**Brief**:
- Test error recovery mechanisms in `browser/error_handler.py`
- Test element not found recovery
- Test session timeout recovery
- Test network error recovery
- Test retry logic with exponential backoff
- Verify error logging works correctly

**Success Criteria**:
- Error recovery tests pass
- Debug Agent approves robustness

---

### Task 6: Sentiment Agent - Real News API Integration
**Status**: PENDING
**Priority**: MEDIUM
**Brief**:
- Integrate real news APIs (Yahoo Finance, Alpha Vantage, or similar)
- Replace placeholder news monitoring in `sentiment/news_monitor.py`
- Implement news feed fetching with rate limiting
- Add economic calendar integration
- Handle API failures gracefully

**Files to Modify**:
- `sentiment/news_monitor.py` - Implement real API integration

**Success Criteria**:
- Can fetch real news data
- Handles API rate limits
- Debug Agent approves implementation

---

### Task 7: Sentiment Agent - Enhanced Sentiment Analysis
**Status**: PENDING
**Priority**: MEDIUM
**Dependencies**: Task 6 must be complete
**Brief**:
- Enhance sentiment analysis in `sentiment/analyzer.py`
- Optionally integrate finBERT or similar financial NLP model
- Improve keyword-based analysis accuracy
- Add sentiment scoring for major events
- Test sentiment accuracy

**Files to Modify**:
- `sentiment/analyzer.py` - Enhance sentiment analysis

**Success Criteria**:
- Sentiment analysis is more accurate
- Debug Agent approves improvements

---

### Task 8: Sentiment Agent - Real-Time News Feed Integration
**Status**: PENDING
**Priority**: LOW
**Dependencies**: Task 6, Task 7 must be complete
**Brief**:
- Implement real-time news feed monitoring
- Add polling mechanism for continuous news updates
- Integrate with sentiment override system
- Test real-time blocking of trades on major events

**Success Criteria**:
- Real-time news monitoring works
- Debug Agent approves integration

---

### Task 9: Integration Agent - Copy/Extend V13 Trading Modules
**Status**: PENDING
**Priority**: MEDIUM
**Brief**:
- Copy trading modules from V13 (if they exist)
- Extend V13 simulator with V14 risk management features
- Integrate with V14 browser automation
- Ensure compatibility with V14 architecture

**Files to Create/Modify**:
- `trading/` modules as needed
- `trading/simulator_v14.py` if not complete

**Success Criteria**:
- V13 trading features work in V14
- Debug Agent approves integration

---

### Task 10: Integration Agent - Copy/Extend V13 UI Modules
**Status**: PENDING
**Priority**: MEDIUM
**Dependencies**: Task 9 should be complete
**Brief**:
- Copy V13 UI modules (graphs, etc.) to V14
- Extend UI to support V14 features
- Integrate with Streamlit UI (if applicable)
- Ensure all V13 UI features available in V14

**Files to Create/Modify**:
- `ui/` modules as needed

**Success Criteria**:
- All V13 UI features available in V14
- Debug Agent approves

---

### Task 11: Integration Agent - Complete Menu System
**Status**: PENDING
**Priority**: HIGH
**Dependencies**: All previous tasks complete
**Brief**:
- Complete menu system in `ui/menu_v14.py`
- Add all V14-specific menu options
- Ensure V13 compatibility
- Test menu navigation and all options

**Files to Modify**:
- `ui/menu_v14.py` - Complete menu implementation

**Success Criteria**:
- Complete menu system works
- All options accessible
- Debug Agent approves

---

### Task 12: Integration Agent - Integration Testing
**Status**: PENDING
**Priority**: HIGH
**Dependencies**: All tasks complete
**Brief**:
- Create comprehensive integration tests
- Test full workflow end-to-end
- Test module interactions
- Verify no breaking changes

**Success Criteria**:
- Integration tests pass
- Debug Agent approves overall system

---

## Current Status

**Active Task**: None (awaiting first commission)
**Last Completed**: None
**Next Task**: Task 1 - Trading212 UI Element Selectors

## Notes
- Each task must wait for Debug Agent approval before proceeding
- Builder Agent should update `AGENT_COORDINATION.md` when starting/completing tasks
- Debug Agent should run tests and verify before approval
- Management Agent coordinates the sequential flow

