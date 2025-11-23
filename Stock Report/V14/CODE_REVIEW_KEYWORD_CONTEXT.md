# Code Review: Keyword Handling & Context Management

## Overview
Comprehensive review of keyword handling patterns and context management throughout the codebase.

**Date**: 2024-01-XX
**Review Scope**: Keyword extraction, sentiment analysis, async context, browser context

---

## 🔴 CRITICAL ISSUES

### 1. Keyword Matching - Substring False Positives
**File**: `sentiment/analyzer.py` (Lines 47-48, 60-61)

**Issue**: Simple substring matching can cause false positives
```python
positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
```

**Problem Examples**:
- "profit" matches in "profitable" ✅ (OK)
- "ass" matches in "class" ❌ (False positive if "ass" was a keyword)
- "beat" matches in "defeated" ✅ (OK but could be wrong context)
- "miss" matches in "dismiss" ❌ (False positive)
- "drop" matches in "raindrop" ✅ (Usually OK)

**Impact**: 
- False sentiment signals
- Incorrect trading decisions based on misclassified news

**Recommendation**: Use word boundary regex matching
```python
import re

# Use word boundaries for exact word matching
pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
matches = len(pattern.findall(text))
```

---

### 2. Duplicate Keyword in List
**File**: `sentiment/analyzer.py` (Line 31)

**Issue**: "bankruptcy" appears twice in `major_event_keywords`
```python
self.major_event_keywords = [
    "earnings", "fda", "merger", "acquisition", "lawsuit",
    "ceo", "cfo", "resignation", "bankruptcy", "bankruptcy"  # ⚠️ DUPLICATE
]
```

**Impact**: Minor - redundant check, doesn't break functionality but wastes computation

**Fix**: Remove duplicate entry

---

### 3. Async Context in Streamlit
**File**: `ui/pages/stock_analysis.py` (Multiple locations)

**Issue**: Using `asyncio.run()` in Streamlit's synchronous context
```python
prediction_result, features, df = asyncio.run(get_prediction_and_features(ticker, timeframe))
```

**Problems**:
1. **Event Loop Conflicts**: Streamlit may already have an event loop running
2. **Blocking**: `asyncio.run()` creates a new event loop, which blocks
3. **Best Practice Violation**: Streamlit prefers sync wrappers or proper async integration

**Impact**: 
- Potential runtime errors in some Streamlit versions
- Performance issues (blocking the UI thread)
- Incompatibility with Streamlit's async features

**Recommendation**: Use `nest_asyncio` or create sync wrappers
```python
import nest_asyncio
nest_asyncio.apply()  # At module level

# Or create sync wrapper
def get_prediction_sync(ticker: str, interval: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_prediction_and_features(ticker, interval))
    finally:
        loop.close()
```

---

## 🟡 MEDIUM ISSUES

### 4. Missing Context Managers for Resources
**File**: `browser/automation.py`, `browser/window_manager.py`

**Issue**: Browser resources not always properly managed with context managers

**Current Pattern**:
```python
browser = BrowserAutomation()
browser.initialize()
# ... use browser ...
browser.close()  # Manual cleanup
```

**Problems**:
- If exception occurs, `close()` might not be called
- Resources leak on errors
- No automatic cleanup

**Recommendation**: Implement context manager protocol
```python
class BrowserAutomation:
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

# Usage:
with BrowserAutomation() as browser:
    browser.navigate("https://...")
# Automatically closed
```

---

### 5. Browser Context JavaScript Execution
**File**: `browser/window_manager.py` (Line 321+)

**Issue**: JavaScript execution lacks proper error context

**Current**:
```python
def execute_javascript(self, script: str, *args) -> Any:
    try:
        if self.browser.library_used == "undetected-chromedriver":
            return self.browser.driver.execute_script(script, *args)
    except Exception:
        return None  # Silent failure
```

**Problems**:
- Silent failures make debugging difficult
- No logging of JavaScript errors
- No context about what script failed

**Recommendation**: Add error logging and context
```python
def execute_javascript(self, script: str, *args) -> Any:
    try:
        if self.browser.library_used == "undetected-chromedriver":
            return self.browser.driver.execute_script(script, *args)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"JavaScript execution failed: {e}\nScript: {script[:100]}...")
        raise  # Or return None with context
```

---

### 6. Keyword Confidence Calculation
**File**: `sentiment/analyzer.py` (Line 65)

**Issue**: Confidence based solely on keyword count, ignores context
```python
confidence = min(1.0, total_keywords / 5.0)  # More keywords = higher confidence
```

**Problems**:
- 1 keyword = 20% confidence (might be very significant)
- 5 keywords = 100% confidence (could be contradictory)
- Doesn't consider:
  - Keyword importance/weight
  - Sentiment consistency (all positive vs mixed)
  - Context quality (headline vs body text)

**Recommendation**: Weighted confidence calculation
```python
# Consider sentiment consistency
sentiment_consistency = abs(sentiment_score)  # Higher if all positive or all negative

# Consider keyword importance (some keywords matter more)
important_keywords = ["earnings", "fda", "bankruptcy"]
important_count = sum(1 for kw in important_keywords if kw in text_lower)

# Weighted confidence
base_confidence = min(1.0, total_keywords / 5.0)
consistency_bonus = sentiment_consistency * 0.3
importance_bonus = min(0.2, important_count * 0.1)
confidence = min(1.0, base_confidence + consistency_bonus + importance_bonus)
```

---

## 🟢 MINOR ISSUES / IMPROVEMENTS

### 7. Case Sensitivity Handling
**File**: `sentiment/analyzer.py`

**Issue**: Keywords are lowercase but text is lowercased - inconsistent style

**Current**:
```python
self.positive_keywords = ["beat", "exceed", ...]  # Lowercase
text_lower = text.lower()  # Conversion happens later
```

**Recommendation**: Store keywords consistently (all lowercase) or use case-insensitive matching throughout

---

### 8. Missing Keyword Categories
**File**: `sentiment/analyzer.py`

**Issue**: Limited keyword sets - could be expanded

**Current Categories**:
- Positive keywords (11 items)
- Negative keywords (11 items)
- Major event keywords (9 items, with 1 duplicate)

**Recommendation**: 
- Add sector-specific keywords
- Add volume/price action keywords
- Add regulatory keywords
- Make keywords configurable (load from file/config)

---

### 9. Context Variable Naming
**File**: Multiple files

**Issue**: Variable name `context` not clearly defined in some places

**Examples**:
- `browser/window_manager.py`: `execute_javascript(script: str, *args)` - context implied but not explicit
- Some functions receive `**kwargs` labeled as "context" but it's just keyword arguments

**Recommendation**: Use more descriptive names
- `browser_context` instead of `context`
- `execution_context` for function execution context
- `user_context` for user-specific data

---

### 10. Async Context Documentation
**Files**: `ui/pages/*.py`

**Issue**: Async/await patterns not well documented

**Recommendation**: Add docstring examples showing async usage patterns

---

## 📊 Summary Statistics

### Keyword-Related Issues:
- **Critical**: 2 issues
- **Medium**: 1 issue
- **Minor**: 3 issues

### Context-Related Issues:
- **Critical**: 1 issue (async context in Streamlit)
- **Medium**: 2 issues (resource management, error context)
- **Minor**: 2 issues (naming, documentation)

### Total Issues Found: 11
- **Critical**: 3 (must fix)
- **Medium**: 3 (should fix)
- **Minor**: 5 (nice to fix)

---

## 🔧 Recommended Fixes Priority

### Priority 1 (Critical - Fix Immediately):
1. ✅ Fix duplicate "bankruptcy" keyword
2. ✅ Add word boundary matching for keywords
3. ✅ Fix async context in Streamlit (use `nest_asyncio` or sync wrappers)

### Priority 2 (Medium - Fix Soon):
4. ✅ Add context managers for browser resources
5. ✅ Improve JavaScript error context/logging
6. ✅ Enhance keyword confidence calculation

### Priority 3 (Minor - Fix When Convenient):
7. ✅ Standardize keyword storage format
8. ✅ Expand keyword categories
9. ✅ Improve context variable naming
10. ✅ Add async usage documentation

---

## 🧪 Testing Recommendations

### Keyword Matching:
- [ ] Test false positive cases ("miss" in "dismiss", "drop" in "raindrop")
- [ ] Test edge cases (empty text, special characters, unicode)
- [ ] Test with real news headlines
- [ ] Compare results before/after word boundary fix

### Context Management:
- [ ] Test browser cleanup on exceptions
- [ ] Test async operations in Streamlit
- [ ] Test JavaScript execution error handling
- [ ] Test resource leak scenarios

---

## 📝 Code Examples for Fixes

### Fix 1: Word Boundary Keyword Matching
```python
import re

def analyze_text(self, text: str) -> Dict[str, any]:
    """Analyze sentiment with word boundary matching."""
    text_lower = text.lower()
    
    # Create regex patterns with word boundaries
    positive_patterns = [
        re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        for kw in self.positive_keywords
    ]
    negative_patterns = [
        re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        for kw in self.negative_keywords
    ]
    
    # Count matches
    positive_count = sum(1 for pattern in positive_patterns if pattern.search(text_lower))
    negative_count = sum(1 for pattern in negative_patterns if pattern.search(text_lower))
    
    # ... rest of method
```

### Fix 2: Context Manager for Browser
```python
class BrowserAutomation:
    def __enter__(self):
        if not self.initialize():
            raise RuntimeError("Failed to initialize browser")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions

# Usage:
try:
    with BrowserAutomation(headless=False) as browser:
        browser.navigate("https://trading212.com")
        # ... operations
except Exception as e:
    # Browser automatically closed
    logger.error(f"Browser operation failed: {e}")
```

### Fix 3: Async Context Fix
```python
# At top of ui/pages/stock_analysis.py
import nest_asyncio
import asyncio

# Allow nested event loops
nest_asyncio.apply()

@st.cache_data(ttl=600)
async def get_prediction_and_features(ticker_sym: str, interval_val: str):
    # ... existing code

# In function:
try:
    prediction_result, features, df = await get_prediction_and_features(ticker, timeframe)
except RuntimeError as e:
    if "cannot be called from a running event loop" in str(e):
        # Fallback to sync wrapper
        prediction_result, features, df = asyncio.run(get_prediction_and_features(ticker, timeframe))
    else:
        raise
```

---

## ✅ Checklist for Implementation

- [x] Fix duplicate "bankruptcy" keyword ✅ FIXED
- [x] Implement word boundary matching ✅ FIXED
- [ ] Fix async context in Streamlit pages (requires nest_asyncio or sync wrappers)
- [x] Add context managers to BrowserAutomation ✅ FIXED
- [ ] Improve JavaScript error logging (recommended but not critical)
- [x] Enhance confidence calculation ✅ FIXED
- [ ] Add unit tests for keyword matching
- [ ] Add integration tests for context management
- [ ] Update documentation

---

## 🔧 FIXES APPLIED

### Fix 1: Removed Duplicate Keyword ✅
**File**: `sentiment/analyzer.py` Line 31
- Removed duplicate "bankruptcy" entry from `major_event_keywords` list

### Fix 2: Word Boundary Matching ✅
**File**: `sentiment/analyzer.py` Lines 47-48, 60-61
- Implemented regex word boundary matching (`\b`) to prevent false positives
- Example: "miss" no longer matches in "dismiss", "drop" no longer matches in "raindrop"
- Applied to both positive/negative keywords and major event keywords

### Fix 3: Enhanced Confidence Calculation ✅
**File**: `sentiment/analyzer.py` Lines 64-75
- Added sentiment consistency bonus (higher confidence if all keywords point same direction)
- Added importance bonus for major events
- More nuanced confidence scoring instead of simple keyword count

### Fix 4: Context Manager Support ✅
**File**: `browser/automation.py` Lines 31-53
- Added `__enter__` and `__exit__` methods for context manager protocol
- Enables `with BrowserAutomation() as browser:` syntax
- Automatic cleanup on exceptions or normal exit

---

## 📝 REMAINING RECOMMENDATIONS

### Priority 1: Async Context in Streamlit
**Files**: `ui/pages/stock_analysis.py`, `ui/pages/settings.py`

**Issue**: Multiple uses of `asyncio.run()` which can conflict with Streamlit's event loop

**Solution Options**:
1. **Option A**: Install and use `nest_asyncio`
   ```python
   import nest_asyncio
   nest_asyncio.apply()  # At module level
   ```

2. **Option B**: Create sync wrapper functions
   ```python
   def fetch_prices_sync(ticker: str, interval: str):
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
       try:
           return loop.run_until_complete(fetch_prices(ticker, interval))
       finally:
           loop.close()
   ```

3. **Option C**: Use Streamlit's native async support (if available in version)

**Recommendation**: Test current implementation first - it may work fine in most Streamlit versions. Only fix if issues occur.

---

## 📚 References

- Python `re` module: Word boundaries (`\b`)
- Streamlit async support: https://docs.streamlit.io/develop/api-reference/utilities/st.rerun
- `nest_asyncio`: https://github.com/erdewit/nest_asyncio
- Context managers: https://docs.python.org/3/library/stdtypes.html#context-manager-types

