# Debugging Reference - Error Patterns & Fixes

**Purpose:** This document serves as a comprehensive reference for debugging agents. It lists all errors encountered, their root causes, and proven fixes. Use this to quickly identify and resolve recurring issues without cycling through the same debugging steps.

**Last Updated:** 2025-01-23  
**Status:** Active Reference Document

---

## Table of Contents

1. [Error Categories](#error-categories)
2. [Common Error Patterns](#common-error-patterns)
3. [Proven Fixes](#proven-fixes)
4. [Self-Rectifying Patterns](#self-rectifying-patterns)
5. [Testing Checklist](#testing-checklist)
6. [Quick Reference](#quick-reference)

---

## Error Categories

### Category 1: Type Hint Evaluation Errors (HIGH PRIORITY)

**Error Pattern:**
```
AttributeError: 'NoneType' object has no attribute 'DataFrame'
AttributeError: 'NoneType' object has no attribute 'Series'
```

**Root Cause:**
- Optional dependencies (e.g., pandas) are set to `None` when not installed
- Type hints evaluate at runtime, causing AttributeError when accessing attributes on `None`
- Example: `pd.DataFrame` in type hint when `pd = None`

**Affected Files:**
- `core/data_fetcher.py` (all functions with pandas type hints)
- Any module using optional dependencies in type hints

**Fix Pattern:**
```python
# ❌ WRONG - Causes AttributeError when pd is None
import pandas as pd  # pd could be None
async def fetch_data(...) -> Optional[pd.DataFrame]:  # Fails!

# ✅ CORRECT - Use TYPE_CHECKING
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd  # Only imported for type checking
else:
    try:
        import pandas as pd
        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False
        pd = None

async def fetch_data(...) -> Optional['pd.DataFrame']:  # String annotation
```

**Self-Rectifying:**
- Type hints no longer evaluated at runtime
- Runtime checks provide clear error messages
- Example: `raise ImportError("pandas is required. Install with: pip install pandas")`

---

### Category 2: API Method Mismatches (MEDIUM PRIORITY)

**Error Pattern:**
```
AttributeError: 'SentimentOverride' object has no attribute 'check_sentiment'
AttributeError: 'SentimentAnalyzer' object has no attribute 'analyze'
```

**Root Cause:**
- Test code or external code calls methods that don't exist
- Method names changed or refactored
- Backward compatibility not maintained

**Affected Files:**
- `sentiment/override.py` - Missing `check_sentiment()` method
- `sentiment/analyzer.py` - Missing `analyze()` alias

**Fix Pattern:**
```python
# ✅ Add wrapper/alias methods for backward compatibility

# Example 1: SentimentOverride.check_sentiment()
def check_sentiment(self, ticker: str) -> Dict[str, Any]:
    """
    Wrapper for should_block_trade() for backward compatibility.
    
    Returns:
        Dictionary with 'blocked' (bool) and 'reason' (str) keys
    """
    should_block, reason = self.should_block_trade(ticker)
    return {"blocked": should_block, "reason": reason}

# Example 2: SentimentAnalyzer.analyze()
def analyze(self, text: str) -> Dict[str, any]:
    """
    Alias for analyze_text() for backward compatibility.
    """
    return self.analyze_text(text)
```

**Self-Rectifying:**
- Methods always available
- Errors propagate naturally from underlying methods
- No silent failures

---

### Category 3: Relative Import Failures (MEDIUM PRIORITY)

**Error Pattern:**
```
ImportError: attempted relative import beyond top-level package
ImportError: No module named 'learning.continuous_service'
```

**Root Cause:**
- Modules use relative imports (`.module`)
- When run from test context or direct execution, relative imports fail
- No fallback to absolute imports

**Affected Files:**
- `learning/__init__.py` (all imports)
- Any `__init__.py` with relative imports
- Modules imported from test scripts

**Fix Pattern:**
```python
# ❌ WRONG - Fails in test context
from .continuous_service import get_continuous_learning_service

# ✅ CORRECT - Try relative, fallback to absolute
try:
    from .continuous_service import get_continuous_learning_service, ContinuousLearningService
except ImportError:
    # Fallback for direct execution or test context
    try:
        from learning.continuous_service import get_continuous_learning_service, ContinuousLearningService
    except ImportError:
        # If still fails, set to None to prevent cascading errors
        get_continuous_learning_service = None
        ContinuousLearningService = None
```

**Self-Rectifying:**
- First tries relative import (normal package usage)
- Falls back to absolute import (test context)
- Sets to None if both fail (prevents cascading errors)
- All imports follow this pattern

---

### Category 4: Missing Dependencies (HIGH PRIORITY)

**Error Pattern:**
```
ModuleNotFoundError: No module named 'pandas'
ModuleNotFoundError: No module named 'aiohttp'
TypeError: 'NoneType' object is not callable
```

**Root Cause:**
- Required dependencies not installed
- Code tries to use modules that are `None`
- No runtime checks before usage

**Affected Files:**
- All modules using optional dependencies
- Functions that don't check availability before use

**Fix Pattern:**
```python
# ✅ Always check availability before use
async def fetch_prices(...):
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp is required for fetching price data. Install with: pip install aiohttp")
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for fetching price data. Install with: pip install pandas")
    
    # Now safe to use
    df = pd.DataFrame(...)
```

**Self-Rectifying:**
- Clear error messages with installation instructions
- Fails fast with helpful guidance
- No silent failures or confusing errors

---

### Category 5: Module Shadowing (MEDIUM PRIORITY)

**Error Pattern:**
```
AttributeError: module 'logging' has no attribute 'getLogger'
AttributeError: module 'asyncio' has no attribute 'run'
```

**Root Cause:**
- Local directory named `logging/` or `asyncio/` shadows standard library
- Python imports local module instead of standard library
- Common in test contexts when script directory is in `sys.path`

**Affected Files:**
- Any script that adds current directory to `sys.path`
- Test scripts that run from project root

**Fix Pattern:**
```python
# ✅ Remove script directory from sys.path before importing standard library
import sys
from pathlib import Path

# Save current directory
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

# Now import standard library
import logging
import asyncio

# Re-add if needed for project imports
sys.path.insert(0, str(SCRIPT_DIR.parent))
```

**Self-Rectifying:**
- Scripts handle path manipulation automatically
- Standard library imports work correctly
- Project imports still work

---

### Category 6: Unicode Encoding Errors (LOW PRIORITY)

**Error Pattern:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 0
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
```

**Root Cause:**
- Print statements use emojis or special Unicode characters
- Windows console uses 'charmap' encoding by default
- Characters not in charmap encoding

**Affected Files:**
- Test scripts with emoji output
- Any script printing to console on Windows

**Fix Pattern:**
```python
# ❌ WRONG - Fails on Windows
print("✅ Test passed")
print("→ Next step")

# ✅ CORRECT - Use ASCII-compatible text
print("[OK] Test passed")
print("-> Next step")
```

**Self-Rectifying:**
- All output uses ASCII-compatible characters
- Works across all platforms
- No encoding issues

---

## Common Error Patterns

### Pattern 1: Type Hint Evaluation on None

**Symptoms:**
- `AttributeError: 'NoneType' object has no attribute 'X'`
- Error occurs during import, not runtime
- Happens even when code path doesn't execute

**Check:**
- Look for type hints using optional dependencies
- Check if dependency is set to `None` when not available

**Fix:**
- Use `TYPE_CHECKING` for conditional imports
- Use string annotations for type hints (`'pd.DataFrame'`)

---

### Pattern 2: Method Not Found

**Symptoms:**
- `AttributeError: 'Class' object has no attribute 'method'`
- Test code calls method that doesn't exist
- Method name changed or refactored

**Check:**
- Compare test code with actual class methods
- Check for method name changes in refactoring

**Fix:**
- Add wrapper/alias methods for backward compatibility
- Update method names to match expected API

---

### Pattern 3: Import Context Issues

**Symptoms:**
- `ImportError: attempted relative import beyond top-level package`
- Works in main app but fails in tests
- Works when run as module but fails when run directly

**Check:**
- Look for relative imports (`.module`)
- Check if script is run from different context

**Fix:**
- Add try/except with fallback to absolute imports
- Set to None if both fail (prevents cascading)

---

### Pattern 4: Missing Dependency Checks

**Symptoms:**
- `TypeError: 'NoneType' object is not callable`
- `ModuleNotFoundError: No module named 'X'`
- Code tries to use module that's `None`

**Check:**
- Look for usage of optional dependencies
- Check if availability is checked before use

**Fix:**
- Add runtime checks before usage
- Raise clear ImportError with installation instructions

---

## Proven Fixes

### Fix 1: Type Hints with Optional Dependencies

**File:** `core/data_fetcher.py`

**Before:**
```python
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

async def fetch_prices(...) -> Optional[pd.DataFrame]:  # AttributeError!
```

**After:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
else:
    try:
        import pandas as pd
        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False
        pd = None

async def fetch_prices(...) -> Optional['pd.DataFrame']:  # String annotation
```

**Status:** ✅ Fixed and verified

---

### Fix 2: API Method Compatibility

**File:** `sentiment/override.py`

**Added:**
```python
def check_sentiment(self, ticker: str) -> Dict[str, Any]:
    """Wrapper for should_block_trade() for backward compatibility."""
    should_block, reason = self.should_block_trade(ticker)
    return {"blocked": should_block, "reason": reason}
```

**File:** `sentiment/analyzer.py`

**Added:**
```python
def analyze(self, text: str) -> Dict[str, any]:
    """Alias for analyze_text() for backward compatibility."""
    return self.analyze_text(text)
```

**Status:** ✅ Fixed and verified

---

### Fix 3: Relative Import Fallbacks

**File:** `learning/__init__.py`

**Before:**
```python
from .continuous_service import get_continuous_learning_service
```

**After:**
```python
try:
    from .continuous_service import get_continuous_learning_service, ContinuousLearningService
except ImportError:
    try:
        from learning.continuous_service import get_continuous_learning_service, ContinuousLearningService
    except ImportError:
        get_continuous_learning_service = None
        ContinuousLearningService = None
```

**Status:** ✅ Fixed and verified

---

## Self-Rectifying Patterns

### Pattern 1: Graceful Import Handling

```python
# Try relative → try absolute → set None
try:
    from .module import Class
except ImportError:
    try:
        from package.module import Class
    except ImportError:
        Class = None  # Prevents cascading errors
```

**Benefits:**
- Works in all contexts (package, test, direct execution)
- Prevents cascading import errors
- Clear failure mode (None instead of crash)

---

### Pattern 2: Runtime Dependency Checks

```python
if not DEPENDENCY_AVAILABLE:
    raise ImportError("dependency is required. Install with: pip install dependency")
```

**Benefits:**
- Fails fast with clear error message
- Provides installation instructions
- No silent failures

---

### Pattern 3: Backward Compatibility Wrappers

```python
def new_method(self, ...):
    """New method name."""
    return self.old_method(...)

def old_method(self, ...):
    """Deprecated: Use new_method()."""
    return self.new_method(...)
```

**Benefits:**
- Both old and new names work
- Easy migration path
- No breaking changes

---

## Testing Checklist

Before marking code as "fixed", verify:

- [ ] **Type hints:** No AttributeError on import (even when dependencies missing)
- [ ] **API methods:** All expected methods exist and are callable
- [ ] **Imports:** Work in package context, test context, and direct execution
- [ ] **Dependencies:** Clear error messages when missing
- [ ] **Unicode:** No encoding errors on Windows console
- [ ] **Module shadowing:** Standard library imports work correctly
- [ ] **Self-rectifying:** Errors provide clear guidance for resolution

---

## Quick Reference

### Type Hints with Optional Dependencies

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optional_module as om
else:
    try:
        import optional_module as om
        MODULE_AVAILABLE = True
    except ImportError:
        MODULE_AVAILABLE = False
        om = None

def function(...) -> Optional['om.Type']:  # String annotation
    if not MODULE_AVAILABLE:
        raise ImportError("optional_module required. Install with: pip install optional_module")
    # Use om here
```

### Relative Import Fallbacks

```python
try:
    from .module import Class
except ImportError:
    try:
        from package.module import Class
    except ImportError:
        Class = None
```

### API Compatibility Wrappers

```python
def new_name(self, ...):
    """New method name."""
    return self.old_name(...)
```

### Runtime Dependency Checks

```python
if not DEPENDENCY_AVAILABLE:
    raise ImportError("dependency required. Install with: pip install dependency")
```

---

## Integration with Test Suite Reports

This debugging reference complements the following test suite documents:

1. **TEST_FAILURES_HANDOFF_REPORT.md** - Detailed failure analysis
2. **QUICK_FIX_SUMMARY.md** - Quick reference for common fixes
3. **FIXES_IMPLEMENTED.md** - Record of all fixes applied

**Usage:**
- When encountering an error, check this document first
- If error matches a pattern, apply the proven fix
- If new error, document it here for future reference
- Update test suite reports with new findings

**For Debugging Agents:**
- Copy relevant error pattern section to debugging agent
- Include code examples from "Proven Fixes" section
- Reference "Self-Rectifying Patterns" for implementation guidance
- Use "Quick Reference" for copy-paste code snippets

---

## Maintenance

**When to Update:**
- New error pattern discovered
- New fix proven effective
- Error pattern resolved permanently
- Test suite reports new categories

**Update Process:**
1. Document error pattern with symptoms
2. Identify root cause
3. Provide proven fix with code examples
4. Mark as self-rectifying if applicable
5. Update quick reference section

---

**End of Debugging Reference**

