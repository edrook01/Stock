# Builder Agent Brief - Task 1: Trading212 UI Element Selectors

## Task Assignment
**Agent**: Builder Agent (Agent 2)
**Task**: Implement Trading212 UI element selectors for login flow
**Status**: IN PROGRESS
**Priority**: HIGH

## Objective
Replace the TODO placeholder in `browser/trade_executor.py`'s `login()` method with actual Trading212 login page element finding and interaction logic.

## Current State
The `login()` method in `browser/trade_executor.py` (lines 28-64) currently:
- Loads credentials correctly
- Navigates to the login page
- Has placeholder comments describing what needs to be done
- Returns `True` without actually performing login

## Requirements

### 1. Element Finding Methods
Add robust element finding methods to `BrowserAutomation` class in `browser/automation.py`:
- `find_element(selector, by='css', timeout=10)` - Generic element finder with multiple selector strategies
- Support CSS selectors, XPath, ID, class name, etc.
- Implement wait logic for dynamic content (WebDriverWait for Selenium, wait_for_selector for Playwright)
- Return element handle or None

### 2. Login Page Elements
Identify and locate these Trading212 login page elements:
- **Username/Email field**: Find input field for email or username
- **Password field**: Find password input field
- **Login button**: Find the submit/login button
- **Error messages**: Optional - locate error message containers for validation

### 3. Element Interaction
Implement interaction with found elements:
- Type username with human-like delays (use `HumanBehavior` class)
- Type password with human-like delays
- Click login button
- Wait for navigation after login

### 4. Login Verification
Verify successful login:
- Check URL change (should navigate away from login page)
- Check for presence of dashboard/trading platform elements
- Handle cases where login fails (wrong credentials, 2FA required, etc.)

### 5. Error Handling
Implement robust error handling:
- Handle element not found errors
- Handle page load timeouts
- Handle invalid credentials
- Return appropriate boolean/error information

## Implementation Details

### Files to Modify
1. **`browser/automation.py`**
   - Add `find_element()` method(s)
   - Add helper methods for waiting, typing, clicking
   - Support both undetected-chromedriver (Selenium) and Playwright backends

2. **`browser/trade_executor.py`**
   - Replace TODO section (lines 53-61) with actual implementation
   - Use element finding methods from `BrowserAutomation`
   - Integrate with `HumanBehavior` for realistic timing
   - Update `logged_in` flag based on actual success

### Selector Strategy
Use multiple selector fallbacks for robustness:
- Primary: CSS selectors (most stable)
- Fallback: XPath selectors
- Fallback: Element IDs
- Fallback: Partial text matching

Example approach:
```python
# Try multiple selectors
selectors = [
    "input[name='email']",
    "input[type='email']",
    "#email",
    "//input[@placeholder='Email']"
]
for selector in selectors:
    element = self.browser.find_element(selector)
    if element:
        break
```

### Wait Strategy
- Wait for page load complete
- Wait for specific elements to be visible/clickable
- Use explicit waits (not fixed time.sleep where possible)
- Timeout after reasonable duration (10-15 seconds)

## Success Criteria

1. ✅ Can locate all required login elements
2. ✅ Can type credentials with human-like behavior
3. ✅ Can submit login form
4. ✅ Can verify successful login
5. ✅ Handles errors gracefully
6. ✅ Works with both undetected-chromedriver and Playwright
7. ✅ Debug Agent approves implementation after testing

## Testing Notes

After implementation:
- Test with valid credentials
- Test with invalid credentials
- Test with network delays
- Test element finding robustness
- Verify human-like behavior timing
- Test with both browser backends

## Next Steps After Approval

Once Debug Agent approves this task:
- Move to Task 2: Test Login Flow
- Create comprehensive login flow tests
- Verify all edge cases handled

## References

- Trading212 login page: https://www.trading212.com/en/login
- Current implementation: `browser/trade_executor.py` lines 28-64
- Browser automation base: `browser/automation.py`
- Human behavior: `browser/human_behavior.py`

---

**Commissioned by**: Management Agent
**Date**: Now
**Awaiting**: Builder Agent completion, then Debug Agent approval

