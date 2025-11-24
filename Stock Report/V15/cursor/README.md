# Test Automation Script for Stock Analyzer V15

This directory contains automated testing scripts that run comprehensive tests and integrate with Cursor for code fixes.

## Files

- `test_automation.py` - Main test automation script
- `README.md` - This file

## Usage

### Running the Test Suite

**Basic Mode** (from V15 directory):
```bash
python cursor/test_automation.py
```

**Autonomous Fix Mode** (automatically triggers Cursor fixes):
```bash
python cursor/test_automation.py --auto-fix
```

**Watch Mode** (automatically re-runs tests after fixes):
```bash
python cursor/test_automation.py --watch
```

**Watch Mode with Custom Retries**:
```bash
python cursor/test_automation.py --watch --max-retries 5
```

Or use the convenience scripts:
```bash
cursor\run_tests.bat  # Windows
./cursor/run_tests.sh  # Linux/Mac
```

### What It Tests

This script runs **ALL** test suites comprehensively:

1. **Root Level Test Files**
   - `test_v15.py` (pytest tests - comprehensive V15 tests)
   - `test_core_functions.py` (direct core function tests)
   - `test_constant_learning.py` (constant learning system tests)

2. **Test Suite Functions** (from `test/` directory)
   - Entrypoint detection tests
   - Function 1: Ticker Analysis comprehensive test suite
   - Function 3: Constant Learning comprehensive test suite
   - All test runner orchestration

3. **Core Functions** (Direct Testing)
   - Function 1: Ticker Analysis
   - Function 2: Autonomous Trading
   - Function 3: Constant Learning
   - Integration between functions

4. **Menu Options**
   - Main menu options (1-5)
   - V15 Features menu options (5A-5F)
   - Verifies all menu handlers exist and are callable

5. **Error Logs**
   - Collects recent error logs from:
     - `logs/error.log`
     - `logging/error.log`
     - `.cursor/debug.log`

**Total Coverage:** All test files, test suites, core functions, menu options, and error logs are tested in a single run.

### Output Files

The script generates several output files:

1. **Test Report (JSON)**
   - `test_report_YYYYMMDD_HHMMSS.json`
   - Complete test results in JSON format

2. **Test Summary (Text)**
   - `test_summary_YYYYMMDD_HHMMSS.txt`
   - Human-readable summary of all tests

3. **Cursor Fix Prompt (Markdown)**
   - `cursor_fix_prompt_YYYYMMDD_HHMMSS.md`
   - Generated only if failures are detected
   - Contains formatted prompt for Cursor to fix issues

### Example Workflows

#### Manual Workflow:
1. **Run Tests**
   ```bash
   python cursor/test_automation.py
   ```

2. **Review Failures**
   - Check the summary file for failed tests
   - Review error logs in the summary

3. **Fix Issues**
   - Open the `AUTO_FIX_REQUEST_*.md` file in Cursor
   - Cursor will automatically process and fix issues
   - Or manually review failures and fix code

4. **Re-run Tests**
   ```bash
   python cursor/test_automation.py
   ```

#### Autonomous Workflow (Recommended):
1. **Run Tests with Auto-Fix**
   ```bash
   python cursor/test_automation.py --auto-fix
   ```

2. **Cursor Automatically:**
   - Detects failures
   - Analyzes errors
   - Creates fix request
   - Processes fixes

3. **Done!** (or re-run if needed)

#### Watch Mode (Fully Autonomous):
1. **Run Tests in Watch Mode**
   ```bash
   python cursor/test_automation.py --watch
   ```

2. **Script Automatically:**
   - Runs tests
   - Detects failures
   - Triggers Cursor fixes
   - Waits for fixes
   - Re-runs tests
   - Repeats until all pass or max retries

3. **Done!** (fully automated)

### Autonomous Error Detection & Fix Integration

The script now **automatically** detects errors and triggers Cursor fixes:

#### 1. **Autonomous Error Analysis**
   - Automatically categorizes errors (import, attribute, timeout, path, etc.)
   - Identifies common patterns across failures
   - Prioritizes fixes (HIGH, MEDIUM, LOW)
   - Provides actionable fix suggestions

#### 2. **Autonomous Fix Request Generation**
   - Creates `AUTO_FIX_REQUEST_*.md` file with:
     - Error analysis and categorization
     - Priority fixes
     - Detailed failure information
     - Error logs
   - Structured format for Cursor to process

#### 3. **Autonomous Cursor Trigger**
   - Updates `.cursorrules` file to trigger Cursor
   - Creates `.cursor/fix_request.json` workspace file
   - Attempts to open fix request in Cursor CLI (if available)
   - Cursor automatically detects and processes fix requests

#### 4. **Watch Mode** (Automatic Re-testing)
   - `--watch` flag enables automatic re-testing after fixes
   - Waits for Cursor to process fixes
   - Re-runs tests automatically
   - Continues until all tests pass or max retries reached

#### Usage Examples:

```bash
# Basic: Run tests and generate fix request
python cursor/test_automation.py

# Autonomous: Run tests and automatically trigger Cursor fixes
python cursor/test_automation.py --auto-fix

# Watch Mode: Automatically re-test after fixes (up to 3 retries)
python cursor/test_automation.py --watch

# Watch Mode: Custom retry count
python cursor/test_automation.py --watch --max-retries 5
```

#### How It Works:

1. **Test Execution**: Runs all test suites
2. **Error Detection**: Automatically detects and categorizes failures
3. **Analysis**: Identifies root causes and common patterns
4. **Fix Request**: Creates structured fix request file
5. **Cursor Trigger**: Automatically triggers Cursor to process fixes
6. **Re-test** (watch mode): Automatically re-runs tests after fixes
7. **Verification**: Confirms all tests pass

### Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed
- `130` - Interrupted by user (Ctrl+C)

### Requirements

- Python 3.7+
- pytest (for test_v15.py)
- All V15 dependencies installed

### Troubleshooting

**Tests timeout:**
- Increase timeout in `test_automation.py` (default: 300 seconds)

**Import errors:**
- Ensure you're running from the V15 directory
- Check that all dependencies are installed

**Menu tests fail:**
- Verify `ui/menu_v15.py` exists and is importable
- Check that all menu handler methods exist

### Test Coverage Summary

The script consolidates and runs:

- ✅ **Root level tests** (3 files)
- ✅ **Test suite functions** (Function 1 & 3 comprehensive suites)
- ✅ **Core function direct tests** (all 3 functions)
- ✅ **Menu option validation** (all menu handlers)
- ✅ **Error log collection** (all log sources)

This ensures **complete test coverage** in a single execution.

### Customization

You can customize the script by modifying:

- `ROOT_TEST_FILES` - Add or remove root level test files
- `TEST_SUITE_FILES` - Add or remove test suite files
- `MENU_OPTIONS` - Update menu structure
- `LOG_DIRS` - Add additional log directories
- Timeout values for test execution

