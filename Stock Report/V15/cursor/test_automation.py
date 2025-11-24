"""
Test Automation Script for Stock Analyzer V15
Automates running test suite and integrates with Cursor for code fixes.

This script:
1. Runs all test files (test_v15.py, test_core_functions.py, test_constant_learning.py)
2. Tests all 3 core functions
3. Tests menu options
4. Reports error logs
5. Provides Cursor integration for fixing failures
"""

import sys
import subprocess
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

# Setup paths
V15_ROOT = Path(__file__).parent.parent
CURSOR_DIR = Path(__file__).parent
SCRIPT_DIR = str(V15_ROOT)

# Add V15 to path
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Test files to run (root level)
ROOT_TEST_FILES = [
    V15_ROOT / "test_v15.py",
    V15_ROOT / "test_core_functions.py",
    V15_ROOT / "test_constant_learning.py"
]

# Test suites in test/ directory
TEST_DIR = V15_ROOT / "test"
TEST_SUITE_FILES = [
    TEST_DIR / "test_function_1_ticker_analysis.py",
    TEST_DIR / "test_function_3_continuous_learning.py",
    TEST_DIR / "test_entrypoint_detector.py"
]

# All test files combined
TEST_FILES = ROOT_TEST_FILES + TEST_SUITE_FILES

# Core functions to test
CORE_FUNCTIONS = [
    "Function 1: Ticker Analysis",
    "Function 2: Autonomous Trading",
    "Function 3: Constant Learning"
]

# Menu options to test (from menu_v15.py)
MENU_OPTIONS = {
    "Main Menu": {
        "1": "Core Analysis",
        "2": "Learning & Training",
        "3": "Data & Logs",
        "4": "System & Maintenance",
        "5": "V15 Features"
    },
    "V15 Features Menu": {
        "5A": "Unified Model - Generate Prediction",
        "5B": "Risk Profile Selection",
        "5C": "Browser Automation Status",
        "5D": "Sentiment Override Settings",
        "5E": "Trade Log Analysis",
        "5F": "Performance Report"
    }
}

# Log directories to check
LOG_DIRS = [
    V15_ROOT / "logs",
    V15_ROOT / "logging",
    V15_ROOT / ".cursor"
]
DEBUG_LOG_PATH = Path(r"c:\Users\edwar\Documents\GitHub\.cursor\debug.log")
CURSOR_AGENT_STARTED = False
CLEANUP_DONE = False


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Optional[Dict] = None) -> None:
    """Append a single NDJSON instrumentation log entry."""
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _ensure_cursor_agent():
    """Start the Cursor agent in a new console window if available."""
    global CURSOR_AGENT_STARTED
    if CURSOR_AGENT_STARTED:
        return
    cursor_commands = [
        r"C:\Program Files\cursor\resources\app\bin\cursor.cmd",
        'cursor',
        'cursor-cli',
        'cursor.exe'
    ]
    for cmd in cursor_commands:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                creation_flags = 0
                if os.name == "nt":
                    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                try:
                    subprocess.Popen(
                        [cmd, 'agent'],
                        cwd=str(V15_ROOT),
                        creationflags=creation_flags
                    )
                    CURSOR_AGENT_STARTED = True
                    print("[INFO] Cursor agent started in a new console window.")
                except Exception as agent_err:
                    print(f"[WARN]  Could not start Cursor agent automatically: {agent_err}")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    print("[WARN]  Cursor CLI not available; automatic agent launch skipped.")


def _cleanup_old_fix_requests():
    """Remove previously completed fix artifacts before a new run."""
    global CLEANUP_DONE
    if CLEANUP_DONE:
        return
    for pattern in ("cursor_fix_prompt_*.md", "AUTO_FIX_REQUEST_*.md"):
        for file_path in CURSOR_DIR.glob(pattern):
            try:
                file_path.unlink()
            except Exception:
                pass
    fix_json = V15_ROOT / ".cursor" / "fix_request.json"
    try:
        if fix_json.exists():
            fix_json.unlink()
    except Exception:
        pass
    cursorrules_file = V15_ROOT / ".cursorrules"
    try:
        if cursorrules_file.exists():
            cursorrules_file.unlink()
    except Exception:
        pass
    CLEANUP_DONE = True


class TestRunner:
    """Main test runner with Cursor integration."""
    
    def __init__(self):
        _ensure_cursor_agent()
        _cleanup_old_fix_requests()
        self.results: Dict[str, any] = {}
        self.failures: List[Dict[str, str]] = []
        self.error_logs: List[str] = []
        self.start_time = datetime.now()
        self.report_file = CURSOR_DIR / f"test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        self.summary_file = CURSOR_DIR / f"test_summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        self._fix_triggered = False
        
    def run_pytest_test(self, test_file: Path) -> Tuple[bool, str, str]:
        """Run a pytest test file and return (success, stdout, stderr)."""
        if not test_file.exists():
            return False, "", f"Test file not found: {test_file}"
        
        try:
            env = os.environ.copy()
            env['PYTHONBREAKPOINT'] = '0'
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env.setdefault('PYTHONUTF8', '1')
            env.setdefault('PYTHONIOENCODING', 'utf-8')
            # Run pytest with verbose output
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,  # 5 minute timeout
                cwd=str(V15_ROOT),
                env=env
            )
            
            success = result.returncode == 0
            if success:
                return True, result.stdout, result.stderr
            if "No module named pytest" in (result.stderr or ""):
                fallback = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300,
                    cwd=str(V15_ROOT),
                    env=env
                )
                success = fallback.returncode == 0
                stderr = (result.stderr or "") + "\n[INFO] pytest not available; executed file directly."
                stderr += "\n" + (fallback.stderr or "")
                stdout = (result.stdout or "") + "\n" + (fallback.stdout or "")
                return success, stdout, stderr
            return False, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Test timed out after 5 minutes"
        except Exception as e:
            return False, "", f"Error running test: {str(e)}"
    
    def run_python_test(self, test_file: Path) -> Tuple[bool, str, str]:
        """Run a Python test file directly and return (success, stdout, stderr)."""
        if not test_file.exists():
            return False, "", f"Test file not found: {test_file}"
        
        try:
            env = os.environ.copy()
            env['PYTHONBREAKPOINT'] = '0'
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env.setdefault('PYTHONUTF8', '1')
            env.setdefault('PYTHONIOENCODING', 'utf-8')
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
                cwd=str(V15_ROOT),
                env=env
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Test timed out after 5 minutes"
        except Exception as e:
            return False, "", f"Error running test: {str(e)}"
    
    def test_core_functions(self) -> Dict[str, bool]:
        """Test all 3 core functions."""
        print("\n" + "=" * 80)
        print("TESTING CORE FUNCTIONS")
        print("=" * 80)
        
        results = {}
        
        # Test Function 1: Ticker Analysis
        print("\n[1/3] Testing Function 1: Ticker Analysis...")
        try:
            from test_core_functions import test_function_1_ticker_analysis
            result = test_function_1_ticker_analysis()
            results["function_1"] = result
            status = "[OK] PASSED" if result else "[FAIL] FAILED"
            print(f"   {status}")
        except Exception as e:
            results["function_1"] = False
            print(f"   [FAIL] FAILED: {e}")
            self.failures.append({
                "test": "Function 1: Ticker Analysis",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        # Test Function 2: Autonomous Trading
        print("\n[2/3] Testing Function 2: Autonomous Trading...")
        try:
            from test_core_functions import test_function_2_autonomous_trading
            result = test_function_2_autonomous_trading()
            results["function_2"] = result
            status = "[OK] PASSED" if result else "[FAIL] FAILED"
            print(f"   {status}")
        except Exception as e:
            results["function_2"] = False
            print(f"   [FAIL] FAILED: {e}")
            self.failures.append({
                "test": "Function 2: Autonomous Trading",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        # Test Function 3: Constant Learning
        print("\n[3/3] Testing Function 3: Constant Learning...")
        try:
            from test_core_functions import test_function_3_constant_learning
            result = test_function_3_constant_learning()
            results["function_3"] = result
            status = "[OK] PASSED" if result else "[FAIL] FAILED"
            print(f"   {status}")
        except Exception as e:
            results["function_3"] = False
            print(f"   [FAIL] FAILED: {e}")
            self.failures.append({
                "test": "Function 3: Constant Learning",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        return results
    
    def test_menu_options(self) -> Dict[str, bool]:
        """Test menu options by checking if functions exist and can be imported."""
        print("\n" + "=" * 80)
        print("TESTING MENU OPTIONS")
        print("=" * 80)
        
        results = {}
        
        try:
            # Import menu controller
            sys.path.insert(0, str(V15_ROOT))
            from ui.menu_v15 import MenuController
            
            menu = MenuController()
            
            # Test main menu options
            print("\n[Main Menu] Testing menu option handlers...")
            menu_methods = {
                "1": ("_handle_analysis_menu", "Core Analysis"),
                "2": ("_handle_learning_menu", "Learning & Training"),
                "3": ("_handle_data_menu", "Data & Logs"),
                "4": ("_handle_system_menu", "System & Maintenance"),
                "5": ("_handle_V15_features_menu", "V15 Features")
            }
            
            for option, (method_name, description) in menu_methods.items():
                try:
                    method = getattr(menu, method_name, None)
                    if method and callable(method):
                        results[f"menu_{option}"] = True
                        print(f"   [OK] Menu {option} ({description}): Handler exists")
                    else:
                        results[f"menu_{option}"] = False
                        print(f"   [FAIL] Menu {option} ({description}): Handler missing")
                        self.failures.append({
                            "test": f"Menu Option {option}: {description}",
                            "error": f"Method {method_name} not found or not callable"
                        })
                except Exception as e:
                    results[f"menu_{option}"] = False
                    print(f"   [FAIL] Menu {option} ({description}): Error - {e}")
                    self.failures.append({
                        "test": f"Menu Option {option}: {description}",
                        "error": str(e)
                    })
            
            # Test V15 features menu methods
            print("\n[V15 Features Menu] Testing feature handlers...")
            v15_methods = {
                "5A": ("_unified_model_prediction", "Unified Model Prediction"),
                "5B": ("_select_risk_profile", "Risk Profile Selection"),
                "5C": ("_browser_automation_status", "Browser Automation Status"),
                "5D": ("_sentiment_override_settings", "Sentiment Override Settings"),
                "5E": ("_trade_log_analysis", "Trade Log Analysis"),
                "5F": ("_performance_report", "Performance Report")
            }
            
            for option, (method_name, description) in v15_methods.items():
                try:
                    method = getattr(menu, method_name, None)
                    if method and callable(method):
                        results[f"menu_{option}"] = True
                        print(f"   [OK] Menu {option} ({description}): Handler exists")
                    else:
                        results[f"menu_{option}"] = False
                        print(f"   [FAIL] Menu {option} ({description}): Handler missing")
                        self.failures.append({
                            "test": f"Menu Option {option}: {description}",
                            "error": f"Method {method_name} not found or not callable"
                        })
                except Exception as e:
                    results[f"menu_{option}"] = False
                    print(f"   [FAIL] Menu {option} ({description}): Error - {e}")
                    self.failures.append({
                        "test": f"Menu Option {option}: {description}",
                        "error": str(e)
                    })
            
        except Exception as e:
            print(f"\n[FAIL] Error testing menu options: {e}")
            traceback.print_exc()
            self.failures.append({
                "test": "Menu Options Import",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        return results
    
    def collect_error_logs(self) -> List[str]:
        """Collect error logs from log directories."""
        print("\n" + "=" * 80)
        print("COLLECTING ERROR LOGS")
        print("=" * 80)
        
        error_logs = []
        
        # Check for error.log files
        error_log_paths = [
            V15_ROOT / "logs" / "error.log",
            V15_ROOT / "logging" / "error.log",
            V15_ROOT / ".cursor" / "debug.log"
        ]
        
        for log_path in error_log_paths:
            if log_path.exists():
                try:
                    # Read last 50 lines of log file
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        recent_lines = lines[-50:] if len(lines) > 50 else lines
                        if recent_lines:
                            error_logs.append(f"\n--- {log_path.name} ({log_path.parent.name}) ---")
                            error_logs.extend(recent_lines)
                            print(f"   [OK] Found: {log_path}")
                except Exception as e:
                    print(f"   [WARN]  Could not read {log_path}: {e}")
        
        if not error_logs:
            print("   ℹ️  No error logs found")
        
        return error_logs
    
    def run_test_suite_functions(self) -> Dict[str, any]:
        """Run test suite functions from test/ directory."""
        print("\n" + "=" * 80)
        print("RUNNING TEST SUITE FUNCTIONS")
        print("=" * 80)
        #region agent log
        _agent_debug_log(
            "H4",
            "cursor/test_automation.py:350",
            "Entered run_test_suite_functions",
            {"current_failures": len(self.failures)}
        )
        #endregion
        
        suite_results = {}
        
        # Test entrypoint detection first
        print("\n[Entrypoint Detection] Testing entrypoint detection...")
        try:
            sys.path.insert(0, str(TEST_DIR))
            from test_entrypoint_detector import detect_and_setup, EntrypointDetector
            
            entrypoint_path, detected_root = detect_and_setup()
            if entrypoint_path:
                suite_results["entrypoint_detection"] = True
                print(f"   [OK] Entrypoint found: {entrypoint_path.name}")
            else:
                suite_results["entrypoint_detection"] = False
                print("   [FAIL] Entrypoint not found")
                self.failures.append({
                    "test": "Entrypoint Detection",
                    "error": "Could not detect main entrypoint file"
                })
        except Exception as e:
            suite_results["entrypoint_detection"] = False
            print(f"   [FAIL] Error: {e}")
            self.failures.append({
                "test": "Entrypoint Detection",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        # Run Function 1 tests from test suite
        print("\n[Test Suite] Running Function 1: Ticker Analysis tests...")
        #region agent log
        _agent_debug_log(
            "H4",
            "cursor/test_automation.py:381",
            "Starting Function 1 suite run",
            {"failures_before_suite": len(self.failures)}
        )
        #endregion
        try:
            from test_function_1_ticker_analysis import run_function_1_tests
            func1_results = run_function_1_tests()
            suite_results["function_1_suite"] = func1_results
            
            # Check if all passed
            all_passed = all(func1_results.values()) if isinstance(func1_results, dict) else False
            if all_passed:
                print(f"   [OK] Function 1 suite: All tests passed")
            else:
                failed = [k for k, v in func1_results.items() if not v] if isinstance(func1_results, dict) else []
                print(f"   [WARN]  Function 1 suite: {len(failed)} test(s) failed")
                self.failures.append({
                    "test": "Function 1 Test Suite",
                    "error": f"Failed tests: {', '.join(failed[:5])}",
                    "details": func1_results
                })
        except Exception as e:
            suite_results["function_1_suite"] = {"error": str(e)}
            print(f"   [FAIL] Error running Function 1 suite: {e}")
            self.failures.append({
                "test": "Function 1 Test Suite",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            #region agent log
            _agent_debug_log(
                "H4",
                "cursor/test_automation.py:400",
                "Function 1 suite raised exception",
                {"error": str(e)}
            )
            #endregion
        
        # Run Function 3 tests from test suite
        print("\n[Test Suite] Running Function 3: Constant Learning tests...")
        try:
            from test_function_3_continuous_learning import run_function_3_tests
            func3_results = run_function_3_tests()
            suite_results["function_3_suite"] = func3_results
            
            # Check if all passed
            all_passed = all(func3_results.values()) if isinstance(func3_results, dict) else False
            if all_passed:
                print(f"   [OK] Function 3 suite: All tests passed")
            else:
                failed = [k for k, v in func3_results.items() if not v] if isinstance(func3_results, dict) else []
                print(f"   [WARN]  Function 3 suite: {len(failed)} test(s) failed")
                self.failures.append({
                    "test": "Function 3 Test Suite",
                    "error": f"Failed tests: {', '.join(failed[:5])}",
                    "details": func3_results
                })
        except Exception as e:
            suite_results["function_3_suite"] = {"error": str(e)}
            print(f"   [FAIL] Error running Function 3 suite: {e}")
            self.failures.append({
                "test": "Function 3 Test Suite",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        return suite_results
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run all tests and collect results."""
        print("\n" + "=" * 80)
        print("STOCK ANALYZER V15 - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"V15 Root: {V15_ROOT}")
        print("=" * 80)
        
        # Run root level pytest tests
        print("\n" + "=" * 80)
        print("RUNNING ROOT LEVEL TESTS")
        print("=" * 80)
        
        for test_file in ROOT_TEST_FILES:
            if test_file.name == "test_v15.py":
                print(f"\n[Python] Running {test_file.name} (pytest fallback)...")
                success, stdout, stderr = self.run_python_test(test_file)
                self.results[f"root_{test_file.name}"] = {
                    "success": success,
                    "stdout": stdout,
                    "stderr": stderr
                }
                if not success:
                    self.failures.append({
                        "test": f"Root Test: {test_file.name}",
                        "error": stderr or "Test failed",
                        "stdout": stdout
                    })
            else:
                # Run as Python scripts
                print(f"\n[Python] Running {test_file.name}...")
                success, stdout, stderr = self.run_python_test(test_file)
                self.results[f"root_{test_file.name}"] = {
                    "success": success,
                    "stdout": stdout,
                    "stderr": stderr
                }
                if not success:
                    self.failures.append({
                        "test": f"Root Test: {test_file.name}",
                        "error": stderr or stdout or "Test failed",
                        "stdout": stdout
                    })
        
        # Run test suite functions (from test/ directory)
        suite_results = self.run_test_suite_functions()
        self.results["test_suite_functions"] = suite_results
        
        # Test core functions (direct function tests)
        core_results = self.test_core_functions()
        self.results["core_functions"] = core_results
        
        # Test menu options
        menu_results = self.test_menu_options()
        self.results["menu_options"] = menu_results
        
        # Collect error logs
        self.error_logs = self.collect_error_logs()
        self.results["error_logs"] = self.error_logs
        
        return self.results
    
    def generate_cursor_prompt(self) -> str:
        """Generate a Cursor prompt for fixing failures."""
        if not self.failures:
            return "All tests passed! No fixes needed."
        
        prompt = """# Test Failures Detected - Please Fix

The following test failures were detected. Please review and fix the issues:

"""
        
        for i, failure in enumerate(self.failures, 1):
            prompt += f"## Failure {i}: {failure['test']}\n\n"
            prompt += f"**Error:**\n```\n{failure.get('error', 'Unknown error')}\n```\n\n"
            
            if 'traceback' in failure:
                prompt += f"**Traceback:**\n```\n{failure['traceback']}\n```\n\n"
            
            if 'stdout' in failure and failure['stdout']:
                prompt += f"**Output:**\n```\n{failure['stdout'][:500]}\n```\n\n"
            
            prompt += "---\n\n"
        
        if self.error_logs:
            prompt += "## Error Logs\n\n"
            prompt += "Recent error logs:\n```\n"
            prompt += "".join(self.error_logs[-500:])  # Last 500 lines
            prompt += "\n```\n\n"
        
        prompt += """
## Instructions

Please:
1. Review each failure above
2. Identify the root cause
3. Fix the code
4. Re-run the test suite to verify fixes

## Files to Review

- Test files: test_v15.py, test_core_functions.py, test_constant_learning.py
- Main application: Stock Analyzer V15.py
- Menu system: ui/menu_v15.py
- Core modules: core/
- Learning modules: learning/
"""
        
        return prompt
    
    def analyze_errors(self) -> Dict[str, any]:
        """Autonomously analyze errors and identify root causes."""
        print("\n" + "=" * 80)
        print("AUTONOMOUS ERROR ANALYSIS")
        print("=" * 80)
        
        error_analysis = {
            "categories": {},
            "common_patterns": [],
            "fix_suggestions": [],
            "priority": []
        }
        
        # Analyze each failure
        for failure in self.failures:
            error_text = failure.get('error', '').lower()
            test_name = failure.get('test', '')
            
            # Categorize errors
            if 'import' in error_text or 'module' in error_text or 'no module' in error_text:
                category = "import_error"
                if category not in error_analysis["categories"]:
                    error_analysis["categories"][category] = []
                error_analysis["categories"][category].append({
                    "test": test_name,
                    "error": failure.get('error', ''),
                    "fix": "Check imports and dependencies"
                })
                error_analysis["fix_suggestions"].append({
                    "type": "dependency",
                    "action": "pip install missing_module",
                    "test": test_name
                })
            
            elif 'attribute' in error_text or 'has no attribute' in error_text:
                category = "attribute_error"
                if category not in error_analysis["categories"]:
                    error_analysis["categories"][category] = []
                error_analysis["categories"][category].append({
                    "test": test_name,
                    "error": failure.get('error', ''),
                    "fix": "Check method/attribute names"
                })
                error_analysis["fix_suggestions"].append({
                    "type": "code_fix",
                    "action": "Fix attribute/method name",
                    "test": test_name
                })
            
            elif 'timeout' in error_text or 'timed out' in error_text:
                category = "timeout_error"
                if category not in error_analysis["categories"]:
                    error_analysis["categories"][category] = []
                error_analysis["categories"][category].append({
                    "test": test_name,
                    "error": failure.get('error', ''),
                    "fix": "Increase timeout or optimize code"
                })
            
            elif 'file not found' in error_text or 'path' in error_text:
                category = "path_error"
                if category not in error_analysis["categories"]:
                    error_analysis["categories"][category] = []
                error_analysis["categories"][category].append({
                    "test": test_name,
                    "error": failure.get('error', ''),
                    "fix": "Check file paths and directory structure"
                })
            
            else:
                category = "other_error"
                if category not in error_analysis["categories"]:
                    error_analysis["categories"][category] = []
                error_analysis["categories"][category].append({
                    "test": test_name,
                    "error": failure.get('error', ''),
                    "fix": "Review error details"
                })
        
        # Identify common patterns
        if len(error_analysis["categories"].get("import_error", [])) > 2:
            error_analysis["common_patterns"].append("Multiple import errors - likely missing dependencies")
            error_analysis["priority"].append({
                "issue": "Missing dependencies",
                "priority": "HIGH",
                "action": "Run: pip install -r requirements.txt"
            })
        
        if len(error_analysis["categories"].get("attribute_error", [])) > 1:
            error_analysis["common_patterns"].append("Multiple attribute errors - likely API mismatch")
            error_analysis["priority"].append({
                "issue": "API method name mismatches",
                "priority": "MEDIUM",
                "action": "Review method names in affected modules"
            })
        
        # Print analysis
        print(f"\n[INFO] Error Categories Found: {len(error_analysis['categories'])}")
        for category, errors in error_analysis["categories"].items():
            print(f"   - {category}: {len(errors)} error(s)")
        
        if error_analysis["common_patterns"]:
            print(f"\n[INFO] Common Patterns:")
            for pattern in error_analysis["common_patterns"]:
                print(f"   - {pattern}")
        
        if error_analysis["priority"]:
            print(f"\n[INFO] Priority Fixes:")
            for item in error_analysis["priority"]:
                print(f"   [{item['priority']}] {item['issue']}")
                print(f"      Action: {item['action']}")
        
        return error_analysis
    
    def create_autonomous_fix_request(self, error_analysis: Dict[str, any]) -> Path:
        """Create an autonomous fix request file for Cursor."""
        fix_request_file = CURSOR_DIR / f"AUTO_FIX_REQUEST_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md"
        
        content = """# AUTONOMOUS FIX REQUEST - Stock Analyzer V15

**Generated:** {timestamp}
**Status:** AUTO-FIX REQUIRED
**Total Failures:** {failure_count}

## Executive Summary

Test suite detected {failure_count} failure(s). This file triggers autonomous fixes.

## Error Analysis

### Categories
{error_categories}

### Common Patterns
{common_patterns}

### Priority Fixes
{priority_fixes}

## Detailed Failures

{detailed_failures}

## Error Logs

{error_logs}

## Autonomous Fix Instructions

**ACTION REQUIRED:** Fix all issues listed above.

### Fix Priority Order:
1. HIGH priority issues first
2. Common patterns (affect multiple tests)
3. Individual test failures

### Expected Actions:
- Fix import errors (install dependencies)
- Fix attribute/method name mismatches
- Fix path/file issues
- Review and fix other errors

### Verification:
After fixes, re-run: `python cursor/test_automation.py`

---
**This file was auto-generated by test_automation.py**
**Cursor should automatically process this file and fix issues**
""".format(
            timestamp=self.start_time.isoformat(),
            failure_count=len(self.failures),
            error_categories=self._format_error_categories(error_analysis),
            common_patterns=self._format_common_patterns(error_analysis),
            priority_fixes=self._format_priority_fixes(error_analysis),
            detailed_failures=self._format_detailed_failures(),
            error_logs=self._format_error_logs()
        )
        
        with open(fix_request_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n[OK] Autonomous fix request created: {fix_request_file}")
        return fix_request_file
    
    def _format_error_categories(self, analysis: Dict) -> str:
        """Format error categories for fix request."""
        if not analysis.get("categories"):
            return "None identified"
        
        lines = []
        for category, errors in analysis["categories"].items():
            lines.append(f"- **{category}**: {len(errors)} error(s)")
            for err in errors[:3]:  # Show first 3
                lines.append(f"  - {err['test']}: {err['error'][:100]}")
        return "\n".join(lines)
    
    def _format_common_patterns(self, analysis: Dict) -> str:
        """Format common patterns for fix request."""
        if not analysis.get("common_patterns"):
            return "None identified"
        return "\n".join(f"- {pattern}" for pattern in analysis["common_patterns"])
    
    def _format_priority_fixes(self, analysis: Dict) -> str:
        """Format priority fixes for fix request."""
        if not analysis.get("priority"):
            return "None identified"
        lines = []
        for item in analysis["priority"]:
            lines.append(f"- **[{item['priority']}]** {item['issue']}")
            lines.append(f"  Action: `{item['action']}`")
        return "\n".join(lines)
    
    def _format_detailed_failures(self) -> str:
        """Format detailed failures for fix request."""
        if not self.failures:
            return "None"
        
        lines = []
        for i, failure in enumerate(self.failures, 1):
            lines.append(f"### Failure {i}: {failure['test']}")
            lines.append(f"**Error:**\n```\n{failure.get('error', 'Unknown')[:500]}\n```")
            if 'traceback' in failure:
                lines.append(f"**Traceback:**\n```\n{failure['traceback'][:500]}\n```")
            lines.append("")
        return "\n".join(lines)
    
    def _format_error_logs(self) -> str:
        """Format error logs for fix request."""
        if not self.error_logs:
            return "No error logs found"
        return "```\n" + "".join(self.error_logs[-200:]) + "\n```"
    
    def trigger_cursor_autonomous_fix(self, fix_request_file: Path) -> bool:
        """Autonomously trigger Cursor to fix issues."""
        print("\n" + "=" * 80)
        print("AUTONOMOUS CURSOR FIX TRIGGER")
        print("=" * 80)
        
        # Method 1: Create a .cursorrules file trigger
        cursorrules_file = V15_ROOT / ".cursorrules"
        trigger_content = f"""
# Auto-generated fix trigger
# This file triggers Cursor to process: {fix_request_file.name}

Please read and process the fix request file: {fix_request_file}
The file contains detailed error analysis and fix instructions.

Fix all issues listed in the fix request file.
"""
        
        try:
            # Append to .cursorrules if it exists, or create new
            if cursorrules_file.exists():
                with open(cursorrules_file, 'r', encoding='utf-8') as f:
                    existing = f.read()
                if fix_request_file.name not in existing:
                    with open(cursorrules_file, 'a', encoding='utf-8') as f:
                        f.write(trigger_content)
                    print(f"[OK] Updated .cursorrules to trigger fix")
            else:
                with open(cursorrules_file, 'w', encoding='utf-8') as f:
                    f.write(trigger_content)
                print(f"[OK] Created .cursorrules to trigger fix")
        except Exception as e:
            print(f"[WARN]  Could not update .cursorrules: {e}")
        
        # Method 2: Create a Cursor workspace file
        cursor_workspace_file = V15_ROOT / ".cursor" / "fix_request.json"
        try:
            cursor_workspace_file.parent.mkdir(parents=True, exist_ok=True)
            workspace_data = {
                "fix_request_file": str(fix_request_file),
                "timestamp": self.start_time.isoformat(),
                "failures": len(self.failures),
                "status": "pending",
                "action": "read_and_fix"
            }
            with open(cursor_workspace_file, 'w', encoding='utf-8') as f:
                json.dump(workspace_data, f, indent=2)
            print(f"[OK] Created Cursor workspace trigger: {cursor_workspace_file}")
        except Exception as e:
            print(f"[WARN]  Could not create workspace trigger: {e}")
        
        # Method 3: Try to call Cursor CLI if available
        cursor_commands = ['cursor', 'cursor-cli', 'cursor.exe']
        for cmd in cursor_commands:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"\n🔧 Cursor CLI detected: {cmd}")
                    print("   Attempting to open fix request...")
                    # Try to open the file in Cursor
                    try:
                        subprocess.run(
                            [cmd, str(fix_request_file)],
                            timeout=10,
                            cwd=str(V15_ROOT)
                        )
                        print("   [OK] Fix request opened in Cursor")
                        return True
                    except Exception as e:
                        print(f"   [WARN]  Could not open in Cursor: {e}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        print("\n[INFO] Fix Request Ready:")
        print(f"   File: {fix_request_file}")
        print("   Cursor will automatically detect and process this file")
        print("   Or manually open it in Cursor to trigger fixes")
        
        return True

    def trigger_autonomous_fix_flow(self):
        """Generate prompt, analyze failures, and trigger Cursor fix."""
        if self._fix_triggered or not self.failures:
            return
        self._fix_triggered = True
        cursor_prompt_file = CURSOR_DIR / f"cursor_fix_prompt_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md"
        prompt = self.generate_cursor_prompt()
        with open(cursor_prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"   - Cursor Fix Prompt: {cursor_prompt_file}")
        error_analysis = self.analyze_errors()
        fix_request_file = self.create_autonomous_fix_request(error_analysis)
        self.trigger_cursor_autonomous_fix(fix_request_file)
        print("\n[INFO] AUTONOMOUS FIX PROCESS INITIATED:")
        print("   1. [OK] Errors analyzed and categorized")
        print("   2. [OK] Fix request file created")
        print("   3. [OK] Cursor fix trigger activated")
        print("\n[INFO] Cursor should automatically process the fix request")
        print("       Re-run this script after fixes to verify")
    
    def save_report(self):
        """Save test report to JSON file."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "results": self.results,
            "failures": self.failures,
            "error_logs": self.error_logs,
            "summary": {
                "total_tests": len(self.results),
                "failed_tests": len(self.failures),
                "passed": len(self.failures) == 0
            }
        }
        
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n[OK] Test report saved: {self.report_file}")
    
    def save_summary(self):
        """Save human-readable summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STOCK ANALYZER V15 - TEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Root level tests
            f.write("\nRoot Level Tests:\n")
            for test_name, result in self.results.items():
                if test_name.startswith("root_") and isinstance(result, dict) and "success" in result:
                    status = "[OK] PASSED" if result["success"] else "[FAIL] FAILED"
                    f.write(f"  {test_name.replace('root_', '')}: {status}\n")
            
            # Test suite functions
            if "test_suite_functions" in self.results:
                f.write("\nTest Suite Functions:\n")
                suite_results = self.results["test_suite_functions"]
                if isinstance(suite_results, dict):
                    for suite_name, suite_result in suite_results.items():
                        if isinstance(suite_result, dict):
                            if "error" in suite_result:
                                f.write(f"  {suite_name}: [FAIL] FAILED ({suite_result['error']})\n")
                            else:
                                all_passed = all(suite_result.values()) if suite_result else False
                                status = "[OK] PASSED" if all_passed else "[WARN]  PARTIAL"
                                f.write(f"  {suite_name}: {status}\n")
                        elif isinstance(suite_result, bool):
                            status = "[OK] PASSED" if suite_result else "[FAIL] FAILED"
                            f.write(f"  {suite_name}: {status}\n")
            
            # Core functions
            if "core_functions" in self.results:
                f.write("\nCore Functions:\n")
                for func_name, passed in self.results["core_functions"].items():
                    status = "[OK] PASSED" if passed else "[FAIL] FAILED"
                    f.write(f"  {func_name}: {status}\n")
            
            # Menu options
            if "menu_options" in self.results:
                f.write("\nMenu Options:\n")
                for menu_name, passed in self.results["menu_options"].items():
                    status = "[OK] PASSED" if passed else "[FAIL] FAILED"
                    f.write(f"  {menu_name}: {status}\n")
            
            # Failures
            if self.failures:
                f.write("\n" + "=" * 80 + "\n")
                f.write("FAILURES\n")
                f.write("=" * 80 + "\n\n")
                for i, failure in enumerate(self.failures, 1):
                    f.write(f"Failure {i}: {failure['test']}\n")
                    f.write(f"Error: {failure.get('error', 'Unknown')}\n")
                    f.write("-" * 80 + "\n\n")
            
            # Error logs summary
            if self.error_logs:
                f.write("\n" + "=" * 80 + "\n")
                f.write("ERROR LOGS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write("".join(self.error_logs))
        
        print(f"[OK] Summary saved: {self.summary_file}")
    
    def print_summary(self):
        """Print test summary to console."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total Failures: {len(self.failures)}")
        print("=" * 80)
        
        if self.failures:
            print("\n[FAIL] FAILURES DETECTED:")
            for i, failure in enumerate(self.failures, 1):
                print(f"\n  {i}. {failure['test']}")
                print(f"     Error: {failure.get('error', 'Unknown')[:100]}")
        else:
            print("\n[OK] ALL TESTS PASSED!")
        
        print(f"\n[INFO] Reports saved:")
        print(f"   - JSON: {self.report_file}")
        print(f"   - Summary: {self.summary_file}")
        
        if self.failures:
            self.trigger_autonomous_fix_flow()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Analyzer V15 Test Automation')
    parser.add_argument('--auto-fix', action='store_true', 
                       help='Automatically trigger Cursor fixes on failure')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode: re-run tests after fixes')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts in watch mode (default: 3)')
    
    args = parser.parse_args()
    
    runner: Optional[TestRunner] = None
    try:
        retry_count = 0
        max_retries = args.max_retries if args.watch else 0
        
        while retry_count <= max_retries:
            if retry_count > 0:
                print("\n" + "=" * 80)
                print(f"RETRY ATTEMPT {retry_count}/{max_retries}")
                print("=" * 80)
                time.sleep(2)  # Brief pause before retry
            runner = TestRunner()
            results = runner.run_all_tests()
            runner.save_report()
            runner.save_summary()
            runner.print_summary()
            
            # If no failures, exit successfully
            if not runner.failures:
                if retry_count > 0:
                    print("\n[OK] All tests passed after fixes!")
                sys.exit(0)
            
            # Failures detected
            runner.trigger_autonomous_fix_flow()
            if args.auto_fix or args.watch:
                print("\n" + "=" * 80)
                print("AUTONOMOUS FIX MODE ACTIVATED")
                print("=" * 80)
                
                if args.watch and retry_count < max_retries:
                    print(f"\n[INFO] Waiting for fixes... (Retry {retry_count + 1}/{max_retries})")
                    print("   Cursor should process the fix request automatically")
                    print("   Re-running tests in 10 seconds...")
                    time.sleep(10)
                    retry_count += 1
                    continue
                else:
                    if args.watch:
                        print(f"\n[WARN]  Max retries ({max_retries}) reached")
                    print("   Please review fixes and re-run manually")
                    sys.exit(1)
            else:
                # Manual mode - just report failures
                sys.exit(1)
            
            break  # Exit loop if not in watch mode
        
    except KeyboardInterrupt:
        print("\n\n[WARN]  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FAIL] FATAL ERROR: {e}")
        traceback.print_exc()
        if runner:
            if not runner.failures:
                runner.failures.append({
                    "test": "Automation Harness",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
            try:
                runner.save_report()
                runner.save_summary()
            except Exception:
                pass
            runner.trigger_autonomous_fix_flow()
        sys.exit(1)


if __name__ == "__main__":
    main()





