"""
Intensive Test Suite Runner for Stock Analyzer V15

Orchestrates all test suites and generates comprehensive reports.
Tests Function 1 (Ticker Analysis) and Function 3 (Continuous Learning).
Function 2 (Trading) is disabled as requested (no linked accounts).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Setup path before imports
from test_entrypoint_detector import detect_and_setup, EntrypointDetector
entrypoint_path, V15_ROOT = detect_and_setup()

# Import test suites
from test_function_1_ticker_analysis import run_function_1_tests
from test_function_3_continuous_learning import run_function_3_tests


class TestRunner:
    """Main test runner for Stock Analyzer V15."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results: Dict[str, Dict[str, bool]] = {}
        self.entrypoint_path = entrypoint_path
        self.V15_ROOT = V15_ROOT
        
    def run_all_tests(self) -> Dict[str, any]:
        """Run all test suites."""
        print("\n" + "=" * 80)
        print("STOCK ANALYZER V15 - INTENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Entrypoint: {self.entrypoint_path}")
        print(f"V15 Root: {self.V15_ROOT}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Detect entrypoint
        print("\n[SETUP] Entrypoint Detection")
        print("-" * 80)
        if self.entrypoint_path:
            print(f"  [OK] Entrypoint found: {self.entrypoint_path.name}")
        else:
            print("  [FAIL] Entrypoint not found!")
            return {"error": "Entrypoint not found"}
        
        # Run Function 1 tests
        print("\n" + "=" * 80)
        print("RUNNING FUNCTION 1 TESTS")
        print("=" * 80)
        try:
            self.results["function_1"] = run_function_1_tests()
        except Exception as e:
            print(f"\n[FAIL] Function 1 tests failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.results["function_1"] = {"error": str(e)}
        
        # Skip Function 2 (Trading) - disabled as requested
        print("\n" + "=" * 80)
        print("FUNCTION 2: AUTONOMOUS TRADING - DISABLED")
        print("=" * 80)
        print("  [WARN] Trading tests disabled (no linked accounts)")
        print("  [OK] Skipping Function 2 tests as requested")
        self.results["function_2"] = {"disabled": True, "reason": "No linked accounts"}
        
        # Run Function 3 tests
        print("\n" + "=" * 80)
        print("RUNNING FUNCTION 3 TESTS")
        print("=" * 80)
        try:
            self.results["function_3"] = run_function_3_tests()
        except Exception as e:
            print(f"\n[FAIL] Function 3 tests failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.results["function_3"] = {"error": str(e)}
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive test report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY REPORT")
        print("=" * 80)
        
        # Calculate statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        function_stats = {}
        
        for function_name, function_results in self.results.items():
            if isinstance(function_results, dict) and "error" not in function_results and "disabled" not in function_results:
                function_total = len(function_results)
                function_passed = sum(1 for v in function_results.values() if v)
                function_failed = function_total - function_passed
                
                total_tests += function_total
                passed_tests += function_passed
                failed_tests += function_failed
                
                function_stats[function_name] = {
                    "total": function_total,
                    "passed": function_passed,
                    "failed": function_failed,
                    "success_rate": (function_passed / function_total * 100) if function_total > 0 else 0
                }
        
        # Print summary
        print(f"\nTest Execution Time: {duration:.2f} seconds")
        print(f"\nOverall Statistics:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)" if total_tests > 0 else "  Passed: 0")
        print(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)" if total_tests > 0 else "  Failed: 0")
        
        # Print function-by-function breakdown
        print(f"\nFunction Breakdown:")
        print("-" * 80)
        
        for function_name, stats in function_stats.items():
            status_icon = "[OK]" if stats["failed"] == 0 else "[WARN]"
            print(f"{status_icon} {function_name.upper().replace('_', ' ')}")
            print(f"    Total: {stats['total']}")
            print(f"    Passed: {stats['passed']}")
            print(f"    Failed: {stats['failed']}")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
        
        # Print disabled functions
        for function_name, function_results in self.results.items():
            if isinstance(function_results, dict) and "disabled" in function_results:
                print(f"\n[WARN] {function_name.upper().replace('_', ' ')}: DISABLED")
                print(f"    Reason: {function_results.get('reason', 'Unknown')}")
        
        # Print detailed results
        print(f"\n" + "=" * 80)
        print("DETAILED TEST RESULTS")
        print("=" * 80)
        
        for function_name, function_results in self.results.items():
            if isinstance(function_results, dict) and "error" not in function_results and "disabled" not in function_results:
                print(f"\n{function_name.upper().replace('_', ' ')}:")
                print("-" * 80)
                for test_name, passed in function_results.items():
                    status = "PASS" if passed else "FAIL"
                    icon = "[OK]" if passed else "[FAIL]"
                    print(f"  {icon} {test_name:50s} {status}")
        
        # Create report dictionary
        report = {
            "entrypoint": str(self.entrypoint_path) if self.entrypoint_path else None,
            "V15_ROOT": str(self.V15_ROOT),
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "overall": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "function_stats": function_stats,
            "detailed_results": self.results
        }
        
        # Save report to file
        report_path = V15_ROOT / "test" / f"test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n[OK] Test report saved to: {report_path}")
        except Exception as e:
            print(f"\n[WARN] Could not save test report: {e}")
        
        return report
    
    def print_recommendations(self):
        """Print recommendations based on test results."""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        # Analyze results and provide recommendations
        recommendations = []
        
        for function_name, function_results in self.results.items():
            if isinstance(function_results, dict) and "error" not in function_results and "disabled" not in function_results:
                failed_tests = [name for name, passed in function_results.items() if not passed]
                
                if failed_tests:
                    recommendations.append(f"{function_name}: Review failed tests: {', '.join(failed_tests[:3])}")
        
        if recommendations:
            for rec in recommendations:
                print(f"  - {rec}")
        else:
            print("  [OK] All tests passed! System is functioning correctly.")
        
        print()


def main():
    """Main entry point for test runner."""
    # Set UTF-8 encoding for Windows console
    import io
    import sys
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    runner = TestRunner()
    report = runner.run_all_tests()
    runner.print_recommendations()
    
    # Exit with appropriate code
    if report.get("overall", {}).get("failed_tests", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

