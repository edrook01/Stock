@echo off
REM Batch script to run test automation on Windows
REM Usage: run_tests.bat

echo ========================================
echo Stock Analyzer V15 - Test Automation
echo ========================================
echo.

cd /d "%~dp0\.."
python cursor\test_automation.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo All tests passed!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Tests failed. Check the summary file.
    echo ========================================
)

pause

