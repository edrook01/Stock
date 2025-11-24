#!/bin/bash
# Shell script to run test automation on Linux/Mac
# Usage: ./run_tests.sh

echo "========================================"
echo "Stock Analyzer V15 - Test Automation"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."
python cursor/test_automation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "All tests passed!"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Tests failed. Check the summary file."
    echo "========================================"
fi

