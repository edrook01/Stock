#!/usr/bin/env python3
"""
Simple test runner script.
Run this from the V14 directory or test directory.
"""

import sys
from pathlib import Path

# Add parent directory to path if running from test directory
if Path(__file__).parent.name == "test":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from test.test_runner import main

if __name__ == "__main__":
    main()

