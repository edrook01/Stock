"""
Wrapper script to run constant learning tests from parent directory.
This allows V14 to be imported as a proper package.
"""

import sys
from pathlib import Path

# Add parent to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

# Change to V14 directory
v14_dir = workspace_root / "Stock Report" / "V14"
import os
os.chdir(str(v14_dir))

# Now run the test
exec(open(v14_dir / "test_constant_learning.py").read())

