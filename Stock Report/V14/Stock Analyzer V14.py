"""
Stock Analyzer V14 - Main Entry Point
Comprehensive trading system with unified ML model, browser automation, and adaptive learning.
"""

import sys
from pathlib import Path

# CRITICAL FIX: Prevent local 'logging' directory from shadowing standard library
# Python automatically adds script directory to sys.path, causing our local
# 'logging' directory to shadow the standard library 'logging' module.
# Solution: Temporarily remove script directory, import standard library modules,
# then add it back.
V14_ROOT = Path(__file__).parent
script_dir = str(V14_ROOT)

# Remove script directory from sys.path temporarily (if present)
if script_dir in sys.path:
    sys.path.remove(script_dir)

# Import standard library modules that might be shadowed
import logging  # Standard library logging
import asyncio  # Uses logging internally

# Now add V14 back to path (after critical imports are done)
sys.path.insert(0, script_dir)

from core.setup import initialize_v14, is_first_run
from core.portable_paths import initialize_structure
from ui.menu_v14 import MenuController


def main():
    """Main entry point for Stock Analyzer V14."""
    # #region agent log
    import json
    from datetime import datetime
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"Stock Analyzer V14.py:main","message":"Function entry","data":{"v14_root":str(V14_ROOT)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    print("=" * 70)
    print("  STOCK ANALYZER V14")
    print("  Unified ML Model | Browser Automation | Adaptive Learning")
    print("=" * 70)
    
    # #region agent log
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V14.py:main","message":"Before initialize_structure","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    # Initialize directory structure
    try:
        initialize_structure()
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V14.py:main","message":"After initialize_structure","data":{"success":True},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
    except Exception as e:
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V14.py:main","message":"initialize_structure failed","data":{"error":str(e)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
        raise
    
    # #region agent log
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"Before is_first_run","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    # Check if first run
    is_first = is_first_run()
    
    # #region agent log
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"After is_first_run","data":{"is_first_run":is_first},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    if is_first:
        print("\nFirst run detected. Initializing V14...")
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"Before initialize_v14","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
        result = initialize_v14()
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"After initialize_v14","data":{"result":result},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
        if result.get("initialized"):
            print("✓ V14 initialized successfully!")
        else:
            print("⚠ Initialization completed with warnings.")
    
    # Initialize menu system
    # #region agent log
    try:
        with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Stock Analyzer V14.py:main","message":"Before MenuController","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except: pass
    # #endregion
    
    try:
        menu = MenuController()
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Stock Analyzer V14.py:main","message":"MenuController created","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
        menu.run()
    except KeyboardInterrupt:
        print("\n\nExiting Stock Analyzer V14...")
        sys.exit(0)
    except Exception as e:
        # #region agent log
        try:
            with open(r'c:\Users\edwar\Documents\Stock Report\V14\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"Stock Analyzer V14.py:main","message":"Fatal error","data":{"error":str(e),"type":type(e).__name__},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except: pass
        # #endregion
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

