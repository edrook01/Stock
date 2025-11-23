"""
Stock Analyzer V14 - Main Entry Point
Comprehensive trading system with unified ML model, browser automation, and adaptive learning.
"""

import sys
import traceback
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
try:
    import logging  # Standard library logging
    import asyncio  # Uses logging internally
except Exception as e:
    print(f"ERROR: Failed to import standard library modules: {e}")
    print("\nThis may be due to module shadowing. Please check for local directories")
    print("named 'logging' or 'asyncio' that might conflict with standard library.")
    input("\nPress Enter to exit...")
    sys.exit(1)

# Now add V14 back to path (after critical imports are done)
sys.path.insert(0, script_dir)

# Import V14 modules with error handling
try:
    from core.setup import initialize_v14, is_first_run
    from core.portable_paths import initialize_structure
    from ui.menu_v14 import MenuController
except ImportError as e:
    print(f"\nERROR: Failed to import V14 modules: {e}")
    print("\nThis may indicate:")
    print("1. Missing dependencies (run: pip install -r requirements.txt)")
    print("2. Incorrect Python path configuration")
    print("3. Corrupted module files")
    print("\nFull error details:")
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: Unexpected error during import: {e}")
    print("\nFull error details:")
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)


def main():
    """Main entry point for Stock Analyzer V14."""
    # #region agent log
    import json
    from datetime import datetime
    # Use correct debug log path based on workspace location
    debug_log_path = V14_ROOT / '.cursor' / 'debug.log'
    try:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"Stock Analyzer V14.py:main","message":"Function entry","data":{"v14_root":str(V14_ROOT)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass  # Debug logging is optional, don't fail if it doesn't work
    # #endregion
    
    print("=" * 70)
    print("  STOCK ANALYZER V14")
    print("  Unified ML Model | Browser Automation | Adaptive Learning")
    print("=" * 70)
    
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V14.py:main","message":"Before initialize_structure","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    # Initialize directory structure
    try:
        initialize_structure()
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V14.py:main","message":"After initialize_structure","data":{"success":True},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
    except Exception as e:
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V14.py:main","message":"initialize_structure failed","data":{"error":str(e)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        print(f"\nERROR: Failed to initialize directory structure: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"Before is_first_run","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    # Check if first run
    try:
        is_first = is_first_run()
    except Exception as e:
        print(f"\nWARNING: Failed to check first run status: {e}")
        print("Assuming first run...")
        is_first = True
    
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"After is_first_run","data":{"is_first_run":is_first},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    if is_first:
        print("\nFirst run detected. Initializing V14...")
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"Before initialize_v14","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        try:
            result = initialize_v14()
            # #region agent log
            try:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V14.py:main","message":"After initialize_v14","data":{"result":result},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
            if result.get("initialized"):
                print("✓ V14 initialized successfully!")
            else:
                print("⚠ Initialization completed with warnings.")
        except Exception as e:
            print(f"\nWARNING: Initialization failed: {e}")
            print("Continuing anyway...")
            traceback.print_exc()
    
    # Initialize menu system
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Stock Analyzer V14.py:main","message":"Before MenuController","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    try:
        menu = MenuController()
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Stock Analyzer V14.py:main","message":"MenuController created","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        menu.run()
    except KeyboardInterrupt:
        print("\n\nExiting Stock Analyzer V14...")
        sys.exit(0)
    except Exception as e:
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"Stock Analyzer V14.py:main","message":"Fatal error","data":{"error":str(e),"type":type(e).__name__},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        print(f"\n\nFATAL ERROR: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("The program encountered a fatal error and must exit.")
        print("Please check the error message above and:")
        print("1. Verify all dependencies are installed")
        print("2. Check that all required files are present")
        print("3. Review the error traceback for details")
        print("=" * 70)
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: Program failed to start: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("The program could not start. This usually indicates:")
        print("1. Missing or corrupted Python installation")
        print("2. Module import failures")
        print("3. System configuration issues")
        print("=" * 70)
        input("\nPress Enter to exit...")
        sys.exit(1)

