"""
Stock Analyzer V15 - Main Entry Point
Comprehensive trading system with unified ML model, browser automation, and adaptive learning.
"""

import sys
import json
import time
import traceback
from pathlib import Path

AGENT_LOG_PATH = Path(r"c:\Users\edwar\Documents\GitHub\.cursor\debug.log")
AGENT_SESSION_ID = "debug-session"
AGENT_RUN_ID = "pre-fix"


def _agent_log(hypothesis_id: str, location: str, message: str, data=None) -> None:
    """Append a single NDJSON instrumentation log entry."""
    payload = {
        "sessionId": AGENT_SESSION_ID,
        "runId": AGENT_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(AGENT_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")
    except Exception:
        pass

# CRITICAL FIX: Prevent local 'logging' directory from shadowing standard library
# Python automatically adds script directory to sys.path, causing our local
# 'logging' directory to shadow the standard library 'logging' module.
# Solution: Temporarily remove script directory, import standard library modules,
# then add it back.
V15_ROOT = Path(__file__).parent
script_dir = str(V15_ROOT)

# Remove script directory from sys.path temporarily (if present)
script_dir_removed = False
if script_dir in sys.path:
    sys.path.remove(script_dir)
    script_dir_removed = True
# #region agent log
_agent_log(
    "H1",
    "Stock Analyzer V15.py:22",
    "Removed script directory from sys.path",
    {"removed": script_dir_removed, "sys_path_head": sys.path[:3]},
)
# #endregion

# Import standard library modules that might be shadowed
try:
    import logging  # Standard library logging
    import asyncio  # Uses logging internally
    # #region agent log
    _agent_log(
        "H2",
        "Stock Analyzer V15.py:31",
        "Standard logging module state",
        {
            "logging_file": getattr(logging, "__file__", None),
            "has_getLogger": hasattr(logging, "getLogger"),
        },
    )
    # #endregion
except Exception as e:
    print(f"ERROR: Failed to import standard library modules: {e}")
    print("\nThis may be due to module shadowing. Please check for local directories")
    print("named 'logging' or 'asyncio' that might conflict with standard library.")
    input("\nPress Enter to exit...")
    sys.exit(1)

# Now add V15 back to path (after critical imports are done)
sys.path.insert(0, script_dir)
# #region agent log
_agent_log(
    "H3",
    "Stock Analyzer V15.py:41",
    "Reinserted script directory",
    {"sys_path_head": sys.path[:3]},
)
# #endregion

# Import V15 modules with error handling
try:
    from core.setup import initialize_v15, is_first_run
    from core.portable_paths import initialize_structure
    from ui.menu_v15 import MenuController
    # Import error logger (after path is set up)
    try:
        from sa_logging.error_logger import log_exception, log_error, log_warning, log_info
    except ImportError:
        # Fallback if error_logger not available
        def log_exception(*args, **kwargs): pass
        def log_error(*args, **kwargs): pass
        def log_warning(*args, **kwargs): pass
        def log_info(*args, **kwargs): pass
    # #region agent log
    
    _agent_log(
        "H2",
        
        "Stock Analyzer V15.py:58",
        "Imported V15 modules successfully",
        {"logging_module": getattr(logging, "__file__", None)},
    )
    # #endregion
except ImportError as e:
    # Try to log before exit
    try:
        from sa_logging.error_logger import log_exception
        log_exception(
            "Failed to import V15 modules",
            e,
            component="main",
            function="import",
            is_hard_error=True
        )
    except:
        pass
    print(f"\nERROR: Failed to import V15 modules: {e}")
    print("\nThis may indicate:")
    print("1. Missing dependencies (run: pip install -r requirements.txt)")
    print("2. Incorrect Python path configuration")
    print("3. Corrupted module files")
    print("\nFull error details:")
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)
except Exception as e:
    # Try to log before exit
    try:
        from sa_logging.error_logger import log_exception
        log_exception(
            "Unexpected error during import",
            e,
            component="main",
            function="import",
            is_hard_error=True
        )
    except:
        pass
    print(f"\nERROR: Unexpected error during import: {e}")
    print("\nFull error details:")
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)


def main():
    """Main entry point for Stock Analyzer V15."""
    # #region agent log
    import json
    from datetime import datetime
    # Use correct debug log path based on workspace location
    debug_log_path = V15_ROOT / '.cursor' / 'debug.log'
    try:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"Stock Analyzer V15.py:main","message":"Function entry","data":{"V15_ROOT":str(V15_ROOT)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass  # Debug logging is optional, don't fail if it doesn't work
    # #endregion
    
    print("=" * 70)
    print("  STOCK ANALYZER V15")
    print("  Unified ML Model | Browser Automation | Adaptive Learning")
    print("=" * 70)
    
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V15.py:main","message":"Before initialize_structure","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    # Initialize directory structure
    try:
        initialize_structure()
        log_info("Directory structure initialized", component="main", function="initialize_structure")
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V15.py:main","message":"After initialize_structure","data":{"success":True},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
    except Exception as e:
        log_exception(
            "Failed to initialize directory structure",
            e,
            component="main",
            function="initialize_structure",
            is_hard_error=True
        )
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Stock Analyzer V15.py:main","message":"initialize_structure failed","data":{"error":str(e)},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
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
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V15.py:main","message":"Before is_first_run","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    # Check if first run
    try:
        is_first = is_first_run()
    except Exception as e:
        log_warning(
            "Failed to check first run status, assuming first run",
            component="main",
            function="is_first_run",
            context={"error": str(e)}
        )
        print(f"\nWARNING: Failed to check first run status: {e}")
        print("Assuming first run...")
        is_first = True
    
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V15.py:main","message":"After is_first_run","data":{"is_first_run":is_first},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    if is_first:
        print("\nFirst run detected. Initializing V15...")
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V15.py:main","message":"Before initialize_v15","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        try:
            result = initialize_v15()
            # #region agent log
            try:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Stock Analyzer V15.py:main","message":"After initialize_v15","data":{"result":result},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
            if result.get("initialized"):
                print("✓ V15 initialized successfully!")
                log_info("V15 initialized successfully", component="main", function="initialize_v15")
            else:
                print("⚠ Initialization completed with warnings.")
                log_warning("Initialization completed with warnings", component="main", function="initialize_v15")
        except Exception as e:
            log_exception(
                "Initialization failed, continuing anyway",
                e,
                component="main",
                function="initialize_v15",
                is_hard_error=False
            )
            print(f"\nWARNING: Initialization failed: {e}")
            print("Continuing anyway...")
            traceback.print_exc()
    
    # Initialize menu system
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Stock Analyzer V15.py:main","message":"Before MenuController","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    
    try:
        menu = MenuController()
        log_info("MenuController created", component="main", function="main")
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Stock Analyzer V15.py:main","message":"MenuController created","data":{},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        menu.run()
    except KeyboardInterrupt:
        log_info("User interrupted program", component="main", function="main")
        print("\n\nExiting Stock Analyzer V15...")
        sys.exit(0)
    except Exception as e:
        log_exception(
            "Fatal error in main menu loop",
            e,
            component="main",
            function="main",
            is_hard_error=True
        )
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"Stock Analyzer V15.py:main","message":"Fatal error","data":{"error":str(e),"type":type(e).__name__},"timestamp":int(datetime.now().timestamp()*1000)})+"\n")
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
        print("4. Check logs/error.log for detailed error information")
        print("=" * 70)
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Try to log critical error
        try:
            from sa_logging.error_logger import log_exception
            log_exception(
                "CRITICAL: Program failed to start",
                e,
                component="main",
                function="__main__",
                is_hard_error=True
            )
        except:
            pass
        print(f"\n\nCRITICAL ERROR: Program failed to start: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("The program could not start. This usually indicates:")
        print("1. Missing or corrupted Python installation")
        print("2. Module import failures")
        print("3. System configuration issues")
        print("4. Check logs/error.log for detailed error information")
        print("=" * 70)
        input("\nPress Enter to exit...")
        sys.exit(1)

