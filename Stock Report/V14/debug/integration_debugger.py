"""
Integration Debugger
Debug utilities for module integration and system-wide functionality.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import importlib
import time
from datetime import datetime

from ..core.portable_paths import get_path, get_root_path
from ..core.setup import initialize_v14, is_first_run
from ..core.portability_check import verify_portability, generate_portability_report


class IntegrationDebugger:
    """Debug integration and system-wide functionality."""
    
    def __init__(self):
        """Initialize integration debugger."""
        pass
    
    def debug_imports(self) -> Dict[str, Any]:
        """
        Debug all module imports.
        
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_imports",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Modules to test
        modules_to_test = [
            ("core.portable_paths", "get_path"),
            ("core.data_fetcher", "fetch_prices"),
            ("core.indicators", "rsi"),
            ("core.timeframes", "is_valid_timeframe"),
            ("risk.volatility", "calculate_atr"),
            ("risk.profiles", "RiskProfile"),
            ("risk.stop_loss", "calculate_stop_loss_distance"),
            ("risk.position_sizing", "calculate_position_size"),
            ("model.unified_model", "get_model"),
            ("model.feature_extractor", "FeatureExtractor"),
            ("learning.trade_tracker", "get_trade_tracker"),
            ("learning.feedback_loop", "get_feedback_loop"),
            ("sentiment.override", "get_sentiment_override"),
            ("logging.trade_logger", "get_trade_logger"),
            ("browser.automation", "BrowserAutomation"),
        ]
        
        successful_imports = []
        failed_imports = []
        
        for module_path, item_name in modules_to_test:
            try:
                module = importlib.import_module(module_path)
                item = getattr(module, item_name)
                successful_imports.append(module_path)
                debug_info["steps"].append({
                    "step": len(debug_info["steps"]) + 1,
                    "action": f"Import {module_path}.{item_name}",
                    "result": "SUCCESS"
                })
            except ImportError as e:
                failed_imports.append((module_path, str(e)))
                debug_info["errors"].append(f"Import failed: {module_path} - {str(e)}")
            except AttributeError as e:
                failed_imports.append((module_path, f"Missing {item_name}"))
                debug_info["warnings"].append(f"Attribute missing: {module_path}.{item_name}")
            except Exception as e:
                failed_imports.append((module_path, str(e)))
                debug_info["errors"].append(f"Unexpected error: {module_path} - {str(e)}")
        
        debug_info["output"] = {
            "successful_imports": len(successful_imports),
            "failed_imports": len(failed_imports),
            "success_rate": len(successful_imports) / len(modules_to_test) * 100,
            "failed_details": failed_imports
        }
        debug_info["success"] = len(failed_imports) == 0
        
        return debug_info
    
    def debug_config_loading(self) -> Dict[str, Any]:
        """
        Debug configuration loading.
        
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_config_loading",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Check config file exists
        config_file = get_path('data') / 'config_v14.json'
        exists = config_file.exists()
        debug_info["steps"].append({
            "step": 1,
            "action": "Check config file exists",
            "result": "EXISTS" if exists else "MISSING"
        })
        
        if not exists:
            debug_info["warnings"].append("Config file not found - will use defaults")
            debug_info["output"] = {"config_exists": False}
            debug_info["success"] = True
            return debug_info
        
        # Step 2: Try to load config
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            debug_info["steps"].append({
                "step": 2,
                "action": "Load config file",
                "result": "SUCCESS"
            })
        except json.JSONDecodeError as e:
            debug_info["errors"].append(f"Invalid JSON: {str(e)}")
            debug_info["success"] = False
            return debug_info
        except Exception as e:
            debug_info["errors"].append(f"Load error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 3: Validate config structure
        required_keys = ["version", "risk_profile", "model", "risk_management"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            debug_info["warnings"].append(f"Missing config keys: {missing_keys}")
        else:
            debug_info["steps"].append({
                "step": 3,
                "action": "Validate config structure",
                "result": "OK"
            })
        
        debug_info["output"] = {
            "config_exists": True,
            "config_keys": list(config.keys()),
            "missing_keys": missing_keys,
            "version": config.get("version")
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_portability(self) -> Dict[str, Any]:
        """
        Debug portability checks.
        
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_portability",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Run portability verification
        try:
            results = verify_portability()
            debug_info["steps"].append({
                "step": 1,
                "action": "Run portability check",
                "result": "PASS" if results["portable"] else "FAIL"
            })
        except Exception as e:
            debug_info["errors"].append(f"Portability check error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Check absolute paths
        if results["absolute_path_issues"]:
            debug_info["warnings"].extend(results["absolute_path_issues"])
            debug_info["steps"].append({
                "step": 2,
                "action": "Check absolute paths",
                "result": f"Found {len(results['absolute_path_issues'])} issues"
            })
        else:
            debug_info["steps"].append({
                "step": 2,
                "action": "Check absolute paths",
                "result": "No issues found"
            })
        
        # Step 3: Check data locations
        data_locations = results["data_locations"]
        missing_locations = [loc for loc, exists in data_locations.items() if not exists]
        if missing_locations:
            debug_info["warnings"].append(f"Missing data locations: {missing_locations}")
        
        debug_info["steps"].append({
            "step": 3,
            "action": "Check data locations",
            "result": f"{len(data_locations) - len(missing_locations)}/{len(data_locations)} exist"
        })
        
        # Step 4: Generate report
        try:
            report = generate_portability_report()
            debug_info["steps"].append({
                "step": 4,
                "action": "Generate portability report",
                "result": f"Report generated ({len(report)} chars)"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Report generation error: {str(e)}")
        
        debug_info["output"] = results
        debug_info["success"] = results["portable"]
        
        return debug_info
    
    def debug_module_communication(
        self,
        test_scenarios: List[str]
    ) -> Dict[str, Any]:
        """
        Debug inter-module communication.
        
        Args:
            test_scenarios: List of scenarios to test
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_module_communication",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "scenarios": test_scenarios
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Test scenario: Model -> Risk -> Logging
        if "model_to_risk" in test_scenarios:
            try:
                from ..model.unified_model import get_model
                from ..risk.profiles import RiskProfile
                from ..risk.stop_loss import calculate_stop_loss_distance
                import pandas as pd
                
                # Create test data
                df = pd.DataFrame({
                    'High': [100, 101, 102],
                    'Low': [99, 100, 101],
                    'Close': [100, 101, 102]
                })
                
                # Model would generate prediction (simplified)
                profile = RiskProfile.MEDIUM
                confidence = 0.7
                
                # Risk calculation
                stop_distance, atr = calculate_stop_loss_distance(
                    df=df,
                    profile=profile,
                    confidence=confidence
                )
                
                debug_info["steps"].append({
                    "step": 1,
                    "action": "Test Model -> Risk communication",
                    "result": f"Stop distance: {stop_distance:.4f}"
                })
            except Exception as e:
                debug_info["errors"].append(f"Model->Risk test failed: {str(e)}")
        
        # Test scenario: Risk -> Learning
        if "risk_to_learning" in test_scenarios:
            try:
                from ..risk.exposure_tracker import ExposureTracker
                from ..learning.trade_tracker import get_trade_tracker
                
                tracker = get_trade_tracker()
                stats = tracker.get_statistics()
                
                debug_info["steps"].append({
                    "step": len(debug_info["steps"]) + 1,
                    "action": "Test Risk -> Learning communication",
                    "result": f"Trade stats available: {stats['total_trades']} trades"
                })
            except Exception as e:
                debug_info["warnings"].append(f"Risk->Learning test warning: {str(e)}")
        
        debug_info["output"] = {
            "scenarios_tested": len(test_scenarios),
            "communication_working": len(debug_info["errors"]) == 0
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_setup_process(self) -> Dict[str, Any]:
        """
        Debug first-run setup process.
        
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_setup_process",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Check if first run
        is_first = is_first_run()
        debug_info["steps"].append({
            "step": 1,
            "action": "Check if first run",
            "result": "YES" if is_first else "NO"
        })
        
        # Step 2: Test initialization
        try:
            result = initialize_v14()
            debug_info["steps"].append({
                "step": 2,
                "action": "Run initialization",
                "result": "SUCCESS" if result.get("initialized") else "PARTIAL"
            })
        except Exception as e:
            debug_info["errors"].append(f"Initialization error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 3: Verify directories created
        from ..core.portable_paths import initialize_structure
        initialize_structure()
        
        required_dirs = ['data', 'logs', 'memory', 'history', 'model', 'cache']
        created_dirs = []
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = get_path(dir_name)
            if dir_path.exists():
                created_dirs.append(dir_name)
            else:
                missing_dirs.append(dir_name)
        
        debug_info["steps"].append({
            "step": 3,
            "action": "Verify directories",
            "result": f"{len(created_dirs)}/{len(required_dirs)} created"
        })
        
        if missing_dirs:
            debug_info["warnings"].append(f"Missing directories: {missing_dirs}")
        
        debug_info["output"] = {
            "is_first_run": is_first,
            "initialization_result": result,
            "directories_created": len(created_dirs),
            "directories_missing": len(missing_dirs)
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info


# Global integration debugger instance
_integration_debugger: Optional[IntegrationDebugger] = None


def get_integration_debugger() -> IntegrationDebugger:
    """Get global integration debugger instance."""
    global _integration_debugger
    if _integration_debugger is None:
        _integration_debugger = IntegrationDebugger()
    return _integration_debugger

