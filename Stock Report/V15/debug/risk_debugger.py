"""
Risk Calculation Debugger
Debug utilities for risk management calculations.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time

# Handle pandas and numpy imports with error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..risk.volatility import calculate_atr, calculate_atr_multiple_periods
from ..risk.stop_loss import calculate_stop_loss_distance, calculate_stop_loss_price, should_skip_trade
from ..risk.position_sizing import calculate_position_size, calculate_position_size_with_profile
from ..risk.exposure_tracker import ExposureTracker, Position as ExposurePosition
from ..risk.profiles import RiskProfile, get_profile_config
from ..risk.trailing_stop import create_trailing_stop, TrailingStop


class RiskDebugger:
    """Debug risk management calculations."""
    
    def __init__(self):
        """Initialize risk debugger."""
        pass
    
    def debug_atr_calculation(
        self,
        df: pd.DataFrame,
        periods: List[int] = [14, 20, 50]
    ) -> Dict[str, Any]:
        """
        Debug ATR calculation step-by-step.
        
        Args:
            df: Price DataFrame
            periods: List of periods to test
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_atr_calculation",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "data_points": len(df),
                "columns": list(df.columns),
                "periods": periods
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": [],
            "performance": {}
        }
        
        start_time = time.time()
        
        # Step 1: Validate input
        debug_info["steps"].append({
            "step": 1,
            "action": "Validate input DataFrame",
            "result": "OK" if df is not None and not df.empty else "FAILED"
        })
        
        if df is None or df.empty:
            debug_info["errors"].append("DataFrame is empty or None")
            debug_info["success"] = False
            return debug_info
        
        # Check required columns
        required_cols = ['High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            debug_info["errors"].append(f"Missing columns: {missing_cols}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Calculate ATR for each period
        atr_results = {}
        for period in periods:
            step_start = time.time()
            try:
                atr = calculate_atr(df, period=period)
                step_duration = (time.time() - step_start) * 1000
                
                atr_results[period] = {
                    "atr": float(atr),
                    "duration_ms": step_duration
                }
                
                debug_info["steps"].append({
                    "step": len(debug_info["steps"]) + 1,
                    "action": f"Calculate ATR(period={period})",
                    "result": f"ATR = {atr:.4f}",
                    "duration_ms": step_duration
                })
            except ValueError as e:
                debug_info["warnings"].append(f"ATR({period}) failed: {str(e)}")
                atr_results[period] = {"error": str(e)}
            except Exception as e:
                debug_info["errors"].append(f"ATR({period}) error: {str(e)}")
                atr_results[period] = {"error": str(e)}
        
        # Step 3: Calculate multiple periods at once
        try:
            multi_atr = calculate_atr_multiple_periods(df, periods)
            debug_info["steps"].append({
                "step": len(debug_info["steps"]) + 1,
                "action": "Calculate ATR for multiple periods",
                "result": f"Calculated {len(multi_atr)} periods"
            })
        except Exception as e:
            debug_info["warnings"].append(f"Multi-period ATR failed: {str(e)}")
        
        total_duration = (time.time() - start_time) * 1000
        debug_info["performance"] = {
            "total_duration_ms": total_duration,
            "slowest_step": max([s.get("duration_ms", 0) for s in debug_info["steps"]], default=0)
        }
        
        debug_info["output"] = {
            "atr_results": atr_results,
            "multi_atr": multi_atr if 'multi_atr' in locals() else {}
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_stop_loss(
        self,
        df: pd.DataFrame,
        profile: RiskProfile,
        confidence: float,
        asset_risk_category: str = "medium"
    ) -> Dict[str, Any]:
        """
        Debug stop-loss calculation.
        
        Args:
            df: Price DataFrame
            profile: Risk profile
            confidence: Model confidence
            asset_risk_category: Asset risk category
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_stop_loss",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "profile": profile.value,
                "confidence": confidence,
                "asset_risk_category": asset_risk_category
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Calculate ATR
        try:
            atr = calculate_atr(df, period=14)
            debug_info["steps"].append({
                "step": 1,
                "action": "Calculate ATR",
                "result": f"ATR = {atr:.4f}"
            })
        except Exception as e:
            debug_info["errors"].append(f"ATR calculation failed: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Get profile config
        config = get_profile_config(profile)
        atr_min, atr_max = config["atr_multiplier_min"], config["atr_multiplier_max"]
        debug_info["steps"].append({
            "step": 2,
            "action": "Get profile ATR multipliers",
            "result": f"Range: {atr_min}x to {atr_max}x"
        })
        
        # Step 3: Calculate stop distance
        try:
            stop_distance, atr_used = calculate_stop_loss_distance(
                df=df,
                profile=profile,
                confidence=confidence,
                asset_risk_category=asset_risk_category
            )
            debug_info["steps"].append({
                "step": 3,
                "action": "Calculate stop distance",
                "result": f"Distance = {stop_distance:.4f}, ATR = {atr_used:.4f}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Stop distance calculation failed: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 4: Calculate stop prices for both directions
        current_price = float(df['Close'].iloc[-1])
        long_stop = calculate_stop_loss_price(current_price, "LONG", stop_distance)
        short_stop = calculate_stop_loss_price(current_price, "SHORT", stop_distance)
        
        debug_info["steps"].append({
            "step": 4,
            "action": "Calculate stop prices",
            "result": f"LONG stop: {long_stop:.2f}, SHORT stop: {short_stop:.2f}"
        })
        
        # Step 5: Check if should skip trade
        should_skip = should_skip_trade(profile, confidence, asset_risk_category)
        debug_info["steps"].append({
            "step": 5,
            "action": "Check if should skip trade",
            "result": "SKIP" if should_skip else "PROCEED"
        })
        
        debug_info["output"] = {
            "atr": float(atr),
            "stop_distance": float(stop_distance),
            "current_price": current_price,
            "long_stop": float(long_stop),
            "short_stop": float(short_stop),
            "should_skip": should_skip
        }
        debug_info["success"] = len(debug_info["errors"]) == 0
        
        return debug_info
    
    def debug_position_sizing(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        profile: RiskProfile,
        confidence: float,
        direction: str = "LONG"
    ) -> Dict[str, Any]:
        """
        Debug position sizing calculation.
        
        Args:
            equity: Account equity
            entry_price: Entry price
            stop_price: Stop-loss price
            profile: Risk profile
            confidence: Model confidence
            direction: Trade direction
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_position_sizing",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "equity": equity,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "profile": profile.value,
                "confidence": confidence,
                "direction": direction
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Validate inputs
        if equity <= 0:
            debug_info["errors"].append("Equity must be positive")
            debug_info["success"] = False
            return debug_info
        
        if entry_price <= 0:
            debug_info["errors"].append("Entry price must be positive")
            debug_info["success"] = False
            return debug_info
        
        debug_info["steps"].append({
            "step": 1,
            "action": "Validate inputs",
            "result": "OK"
        })
        
        # Step 2: Calculate stop distance
        if direction.upper() == "LONG":
            stop_distance = entry_price - stop_price
        else:
            stop_distance = stop_price - entry_price
        
        if stop_distance <= 0:
            debug_info["errors"].append("Invalid stop price (must be below entry for LONG, above for SHORT)")
            debug_info["success"] = False
            return debug_info
        
        debug_info["steps"].append({
            "step": 2,
            "action": "Calculate stop distance",
            "result": f"Distance = {stop_distance:.4f}"
        })
        
        # Step 3: Get risk range for profile
        from ..risk.profiles import get_equity_risk_range
        risk_min, risk_max = get_equity_risk_range(profile)
        debug_info["steps"].append({
            "step": 3,
            "action": "Get risk range for profile",
            "result": f"Range: {risk_min}% to {risk_max}%"
        })
        
        # Step 4: Calculate position size with profile
        try:
            position_size, risk_amount, reason = calculate_position_size_with_profile(
                equity=equity,
                entry_price=entry_price,
                stop_price=stop_price,
                profile=profile,
                confidence=confidence,
                direction=direction
            )
            
            if position_size is None:
                debug_info["warnings"].append(f"Position sizing failed: {reason}")
                debug_info["output"] = {"reason": reason}
                debug_info["success"] = False
                return debug_info
            
            debug_info["steps"].append({
                "step": 4,
                "action": "Calculate position size",
                "result": f"Size = {position_size:.2f}, Risk = ${risk_amount:.2f}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Position sizing error: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 5: Calculate position value
        position_value = position_size * entry_price
        risk_percentage = (risk_amount / equity) * 100.0
        
        debug_info["steps"].append({
            "step": 5,
            "action": "Calculate position metrics",
            "result": f"Value = ${position_value:.2f}, Risk = {risk_percentage:.2f}%"
        })
        
        debug_info["output"] = {
            "position_size": float(position_size),
            "risk_amount": float(risk_amount),
            "position_value": float(position_value),
            "risk_percentage": float(risk_percentage),
            "stop_distance": float(stop_distance),
            "reason": reason
        }
        debug_info["success"] = True
        
        return debug_info
    
    def debug_exposure(
        self,
        equity: float,
        profile: RiskProfile,
        positions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Debug exposure tracking.
        
        Args:
            equity: Account equity
            profile: Risk profile
            positions: List of position dictionaries
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_exposure",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "equity": equity,
                "profile": profile.value,
                "position_count": len(positions)
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Create exposure tracker
        tracker = ExposureTracker(equity, profile)
        debug_info["steps"].append({
            "step": 1,
            "action": "Initialize exposure tracker",
            "result": "OK"
        })
        
        # Step 2: Add positions
        for i, pos_data in enumerate(positions):
            try:
                exp_position = ExposurePosition(
                    position_id=pos_data.get("position_id", f"pos_{i}"),
                    ticker=pos_data.get("ticker", "UNKNOWN"),
                    direction=pos_data.get("direction", "LONG"),
                    entry_price=pos_data.get("entry_price", 0.0),
                    quantity=pos_data.get("quantity", 0.0),
                    stop_price=pos_data.get("stop_price", 0.0),
                    current_price=pos_data.get("current_price", pos_data.get("entry_price", 0.0))
                )
                tracker.add_position(exp_position)
                debug_info["steps"].append({
                    "step": len(debug_info["steps"]) + 1,
                    "action": f"Add position {i+1}",
                    "result": f"Ticker: {exp_position.ticker}, Risk: ${exp_position.get_risk_amount():.2f}"
                })
            except Exception as e:
                debug_info["warnings"].append(f"Failed to add position {i+1}: {str(e)}")
        
        # Step 3: Calculate total exposure
        total_exposure = tracker.get_total_exposure()
        worst_case_loss = tracker.get_worst_case_loss()
        max_exposure = tracker.max_exposure
        
        debug_info["steps"].append({
            "step": len(debug_info["steps"]) + 1,
            "action": "Calculate total exposure",
            "result": f"Exposure: {total_exposure:.2f}%, Max: {max_exposure:.2f}%"
        })
        
        # Step 4: Check if new position can be opened
        test_risk = 100.0  # Test with $100 risk
        can_open, reason = tracker.can_open_new_position(test_risk)
        debug_info["steps"].append({
            "step": len(debug_info["steps"]) + 1,
            "action": "Test new position",
            "result": "CAN OPEN" if can_open else f"Cannot open: {reason}"
        })
        
        # Step 5: Get summary
        summary = tracker.get_summary()
        
        debug_info["output"] = {
            "total_exposure_pct": float(total_exposure),
            "worst_case_loss": float(worst_case_loss),
            "max_exposure": float(max_exposure),
            "exposure_remaining": float(max_exposure - total_exposure),
            "can_open_new": can_open,
            "summary": summary
        }
        debug_info["success"] = True
        
        return debug_info
    
    def debug_trailing_stop(
        self,
        entry_price: float,
        direction: str,
        initial_stop: float,
        df: pd.DataFrame,
        price_sequence: List[float]
    ) -> Dict[str, Any]:
        """
        Debug trailing stop behavior.
        
        Args:
            entry_price: Entry price
            direction: Trade direction
            initial_stop: Initial stop price
            df: Price DataFrame
            price_sequence: Sequence of prices to test
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_trailing_stop",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "entry_price": entry_price,
                "direction": direction,
                "initial_stop": initial_stop,
                "price_sequence_length": len(price_sequence)
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Create trailing stop
        try:
            trailing_stop = create_trailing_stop(
                entry_price=entry_price,
                direction=direction,
                initial_stop=initial_stop,
                df=df
            )
            debug_info["steps"].append({
                "step": 1,
                "action": "Create trailing stop",
                "result": f"Initial stop: {trailing_stop.current_stop:.4f}"
            })
        except Exception as e:
            debug_info["errors"].append(f"Failed to create trailing stop: {str(e)}")
            debug_info["success"] = False
            return debug_info
        
        # Step 2: Update with price sequence
        stop_history = []
        for i, price in enumerate(price_sequence):
            updated = trailing_stop.update(price)
            stop_history.append({
                "price": price,
                "stop": trailing_stop.current_stop,
                "updated": updated,
                "is_profitable": trailing_stop.is_profitable,
                "is_breakeven": trailing_stop.is_breakeven,
                "profit_locked": trailing_stop.get_profit_locked()
            })
            
            if updated:
                debug_info["steps"].append({
                    "step": len(debug_info["steps"]) + 1,
                    "action": f"Update with price {price:.2f}",
                    "result": f"Stop updated to {trailing_stop.current_stop:.4f}"
                })
            
            # Check if triggered
            if trailing_stop.is_triggered(price):
                debug_info["steps"].append({
                    "step": len(debug_info["steps"]) + 1,
                    "action": f"Check trigger at price {price:.2f}",
                    "result": "TRIGGERED"
                })
                break
        
        debug_info["output"] = {
            "initial_stop": float(initial_stop),
            "final_stop": float(trailing_stop.current_stop),
            "best_price": float(trailing_stop.best_price),
            "is_profitable": trailing_stop.is_profitable,
            "is_breakeven": trailing_stop.is_breakeven,
            "profit_locked": float(trailing_stop.get_profit_locked()),
            "stop_history": stop_history
        }
        debug_info["success"] = True
        
        return debug_info
    
    def debug_risk_profile(
        self,
        profile: RiskProfile,
        test_cases: List[Dict]
    ) -> Dict[str, Any]:
        """
        Debug risk profile settings and validation.
        
        Args:
            profile: Risk profile to test
            test_cases: List of test case dictionaries
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "test": "debug_risk_profile",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "profile": profile.value,
                "test_cases": len(test_cases)
            },
            "steps": [],
            "output": {},
            "errors": [],
            "warnings": []
        }
        
        # Step 1: Get profile config
        config = get_profile_config(profile)
        debug_info["steps"].append({
            "step": 1,
            "action": "Load profile configuration",
            "result": f"Config loaded: {len(config)} settings"
        })
        
        # Step 2: Test each test case
        results = []
        for i, test_case in enumerate(test_cases):
            asset_category = test_case.get("asset_risk_category", "medium")
            confidence = test_case.get("confidence", 0.7)
            
            from ..risk.profiles import is_asset_allowed, get_confidence_threshold
            
            is_allowed = is_asset_allowed(profile, asset_category)
            threshold = get_confidence_threshold(profile)
            meets_threshold = confidence >= threshold
            
            result = {
                "test_case": i + 1,
                "asset_category": asset_category,
                "confidence": confidence,
                "is_allowed": is_allowed,
                "meets_threshold": meets_threshold,
                "can_trade": is_allowed and meets_threshold
            }
            results.append(result)
            
            debug_info["steps"].append({
                "step": len(debug_info["steps"]) + 1,
                "action": f"Test case {i+1}",
                "result": "CAN TRADE" if result["can_trade"] else "BLOCKED"
            })
        
        debug_info["output"] = {
            "profile_config": config,
            "test_results": results,
            "summary": {
                "total_tests": len(test_cases),
                "allowed": sum(1 for r in results if r["can_trade"]),
                "blocked": sum(1 for r in results if not r["can_trade"])
            }
        }
        debug_info["success"] = True
        
        return debug_info


# Global risk debugger instance
_risk_debugger: Optional[RiskDebugger] = None


def get_risk_debugger() -> RiskDebugger:
    """Get global risk debugger instance."""
    global _risk_debugger
    if _risk_debugger is None:
        _risk_debugger = RiskDebugger()
    return _risk_debugger

