"""
Diagnostic Analysis
Analyzes failed trades to determine causes and generate reports.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from .failure_tracker import get_failure_tracker
from ..risk.volatility import calculate_atr
from ..core.data_fetcher import fetch_prices
import asyncio


class TradeDiagnostic:
    """Performs diagnostic analysis on failed trades."""
    
    def __init__(self):
        """Initialize trade diagnostic."""
        self.failure_tracker = get_failure_tracker()
    
    async def diagnose_failure(self, failure: Dict) -> Dict:
        """
        Diagnose a failed trade to determine cause.
        
        Args:
            failure: Failure dictionary from FailureTracker
            
        Returns:
            Dictionary with diagnostic results
        """
        diagnosis = {
            "trade_id": failure.get("trade_id"),
            "timestamp": datetime.now().isoformat(),
            "causes": [],
            "severity": "medium",
            "recommendations": []
        }
        
        ticker = failure.get("ticker")
        entry_price = failure.get("entry_price")
        exit_price = failure.get("exit_price")
        entry_time = failure.get("entry_time")
        
        # Check for volatility spikes
        if ticker and entry_time:
            try:
                # Fetch price data around the failure time
                df = await fetch_prices(ticker, "1h")
                if df is not None and len(df) > 0:
                    # Calculate ATR
                    try:
                        atr = calculate_atr(df, period=14)
                        
                        # Check if there was a volatility spike
                        if entry_price and exit_price:
                            price_move = abs(exit_price - entry_price)
                            if price_move > atr * 3:  # Large move relative to ATR
                                diagnosis["causes"].append({
                                    "type": "volatility_spike",
                                    "description": f"Large price move ({price_move:.2f}) relative to ATR ({atr:.2f})",
                                    "severity": "high"
                                })
                                diagnosis["severity"] = "high"
                    except ValueError:
                        pass
            except Exception:
                pass
        
        # Check for slippage
        slippage = failure.get("slippage", 0.0)
        if slippage > 0:
            planned_loss = failure.get("planned_loss", 0.0)
            if slippage > planned_loss * 0.3:  # Significant slippage
                diagnosis["causes"].append({
                    "type": "slippage",
                    "description": f"Significant slippage: ${slippage:.2f}",
                    "severity": "medium"
                })
                diagnosis["recommendations"].append("Consider using limit orders instead of market orders")
        
        # Check failure type
        failure_type = failure.get("failure_type", "")
        if failure_type == "execution_error":
            diagnosis["causes"].append({
                "type": "execution_error",
                "description": "Execution error - loss much larger than planned",
                "severity": "high"
            })
            diagnosis["severity"] = "high"
            diagnosis["recommendations"].append("Review execution system and order routing")
        
        # Check if stop-loss was placed correctly
        planned_stop = failure.get("planned_stop")
        exit_price = failure.get("exit_price")
        if planned_stop and exit_price:
            direction = failure.get("direction", "LONG").upper()
            
            if direction == "LONG":
                if exit_price < planned_stop:
                    diagnosis["causes"].append({
                        "type": "stop_not_honored",
                        "description": f"Price fell below stop ({planned_stop:.2f}) to {exit_price:.2f}",
                        "severity": "high"
                    })
                    diagnosis["severity"] = "high"
                    diagnosis["recommendations"].append("Consider using guaranteed stop-loss orders")
            else:  # SHORT
                if exit_price > planned_stop:
                    diagnosis["causes"].append({
                        "type": "stop_not_honored",
                        "description": f"Price rose above stop ({planned_stop:.2f}) to {exit_price:.2f}",
                        "severity": "high"
                    })
                    diagnosis["severity"] = "high"
                    diagnosis["recommendations"].append("Consider using guaranteed stop-loss orders")
        
        # Generate recommendations based on causes
        if not diagnosis["recommendations"]:
            if diagnosis["severity"] == "high":
                diagnosis["recommendations"].append("Review risk management parameters")
                diagnosis["recommendations"].append("Consider reducing position size for this asset")
            else:
                diagnosis["recommendations"].append("Monitor similar trades closely")
        
        return diagnosis
    
    async def diagnose_all_failures(self) -> List[Dict]:
        """
        Diagnose all failed trades.
        
        Returns:
            List of diagnostic dictionaries
        """
        failures = self.failure_tracker.get_failed_trades()
        diagnostics = []
        
        for failure in failures:
            diagnosis = await self.diagnose_failure(failure)
            diagnostics.append(diagnosis)
        
        return diagnostics
    
    def generate_diagnostic_report(self, diagnostics: List[Dict]) -> str:
        """
        Generate a text diagnostic report.
        
        Args:
            diagnostics: List of diagnostic dictionaries
            
        Returns:
            Formatted diagnostic report string
        """
        if not diagnostics:
            return "No failures to diagnose."
        
        report = f"""
Trade Failure Diagnostic Report
{'=' * 60}
Total Failures Analyzed: {len(diagnostics)}

"""
        
        high_severity = [d for d in diagnostics if d.get("severity") == "high"]
        medium_severity = [d for d in diagnostics if d.get("severity") == "medium"]
        low_severity = [d for d in diagnostics if d.get("severity") == "low"]
        
        report += f"Severity Breakdown:\n"
        report += f"  High: {len(high_severity)}\n"
        report += f"  Medium: {len(medium_severity)}\n"
        report += f"  Low: {len(low_severity)}\n\n"
        
        # Common causes
        all_causes = []
        for d in diagnostics:
            all_causes.extend(d.get("causes", []))
        
        cause_types = {}
        for cause in all_causes:
            ctype = cause.get("type", "unknown")
            cause_types[ctype] = cause_types.get(ctype, 0) + 1
        
        if cause_types:
            report += "Common Causes:\n"
            for ctype, count in sorted(cause_types.items(), key=lambda x: x[1], reverse=True):
                report += f"  {ctype}: {count}\n"
            report += "\n"
        
        # Recommendations
        all_recommendations = []
        for d in diagnostics:
            all_recommendations.extend(d.get("recommendations", []))
        
        if all_recommendations:
            unique_recommendations = list(set(all_recommendations))
            report += "Recommendations:\n"
            for rec in unique_recommendations[:10]:  # Top 10
                report += f"  - {rec}\n"
        
        return report


# Global diagnostic instance
_diagnostic: Optional[TradeDiagnostic] = None


def get_diagnostic() -> TradeDiagnostic:
    """Get global diagnostic instance."""
    global _diagnostic
    if _diagnostic is None:
        _diagnostic = TradeDiagnostic()
    return _diagnostic

