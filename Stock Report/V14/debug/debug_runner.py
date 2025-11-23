"""
Debug Runner
Run all debuggers and generate comprehensive reports.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from .prediction_debugger import get_prediction_debugger
from .risk_debugger import get_risk_debugger
from .browser_debugger import get_browser_debugger
from .learning_debugger import get_learning_debugger
from .integration_debugger import get_integration_debugger
from .sentiment_debugger import get_sentiment_debugger
from .model_debugger import get_model_debugger
from ..core.portable_paths import get_path


class DebugRunner:
    """Run all debuggers."""
    
    def __init__(self):
        """Initialize debug runner."""
        self.debuggers = {
            "prediction": get_prediction_debugger(),
            "risk": get_risk_debugger(),
            "browser": get_browser_debugger(),
            "learning": get_learning_debugger(),
            "integration": get_integration_debugger(),
            "sentiment": get_sentiment_debugger(),
            "model": get_model_debugger()
        }
    
    async def run_all_debuggers(
        self,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run all debuggers.
        
        Args:
            options: Optional configuration for debuggers
            
        Returns:
            Dictionary with all debug results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "debuggers_run": [],
            "results": {},
            "summary": {}
        }
        
        # Run each debugger
        for name, debugger in self.debuggers.items():
            try:
                if name == "prediction":
                    # Would need ticker and df
                    pass
                elif name == "integration":
                    result = debugger.debug_imports()
                    results["results"][name] = result
                    results["debuggers_run"].append(name)
                # Add other debuggers as needed
            except Exception as e:
                results["results"][name] = {"error": str(e)}
        
        # Generate summary
        results["summary"] = {
            "total_run": len(results["debuggers_run"]),
            "successful": sum(1 for r in results["results"].values() if r.get("success", False)),
            "failed": sum(1 for r in results["results"].values() if not r.get("success", False))
        }
        
        return results
    
    def save_debug_report(
        self,
        results: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save debug report to file.
        
        Args:
            results: Debug results dictionary
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_report_{timestamp}.json"
        
        report_dir = get_path('logs') / 'debug_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / filename
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return report_file


def get_debug_runner() -> DebugRunner:
    """Get global debug runner instance."""
    global _debug_runner
    if _debug_runner is None:
        _debug_runner = DebugRunner()
    return _debug_runner

