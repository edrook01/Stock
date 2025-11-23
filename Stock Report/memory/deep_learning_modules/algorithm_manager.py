#!/usr/bin/env python3
"""
Algorithm Manager - Self-Contained Module
Manages swappable learned algorithms.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Optional, Any


class AlgorithmManager:
    """Manages multiple algorithm instances."""
    
    def __init__(self):
        self.algorithms = {}
        self.algorithm_performance = {}
    
    def discover_algorithms(self, directory: str = "memory") -> List[str]:
        """Discover algorithm files (algorithm1.py, algorithm2.py, etc.)."""
        algorithm_files = []
        for filename in os.listdir(directory):
            if filename.startswith("algorithm") and filename.endswith(".py"):
                algorithm_files.append(os.path.join(directory, filename))
        return sorted(algorithm_files)
    
    def load_algorithm(self, filepath: str) -> Optional[Any]:
        """Load an algorithm module."""
        try:
            spec = importlib.util.spec_from_file_location("algorithm_module", filepath)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            return None
    
    def get_best_algorithm(self) -> Optional[Any]:
        """Get best performing algorithm."""
        if not self.algorithm_performance:
            return None
        best = max(self.algorithm_performance.items(), key=lambda x: x[1])
        return self.algorithms.get(best[0])


# Global algorithm manager instance
_algorithm_manager = None


def get_algorithm_manager() -> AlgorithmManager:
    """Get or create global algorithm manager instance."""
    global _algorithm_manager
    if _algorithm_manager is None:
        _algorithm_manager = AlgorithmManager()
    return _algorithm_manager
