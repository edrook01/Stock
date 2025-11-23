"""
Entrypoint Detection Utility
Automatically identifies the main entrypoint for the Stock Analyzer program.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple


class EntrypointDetector:
    """Detects the main entrypoint file for Stock Analyzer V14."""
    
    # Common entrypoint filenames to search for
    ENTRYPOINT_CANDIDATES = [
        "Stock Analyzer V14.py",
        "stock_analyzer_v14.py",
        "main.py",
        "app.py",
        "run.py"
    ]
    
    @classmethod
    def find_entrypoint(cls, search_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[str]]:
        """
        Find the main entrypoint file.
        
        Args:
            search_dir: Directory to search in (defaults to test directory's parent)
            
        Returns:
            Tuple of (entrypoint_path, entrypoint_name) or (None, None) if not found
        """
        if search_dir is None:
            # Default: search in parent of test directory (V14 root)
            test_dir = Path(__file__).parent
            search_dir = test_dir.parent
        
        search_dir = Path(search_dir).resolve()
        
        # First, try exact matches in the search directory
        for candidate in cls.ENTRYPOINT_CANDIDATES:
            candidate_path = search_dir / candidate
            if candidate_path.exists() and candidate_path.is_file():
                # Verify it has main() or if __name__ == "__main__"
                if cls._is_valid_entrypoint(candidate_path):
                    return candidate_path, candidate
        
        # Search recursively for entrypoint files
        for candidate in cls.ENTRYPOINT_CANDIDATES:
            for path in search_dir.rglob(candidate):
                if path.is_file() and cls._is_valid_entrypoint(path):
                    return path, candidate
        
        return None, None
    
    @classmethod
    def _is_valid_entrypoint(cls, file_path: Path) -> bool:
        """
        Check if a file is a valid entrypoint.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file appears to be a valid entrypoint
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for common entrypoint patterns
            has_main = 'def main(' in content or 'if __name__ == "__main__"' in content
            has_stock_analyzer = 'stock' in content.lower() and 'analyzer' in content.lower()
            
            # Must have main function or main guard
            return has_main and (has_stock_analyzer or 'v14' in content.lower())
        except Exception:
            return False
    
    @classmethod
    def get_v14_root(cls) -> Path:
        """
        Get the V14 root directory.
        
        Returns:
            Path to V14 root directory
        """
        test_dir = Path(__file__).parent
        return test_dir.parent
    
    @classmethod
    def setup_sys_path(cls, v14_root: Optional[Path] = None) -> None:
        """
        Setup sys.path to include V14 root for imports.
        
        Args:
            v14_root: V14 root directory (auto-detected if None)
        """
        if v14_root is None:
            v14_root = cls.get_v14_root()
        
        v14_root_str = str(v14_root.resolve())
        
        # Remove if already present to avoid duplicates
        if v14_root_str in sys.path:
            sys.path.remove(v14_root_str)
        
        # Insert at beginning for priority
        sys.path.insert(0, v14_root_str)


def detect_and_setup() -> Tuple[Optional[Path], Path]:
    """
    Detect entrypoint and setup sys.path in one call.
    
    Returns:
        Tuple of (entrypoint_path, v14_root)
    """
    detector = EntrypointDetector()
    v14_root = detector.get_v14_root()
    entrypoint_path, _ = detector.find_entrypoint(v14_root)
    
    # Setup sys.path
    detector.setup_sys_path(v14_root)
    
    return entrypoint_path, v14_root

