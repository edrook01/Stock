"""
Portability Check for V14
Verifies that all paths are relative and the system is portable.
"""

from pathlib import Path
import ast
import os
from typing import List, Dict

from .portable_paths import get_path, get_root_path


def check_absolute_paths() -> List[str]:
    """
    Check for hardcoded absolute paths in Python files.
    
    Returns:
        List of files with absolute paths found
    """
    issues = []
    root = get_root_path()
    
    # Check all Python files in V14
    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common absolute path patterns
            if 'C:\\' in content or '/home/' in content or '/Users/' in content:
                # Check if it's in a comment or string literal
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if ('C:\\' in line or '/home/' in line or '/Users/' in line) and not line.strip().startswith('#'):
                        issues.append(f"{py_file.relative_to(root)}: Line {i}")
        except Exception:
            pass
    
    return issues


def check_data_locations() -> Dict[str, bool]:
    """
    Check that all data is stored in V14 folder.
    
    Returns:
        Dictionary mapping data type to whether it's in V14 folder
    """
    root = get_root_path()
    
    checks = {
        "config": (get_path('data') / 'config_v14.json').exists(),
        "model_weights": get_path('model_weights').exists(),
        "history": get_path('history').exists(),
        "logs": get_path('logs').exists(),
        "memory": get_path('memory').exists(),
        "cache": get_path('cache').exists(),
    }
    
    return checks


def verify_portability() -> Dict:
    """
    Verify that V14 is portable.
    
    Returns:
        Dictionary with portability check results
    """
    absolute_path_issues = check_absolute_paths()
    data_locations = check_data_locations()
    
    all_data_in_v14 = all(data_locations.values())
    no_absolute_paths = len(absolute_path_issues) == 0
    
    return {
        "portable": all_data_in_v14 and no_absolute_paths,
        "absolute_path_issues": absolute_path_issues,
        "data_locations": data_locations,
        "all_data_in_v14": all_data_in_v14,
        "no_absolute_paths": no_absolute_paths
    }


def generate_portability_report() -> str:
    """
    Generate a portability report.
    
    Returns:
        Formatted portability report string
    """
    results = verify_portability()
    
    report = f"""
Portability Check Report
{'=' * 50}
Overall Status: {'PASS' if results['portable'] else 'FAIL'}

Data Locations:
"""
    for location, exists in results['data_locations'].items():
        status = "✓" if exists else "✗"
        report += f"  {status} {location}\n"
    
    if results['absolute_path_issues']:
        report += f"\nAbsolute Path Issues Found: {len(results['absolute_path_issues'])}\n"
        for issue in results['absolute_path_issues'][:10]:  # Show first 10
            report += f"  - {issue}\n"
    else:
        report += "\nNo absolute paths found.\n"
    
    return report

