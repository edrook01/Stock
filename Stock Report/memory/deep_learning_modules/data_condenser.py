#!/usr/bin/env python3
"""
Data Condenser - Self-Contained Module
Condenses data for faster processing.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any

try:
    import pandas as pd
except ImportError:
    pd = None


def condense_data(df, target_size: int = 100) -> Optional[Any]:
    """Condense dataframe to target size while preserving key information."""
    if pd is None or df is None:
        return None
    
    if len(df) <= target_size:
        return df
    
    # Sample or aggregate to target size
    step = len(df) // target_size
    condensed = df.iloc[::step].copy()
    
    return condensed
