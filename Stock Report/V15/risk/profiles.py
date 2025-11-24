"""
Risk Profile System
Defines risk profiles (Low, Medium, High) and their associated settings.
"""

from typing import Dict, List, Optional
from enum import Enum


class RiskProfile(Enum):
    """Risk profile enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Risk profile configurations
RISK_PROFILE_CONFIG: Dict[RiskProfile, Dict] = {
    RiskProfile.LOW: {
        "equity_risk_per_trade_min": 0.5,  # 0.5% minimum
        "equity_risk_per_trade_max": 1.0,   # 1.0% maximum
        "atr_multiplier_min": 1.5,         # 1.5× ATR for stops
        "atr_multiplier_max": 2.0,         # 2.0× ATR for stops
        "max_combined_exposure": 5.0,      # 5% total exposure
        "allowed_assets": "stable",        # Only stable assets
        "confidence_threshold": 0.75,       # Higher confidence required
    },
    RiskProfile.MEDIUM: {
        "equity_risk_per_trade_min": 1.0,  # 1.0% minimum
        "equity_risk_per_trade_max": 1.0,  # 1.0% maximum
        "atr_multiplier_min": 2.0,         # 2.0× ATR for stops
        "atr_multiplier_max": 2.0,         # 2.0× ATR for stops
        "max_combined_exposure": 10.0,     # 10% total exposure
        "allowed_assets": "moderate",       # Moderate volatility assets
        "confidence_threshold": 0.65,      # Standard confidence
    },
    RiskProfile.HIGH: {
        "equity_risk_per_trade_min": 1.0,  # 1.0% minimum
        "equity_risk_per_trade_max": 2.0,  # 2.0% maximum
        "atr_multiplier_min": 3.0,         # 3.0× ATR for stops
        "atr_multiplier_max": 4.0,         # 4.0× ATR for stops
        "max_combined_exposure": 10.0,     # 10% total exposure (same as medium)
        "allowed_assets": "all",            # All assets including volatile
        "confidence_threshold": 0.60,      # Lower confidence threshold
    },
}


def get_risk_profile(profile_name: str) -> Optional[RiskProfile]:
    """
    Get RiskProfile enum from string name.
    
    Args:
        profile_name: Profile name ("low", "medium", "high")
        
    Returns:
        RiskProfile enum, or None if invalid
    """
    profile_name_lower = profile_name.lower()
    for profile in RiskProfile:
        if profile.value == profile_name_lower:
            return profile
    return None


def get_profile_config(profile: RiskProfile) -> Dict:
    """
    Get configuration for a risk profile.
    
    Args:
        profile: RiskProfile enum
        
    Returns:
        Dictionary of profile configuration
    """
    return RISK_PROFILE_CONFIG.get(profile, RISK_PROFILE_CONFIG[RiskProfile.MEDIUM])


def get_equity_risk_range(profile: RiskProfile) -> tuple:
    """
    Get equity risk percentage range for a profile.
    
    Args:
        profile: RiskProfile enum
        
    Returns:
        Tuple of (min, max) equity risk percentages
    """
    config = get_profile_config(profile)
    return (config["equity_risk_per_trade_min"], config["equity_risk_per_trade_max"])


def get_atr_multiplier_range(profile: RiskProfile) -> tuple:
    """
    Get ATR multiplier range for stop-loss calculation.
    
    Args:
        profile: RiskProfile enum
        
    Returns:
        Tuple of (min, max) ATR multipliers
    """
    config = get_profile_config(profile)
    return (config["atr_multiplier_min"], config["atr_multiplier_max"])


def get_max_combined_exposure(profile: RiskProfile) -> float:
    """
    Get maximum combined exposure percentage.
    
    Args:
        profile: RiskProfile enum
        
    Returns:
        Maximum combined exposure as percentage
    """
    config = get_profile_config(profile)
    return config["max_combined_exposure"]


def get_confidence_threshold(profile: RiskProfile) -> float:
    """
    Get minimum confidence threshold for trades.
    
    Args:
        profile: RiskProfile enum
        
    Returns:
        Minimum confidence score (0-1)
    """
    config = get_profile_config(profile)
    return config["confidence_threshold"]


def is_asset_allowed(profile: RiskProfile, asset_risk_category: str) -> bool:
    """
    Check if an asset is allowed for a given risk profile.
    
    Args:
        profile: RiskProfile enum
        asset_risk_category: Asset risk category ("low", "medium", "high", "stable", "moderate", "all")
        
    Returns:
        True if asset is allowed, False otherwise
    """
    config = get_profile_config(profile)
    allowed = config["allowed_assets"]
    
    if allowed == "all":
        return True
    elif allowed == "stable":
        return asset_risk_category.lower() in ["low", "stable"]
    elif allowed == "moderate":
        return asset_risk_category.lower() in ["low", "stable", "medium", "moderate"]
    else:
        return False

