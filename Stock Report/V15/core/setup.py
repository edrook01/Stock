"""
First-Run Setup
Detects first run and initializes V15 directory structure and configuration.
"""

from pathlib import Path
import json
from typing import Dict, Optional

from .portable_paths import get_path, initialize_structure, get_data_path


def is_first_run() -> bool:
    """
    Check if this is the first run of V15.
    
    Returns:
        True if first run, False otherwise
    """
    config_file = get_data_path() / 'config_v15.json'
    return not config_file.exists()


def initialize_v15() -> Dict:
    """
    Initialize V15 on first run.
    
    Returns:
        Dictionary with initialization status
    """
    # Create directory structure
    initialize_structure()
    
    # Create model weights directory
    model_weights_dir = get_path('model_weights')
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create default configuration
    config_file = get_data_path() / 'config_v15.json'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Create default config
        config = {
            "version": "15.0",
            "risk_profile": "medium",
            "model": {
                "use_gpu": False,
                "confidence_threshold": 0.65,
                "retrain_interval_days": 7
            },
            "risk_management": {
                "max_equity_risk_per_trade": 2.0,
                "max_combined_exposure": 10.0,
                "min_position_value": 10.0
            },
            "browser_automation": {
                "library": "undetected-chromedriver",
                "headless": False,
                "human_like_delays": True,
                "trading212": {
                    "demo_mode": True,
                    "credentials_encrypted": False
                }
            },
            "timeframes": {
                "cfd": ["1m", "5m", "10m", "15m", "1h"],
                "investment": ["1d", "1w"]
            },
            "sentiment": {
                "enabled": True,
                "override_threshold": 0.7,
                "news_sources": ["yahoo_finance"]
            },
            "logging": {
                "log_level": "INFO",
                "log_trades": True,
                "log_predictions": True
            },
            "portability": {
                "use_relative_paths": True,
                "data_in_project_folder": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    return {
        "initialized": True,
        "config_loaded": True,
        "directories_created": True
    }


def prompt_trading212_credentials() -> Optional[Dict[str, str]]:
    """
    Prompt user for Trading212 credentials.
    
    Returns:
        Dictionary with username and password, or None if cancelled
    """
    print("\nTrading212 Credentials Setup")
    print("=" * 50)
    print("Enter Trading212 credentials (or press Enter to skip)")
    
    username = input("Username/Email: ").strip()
    if not username:
        return None
    
    password = input("Password: ").strip()
    if not password:
        return None
    
    return {
        "username": username,
        "password": password
    }


def save_credentials(credentials: Dict[str, str], encrypted: bool = False) -> bool:
    """
    Save Trading212 credentials.
    
    Args:
        credentials: Dictionary with username and password
        encrypted: Whether to encrypt credentials (requires cryptography)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        config_file = get_data_path() / 'config_v15.json'
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if encrypted:
            # TODO: Implement encryption using cryptography library
            # For now, store in plain text (not recommended for production)
            config["browser_automation"]["trading212"]["username"] = credentials["username"]
            config["browser_automation"]["trading212"]["password"] = credentials["password"]
            config["browser_automation"]["trading212"]["credentials_encrypted"] = True
        else:
            config["browser_automation"]["trading212"]["username"] = credentials["username"]
            config["browser_automation"]["trading212"]["password"] = credentials["password"]
            config["browser_automation"]["trading212"]["credentials_encrypted"] = False
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
    except Exception:
        return False

