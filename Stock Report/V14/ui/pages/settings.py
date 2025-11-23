"""
Settings Page
Application configuration and settings management.
"""

import streamlit as st
import json
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.portable_paths import get_data_path
from core.setup import initialize_v14
from risk.profiles import RiskProfile, get_risk_profile
from browser.automation import BrowserAutomation
from sentiment.override import get_sentiment_override
from core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES


def show_settings():
    """Display settings page."""
    
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Load current configuration
    config = _load_config()
    
    if config is None:
        st.error("❌ Could not load configuration. Initializing default configuration...")
        initialize_v14()
        config = _load_config()
        if config is None:
            st.error("❌ Failed to initialize configuration")
            return
    
    # Risk Profile Section
    _display_risk_profile_settings(config)
    
    st.markdown("---")
    
    # Model Settings
    _display_model_settings(config)
    
    st.markdown("---")
    
    # Risk Management Settings
    _display_risk_management_settings(config)
    
    st.markdown("---")
    
    # Browser Automation Settings
    _display_browser_automation_settings(config)
    
    st.markdown("---")
    
    # Sentiment Override Settings
    _display_sentiment_settings(config)
    
    st.markdown("---")
    
    # Timeframe Configuration
    _display_timeframe_settings(config)
    
    st.markdown("---")
    
    # Logging Settings
    _display_logging_settings(config)
    
    st.markdown("---")
    
    # Data Provider Settings
    _display_data_provider_settings(config)
    
    st.markdown("---")
    
    # Cache Management Settings
    _display_cache_settings(config)
    
    st.markdown("---")
    
    # Ticker Validation Settings
    _display_ticker_validation_settings(config)
    
    st.markdown("---")
    
    # Save/Reset buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            if _save_config(config):
                st.success("✅ Settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            if _reset_to_defaults():
                st.success("✅ Settings reset to defaults")
                st.rerun()
            else:
                st.error("❌ Failed to reset settings")
    
    with col3:
        if st.button("📋 View Raw Config", use_container_width=True):
            st.json(config)


def _load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        config_file = get_data_path() / 'config_v14.json'
        if not config_file.exists():
            return None
        
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    try:
        config_file = get_data_path() / 'config_v14.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Risk profile is applied when config is loaded
        # No need to set it here as it's read from config
        
        return True
    except Exception:
        return False


def _reset_to_defaults() -> bool:
    """Reset configuration to defaults."""
    try:
        initialize_v14()
        return True
    except Exception:
        return False


def _display_risk_profile_settings(config: Dict[str, Any]):
    """Display risk profile settings."""
    st.subheader("🎯 Risk Profile")
    
    current_profile = config.get("risk_profile", "medium").lower()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        profile_options = {
            "Low": "low",
            "Medium": "medium",
            "High": "high"
        }
        
        selected = st.selectbox(
            "Risk Profile",
            options=list(profile_options.keys()),
            index=list(profile_options.values()).index(current_profile) if current_profile in profile_options.values() else 1
        )
        
        config["risk_profile"] = profile_options[selected]
    
    with col2:
        if selected == "Low":
            st.info("""
            **Low Risk Profile**
            - Equity risk per trade: 0.5-1%
            - Stable assets only
            - Tight stop-losses
            - Conservative approach
            """)
        elif selected == "Medium":
            st.info("""
            **Medium Risk Profile**
            - Equity risk per trade: 1%
            - Moderate assets
            - Balanced approach
            - Standard stop-losses
            """)
        else:
            st.info("""
            **High Risk Profile**
            - Equity risk per trade: 1-2%
            - All assets allowed
            - Wider stop-losses
            - Aggressive approach
            """)


def _display_model_settings(config: Dict[str, Any]):
    """Display model settings."""
    st.subheader("🤖 Model Settings")
    
    model_config = config.setdefault("model", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_gpu = st.checkbox(
            "Use GPU Acceleration",
            value=model_config.get("use_gpu", False),
            help="Enable GPU acceleration if CUDA-compatible GPU is available"
        )
        model_config["use_gpu"] = use_gpu
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=model_config.get("confidence_threshold", 0.65),
            step=0.05,
            help="Minimum confidence required for trade execution"
        )
        model_config["confidence_threshold"] = confidence_threshold
    
    with col3:
        retrain_interval = st.number_input(
            "Retrain Interval (days)",
            min_value=1,
            max_value=30,
            value=model_config.get("retrain_interval_days", 7),
            help="Days between model retraining"
        )
        model_config["retrain_interval_days"] = retrain_interval


def _display_risk_management_settings(config: Dict[str, Any]):
    """Display risk management settings."""
    st.subheader("⚠️ Risk Management")
    
    risk_config = config.setdefault("risk_management", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_equity_risk = st.number_input(
            "Max Equity Risk per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=risk_config.get("max_equity_risk_per_trade", 2.0),
            step=0.1,
            help="Maximum percentage of equity risk per trade"
        )
        risk_config["max_equity_risk_per_trade"] = max_equity_risk
    
    with col2:
        max_combined_exposure = st.number_input(
            "Max Combined Exposure (%)",
            min_value=1.0,
            max_value=50.0,
            value=risk_config.get("max_combined_exposure", 10.0),
            step=0.5,
            help="Maximum combined exposure across all positions"
        )
        risk_config["max_combined_exposure"] = max_combined_exposure
    
    with col3:
        min_position_value = st.number_input(
            "Min Position Value ($)",
            min_value=1.0,
            max_value=1000.0,
            value=risk_config.get("min_position_value", 10.0),
            step=1.0,
            help="Minimum position value in dollars"
        )
        risk_config["min_position_value"] = min_position_value


def _display_browser_automation_settings(config: Dict[str, Any]):
    """Display browser automation settings."""
    st.subheader("🌐 Browser Automation")
    
    browser_config = config.setdefault("browser_automation", {})
    trading212_config = browser_config.setdefault("trading212", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Browser Settings**")
        
        library = st.selectbox(
            "Automation Library",
            options=["undetected-chromedriver", "playwright"],
            index=0 if browser_config.get("library", "undetected-chromedriver") == "undetected-chromedriver" else 1,
            help="Browser automation library to use"
        )
        browser_config["library"] = library
        
        headless = st.checkbox(
            "Headless Mode",
            value=browser_config.get("headless", False),
            help="Run browser in headless mode (no visible window)"
        )
        browser_config["headless"] = headless
        
        human_like = st.checkbox(
            "Human-Like Delays",
            value=browser_config.get("human_like_delays", True),
            help="Add randomized delays to simulate human behavior"
        )
        browser_config["human_like_delays"] = human_like
    
    with col2:
        st.write("**Trading212 Settings**")
        
        demo_mode = st.checkbox(
            "Demo Mode",
            value=trading212_config.get("demo_mode", True),
            help="Use Trading212 demo account"
        )
        trading212_config["demo_mode"] = demo_mode
        
        st.write("**Credentials**")
        st.caption("⚠️ Credentials are stored in plain text. Use encryption for production.")
        
        username = st.text_input(
            "Username/Email",
            value=trading212_config.get("username", ""),
            type="default",
            help="Trading212 username or email"
        )
        if username:
            trading212_config["username"] = username
        
        password = st.text_input(
            "Password",
            value=trading212_config.get("password", ""),
            type="password",
            help="Trading212 password"
        )
        if password:
            trading212_config["password"] = password
    
    # Browser automation status
    st.write("**Browser Status**")
    try:
        browser_automation = BrowserAutomation(
            headless=browser_config.get("headless", False),
            use_playwright=(library == "playwright")
        )
        
        if browser_automation.is_ready():
            st.success(f"✅ Browser automation ready ({browser_automation.library_used})")
        else:
            st.warning("⚠️ Browser automation not initialized")
    except Exception as e:
        st.warning(f"⚠️ Could not check browser status: {str(e)}")


def _display_sentiment_settings(config: Dict[str, Any]):
    """Display sentiment override settings."""
    st.subheader("📰 Sentiment Override")
    
    sentiment_config = config.setdefault("sentiment", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        enabled = st.checkbox(
            "Enable Sentiment Override",
            value=sentiment_config.get("enabled", True),
            help="Enable sentiment-based trade blocking"
        )
        sentiment_config["enabled"] = enabled
        
        override_threshold = st.slider(
            "Override Threshold",
            min_value=0.0,
            max_value=1.0,
            value=sentiment_config.get("override_threshold", 0.7),
            step=0.05,
            help="Sentiment threshold for trade blocking"
        )
        sentiment_config["override_threshold"] = override_threshold
    
    with col2:
        st.write("**News Sources**")
        
        news_sources = sentiment_config.get("news_sources", ["yahoo_finance"])
        
        yahoo_finance = st.checkbox(
            "Yahoo Finance",
            value="yahoo_finance" in news_sources
        )
        
        # Update news sources list
        new_sources = []
        if yahoo_finance:
            new_sources.append("yahoo_finance")
        sentiment_config["news_sources"] = new_sources
    
    # Sentiment override status
    st.write("**Override Status**")
    try:
        sentiment_override = get_sentiment_override()
        status = sentiment_override.get_override_status()
        
        col1, col2 = st.columns(2)
        with col1:
            if status.get("protective_mode", False):
                st.warning("🛡️ Protective Mode: ACTIVE")
            else:
                st.success("✅ Protective Mode: Inactive")
        
        with col2:
            blocked_count = len(status.get("blocked_tickers", {}))
            st.metric("Blocked Tickers", blocked_count)
    except Exception as e:
        st.warning(f"⚠️ Could not load sentiment status: {str(e)}")


def _display_timeframe_settings(config: Dict[str, Any]):
    """Display timeframe configuration."""
    st.subheader("⏰ Timeframe Configuration")
    
    timeframe_config = config.setdefault("timeframes", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**CFD Timeframes**")
        cfd_timeframes = timeframe_config.get("cfd", CFD_TIMEFRAMES)
        
        selected_cfd = st.multiselect(
            "Select CFD Timeframes",
            options=CFD_TIMEFRAMES,
            default=cfd_timeframes,
            help="Timeframes for CFD trading"
        )
        timeframe_config["cfd"] = selected_cfd
    
    with col2:
        st.write("**Investment Timeframes**")
        investment_timeframes = timeframe_config.get("investment", INVESTMENT_TIMEFRAMES)
        
        selected_investment = st.multiselect(
            "Select Investment Timeframes",
            options=INVESTMENT_TIMEFRAMES,
            default=investment_timeframes,
            help="Timeframes for investment analysis"
        )
        timeframe_config["investment"] = selected_investment


def _display_logging_settings(config: Dict[str, Any]):
    """Display logging settings."""
    st.subheader("📝 Logging Settings")
    
    logging_config = config.setdefault("logging", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                logging_config.get("log_level", "INFO")
            ),
            help="Minimum log level to record"
        )
        logging_config["log_level"] = log_level
    
    with col2:
        log_trades = st.checkbox(
            "Log Trades",
            value=logging_config.get("log_trades", True),
            help="Log all trade entries and exits"
        )
        logging_config["log_trades"] = log_trades
    
    with col3:
        log_predictions = st.checkbox(
            "Log Predictions",
            value=logging_config.get("log_predictions", True),
            help="Log model predictions"
        )
        logging_config["log_predictions"] = log_predictions


def _display_data_provider_settings(config: Dict[str, Any]):
    """Display data provider settings."""
    st.subheader("🌐 Data Provider Settings")
    
    provider_config = config.setdefault("data_providers", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Provider Configuration**")
        
        primary = st.selectbox(
            "Primary Provider",
            options=["yahoo_finance", "alpha_vantage", "polygon"],
            index=0 if provider_config.get("primary", "yahoo_finance") == "yahoo_finance" else 1,
            help="Primary data provider to use"
        )
        provider_config["primary"] = primary
        
        use_multiple = st.checkbox(
            "Use Multiple Providers (Fallback)",
            value=provider_config.get("use_multiple_providers", True),
            help="Try fallback providers if primary fails"
        )
        provider_config["use_multiple_providers"] = use_multiple
        
        retry_enabled = st.checkbox(
            "Enable Retry Logic",
            value=provider_config.get("retry_enabled", True),
            help="Retry failed requests with exponential backoff"
        )
        provider_config["retry_enabled"] = retry_enabled
        
        if retry_enabled:
            max_retries = st.number_input(
                "Max Retries",
                min_value=1,
                max_value=5,
                value=provider_config.get("max_retries", 3),
                help="Maximum number of retry attempts"
            )
            provider_config["max_retries"] = max_retries
    
    with col2:
        st.write("**API Keys**")
        st.caption("Enter API keys for additional data providers (optional)")
        
        alpha_key = st.text_input(
            "Alpha Vantage API Key",
            value=provider_config.get("alpha_vantage_api_key", ""),
            type="password",
            help="Get free API key from alphavantage.co"
        )
        if alpha_key:
            provider_config["alpha_vantage_api_key"] = alpha_key
        
        polygon_key = st.text_input(
            "Polygon.io API Key",
            value=provider_config.get("polygon_api_key", ""),
            type="password",
            help="Get API key from polygon.io"
        )
        if polygon_key:
            provider_config["polygon_api_key"] = polygon_key
        
        # Show provider status
        st.write("**Provider Status**")
        try:
            from core.data_providers import get_available_providers
            providers = get_available_providers()
            for provider in providers:
                status = "✅ Available" if provider.is_available() else "❌ Unavailable"
                st.caption(f"{provider.get_name()}: {status}")
        except Exception as e:
            st.warning(f"Could not check provider status: {e}")


def _display_cache_settings(config: Dict[str, Any]):
    """Display cache management settings."""
    st.subheader("💾 Cache Management")
    
    cache_config = config.setdefault("cache", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cache Limits**")
        
        max_age = st.number_input(
            "Max Cache Age (days)",
            min_value=1,
            max_value=365,
            value=cache_config.get("max_age_days", 30),
            help="Maximum age for cached files"
        )
        cache_config["max_age_days"] = max_age
        
        max_size = st.number_input(
            "Max Cache Size (MB)",
            min_value=100,
            max_value=10000,
            value=int(cache_config.get("max_size_mb", 1000)),
            help="Maximum total cache size"
        )
        cache_config["max_size_mb"] = float(max_size)
    
    with col2:
        st.write("**Auto Pruning**")
        
        auto_prune = st.checkbox(
            "Enable Auto Pruning",
            value=cache_config.get("auto_prune", False),
            help="Automatically prune cache when size exceeds threshold"
        )
        cache_config["auto_prune"] = auto_prune
        
        if auto_prune:
            threshold = st.number_input(
                "Auto Prune Threshold (MB)",
                min_value=500,
                max_value=5000,
                value=int(cache_config.get("auto_prune_threshold_mb", 2000)),
                help="Prune cache when size exceeds this threshold"
            )
            cache_config["auto_prune_threshold_mb"] = float(threshold)
    
    # Cache statistics
    st.write("**Cache Statistics**")
    try:
        from core.cache_manager import get_cache_manager
        manager = get_cache_manager()
        stats = manager.get_cache_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Size", f"{stats['total_size_mb']:.2f} MB")
        with col2:
            st.metric("File Count", stats['file_count'])
        with col3:
            st.metric("Avg Age", f"{stats['average_file_age_days']:.1f} days")
        
        if stats['recommendations']:
            st.warning("⚠️ " + " | ".join(stats['recommendations']))
        
        # Cache actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Prune Old Files (>30 days)", use_container_width=True):
                result = manager.prune_cache(max_age_days=30, dry_run=False)
                st.success(f"✅ Pruned {result['removed_count']} files, freed {result['freed_mb']:.2f} MB")
                st.rerun()
        
        with col2:
            if st.button("Clear All Cache", use_container_width=True):
                if manager.clear_cache(confirm=True):
                    st.success("✅ Cache cleared")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear cache")
    
    except Exception as e:
        st.warning(f"Could not load cache statistics: {e}")


def _display_ticker_validation_settings(config: Dict[str, Any]):
    """Display ticker validation settings."""
    st.subheader("🔍 Ticker Validation")
    
    validation_config = config.setdefault("ticker_validation", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        cache_duration = st.number_input(
            "Validation Cache Duration (hours)",
            min_value=1,
            max_value=168,
            value=validation_config.get("cache_duration_hours", 24),
            help="How long to cache validation results"
        )
        validation_config["cache_duration_hours"] = cache_duration
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=50,
            value=validation_config.get("batch_size", 10),
            help="Number of tickers to validate concurrently"
        )
        validation_config["batch_size"] = batch_size
    
    with col2:
        auto_audit = st.checkbox(
            "Enable Auto Audit",
            value=validation_config.get("auto_audit", False),
            help="Automatically audit ticker list on startup"
        )
        validation_config["auto_audit"] = auto_audit
        
        if auto_audit:
            audit_interval = st.number_input(
                "Audit Interval (days)",
                min_value=1,
                max_value=90,
                value=validation_config.get("audit_interval_days", 30),
                help="Days between automatic audits"
            )
            validation_config["audit_interval_days"] = audit_interval
    
    # Ticker audit action
    st.write("**Ticker List Audit**")
    ticker_file = st.text_input(
        "Ticker List File",
        value="data/tickers.txt",
        help="Path to ticker list file to audit"
    )
    
    if st.button("Run Ticker Audit", use_container_width=True):
        try:
            from core.ticker_auditor import get_ticker_auditor
            import asyncio
            from pathlib import Path
            
            auditor = get_ticker_auditor()
            ticker_path = Path(ticker_file)
            
            if not ticker_path.exists():
                st.error(f"❌ File not found: {ticker_file}")
            else:
                with st.spinner("Auditing ticker list..."):
                    tickers = auditor._load_ticker_list(ticker_path)
                    result = asyncio.run(auditor.audit_ticker_list(tickers, auto_fix=False))
                    
                    st.success("✅ Audit complete!")
                    st.text(result["report"])
                    
                    if st.button("Update Ticker List", use_container_width=True):
                        updated = asyncio.run(auditor.update_ticker_list(ticker_path, remove_invalid=True))
                        st.success(f"✅ Updated: {updated['cleaned_count']} valid tickers (removed {updated['removed_count']})")
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error during audit: {e}")
            st.exception(e)

