"""
V15 Menu System
Extended menu controller with V15-specific features.
"""

import sys
import asyncio
import time
import os
from typing import Dict, List, Optional
from pathlib import Path

# Import V15 modules
# First, ensure V15 root is in sys.path for absolute imports
v15_root = Path(__file__).parent.parent
if str(v15_root) not in sys.path:
    sys.path.insert(0, str(v15_root))

# Use absolute imports since we've set up sys.path
# This avoids issues with relative imports when running as a script
try:
    # Fallback for direct execution
    # v15_root already set above, just ensure it's in path
    if str(v15_root) not in sys.path:
        sys.path.insert(0, str(v15_root))
    from core.portable_paths import get_path
    from core.ticker_universe import get_trading212_tickers
    from core.prediction_scheduler import get_prediction_scheduler
    from core.data_fetcher import fetch_prices
    from core.timeframes import (
        ALL_TIMEFRAMES,
        CFD_TIMEFRAMES,
        INVESTMENT_TIMEFRAMES,
        CONSTANT_LEARNING_INTERVALS,
        is_cfd_timeframe,
        is_investment_timeframe,
        is_valid_timeframe,
        get_timeframe_duration_seconds,
    )
    from core.indicators import rsi, sma, ema
    from model.unified_model import get_model
    from risk.profiles import RiskProfile, get_risk_profile
    from risk.stop_loss import (
        calculate_stop_loss_distance,
        calculate_stop_loss_price,
        should_skip_trade,
    )
    from risk.position_sizing import calculate_position_size_with_profile
    # Import risk.equity_monitor with special handling since it also uses relative imports
    try:
        from risk.equity_monitor import get_equity_monitor
    except (ImportError, ValueError, SystemError):
        # If that fails, manually import it using importlib to bypass relative import issues
        import importlib.util
        import types
        
        # Set up package structure so relative imports in equity_monitor work
        # First, ensure 'core' package exists
        if 'core' not in sys.modules:
            core_package = types.ModuleType('core')
            core_package.__path__ = [str(v15_root / "core")]
            sys.modules['core'] = core_package
        
        # Ensure risk package is in sys.modules
        if 'risk' not in sys.modules:
            risk_package = types.ModuleType('risk')
            risk_package.__path__ = [str(v15_root / "risk")]
            sys.modules['risk'] = risk_package
        
        # Now load equity_monitor with proper package context
        equity_monitor_spec = importlib.util.spec_from_file_location(
            "risk.equity_monitor", v15_root / "risk" / "equity_monitor.py"
        )
        equity_monitor_module = importlib.util.module_from_spec(equity_monitor_spec)
        equity_monitor_module.__package__ = 'risk'
        equity_monitor_module.__name__ = 'risk.equity_monitor'
        sys.modules['risk.equity_monitor'] = equity_monitor_module
        equity_monitor_spec.loader.exec_module(equity_monitor_module)
        get_equity_monitor = equity_monitor_module.get_equity_monitor
    from browser.automation import BrowserAutomation
    from sentiment.override import get_sentiment_override
    from learning.prediction_generator import get_prediction_generator
    # Import local logging modules using importlib for standalone execution
    import importlib.util
    import sys as sys_module

    trade_logger_spec = importlib.util.spec_from_file_location(
        "sa_logging.trade_logger", v15_root / "sa_logging" / "trade_logger.py"
    )
    trade_logger_module = importlib.util.module_from_spec(trade_logger_spec)
    sys_module.modules['sa_logging.trade_logger'] = trade_logger_module
    trade_logger_spec.loader.exec_module(trade_logger_module)
    get_trade_logger = trade_logger_module.get_trade_logger

    analyzer_spec = importlib.util.spec_from_file_location(
        "sa_logging.analyzer", v15_root / "sa_logging" / "analyzer.py"
    )
    analyzer_module = importlib.util.module_from_spec(analyzer_spec)
    sys_module.modules['sa_logging.analyzer'] = analyzer_module
    sys_module.modules['sa_logging.trade_logger'] = trade_logger_module
    analyzer_spec.loader.exec_module(analyzer_module)
    generate_performance_report = analyzer_module.generate_performance_report

    # Import error logger in fallback path
    try:
        error_logger_spec = importlib.util.spec_from_file_location(
            "sa_logging.error_logger", v15_root / "sa_logging" / "error_logger.py"
        )
        error_logger_module = importlib.util.module_from_spec(error_logger_spec)
        sys_module.modules['sa_logging.error_logger'] = error_logger_module
        error_logger_spec.loader.exec_module(error_logger_module)
        log_exception = error_logger_module.log_exception
        log_error = error_logger_module.log_error
        log_warning = error_logger_module.log_warning
        log_info = error_logger_module.log_info
    except Exception as e:
        # Fallback error logger functions if import fails
        # These will write directly to error log if needed
        def _write_error_direct(message, error=None, component=None, function=None):
            try:
                from pathlib import Path
                from datetime import datetime
                logs_dir = v15_root.parent / 'logs'
                logs_dir.mkdir(parents=True, exist_ok=True)
                error_log = logs_dir / 'error.log'
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(error_log, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] ERROR")
                    if component:
                        f.write(f" [{component}]")
                    if function:
                        f.write(f" [{function}]")
                    f.write(f": {message}")
                    if error:
                        f.write(f" | Exception: {type(error).__name__}: {str(error)}\n")
                        import traceback
                        f.write(traceback.format_exc())
                    else:
                        f.write("\n")
            except Exception:
                pass
        
        def log_exception(msg, err, component=None, function=None, **kwargs):
            _write_error_direct(msg, err, component, function)
        def log_error(msg, error=None, component=None, function=None, **kwargs):
            _write_error_direct(msg, error, component, function)
        def log_warning(msg, component=None, function=None, **kwargs):
            _write_error_direct(msg, None, component, function)
        def log_info(msg, component=None, function=None, **kwargs):
            pass  # Info logging is optional
except Exception as e:
    # If imports fail completely, provide minimal fallbacks
    import traceback
    print(f"ERROR: Failed to import V15 modules in menu_v15.py: {e}")
    traceback.print_exc()
    # Define minimal fallbacks to prevent further errors
    def get_path(*args, **kwargs): return Path(__file__).parent.parent
    def log_exception(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass
    def log_warning(*args, **kwargs): pass
    def log_info(*args, **kwargs): pass
    # These will cause errors if used, but at least the module can load
    ALL_TIMEFRAMES = []
    CFD_TIMEFRAMES = []
    INVESTMENT_TIMEFRAMES = []
    CONSTANT_LEARNING_INTERVALS = []
    def is_cfd_timeframe(*args): return False
    def is_investment_timeframe(*args): return False
    def is_valid_timeframe(*args): return False
    from enum import Enum
    class RiskProfile(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    def get_risk_profile(*args): return RiskProfile.MEDIUM
    def get_equity_monitor(*args): return None
    def get_trading212_tickers(*args): return []
    def get_prediction_scheduler(*args): return None
    def get_model(*args): return None
    def get_prediction_generator(*args): return None
    def get_trade_logger(*args): return None
    def generate_performance_report(*args): return {}
    class BrowserAutomation:
        pass
    def get_sentiment_override(*args): return None
    def rsi(*args): return 50.0
    def sma(*args): return 0.0
    def ema(*args): return 0.0
    def calculate_stop_loss_distance(*args): return 0.0
    def calculate_stop_loss_price(*args): return 0.0
    def should_skip_trade(*args): return False
    def calculate_position_size_with_profile(*args): return 0.0
    def fetch_prices(*args): return {}


class MenuController:
    """Main menu controller for V15."""
    
    def __init__(self):
        """Initialize menu controller."""
        self.running = True
        self.current_profile = RiskProfile.MEDIUM
        self.browser_automation = None
        self._prediction_generator = None
        # Rotate through ticker/timeframe batches when seeding prediction feed
        self._seed_rotation_ticker = 0
        self._seed_rotation_interval = 0
        
        # Load risk profile from config
        try:
            from ..core.portable_paths import get_data_path
            import json
            config_file = get_data_path() / 'config_v15.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                profile_name = config.get("risk_profile", "medium")
                profile = get_risk_profile(profile_name)
                if profile:
                    self.current_profile = profile
        except Exception:
            # Use default if config can't be loaded
            pass

    @property
    def prediction_generator(self):
        if self._prediction_generator is None:
            try:
                self._prediction_generator = get_prediction_generator()
            except Exception as gen_error:
                try:
                    log_warning(
                        "Prediction generator unavailable",
                        component="menu",
                        function="prediction_generator",
                        context={"error": str(gen_error)},
                    )
                except Exception:
                    pass
                self._prediction_generator = None
        return self._prediction_generator

    def _get_active_tickers(self) -> List[str]:
        try:
            from ..learning.constant_learning_engine import get_constant_learning_engine
        except (ImportError, ValueError):
            from learning.constant_learning_engine import get_constant_learning_engine
        try:
            engine = get_constant_learning_engine()
            if getattr(engine, "active_tickers", None):
                return list(engine.active_tickers)
        except Exception:
            pass
        try:
            return get_trading212_tickers()
        except Exception:
            return ["AAPL", "MSFT", "TSLA"]

    def _get_watch_intervals(self) -> List[str]:
        try:
            return list(CONSTANT_LEARNING_INTERVALS)
        except Exception:
            return ["1m", "5m", "10m", "15m", "1h", "1d", "1mo", "3mo", "1y"]

    def _seed_prediction_feed(
        self,
        storage,
        per_interval: int = 1,
        max_tickers: int = 12,
        max_intervals: int = 4,
    ) -> int:
        """
        Populate the live prediction feed without freezing the UI.

        The ticker universe can include hundreds of symbols, so this function
        rotates through small batches to keep each seeding call fast while
        eventually covering all available tickers/intervals.
        """
        generator = self.prediction_generator
        if generator is None:
            return 0

        tickers = self._get_active_tickers()
        intervals = self._get_watch_intervals()
        if not tickers or not intervals:
            return 0

        max_tickers = max(1, min(max_tickers, len(tickers)))
        max_intervals = max(1, min(max_intervals, len(intervals)))

        def _take_with_wrap(items, start_index, count):
            if count >= len(items):
                return list(items)
            collected = []
            idx = start_index % len(items)
            for _ in range(count):
                collected.append(items[idx])
                idx = (idx + 1) % len(items)
            return collected

        selected_tickers = _take_with_wrap(tickers, self._seed_rotation_ticker, max_tickers)
        selected_intervals = _take_with_wrap(intervals, self._seed_rotation_interval, max_intervals)

        self._seed_rotation_ticker = (self._seed_rotation_ticker + max_tickers) % len(tickers)
        self._seed_rotation_interval = (self._seed_rotation_interval + max_intervals) % len(intervals)

        try:
            created = generator.ensure_predictions(
                tickers=selected_tickers,
                intervals=selected_intervals,
                per_interval=max(1, per_interval),
            )
            return created
        except Exception as seed_error:
            try:
                log_warning(
                    "Failed to seed prediction feed",
                    component="menu",
                    function="_seed_prediction_feed",
                    context={"error": str(seed_error)},
                )
            except Exception:
                pass
            return 0
    
    def display_main_menu(self):
        """Display main menu."""
        try:
            print("\n" + "=" * 70)
            print("  STOCK ANALYZER V15 - MAIN MENU")
            print("=" * 70)
            profile_value = self.current_profile.value if hasattr(self.current_profile, 'value') else str(self.current_profile)
            print(f"\nCurrent Risk Profile: {profile_value.upper()}")
            print("\n1. Core Analysis")
            print("2. Learning & Training")
            print("3. Data & Logs")
            print("4. System & Maintenance")
            print("5. V15 Features")
            print("\n0. Exit")
            print("-" * 70)
        except Exception as e:
            log_exception(
                "Error displaying menu",
                e,
                component="menu",
                function="display_main_menu",
                is_hard_error=False
            )
            print(f"\nERROR displaying menu: {e}")
            # Fallback to basic menu
            print("\n" + "=" * 70)
            print("  STOCK ANALYZER V15 - MAIN MENU")
            print("=" * 70)
            print("\n1. Core Analysis")
            print("2. Learning & Training")
            print("3. Data & Logs")
            print("4. System & Maintenance")
            print("5. V15 Features")
            print("\n0. Exit")
            print("-" * 70)
    
    def display_V15_features_menu(self):
        """Display V15-specific features menu."""
        print("\n" + "=" * 70)
        print("  V15 FEATURES MENU")
        print("=" * 70)
        print("\n5A. Unified Model - Generate Prediction")
        print("5B. Risk Profile Selection")
        print("5C. Browser Automation Status")
        print("5D. Sentiment Override Settings")
        print("5E. Trade Log Analysis")
        print("5F. Performance Report")
        print("\n0. Back to Main Menu")
        print("-" * 70)
    
    def run(self):
        """Run main menu loop."""
        try:
            while self.running:
                try:
                    self.display_main_menu()
                    choice = input("\nEnter choice: ").strip().upper()
                    
                    if choice == "0":
                        self.running = False
                    elif choice == "1":
                        self._handle_analysis_menu()
                    elif choice == "2":
                        self._handle_learning_menu()
                    elif choice == "3":
                        self._handle_data_menu()
                    elif choice == "4":
                        self._handle_system_menu()
                    elif choice == "5":
                        self._handle_V15_features_menu()
                    else:
                        print("Invalid choice. Please try again.")
                except KeyboardInterrupt:
                    log_info("User interrupted menu", component="menu", function="run")
                    print("\n\nExiting menu...")
                    self.running = False
                except Exception as e:
                    log_exception(
                        "Error in main menu loop",
                        e,
                        component="menu",
                        function="run",
                        is_hard_error=False
                    )
                    print(f"\nERROR in menu: {e}")
                    import traceback
                    traceback.print_exc()
                    input("\nPress Enter to continue...")
        except Exception as e:
            log_exception(
                "FATAL ERROR in menu system",
                e,
                component="menu",
                function="run",
                is_hard_error=True
            )
            print(f"\nFATAL ERROR in menu system: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _handle_V15_features_menu(self):
        """Handle V15 features menu."""
        while True:
            try:
                self.display_V15_features_menu()
                choice = input("\nEnter choice: ").strip().upper()
                
                if choice == "0":
                    break
                elif choice == "5A":
                    self._unified_model_prediction()
                elif choice == "5B":
                    self._select_risk_profile()
                elif choice == "5C":
                    self._browser_automation_status()
                elif choice == "5D":
                    self._sentiment_override_settings()
                elif choice == "5E":
                    self._trade_log_analysis()
                elif choice == "5F":
                    try:
                        self._performance_report()
                    except Exception as e:
                        # Ensure error is logged even if _performance_report fails
                        try:
                            log_exception(
                                "Error in performance report (5F)",
                                e,
                                component="menu",
                                function="_handle_V15_features_menu",
                                is_hard_error=False
                            )
                        except Exception:
                            # If logger fails, write directly to error log
                            try:
                                from pathlib import Path
                                from datetime import datetime
                                logs_dir = Path(__file__).parent.parent.parent / 'logs'
                                logs_dir.mkdir(parents=True, exist_ok=True)
                                error_log = logs_dir / 'error.log'
                                with open(error_log, 'a', encoding='utf-8') as f:
                                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR [menu][_handle_V15_features_menu]: Error in performance report (5F) | Exception: {type(e).__name__}: {str(e)}\n")
                                    import traceback
                                    f.write(traceback.format_exc() + "\n")
                            except Exception:
                                pass
                        print(f"\n❌ Error in performance report: {e}")
                        import traceback
                        traceback.print_exc()
                        input("\nPress Enter to continue...")
                else:
                    print("Invalid choice.")
            except KeyboardInterrupt:
                try:
                    log_info("User interrupted V15 features menu", component="menu", function="_handle_V15_features_menu")
                except Exception:
                    pass
                print("\n\nReturning to main menu...")
                break
            except Exception as e:
                try:
                    log_exception(
                        "Error in V15 features menu",
                        e,
                        component="menu",
                        function="_handle_V15_features_menu",
                        is_hard_error=False
                    )
                except Exception:
                    # If logger fails, write directly to error log
                    try:
                        from pathlib import Path
                        from datetime import datetime
                        logs_dir = Path(__file__).parent.parent.parent / 'logs'
                        logs_dir.mkdir(parents=True, exist_ok=True)
                        error_log = logs_dir / 'error.log'
                        with open(error_log, 'a', encoding='utf-8') as f:
                            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR [menu][_handle_V15_features_menu]: Error in V15 features menu | Exception: {type(e).__name__}: {str(e)}\n")
                            import traceback
                            f.write(traceback.format_exc() + "\n")
                    except Exception:
                        pass
                print(f"\n❌ Error in menu: {e}")
                import traceback
                traceback.print_exc()
                input("\nPress Enter to continue...")
    
    def _unified_model_prediction(self):
        """Generate prediction using unified model."""
        ticker = input("Enter ticker symbol: ").strip().upper()
        if not ticker:
            print("Invalid ticker.")
            input("\nPress Enter to continue...")
            return
        
        timeframe = input("Enter timeframe (1m, 5m, 1h, 1d, etc.): ").strip().lower()
        if not timeframe:
            timeframe = "1d"
        
        print(f"\nGenerating prediction for {ticker} ({timeframe})...")
        print("This may take a few moments...")
        
        try:
            model = get_model(timeframe)
            prediction = asyncio.run(model.predict(ticker))
            
            if not prediction:
                print("\n⚠️  Could not generate prediction. Model may not be trained.")
                print("Prediction will use default values until model is trained.")
            else:
                print(f"\n{'=' * 50}")
                print(f"PREDICTION RESULTS")
                print(f"{'=' * 50}")
                print(f"Ticker: {ticker}")
                print(f"Timeframe: {timeframe}")
                print(f"\nPredicted Movement: {prediction.get('prediction', 0):.2f}%")
                print(f"Confidence: {prediction.get('confidence', 0):.2%}")
                print(f"Range: {prediction.get('range_low', 0):.2f}% to {prediction.get('range_high', 0):.2f}%")
                if prediction.get('model_agreement'):
                    print(f"Model Agreement: {prediction.get('model_agreement', 0):.2%}")
                print(f"{'=' * 50}")
                
                if not model.is_trained:
                    print("\n⚠️  Note: Model is not yet trained. Using default predictions.")
                    print("Train the model with historical data for better accuracy.")
                try:
                    generator = self.prediction_generator
                    if generator:
                        if generator.record_external_prediction(ticker, timeframe, prediction, source="manual_analysis"):
                            print("\n📡 Prediction stored for live monitoring and evaluation.")
                except Exception:
                    pass
        except ImportError as e:
            print(f"\n❌ Error: Missing required dependency: {e}")
            print("Please install required packages: pip install scikit-learn pandas numpy")
        except Exception as e:
            log_exception(
                "Error generating prediction",
                e,
                component="menu",
                function="_unified_model_prediction",
                is_hard_error=False
            )
            print(f"\n❌ Error generating prediction: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _select_risk_profile(self):
        """Select risk profile."""
        try:
            print("\n" + "=" * 50)
            print("RISK PROFILE SELECTION")
            print("=" * 50)
            profile_value = self.current_profile.value if hasattr(self.current_profile, 'value') else str(self.current_profile)
            print("\nCurrent Profile: " + profile_value.upper())
            print("\nAvailable Profiles:")
            print("1. LOW    - 0.5-1% equity risk, stable assets only, tight stops")
            print("2. MEDIUM - 1% equity risk, moderate assets, balanced approach")
            print("3. HIGH   - 1-2% equity risk, all assets, wider stops")
            print("\n0. Cancel")
            print("-" * 50)
            
            choice = input("\nSelect profile (1-3): ").strip()
            
            if choice == "0":
                print("Cancelled.")
            elif choice == "1":
                self.current_profile = RiskProfile.LOW
                # Save to config
                try:
                    from ..core.portable_paths import get_data_path
                    import json
                    config_file = get_data_path() / 'config_v15.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        config['risk_profile'] = 'low'
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                except Exception:
                    pass
                print(f"\n✅ Risk profile set to: LOW")
            elif choice == "2":
                self.current_profile = RiskProfile.MEDIUM
                try:
                    from ..core.portable_paths import get_data_path
                    import json
                    config_file = get_data_path() / 'config_v15.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        config['risk_profile'] = 'medium'
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                except Exception:
                    pass
                print(f"\n✅ Risk profile set to: MEDIUM")
            elif choice == "3":
                self.current_profile = RiskProfile.HIGH
                try:
                    from ..core.portable_paths import get_data_path
                    import json
                    config_file = get_data_path() / 'config_v15.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        config['risk_profile'] = 'high'
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                except Exception:
                    pass
                print(f"\n✅ Risk profile set to: HIGH")
            else:
                print("❌ Invalid choice.")
        except Exception as e:
            log_exception(
                "Error in menu function",
                e,
                component="menu",
                function="_select_risk_profile",
                is_hard_error=False
            )
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _browser_automation_status(self):
        """Display browser automation status."""
        try:
            print("\n" + "=" * 50)
            print("BROWSER AUTOMATION STATUS")
            print("=" * 50)
            
            if self.browser_automation is None:
                print("\nStatus: Not initialized")
                print("\nBrowser automation allows automated Trading212 CFD trading.")
                print("Requirements:")
                print("  - Google Chrome installed")
                print("  - undetected-chromedriver or playwright library")
                print("  - Trading212 credentials in config_v15.json")
                
                init = input("\nInitialize browser automation? (y/n): ").strip().lower()
                if init == 'y':
                    print("\nInitializing browser automation...")
                    try:
                        self.browser_automation = BrowserAutomation()
                        if self.browser_automation.initialize():
                            print("✅ Browser automation initialized successfully!")
                            print(f"   Library: {self.browser_automation.library_used}")
                        else:
                            print("❌ Failed to initialize browser automation.")
                            print("   Please check:")
                            print("   1. Chrome is installed")
                            print("   2. Required library is installed (pip install undetected-chromedriver)")
                            self.browser_automation = None
                    except Exception as e:
                        log_exception(
                            "Error initializing browser automation",
                            e,
                            component="menu",
                            function="_browser_automation_status",
                            is_hard_error=False
                        )
                        print(f"❌ Error initializing: {e}")
                        self.browser_automation = None
            else:
                print(f"\nStatus: {'✅ Ready' if self.browser_automation.is_ready() else '❌ Not ready'}")
                print(f"Library: {self.browser_automation.library_used}")
                print(f"Initialized: {'Yes' if self.browser_automation.is_initialized else 'No'}")
                
                action = input("\nClose browser? (y/n): ").strip().lower()
                if action == 'y':
                    self.browser_automation.close()
                    self.browser_automation = None
                    print("✅ Browser closed")
        except Exception as e:
            log_exception(
                "Error in menu function",
                e,
                component="menu",
                function="_select_risk_profile",
                is_hard_error=False
            )
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _sentiment_override_settings(self):
        """Display sentiment override settings."""
        try:
            override = get_sentiment_override()
            status = override.get_override_status()
            
            print("\n" + "=" * 50)
            print("SENTIMENT OVERRIDE SETTINGS")
            print("=" * 50)
            print(f"\nProtective Mode: {'🟢 Active' if status['protective_mode'] else '⚪ Inactive'}")
            print(f"Blocked Tickers: {len(status['blocked_tickers'])}")
            if status['blocked_tickers']:
                print("\nCurrently Blocked:")
                for ticker, until in status['blocked_tickers'].items():
                    print(f"  - {ticker}: until {until}")
            print(f"Override Threshold: {status['override_threshold']:.2f}")
            
            print("\nOptions:")
            print("1. Toggle Protective Mode")
            print("2. View Blocked Tickers")
            print("3. Unblock Ticker")
            print("0. Back")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                if status['protective_mode']:
                    override.disable_protective_mode()
                    print("✅ Protective mode disabled")
                else:
                    override.enable_protective_mode()
                    print("✅ Protective mode enabled")
            elif choice == "2":
                if status['blocked_tickers']:
                    print("\nBlocked Tickers:")
                    for ticker, until in status['blocked_tickers'].items():
                        print(f"  {ticker}: until {until}")
                else:
                    print("\nNo tickers currently blocked")
            elif choice == "3":
                ticker = input("Enter ticker to unblock: ").strip().upper()
                if ticker in status['blocked_tickers']:
                    # Note: This requires adding an unblock method to SentimentOverride
                    print(f"⚠️  Unblocking not yet implemented. Ticker {ticker} will unblock automatically when block expires.")
                else:
                    print(f"Ticker {ticker} is not blocked")
        except Exception as e:
            log_exception(
                "Error in menu function",
                e,
                component="menu",
                function="_select_risk_profile",
                is_hard_error=False
            )
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _trade_log_analysis(self):
        """Display trade log analysis."""
        try:
            from ..sa_logging.analyzer import calculate_performance_metrics, compare_predicted_vs_actual, identify_patterns
            
            logger = get_trade_logger()
            trades = logger.get_trades()
            
            print("\n" + "=" * 50)
            print("TRADE LOG ANALYSIS")
            print("=" * 50)
            
            if not trades:
                print("\n⚠️  No trades logged yet.")
                print("Trades will appear here once you start trading.")
            else:
                completed = [t for t in trades if t.get("exit_time")]
                open_trades = [t for t in trades if not t.get("exit_time")]
                
                print(f"\nTotal Trades: {len(trades)}")
                print(f"  ✅ Completed: {len(completed)}")
                print(f"  ⏳ Open: {len(open_trades)}")
                
                if completed:
                    metrics = calculate_performance_metrics(trades)
                    comparison = compare_predicted_vs_actual(trades)
                    patterns = identify_patterns(trades)
                    
                    print("\n" + "-" * 50)
                    print("PERFORMANCE METRICS")
                    print("-" * 50)
                    print(f"Win Rate: {metrics['win_rate']:.2%}")
                    print(f"Wins: {metrics['wins']}")
                    print(f"Losses: {metrics['losses']}")
                    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                    print(f"Total P/L: ${metrics['total_pnl']:.2f}")
                    print(f"Average P/L: ${metrics['avg_pnl']:.2f}")
                    print(f"Max Drawdown: ${metrics['max_drawdown']:.2f}")
                    
                    print("\n" + "-" * 50)
                    print("PREDICTION ACCURACY")
                    print("-" * 50)
                    print(f"Accuracy: {comparison['accuracy']:.2%}")
                    print(f"Average Error: {comparison['avg_prediction_error']:.4f}")
                    
                    if patterns:
                        print("\n" + "-" * 50)
                        print("PATTERNS")
                        print("-" * 50)
                        print(f"High Confidence Win Rate: {patterns.get('high_confidence_win_rate', 0):.2%}")
                        if patterns.get('timeframe_stats'):
                            print("\nBy Timeframe:")
                            for tf, stats in patterns['timeframe_stats'].items():
                                total = stats['wins'] + stats['losses']
                                wr = stats['wins'] / total if total > 0 else 0
                                print(f"  {tf}: {stats['wins']}W / {stats['losses']}L ({wr:.2%})")
                
                # Show recent trades
                print("\n" + "-" * 50)
                print("RECENT TRADES (Last 5)")
                print("-" * 50)
                recent = sorted(trades, key=lambda t: t.get("entry_time", ""), reverse=True)[:5]
                for trade in recent:
                    status = "✅ Closed" if trade.get("exit_time") else "⏳ Open"
                    pnl_str = f"${trade.get('pnl', 0):.2f}" if trade.get("pnl") is not None else "N/A"
                    print(f"{status} | {trade.get('ticker', 'N/A')} | {trade.get('side', 'N/A')} | P/L: {pnl_str}")
        except Exception as e:
            log_exception(
                "Error in menu function",
                e,
                component="menu",
                function="_select_risk_profile",
                is_hard_error=False
            )
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _performance_report(self):
        """Display performance report."""
        try:
            print("\nGenerating performance report...")
            print("This may take a moment...")
            
            # Ask if user wants to filter by ticker
            ticker_filter = input("Filter by ticker? (Enter ticker symbol or press Enter for all): ").strip().upper()
            ticker = ticker_filter if ticker_filter else None
            
            # Call with error handling wrapper
            try:
                report = generate_performance_report(ticker=ticker)
            except KeyError as ke:
                # Handle KeyError specifically - might be from cached bytecode
                log_exception(
                    f"KeyError in performance report (possibly cached bytecode): {ke}",
                    ke,
                    component="menu",
                    function="_performance_report",
                    is_hard_error=False
                )
                report = f"""
Performance Report - Error
{'=' * 50}
Error: Missing key in metrics: {ke}
This may be due to cached Python bytecode (.pyc files).

Please try:
1. Delete __pycache__ folders in V15 directory
2. Restart the application

Error details logged to logs/error.log
"""
            except Exception as e:
                # Catch any other errors
                log_exception(
                    "Error generating performance report",
                    e,
                    component="menu",
                    function="_performance_report",
                    is_hard_error=False
                )
                report = f"""
Performance Report - Error
{'=' * 50}
Error generating performance report: {type(e).__name__}: {str(e)}

Please check logs/error.log for details.
"""
            
            print("\n" + "=" * 70)
            print(report)
            print("=" * 70)
            
            # Option to export
            export = input("\nExport report to file? (y/n): ").strip().lower()
            if export == 'y':
                try:
                    from ..core.portable_paths import get_path
                    from datetime import datetime
                    history_dir = get_path('history')
                    report_file = history_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(report_file, 'w') as f:
                        f.write(report)
                    print(f"✅ Report saved to: {report_file}")
                except Exception as e:
                    log_exception(
                        "Error saving performance report",
                        e,
                        component="menu",
                        function="_performance_report",
                        is_hard_error=False
                    )
                    print(f"❌ Error saving report: {e}")
        except Exception as e:
            # Ensure logger is available before logging
            try:
                log_exception(
                    "Error generating performance report",
                    e,
                    component="menu",
                    function="_performance_report",
                    is_hard_error=False
                )
            except Exception as log_err:
                # If logging fails, at least print to console
                print(f"\n⚠️  Error logger also failed: {log_err}")
            print(f"\n❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _handle_analysis_menu(self):
        """Handle core analysis menu with trading interface."""
        while True:
            self._display_analysis_menu()
            choice = input("\nEnter choice: ").strip().upper()

            if choice == "0":
                break
            elif choice == "1":
                self._market_assessment()
            elif choice == "2":
                self._strategy_trade_interface(strategy_type="investment")
            elif choice == "3":
                self._strategy_trade_interface(strategy_type="cfd")
            else:
                print("Invalid choice. Please try again.")

    def _display_analysis_menu(self):
        """Render the Core Analysis submenu."""
        print("\n" + "=" * 70)
        print("  CORE ANALYSIS & TRADING INTERFACE")
        print("=" * 70)
        print("\n1. Market Assessment & Prediction Overview")
        print("2. Investment Strategy Trade Suggestions")
        print("3. CFD Strategy Trade Suggestions")
        print("\n0. Back to Main Menu")
        print("-" * 70)

    def _market_assessment(self):
        """Generate a quick market snapshot for one or more tickers."""
        tickers = self._prompt_for_tickers()
        if not tickers:
            return

        timeframe_input = input(
            "Enter timeframe(s) (comma separated, 'all', or leave blank for 1d): "
        ).strip()
        intervals = self._resolve_timeframes(timeframe_input, strategy_type=None)
        if not intervals:
            print("No valid timeframes provided.")
            return

        print("\nFetching market data. This may take a few moments...\n")
        for ticker in tickers:
            for interval in intervals:
                snapshot = self._generate_market_snapshot(ticker, interval)
                self._render_market_snapshot(snapshot)

        input("\nPress Enter to continue...")

    def _strategy_trade_interface(self, strategy_type: str):
        """
        Suggest trades for the requested strategy type.

        Args:
            strategy_type: "investment" or "cfd"
        """
        strategy_label = "Investment" if strategy_type == "investment" else "CFD"
        tickers = self._prompt_for_tickers()
        if not tickers:
            return

        default_prompt = (
            "1d" if strategy_type == "investment" else "1h"
        )
        timeframe_input = input(
            f"Enter {strategy_label} timeframe(s) "
            f"(comma separated, 'all', or leave blank for {default_prompt}): "
        ).strip()
        intervals = self._resolve_timeframes(timeframe_input, strategy_type=strategy_type)
        if not intervals:
            print("No valid timeframes provided for this strategy.")
            return

        equity = self._prompt_for_equity()
        print(f"\nGenerating {strategy_label} trade suggestions...\n")

        for ticker in tickers:
            for interval in intervals:
                recommendation = self._build_trade_recommendation(
                    ticker=ticker,
                    timeframe=interval,
                    strategy_type=strategy_type,
                    equity=equity,
                )
                self._render_trade_recommendation(recommendation)

        input("\nPress Enter to continue...")

    def _prompt_for_tickers(self) -> List[str]:
        """Prompt user for tickers."""
        raw = input(
            "\nEnter ticker symbol(s) (comma separated, e.g., AAPL, MSFT): "
        ).strip().upper()
        if not raw:
            print("⚠️  No tickers provided.")
            return []
        tickers = [t.strip() for t in raw.split(",") if t.strip()]
        if not tickers:
            print("⚠️  Could not parse any tickers.")
            return []
        return tickers

    def _resolve_timeframes(self, user_input: str, strategy_type: Optional[str]) -> List[str]:
        """
        Resolve user timeframe input into a validated list.

        Args:
            user_input: Raw string from prompt
            strategy_type: Optional strategy filter ("investment", "cfd", or None)
        """
        if not user_input:
            default = "1d" if strategy_type == "investment" else "1h"
            normalized = self._normalize_timeframe(default)
            return [normalized] if normalized else []

        token = user_input.strip().lower()
        if token in {"all", "*"}:
            base = ALL_TIMEFRAMES
        elif token in {"investment", "invest", "long"}:
            base = INVESTMENT_TIMEFRAMES
        elif token in {"cfd", "trading", "short"}:
            base = CFD_TIMEFRAMES
        else:
            intervals = []
            for chunk in user_input.split(","):
                normalized = self._normalize_timeframe(chunk.strip())
                if not normalized:
                    print(f"⚠️  Skipping invalid timeframe: {chunk.strip()}")
                    continue
                if strategy_type == "investment" and not is_investment_timeframe(normalized):
                    print(f"⚠️  {normalized} is not an investment timeframe.")
                    continue
                if strategy_type == "cfd" and not is_cfd_timeframe(normalized):
                    print(f"⚠️  {normalized} is not a CFD timeframe.")
                    continue
                intervals.append(normalized)
            return list(dict.fromkeys(intervals))  # Preserve order, remove duplicates

        if strategy_type == "investment":
            base = [tf for tf in base if is_investment_timeframe(tf)]
        elif strategy_type == "cfd":
            base = [tf for tf in base if is_cfd_timeframe(tf)]

        return list(base)

    def _normalize_timeframe(self, timeframe: str) -> Optional[str]:
        """Normalize timeframe input to canonical form."""
        if not timeframe:
            return None
        lowercase = timeframe.strip().lower()
        for tf in ALL_TIMEFRAMES:
            if tf.lower() == lowercase:
                return tf
        return None

    def _generate_market_snapshot(self, ticker: str, timeframe: str) -> Dict:
        """Generate a quick market snapshot with indicators and prediction."""
        snapshot: Dict[str, Optional[float]] = {
            "ticker": ticker,
            "timeframe": timeframe,
            "error": None,
            "current_price": None,
            "change_pct": None,
            "rsi": None,
            "sma": None,
            "ema": None,
            "prediction": None,
        }

        try:
            df = self._execute_async(fetch_prices, ticker, timeframe)
        except Exception as e:
            snapshot["error"] = f"Data fetch error: {e}"
            log_exception(
                "Market assessment fetch failure",
                e,
                component="menu",
                function="_generate_market_snapshot",
            )
            return snapshot

        if df is None or df.empty:
            snapshot["error"] = "No market data returned."
            return snapshot

        if "Close" not in df.columns:
            snapshot["error"] = "Price data missing 'Close' column."
            return snapshot

        close_series = df["Close"]
        snapshot["current_price"] = float(close_series.iloc[-1])
        if len(close_series) > 1:
            prev = float(close_series.iloc[-2])
            snapshot["change_pct"] = (
                ((snapshot["current_price"] - prev) / prev) * 100 if prev else 0.0
            )
        else:
            snapshot["change_pct"] = 0.0

        try:
            if len(close_series) >= 15:
                snapshot["rsi"] = float(rsi(close_series, period=14).iloc[-1])
        except Exception:
            snapshot["rsi"] = None

        try:
            if len(close_series) >= 20:
                snapshot["sma"] = float(sma(close_series, period=20).iloc[-1])
                snapshot["ema"] = float(ema(close_series, period=20).iloc[-1])
        except Exception:
            snapshot["sma"] = snapshot["ema"] = None

        try:
            model = get_model(timeframe)
            prediction = self._execute_async(model.predict, ticker, df)
            snapshot["prediction"] = prediction or self._default_prediction(timeframe, ticker)
        except Exception as e:
            snapshot["prediction"] = self._default_prediction(timeframe, ticker)
            log_exception(
                "Market assessment prediction failure",
                e,
                component="menu",
                function="_generate_market_snapshot",
                is_hard_error=False,
            )

        return snapshot

    def _render_market_snapshot(self, snapshot: Dict) -> None:
        """Print market snapshot information."""
        print("-" * 70)
        print(f"{snapshot.get('ticker')} @ {snapshot.get('timeframe')}")

        if snapshot.get("error"):
            print(f"  ❌ {snapshot['error']}")
            return

        price = snapshot.get("current_price")
        if price is not None:
            change = snapshot.get("change_pct", 0.0) or 0.0
            print(f"  Price: ${price:.2f} ({change:+.2f}%)")

        if snapshot.get("rsi") is not None:
            print(f"  RSI(14): {snapshot['rsi']:.2f}")
        if snapshot.get("sma") is not None and snapshot.get("ema") is not None:
            diff = price - snapshot["sma"] if price and snapshot["sma"] else 0.0
            print(
                f"  SMA20: ${snapshot['sma']:.2f} | EMA20: ${snapshot['ema']:.2f} "
                f"| Price vs SMA: {diff:+.2f}"
            )

        prediction = snapshot.get("prediction") or {}
        movement = prediction.get("prediction", 0.0)
        confidence = prediction.get("confidence", 0.5) * 100
        range_low = prediction.get("range_low", 0.0)
        range_high = prediction.get("range_high", 0.0)
        print(
            f"  Model: {movement:+.2f}% move | Confidence {confidence:.1f}% "
            f"| Range [{range_low:.2f}%, {range_high:.2f}%]"
        )

    def _prompt_for_equity(self) -> float:
        """Prompt user for account equity with fallback."""
        default_equity = self._get_cached_equity()
        prompt = f"Enter account equity (press Enter to use ${default_equity:,.2f}): "
        raw = input(prompt).strip()
        if raw:
            try:
                equity = float(raw)
                if equity <= 0:
                    raise ValueError
                self._cached_equity = equity
                return equity
            except ValueError:
                print("⚠️  Invalid equity input. Using default.")
        return default_equity

    def _get_cached_equity(self) -> float:
        """Retrieve cached equity or load from equity monitor/config."""
        if hasattr(self, "_cached_equity") and self._cached_equity > 0:
            return self._cached_equity
        try:
            equity_monitor = get_equity_monitor()
            equity = equity_monitor.get_current_equity()
        except Exception:
            equity = 0.0
        if equity <= 0:
            equity = 10000.0  # Sensible default when no data is available
        self._cached_equity = equity
        return equity

    def _build_trade_recommendation(
        self,
        ticker: str,
        timeframe: str,
        strategy_type: str,
        equity: float,
    ) -> Dict:
        """Assemble a trade suggestion for the requested strategy."""
        recommendation: Dict = {
            "ticker": ticker,
            "timeframe": timeframe,
            "strategy": strategy_type,
            "error": None,
            "prediction": None,
            "direction": "NEUTRAL",
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "position_size": None,
            "risk_amount": None,
            "block_reason": None,
            "skip_reason": None,
        }

        asset_category = "stable" if strategy_type == "investment" else "high"

        try:
            df = self._execute_async(fetch_prices, ticker, timeframe)
        except Exception as e:
            recommendation["error"] = f"Data fetch error: {e}"
            log_exception(
                "Trade interface fetch failure",
                e,
                component="menu",
                function="_build_trade_recommendation",
                is_hard_error=False,
            )
            return recommendation

        if df is None or df.empty or "Close" not in df.columns:
            recommendation["error"] = "No price data available."
            return recommendation

        recommendation["entry_price"] = float(df["Close"].iloc[-1])

        try:
            model = get_model(timeframe)
            prediction = self._execute_async(model.predict, ticker, df)
        except Exception as e:
            prediction = None
            log_exception(
                "Trade interface prediction failure",
                e,
                component="menu",
                function="_build_trade_recommendation",
                is_hard_error=False,
            )

        if not prediction:
            prediction = self._default_prediction(timeframe, ticker)

        recommendation["prediction"] = prediction
        movement = prediction.get("prediction", 0.0)
        confidence = prediction.get("confidence", 0.5)

        # Sentiment/override gate
        try:
            sentiment_override = get_sentiment_override()
            blocked, reason = sentiment_override.should_block_trade(ticker)
            if blocked:
                recommendation["block_reason"] = reason
        except Exception:
            recommendation["block_reason"] = None

        thresholds = self._get_strategy_thresholds(strategy_type)
        if movement >= thresholds["long"]:
            recommendation["direction"] = "LONG"
        elif movement <= thresholds["short"]:
            recommendation["direction"] = "SHORT"
        else:
            recommendation["direction"] = "NEUTRAL"

        if recommendation["direction"] == "NEUTRAL":
            recommendation["skip_reason"] = "Signal not strong enough for this strategy."
            return recommendation

        if recommendation["block_reason"]:
            return recommendation

        if should_skip_trade(self.current_profile, confidence, asset_category):
            recommendation["skip_reason"] = (
                "Risk profile blocked trade (confidence/profile constraints)."
            )
            return recommendation

        try:
            stop_distance, atr_value = calculate_stop_loss_distance(
                df=df,
                profile=self.current_profile,
                confidence=confidence,
                asset_risk_category=asset_category,
            )
            stop_price = calculate_stop_loss_price(
                entry_price=recommendation["entry_price"],
                direction=recommendation["direction"],
                stop_distance=stop_distance,
            )
            recommendation["stop_price"] = stop_price
            recommendation["atr"] = atr_value
        except Exception as e:
            recommendation["error"] = f"Stop-loss calculation failed: {e}"
            return recommendation

        target_price = self._calculate_target_price(
            entry_price=recommendation["entry_price"],
            movement_pct=movement,
            direction=recommendation["direction"],
        )
        recommendation["target_price"] = target_price

        position_size, risk_amount, sizing_reason = calculate_position_size_with_profile(
            equity=equity,
            entry_price=recommendation["entry_price"],
            stop_price=recommendation["stop_price"],
            profile=self.current_profile,
            confidence=confidence,
            direction=recommendation["direction"],
        )

        recommendation["position_size"] = position_size
        recommendation["risk_amount"] = risk_amount
        recommendation["sizing_reason"] = sizing_reason

        if position_size is None:
            recommendation["skip_reason"] = sizing_reason or "Position sizing failed."

        return recommendation

    def _calculate_target_price(
        self,
        entry_price: float,
        movement_pct: float,
        direction: str,
    ) -> Optional[float]:
        """Estimate target price based on predicted movement."""
        if entry_price is None:
            return None
        move = abs(movement_pct) / 100.0
        if move == 0:
            move = 0.01  # Provide minimal movement target
        if direction == "LONG":
            return entry_price * (1 + move)
        if direction == "SHORT":
            return entry_price * (1 - move)
        return None

    def _get_strategy_thresholds(self, strategy_type: str) -> Dict[str, float]:
        """Return prediction thresholds for strategy decisioning."""
        if strategy_type == "investment":
            return {"long": 1.0, "short": -1.0}
        return {"long": 0.3, "short": -0.3}

    def _render_trade_recommendation(self, recommendation: Dict) -> None:
        """Print trade suggestion details."""
        print("-" * 70)
        header = (
            f"{recommendation.get('ticker')} @ {recommendation.get('timeframe')} "
            f"({recommendation.get('strategy').upper()} STRATEGY)"
        )
        print(header)

        if recommendation.get("error"):
            print(f"  ❌ {recommendation['error']}")
            return

        prediction = recommendation.get("prediction") or {}
        print(
            f"  Model Signal: {prediction.get('prediction', 0.0):+.2f}% "
            f"(Confidence {prediction.get('confidence', 0.5)*100:.1f}%)"
        )

        if recommendation.get("block_reason"):
            print(f"  ⚠️  Trade blocked by sentiment override: {recommendation['block_reason']}")
            return

        if recommendation.get("skip_reason"):
            print(f"  ℹ️  {recommendation['skip_reason']}")

        print(f"  Suggested Direction: {recommendation.get('direction')}")

        entry = recommendation.get("entry_price")
        stop = recommendation.get("stop_price")
        target = recommendation.get("target_price")
        if entry:
            print(f"  Entry: ${entry:.2f}")
        if stop:
            print(f"  Stop: ${stop:.2f}")
        if target:
            print(f"  Target: ${target:.2f}")

        pos_size = recommendation.get("position_size")
        if pos_size:
            risk_amt = recommendation.get("risk_amount") or 0.0
            print(f"  Position Size: {pos_size:,.2f} units | Risk ${risk_amt:,.2f}")
        elif recommendation.get("sizing_reason"):
            print(f"  Position sizing note: {recommendation['sizing_reason']}")

    @staticmethod
    def _execute_async(async_fn, *args, **kwargs):
        """Run async functions safely from sync context."""
        try:
            return asyncio.run(async_fn(*args, **kwargs))
        except RuntimeError as exc:
            if "asyncio.run()" in str(exc):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(async_fn(*args, **kwargs))
                finally:
                    loop.close()
            raise

    @staticmethod
    def _default_prediction(timeframe: str, ticker: str) -> Dict:
        """Fallback prediction dictionary."""
        return {
            "prediction": 0.0,
            "confidence": 0.5,
            "range_low": -1.0,
            "range_high": 1.0,
            "timeframe": timeframe,
            "ticker": ticker,
            "model_agreement": 0.0,
            "is_default": True,
        }
    
    def _handle_learning_menu(self):
        """Handle learning menu."""
        while True:
            print("\n" + "=" * 70)
            print("  LEARNING & TRAINING MENU")
            print("=" * 70)
            print("\n1. Start/Stop Continuous Training")
            print("2. Review Training Performance")
            print("3. Trigger Manual Retraining")
            print("4. Reset Learned Model")
            print("\n0. Back to Main Menu")
            print("-" * 70)
            
            choice = input("\nEnter choice: ").strip().upper()
            
            if choice == "0":
                break
            elif choice == "1":
                self._toggle_continuous_training()
            elif choice == "2":
                self._review_training_performance()
            elif choice == "3":
                self._trigger_manual_retraining()
            elif choice == "4":
                self._reset_learned_model()
            else:
                print("Invalid choice.")
    
    def _toggle_continuous_training(self):
        """Start or stop continuous training."""
        skip_pause = False
        try:
            # Try relative import first, fallback to absolute
            try:
                from ..learning.continuous_service import get_continuous_learning_service
            except (ImportError, ValueError):
                from learning.continuous_service import get_continuous_learning_service
            try:
                from ..learning.prediction_storage import get_prediction_storage
            except (ImportError, ValueError):
                from learning.prediction_storage import get_prediction_storage
            
            service = get_continuous_learning_service()
            status = service.get_status()
            
            print(f"\nContinuous Learning Status:")
            print(f"  Running: {'✅ Yes' if status['running'] else '❌ No'}")
            print(f"  Check Interval: {status['check_interval_hours']} hours")
            print(f"  Last Check: {status['last_check'] or 'Never'}")
            print(f"  Last Retrain: {status['last_retrain'] or 'Never'}")
            print(f"  Available Trades: {status.get('available_trades', 0)}")
            if status.get("available_prediction_samples") is not None:
                print(f"  Prediction Samples: {status['available_prediction_samples']}")
            if status.get("total_samples") is not None:
                print(f"  Total Samples: {status['total_samples']}")
            print(f"  Should Retrain: {'Yes' if status['should_retrain'] else 'No'}")
            refresh_cycles = 3 if status['running'] else 1
            self._display_live_prediction_feed(refresh_cycles=refresh_cycles)
            self._print_prediction_history(get_prediction_storage())
            
            if status['running']:
                action = input("\nStop continuous training? (y/n): ").strip().lower()
                if action == 'y':
                    if service.stop():
                        print("✅ Continuous training stopped")
                    else:
                        print("❌ Failed to stop continuous training")
            else:
                action = input("\nStart continuous training? (y/n): ").strip().lower()
                if action == 'y':
                    if service.start():
                        print("✅ Continuous training started")
                        try:
                            storage = get_prediction_storage()
                        except Exception:
                            storage = None
                        if storage:
                            self._seed_prediction_feed(storage, per_interval=1)
                        else:
                            storage = get_prediction_storage()
                        self._stream_interval_predictions(storage)
                        skip_pause = True
                    else:
                        print("❌ Failed to start continuous training")
        
        except Exception as e:
            log_exception(
                "Error toggling continuous training",
                e,
                component="menu",
                function="_toggle_continuous_training",
                is_hard_error=False
            )
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        if not skip_pause:
            input("\nPress Enter to continue...")

    def _print_prediction_history(self, storage, limit: int = 6) -> None:
        """Display recent prediction records with live readout fallback."""
        live_entries = self._fetch_live_feed_entries(limit)
        if live_entries:
            self._render_live_feed(live_entries, title="LIVE PREDICTION FEED (snapshot)")
            return

        try:
            records = storage.get_predictions()
        except Exception:
            records = []
        if not records:
            print("\nNo prediction history available yet.")
            return

        sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]

        print("\n" + "-" * 70)
        print("RECENT PREDICTIONS (Most recent first)")
        print("-" * 70)
        header = f"{'Ticker':<8} {'Interval':<8} {'Prediction%':>11} {'Confidence':>11} {'Status':>12} {'Recorded':>18}"
        print(header)
        print("-" * len(header))

        for record in sorted_records:
            prediction_pct = f"{record.predicted_price:.2f}" if record.predicted_price is not None else "N/A"
            confidence_pct = self._format_confidence(record.confidence)
            status = record.evaluation_status or "pending"
            timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M")
            print(f"{record.ticker[:8]:<8} {record.interval[:8]:<8} {prediction_pct:>11} {confidence_pct:>11} {status:>12} {timestamp:>18}")

    def _display_live_prediction_feed(self, refresh_cycles: int = 1, refresh_delay: float = 1.0, limit: int = 6) -> None:
        """Stream the live prediction feed for the requested number of refresh cycles."""
        refresh_cycles = max(1, refresh_cycles)
        for cycle in range(refresh_cycles):
            entries = self._fetch_live_feed_entries(limit)
            if not entries:
                if cycle == 0:
                    print("\nNo live prediction activity yet. Predictions will appear here once generated.")
                break
            if cycle > 0:
                print("\n⏳ Refreshing live prediction feed...\n")
            self._render_live_feed(entries, title="LIVE PREDICTION FEED (auto-refresh)")
            if cycle < refresh_cycles - 1:
                time.sleep(max(0.5, refresh_delay))

    def _render_live_feed(self, entries, title: str) -> None:
        """Nicely format live prediction feed entries."""
        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)
        header = f"{'Ticker':<8} {'Interval':<8} {'Price':>9} {'Confidence':>11} {'Event':>10} {'Accuracy':>9} {'Updated':>10}"
        print(header)
        print("-" * len(header))
        for entry in entries:
            ticker = (entry.get("ticker") or "")[:8]
            interval = (entry.get("interval") or "")[:8]
            price = entry.get("predicted_price")
            price_txt = f"{price:.2f}" if isinstance(price, (int, float)) else "N/A"
            confidence_txt = self._format_confidence(entry.get("confidence"))
            event = entry.get("event", entry.get("status", "pending")) or "pending"
            accuracy = entry.get("accuracy")
            accuracy_txt = f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else "--"
            timestamp = entry.get("timestamp", "")[-8:]
            print(f"{ticker:<8} {interval:<8} {price_txt:>9} {confidence_txt:>11} {event:>10} {accuracy_txt:>9} {timestamp:>10}")

    def _fetch_live_feed_entries(self, limit: int) -> list:
        """Fetch live feed entries if the live readout module is available."""
        try:
            try:
                from ..learning.live_readout import get_live_prediction_readout
            except (ImportError, ValueError):
                from learning.live_readout import get_live_prediction_readout
        except (ImportError, ValueError):
            return []

        try:
            feed = get_live_prediction_readout()
            return feed.get_recent_entries(limit)
        except Exception:
            return []

    @staticmethod
    def _format_confidence(confidence: Optional[float]) -> str:
        """Render confidence as a 1-10 rating whether stored as 0-1 or 1-10."""
        if confidence is None:
            return "N/A"
        if 0 <= confidence <= 1:
            # Convert 0-1 scale to 1-10 scale: 0.0 -> 1.0, 0.5 -> 5.5, 1.0 -> 10.0
            rating = 1 + (confidence * 9)
            return f"{rating:.1f}"
        # Already on 1-10 scale
        return f"{confidence:.1f}"

    def _stream_interval_predictions(self, storage, refresh_seconds: float = 3.0) -> None:
        """Continuously display interval predictions while learning runs.
        
        Only prints each interval's prediction once per interval:
        - On startup: prints all predictions
        - After that: prints each prediction only when its interval elapses
        (e.g., 1m predictions print once per minute, 1d predictions print once per day)
        """
        print("\nStreaming interval predictions. Press Ctrl+C to stop.\n")
        self._seed_prediction_feed(storage, per_interval=1)
        
        # Track when each interval's prediction was last printed
        # Key: interval string (e.g., "1m", "1d"), Value: timestamp of last print
        last_print_times: Dict[str, float] = {}
        is_startup = True  # Flag to print all predictions on startup
        
        try:
            while True:
                current_time = time.time()
                predictions = storage.get_predictions()
                
                if not predictions:
                    created = self._seed_prediction_feed(storage, per_interval=1)
                    if created:
                        predictions = storage.get_predictions()
                
                # Separate active and elapsed predictions
                all_active_predictions = [p for p in predictions if p.evaluation_status != "evaluated"]
                elapsed_predictions = [p for p in predictions if p.evaluation_status == "evaluated"]
                
                # Filter active predictions to only show those that should be printed
                active_predictions_to_show = []
                
                if is_startup:
                    # On startup, show all active predictions
                    active_predictions_to_show = all_active_predictions
                    # Initialize last_print_times for all intervals found
                    for record in all_active_predictions:
                        interval = record.interval
                        if interval not in last_print_times:
                            last_print_times[interval] = current_time
                    is_startup = False
                else:
                    # After startup, only show predictions whose interval has elapsed
                    for record in all_active_predictions:
                        interval = record.interval
                        interval_duration = get_timeframe_duration_seconds(interval)
                        
                        if interval_duration is None:
                            # Unknown interval, show it (fallback behavior)
                            active_predictions_to_show.append(record)
                            continue
                        
                        # Check if we've printed this interval before
                        if interval not in last_print_times:
                            # First time seeing this interval, print it
                            active_predictions_to_show.append(record)
                            last_print_times[interval] = current_time
                        else:
                            # Check if interval has elapsed since last print
                            time_since_last_print = current_time - last_print_times[interval]
                            if time_since_last_print >= interval_duration:
                                # Interval has elapsed, print this prediction
                                active_predictions_to_show.append(record)
                                last_print_times[interval] = current_time
                
                # Only update display if there are predictions to show
                if active_predictions_to_show or elapsed_predictions:
                    self._clear_console()
                    
                    # Display active predictions
                    if active_predictions_to_show:
                        print("ACTIVE INTERVAL PREDICTIONS")
                        print("=" * 80)
                        header = (
                            f"{'Ticker':<8} {'Interval':<8} {'Target':>10} {'Low':>10} "
                            f"{'High':>10} {'Rating':>12} {'Accuracy':>10} {'Status':>10}"
                        )
                        print(header)
                        print("-" * len(header))
                        sorted_predictions = sorted(
                            active_predictions_to_show,
                            key=lambda r: (r.interval, r.timestamp),
                            reverse=True
                        )
                        for record in sorted_predictions:
                            target = f"{record.predicted_price:.2f}" if record.predicted_price is not None else "N/A"
                            low = f"{record.predicted_range_low:.2f}" if record.predicted_range_low is not None else "N/A"
                            high = f"{record.predicted_range_high:.2f}" if record.predicted_range_high is not None else "N/A"
                            confidence = self._format_confidence(record.confidence)
                            accuracy = ""
                            status = record.evaluation_status or "pending"
                            print(
                                f"{record.ticker[:8]:<8} {record.interval[:8]:<8} {target:>10} {low:>10} "
                                f"{high:>10} {confidence:>12} {accuracy:>10} {status:>10}"
                            )
                    elif not predictions:
                        print("ACTIVE INTERVAL PREDICTIONS")
                        print("=" * 80)
                        print("No predictions available yet. Waiting for the learning engine...")
                    
                    # Display elapsed predictions (always show if available)
                    if elapsed_predictions:
                        print("\nELAPSED PREDICTIONS (Evaluated)")
                        print("=" * 80)
                        elapsed_header = (
                            f"{'Ticker':<8} {'Interval':<8} {'Predicted':>12} {'Actual High':>12} "
                            f"{'Actual Low':>12} {'Actual Close':>12} {'Accuracy':>10}"
                        )
                        print(elapsed_header)
                        print("-" * len(elapsed_header))
                        sorted_elapsed = sorted(
                            elapsed_predictions,
                            key=lambda r: (r.interval, r.timestamp),
                            reverse=True
                        )
                        for record in sorted_elapsed:
                            predicted = f"{record.predicted_price:.2f}" if record.predicted_price is not None else "N/A"
                            actual_high = (
                                f"{record.actual_high:.2f}" 
                                if record.actual_high is not None 
                                else "N/A"
                            )
                            actual_low = (
                                f"{record.actual_low:.2f}" 
                                if record.actual_low is not None 
                                else "N/A"
                            )
                            actual_close = (
                                f"{record.actual_close:.2f}" 
                                if record.actual_close is not None 
                                else (f"{record.actual_price:.2f}" if record.actual_price is not None else "N/A")
                            )
                            accuracy = (
                                f"{record.accuracy_score:.2f}"
                                if record.accuracy_score is not None
                                else "N/A"
                            )
                            print(
                                f"{record.ticker[:8]:<8} {record.interval[:8]:<8} {predicted:>12} {actual_high:>12} "
                                f"{actual_low:>12} {actual_close:>12} {accuracy:>10}"
                            )
                    
                    print("\n(Streaming... Press Ctrl+C to return to the menu.)")
                
                self._seed_prediction_feed(storage, per_interval=1)
                time.sleep(max(1.0, refresh_seconds))
        except KeyboardInterrupt:
            print("\n\nStopping prediction stream. Returning to menu...")
        except Exception as stream_error:
            print(f"\n❌ Prediction stream error: {stream_error}")

    @staticmethod
    def _clear_console() -> None:
        """Best-effort console clear for streaming displays."""
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            print("\n" * 2)
    
    def _review_training_performance(self):
        """Review training performance and history."""
        try:
            # Try relative import first, fallback to absolute
            try:
                from ..learning.model_updater import get_model_updater
            except (ImportError, ValueError):
                from learning.model_updater import get_model_updater
            
            updater = get_model_updater()
            history = updater.get_model_version_history()
            
            print("\nTraining Performance History")
            print("-" * 70)
            
            if not history:
                print("No training history available.")
            else:
                for i, version in enumerate(history):
                    print(f"\nVersion {i+1}: {version.get('model_version', 'Unknown')}")
                    print(f"  Trained: {version.get('timestamp', 'Unknown')}")
                    print(f"  Training Samples: {version.get('training_data_size', 0)}")
                    metrics = version.get('performance_metrics', {})
                    print(f"  Timeframes Trained: {metrics.get('timeframes_trained', 0)}")
            
            latest = updater.get_latest_model_version()
            if latest:
                print(f"\nLatest Model: {latest.get('model_version', 'Unknown')}")
                print(f"  Trained: {latest.get('timestamp', 'Unknown')}")
        
        except Exception as e:
            log_exception(
                "Error reviewing training performance",
                e,
                component="menu",
                function="_review_training_performance",
                is_hard_error=False
            )
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _trigger_manual_retraining(self):
        """Manually trigger model retraining."""
        try:
            # Try relative import first, fallback to absolute
            try:
                from ..learning.continuous_service import get_continuous_learning_service
            except (ImportError, ValueError):
                from learning.continuous_service import get_continuous_learning_service
            
            service = get_continuous_learning_service()
            
            print("\nTriggering manual retraining...")
            print("This may take several minutes...")
            
            result = service.trigger_retrain()
            
            if result.get("retrained"):
                print(f"\n✅ Retraining completed!")
                print(f"  Model Version: {result.get('model_version')}")
                print(f"  Training Samples: {result.get('training_samples')}")
                print(f"  Timeframes Trained: {result.get('timeframes_trained')}")
            else:
                print(f"\n❌ Retraining not performed")
                print(f"  Reason: {result.get('reason', result.get('error', 'Unknown'))}")
                if 'available_trades' in result:
                    print(f"  Available Trades: {result['available_trades']}")
                if 'available_predictions' in result:
                    print(f"  Prediction Samples: {result['available_predictions']}")
                if 'total_samples' in result:
                    print(f"  Total Samples: {result['total_samples']}")
        
        except Exception as e:
            log_exception(
                "Error triggering manual retraining",
                e,
                component="menu",
                function="_trigger_manual_retraining",
                is_hard_error=False
            )
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _reset_learned_model(self):
        """Reset learned model to initial state."""
        try:
            # Try relative imports first, fallback to absolute
            try:
                from ..learning.model_updater import get_model_updater
                from ..model.unified_model import get_model
                from ..core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES
            except (ImportError, ValueError):
                from learning.model_updater import get_model_updater
                from model.unified_model import get_model
                from core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES
            
            confirm = input("\n⚠️  Are you sure you want to reset all learned models? This cannot be undone. Type 'yes' to confirm: ").strip().lower()
            
            if confirm != 'yes':
                print("Reset cancelled.")
                input("\nPress Enter to continue...")
                return
            
            print("\nResetting models...")
            
            # Reset each model
            all_timeframes = CFD_TIMEFRAMES + INVESTMENT_TIMEFRAMES
            reset_count = 0
            
            for timeframe in all_timeframes:
                try:
                    model = get_model(timeframe)
                    # Clear model weights/data
                    model.is_trained = False
                    model.save()
                    reset_count += 1
                except Exception:
                    pass
            
            # Clear model updater history
            updater = get_model_updater()
            updater.model_versions = []
            updater.last_retrain_date = None
            updater._save_model_history()
            
            print(f"✅ Reset {reset_count} models")
            print("Models will need to be retrained from scratch.")
        
        except Exception as e:
            log_exception(
                "Error resetting learned model",
                e,
                component="menu",
                function="_reset_learned_model",
                is_hard_error=False
            )
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _handle_data_menu(self):
        """Handle data menu (placeholder - extend from V13)."""
        print("\nData menu (V13 functionality - to be extended)")
        input("Press Enter to continue...")
    
    def _handle_system_menu(self):
        """Handle system menu."""
        while True:
            print("\n" + "=" * 70)
            print("  SYSTEM & MAINTENANCE MENU")
            print("=" * 70)
            print("\n1. Ticker List Audit/Refresh")
            print("2. Cache Management")
            print("3. Update Data Providers/API Keys")
            print("4. Check for Updates/Patchnotes")
            print("5. Settings (Risk Level & Constant Learning)")
            print("\n0. Back to Main Menu")
            print("-" * 70)
            
            choice = input("\nEnter choice: ").strip().upper()
            
            if choice == "0":
                break
            elif choice == "1":
                self._ticker_list_audit()
            elif choice == "2":
                self._cache_management()
            elif choice == "3":
                self._update_data_providers()
            elif choice == "4":
                self._check_for_updates()
            elif choice == "5":
                self._settings_menu()
            else:
                print("Invalid choice.")
    
    def _ticker_list_audit(self):
        """Ticker list audit and refresh."""
        print("\nTicker List Audit")
        print("-" * 70)
        
        # Get ticker list file path
        ticker_file = input("Enter ticker list file path (or press Enter for data/tickers.txt): ").strip()
        if not ticker_file:
            from core.portable_paths import get_data_path
            ticker_file = str(get_data_path() / 'tickers.txt')
        
        ticker_path = Path(ticker_file)
        if not ticker_path.exists():
            print(f"❌ File not found: {ticker_file}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\nAuditing ticker list: {ticker_file}")
        print("This may take a few minutes...")
        
        try:
            from core.ticker_auditor import get_ticker_auditor
            import asyncio
            
            auditor = get_ticker_auditor()
            
            # Load ticker list
            tickers = auditor._load_ticker_list(ticker_path)
            print(f"Loaded {len(tickers)} tickers")
            
            # Audit
            result = asyncio.run(auditor.audit_ticker_list(tickers, auto_fix=False))
            
            # Display results
            print("\n" + result["report"])
            
            # Ask if user wants to update
            update = input("\nUpdate ticker list? (y/n): ").strip().lower()
            if update == 'y':
                updated_result = asyncio.run(auditor.update_ticker_list(ticker_path, remove_invalid=True))
                print(f"\n✅ Updated ticker list:")
                print(f"   Original: {updated_result['original_count']} tickers")
                print(f"   Cleaned: {updated_result['cleaned_count']} tickers")
                print(f"   Removed: {updated_result['removed_count']} tickers")
        
        except Exception as e:
            print(f"❌ Error during audit: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _cache_management(self):
        """Cache management."""
        print("\nCache Management")
        print("-" * 70)
        
        try:
            from core.cache_manager import get_cache_manager
            
            manager = get_cache_manager()
            stats = manager.get_cache_statistics()
            
            print(f"\nCache Statistics:")
            print(f"  Total Size: {stats['total_size_mb']:.2f} MB")
            print(f"  File Count: {stats['file_count']}")
            print(f"  Average Age: {stats['average_file_age_days']:.1f} days")
            print(f"  Oldest File: {stats['oldest_file_age_days']:.1f} days")
            
            if stats['recommendations']:
                print("\nRecommendations:")
                for rec in stats['recommendations']:
                    print(f"  ⚠️  {rec}")
            
            print("\nOptions:")
            print("1. Prune old cache files (>30 days)")
            print("2. Prune cache to size limit (1GB)")
            print("3. Clear all cache")
            print("0. Back")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                result = manager.prune_cache(max_age_days=30, dry_run=False)
                print(f"\n✅ Pruned {result['removed_count']} files")
                print(f"   Freed: {result['freed_mb']:.2f} MB")
            elif choice == "2":
                result = manager.prune_cache(max_size_mb=1000.0, dry_run=False)
                print(f"\n✅ Pruned {result['removed_count']} files")
                print(f"   Freed: {result['freed_mb']:.2f} MB")
                print(f"   Remaining: {result['remaining_size_mb']:.2f} MB")
            elif choice == "3":
                confirm = input("⚠️  Are you sure? Type 'yes' to confirm: ").strip().lower()
                if confirm == 'yes':
                    if manager.clear_cache(confirm=True):
                        print("\n✅ Cache cleared")
                    else:
                        print("\n❌ Failed to clear cache")
                else:
                    print("\nCancelled")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _update_data_providers(self):
        """Update data provider settings."""
        print("\nUpdate Data Providers/API Keys")
        print("-" * 70)
        print("\nCurrent provider settings are in data/config_v15.json")
        print("Edit the file directly or use the Settings page in Streamlit UI.")
        print("\nSupported providers:")
        print("  - Yahoo Finance (no API key required)")
        print("  - Alpha Vantage (requires API key)")
        print("  - Polygon.io (requires API key)")
        input("\nPress Enter to continue...")
    
    def _check_for_updates(self):
        """Check for updates and display patchnotes."""
        print("\nCheck for Updates/Patchnotes")
        print("-" * 70)
        
        try:
            from pathlib import Path
            patchnotes_file = Path(__file__).parent.parent / 'PATCHNOTES.md'
            
            if patchnotes_file.exists():
                print("\nRecent Updates:")
                print("-" * 70)
                with open(patchnotes_file, 'r') as f:
                    lines = f.readlines()
                    # Display first 50 lines
                    for line in lines[:50]:
                        print(line.rstrip())
                    if len(lines) > 50:
                        print("\n... (see PATCHNOTES.md for full changelog)")
            else:
                print("⚠️  PATCHNOTES.md not found")
            
            # Check version
            try:
                from core.setup import get_data_path
                import json
                config_file = get_data_path() / 'config_v15.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        version = config.get("version", "unknown")
                        print(f"\nCurrent Version: {version}")
            except:
                pass
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def _settings_menu(self):
        """Settings menu with risk level and constant learning controls."""
        while True:
            print("\n" + "=" * 70)
            print("  SETTINGS MENU")
            print("=" * 70)
            print("\n1. Risk Level Configuration")
            print("2. Constant Learning (Function 3) Settings")
            print("3. View Constant Learning Statistics")
            print("\n0. Back to System Menu")
            print("-" * 70)
            
            choice = input("\nEnter choice: ").strip().upper()
            
            if choice == "0":
                break
            elif choice == "1":
                self._select_risk_profile()
            elif choice == "2":
                self._constant_learning_settings()
            elif choice == "3":
                self._view_constant_learning_stats()
            else:
                print("Invalid choice.")
    
    def _constant_learning_settings(self):
        """Constant learning settings configuration."""
        print("\n" + "=" * 70)
        print("  CONSTANT LEARNING SETTINGS (Function 3)")
        print("=" * 70)
        
        try:
            # Try relative imports first, fallback to absolute
            try:
                from ..learning.constant_learning_engine import get_constant_learning_engine
                from ..learning.parameter_optimizer import get_parameter_optimizer
                from ..core.timeframes import CONSTANT_LEARNING_INTERVALS
                from ..core.portable_paths import get_data_path
            except (ImportError, ValueError):
                from learning.constant_learning_engine import get_constant_learning_engine
                from learning.parameter_optimizer import get_parameter_optimizer
                from core.timeframes import CONSTANT_LEARNING_INTERVALS
                from core.portable_paths import get_data_path
            import json
            
            engine = get_constant_learning_engine()
            optimizer = get_parameter_optimizer()
            
            # Load config
            config_file = get_data_path() / 'config_v15.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            if "constant_learning" not in config:
                config["constant_learning"] = {
                    "enabled": False,
                    "intervals": CONSTANT_LEARNING_INTERVALS.copy(),
                    "evaluation_frequency_seconds": 5.0,
                    "trade_outcome_weight": 3.0,
                    "max_predictions_per_cycle": 10
                }
            
            cl_config = config["constant_learning"]
            
            # Current status
            status = engine.get_status()
            print(f"\nCurrent Status:")
            print(f"  Enabled: {'✅ Yes' if status['enabled'] else '❌ No'}")
            print(f"  Running: {'✅ Yes' if status['running'] else '❌ No'}")
            print(f"  Active Intervals: {', '.join(status['active_intervals'])}")
            print(f"  Active Tickers: {status['active_tickers_count']}")
            
            # Enable/Disable
            print("\n" + "-" * 70)
            enable = input("Enable constant learning? (y/n) [current: {}]: ".format(
                "Yes" if cl_config.get("enabled", False) else "No"
            )).strip().lower()
            
            if enable == 'y':
                cl_config["enabled"] = True
                engine.set_enabled(True)
            elif enable == 'n':
                cl_config["enabled"] = False
                engine.set_enabled(False)
            
            if cl_config["enabled"]:
                # Intervals selection
                print(f"\nAvailable intervals: {', '.join(CONSTANT_LEARNING_INTERVALS)}")
                intervals_input = input(f"Select intervals (comma-separated, or Enter for all): ").strip()
                if intervals_input:
                    selected = [i.strip() for i in intervals_input.split(',')]
                    valid_intervals = [i for i in selected if i in CONSTANT_LEARNING_INTERVALS]
                    if valid_intervals:
                        cl_config["intervals"] = valid_intervals
                        engine.set_active_intervals(valid_intervals)
                else:
                    cl_config["intervals"] = CONSTANT_LEARNING_INTERVALS.copy()
                    engine.set_active_intervals(CONSTANT_LEARNING_INTERVALS.copy())
                
                # Evaluation frequency
                freq_input = input(f"Evaluation frequency in seconds [current: {cl_config.get('evaluation_frequency_seconds', 5.0)}]: ").strip()
                if freq_input:
                    try:
                        freq = float(freq_input)
                        cl_config["evaluation_frequency_seconds"] = freq
                        engine.evaluation_frequency_seconds = freq
                    except ValueError:
                        print("Invalid input, keeping current value")
                
                # Trade outcome weight
                weight_input = input(f"Trade outcome weight (1.0-5.0) [current: {cl_config.get('trade_outcome_weight', 3.0)}]: ").strip()
                if weight_input:
                    try:
                        weight = float(weight_input)
                        if 1.0 <= weight <= 5.0:
                            cl_config["trade_outcome_weight"] = weight
                            optimizer.set_trade_outcome_weight(weight)
                        else:
                            print("Weight must be between 1.0 and 5.0")
                    except ValueError:
                        print("Invalid input, keeping current value")
            
            # Save config
            config["constant_learning"] = cl_config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("\n✅ Settings saved!")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _view_constant_learning_stats(self):
        """View constant learning statistics."""
        print("\n" + "=" * 70)
        print("  CONSTANT LEARNING STATISTICS")
        print("=" * 70)
        
        try:
            # Try relative import first, fallback to absolute
            try:
                from ..learning.learning_statistics import get_learning_statistics
            except (ImportError, ValueError):
                from learning.learning_statistics import get_learning_statistics
            
            stats = get_learning_statistics()
            report = stats.generate_report()
            print("\n" + report)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")

