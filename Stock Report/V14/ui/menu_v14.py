"""
V14 Menu System
Extended menu controller with V14-specific features.
"""

import sys
import asyncio
from typing import Optional
from pathlib import Path

# Import V14 modules
try:
    from ..core.portable_paths import get_path
    from ..core.prediction_scheduler import get_prediction_scheduler
    from ..model.unified_model import get_model
    from ..risk.profiles import RiskProfile, get_risk_profile
    from ..browser.automation import BrowserAutomation
    from ..sentiment.override import get_sentiment_override
    from ..logging.trade_logger import get_trade_logger
    from ..logging.analyzer import generate_performance_report
    # Import error logger
    try:
        from ..logging.error_logger import log_exception, log_error, log_warning, log_info
    except ImportError:
        # Fallback if error_logger not available
        def log_exception(*args, **kwargs): pass
        def log_error(*args, **kwargs): pass
        def log_warning(*args, **kwargs): pass
        def log_info(*args, **kwargs): pass
except ImportError:
    # Fallback for direct execution
    v14_root = Path(__file__).parent.parent
    sys.path.insert(0, str(v14_root))
    from core.portable_paths import get_path
    from core.prediction_scheduler import get_prediction_scheduler
    from model.unified_model import get_model
    from risk.profiles import RiskProfile, get_risk_profile
    from browser.automation import BrowserAutomation
    from sentiment.override import get_sentiment_override
    # Import local logging modules using importlib to avoid standard library shadowing
    import importlib.util
    import sys as sys_module
    # Temporarily remove standard library logging from cache to allow our module
    if 'logging' in sys_module.modules and hasattr(sys_module.modules['logging'], '__file__'):
        # Only remove if it's the standard library (no __file__ or in site-packages)
        logging_file = sys_module.modules['logging'].__file__
        if logging_file and 'site-packages' in logging_file:
            del sys_module.modules['logging']
    # Now import our local logging modules
    trade_logger_spec = importlib.util.spec_from_file_location(
        "v14_logging.trade_logger", v14_root / "logging" / "trade_logger.py"
    )
    trade_logger_module = importlib.util.module_from_spec(trade_logger_spec)
    sys_module.modules['v14_logging.trade_logger'] = trade_logger_module
    trade_logger_spec.loader.exec_module(trade_logger_module)
    get_trade_logger = trade_logger_module.get_trade_logger
    # Same for analyzer - need to set up the module path correctly
    analyzer_spec = importlib.util.spec_from_file_location(
        "v14_logging.analyzer", v14_root / "logging" / "analyzer.py"
    )
    analyzer_module = importlib.util.module_from_spec(analyzer_spec)
    sys_module.modules['v14_logging.analyzer'] = analyzer_module
    # Set up the trade_logger import for analyzer
    sys_module.modules['v14_logging.trade_logger'] = trade_logger_module
    analyzer_spec.loader.exec_module(analyzer_module)
    generate_performance_report = analyzer_module.generate_performance_report
    # Import error logger in fallback path
    try:
        error_logger_spec = importlib.util.spec_from_file_location(
            "v14_logging.error_logger", v14_root / "logging" / "error_logger.py"
        )
        error_logger_module = importlib.util.module_from_spec(error_logger_spec)
        sys_module.modules['v14_logging.error_logger'] = error_logger_module
        # Set up sys.modules for error_logger's imports
        sys_module.modules['logging.error_logger'] = error_logger_module
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
                logs_dir = v14_root.parent / 'logs'
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


class MenuController:
    """Main menu controller for V14."""
    
    def __init__(self):
        """Initialize menu controller."""
        self.running = True
        self.current_profile = RiskProfile.MEDIUM
        self.browser_automation = None
        
        # Load risk profile from config
        try:
            from ..core.portable_paths import get_data_path
            import json
            config_file = get_data_path() / 'config_v14.json'
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
    
    def display_main_menu(self):
        """Display main menu."""
        try:
            print("\n" + "=" * 70)
            print("  STOCK ANALYZER V14 - MAIN MENU")
            print("=" * 70)
            print(f"\nCurrent Risk Profile: {self.current_profile.value.upper()}")
            print("\n1. Core Analysis")
            print("2. Learning & Training")
            print("3. Data & Logs")
            print("4. System & Maintenance")
            print("5. V14 Features")
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
            print("  STOCK ANALYZER V14 - MAIN MENU")
            print("=" * 70)
            print("\n1. Core Analysis")
            print("2. Learning & Training")
            print("3. Data & Logs")
            print("4. System & Maintenance")
            print("5. V14 Features")
            print("\n0. Exit")
            print("-" * 70)
    
    def display_v14_features_menu(self):
        """Display V14-specific features menu."""
        print("\n" + "=" * 70)
        print("  V14 FEATURES MENU")
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
                        self._handle_v14_features_menu()
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
    
    def _handle_v14_features_menu(self):
        """Handle V14 features menu."""
        while True:
            try:
                self.display_v14_features_menu()
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
                                function="_handle_v14_features_menu",
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
                                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR [menu][_handle_v14_features_menu]: Error in performance report (5F) | Exception: {type(e).__name__}: {str(e)}\n")
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
                    log_info("User interrupted V14 features menu", component="menu", function="_handle_v14_features_menu")
                except Exception:
                    pass
                print("\n\nReturning to main menu...")
                break
            except Exception as e:
                try:
                    log_exception(
                        "Error in V14 features menu",
                        e,
                        component="menu",
                        function="_handle_v14_features_menu",
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
                            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR [menu][_handle_v14_features_menu]: Error in V14 features menu | Exception: {type(e).__name__}: {str(e)}\n")
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
            print("\nCurrent Profile: " + self.current_profile.value.upper())
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
                    config_file = get_data_path() / 'config_v14.json'
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
                    config_file = get_data_path() / 'config_v14.json'
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
                    config_file = get_data_path() / 'config_v14.json'
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
                print("  - Trading212 credentials in config_v14.json")
                
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
            from ..logging.analyzer import calculate_performance_metrics, compare_predicted_vs_actual, identify_patterns
            
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
1. Delete __pycache__ folders in V14 directory
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
        """Handle analysis menu (placeholder - extend from V13)."""
        print("\nAnalysis menu (V13 functionality - to be extended)")
        input("Press Enter to continue...")
    
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
        try:
            # Try relative import first, fallback to absolute
            try:
                from ..learning.continuous_service import get_continuous_learning_service
            except (ImportError, ValueError):
                from learning.continuous_service import get_continuous_learning_service
            
            service = get_continuous_learning_service()
            status = service.get_status()
            
            print(f"\nContinuous Learning Status:")
            print(f"  Running: {'✅ Yes' if status['running'] else '❌ No'}")
            print(f"  Check Interval: {status['check_interval_hours']} hours")
            print(f"  Last Check: {status['last_check'] or 'Never'}")
            print(f"  Last Retrain: {status['last_retrain'] or 'Never'}")
            print(f"  Available Trades: {status['available_trades']}")
            print(f"  Should Retrain: {'Yes' if status['should_retrain'] else 'No'}")
            
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
        
        input("\nPress Enter to continue...")
    
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
        print("\nCurrent provider settings are in data/config_v14.json")
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
                config_file = get_data_path() / 'config_v14.json'
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
            config_file = get_data_path() / 'config_v14.json'
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

