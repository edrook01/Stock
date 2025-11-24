"""
Extended logging utilities for Stock Analyzer V15.

This package intentionally shares its name with Python's standard ``logging``
module so that imports such as ``from logging import LogRecord`` continue to
work. We load the stdlib implementation under this namespace and then expose
the Stock Analyzer helpers from ``sa_logging``.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path

# Load the CPython logging implementation under a private module name so we can
# re-export its API without losing this package's submodules.
_STD_LOGGING_PATH = Path(sysconfig.get_paths()["stdlib"]) / "logging" / "__init__.py"
if not _STD_LOGGING_PATH.exists():
    raise ImportError(f"Unable to locate standard logging module at {_STD_LOGGING_PATH}")

_STD_SPEC = importlib.util.spec_from_file_location(
    "_stock_analyzer_std_logging", str(_STD_LOGGING_PATH)
)
if _STD_SPEC is None or _STD_SPEC.loader is None:
    raise ImportError("Unable to load standard logging module spec")

_stdlib_logging = importlib.util.module_from_spec(_STD_SPEC)
sys.modules.setdefault("_stock_analyzer_std_logging", _stdlib_logging)
_STD_SPEC.loader.exec_module(_stdlib_logging)

for _name, _value in vars(_stdlib_logging).items():
    # Preserve package metadata (__file__, __path__, etc.) defined by importlib.
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    if _name in globals():
        continue
    globals()[_name] = _value

# Project specific loggers and helpers live under sa_logging to avoid circular
# imports during bootstrap. Re-export them here for backwards compatibility.
from sa_logging.trade_logger import TradeLogger, get_trade_logger  # noqa: E402
from sa_logging.analyzer import (  # noqa: E402
    calculate_performance_metrics,
    compare_predicted_vs_actual,
    identify_patterns,
    generate_performance_report,
)
from sa_logging.prediction_logger import PredictionLogger, get_prediction_logger  # noqa: E402
from sa_logging.model_logger import ModelLogger, get_model_logger  # noqa: E402
from sa_logging.market_logger import MarketLogger, get_market_logger  # noqa: E402
from sa_logging.system_logger import SystemLogger, get_system_logger  # noqa: E402

_CUSTOM_EXPORTS = [
    "TradeLogger",
    "get_trade_logger",
    "calculate_performance_metrics",
    "compare_predicted_vs_actual",
    "identify_patterns",
    "generate_performance_report",
    "PredictionLogger",
    "get_prediction_logger",
    "ModelLogger",
    "get_model_logger",
    "MarketLogger",
    "get_market_logger",
    "SystemLogger",
    "get_system_logger",
]

_STD_ALL = list(getattr(_stdlib_logging, "__all__", []))
__all__ = list(dict.fromkeys(_STD_ALL + _CUSTOM_EXPORTS))
