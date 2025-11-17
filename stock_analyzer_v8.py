#!/usr/bin/env python3
"""Stock Analyzer V8 – Predictive (Complete Build).

This release folds in all improvements that landed after the previous merge
and ships as a single, self-contained file. The deterministic bootstrapper is
still here, but the CLI has been expanded with argument-based automation, an
offline self-test harness, and a diagnostics workflow so the tool is easier to
debug on fresh machines.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import random
import os
import sys
import subprocess
import importlib
import time
import signal
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

VERSION = "V8"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(BASE_DIR, "libs")
MEM_DIR = os.path.join(BASE_DIR, "memory")
DATA_DIR = os.path.join(BASE_DIR, "data")


def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LIB_DIR, exist_ok=True)
    os.makedirs(MEM_DIR, exist_ok=True)


def ensure_lib_path() -> None:
    if LIB_DIR not in sys.path:
        sys.path.insert(0, LIB_DIR)


def print_banner() -> None:
    print(f"\n=== Stock Analyzer {VERSION} - Predictive (Complete Build) ===\n")

MEM_FILE = os.path.join(MEM_DIR, "predictions_log.csv")
DEFAULT_BAND_PERCENT = 5.0
MIN_CONFIDENCE_SATISFACTION = 8.5
DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "TSLA", "NVDA"]
DEFAULT_TICKER_CSV = os.environ.get(
    "TICKER_UNIVERSE_CSV", os.path.join(DATA_DIR, "trading212_listings.csv")
)
TICKER_UNIVERSE_API = os.environ.get("TICKER_UNIVERSE_API")
TRADING212_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSLA",
    "SPY",
    "QQQ",
    "XLK",
    "AVGO",
    "ASML",
]


# =============================================================
# DEPENDENCIES
# =============================================================

PACKAGE_SPECS: Dict[str, Tuple[str, Optional[str]]] = {
    "yfinance": ("yfinance", "0.2.66"),
    "pandas": ("pandas", "2.2.2"),
    "numpy": ("numpy", "1.26.4"),
    "ta": ("ta", "0.11.0"),
    "pandas_ta": ("pandas_ta", "0.4.71b0"),
    "sklearn": ("scikit-learn", "1.3.2"),
    "requests": ("requests", "2.32.3"),
    "tabulate": ("tabulate", "0.9.0"),
    "matplotlib": ("matplotlib", "3.8.4"),
    "torch": ("torch", None),  # optional – used for NN if available.
}

yfinance = pd = np = ta_mod = pta = skl = requests_mod = tabulate_mod = torch_mod = plt_mod = None  # type: ignore[assignment]
_INFO_CACHE: Dict[str, Dict] = {}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def divider(char: str = "─", width: int = 70) -> str:
    return char * width


def print_panel(title: str, lines: Iterable[str]) -> None:
    border = divider()
    print(border)
    print(f"{title}")
    print(border)
    for line in lines:
        print(f"• {line}")
    print(border)


def clear_screen() -> None:
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)


def guided_prompt(
    label: str,
    expectation: str,
    example: str,
    default: Optional[str] = None,
) -> str:
    default_hint = f" [default: {default}]" if default else ""
    print(
        f"\n{divider()}\n{label}{default_hint}\n- Expected: {expectation}\n- Example:  {example}\n{divider()}"
    )
    return prompt(f"{label}{default_hint}: ", default)


def install_packages_local(wheel_dir: Optional[str] = None) -> None:
    log("Checking dependencies...")

    if wheel_dir:
        if not os.path.isdir(wheel_dir):
            log(f"!! Wheel directory not found: {wheel_dir}")
            sys.exit(1)
        log(f"Using local wheel/cache directory: {wheel_dir}")

    for import_name, (pip_name, version) in PACKAGE_SPECS.items():
        optional = import_name == "torch"

        try:
            importlib.import_module(import_name)
            log(f"✔ {import_name} OK")
            continue
        except Exception:
            log(f"✘ Installing {pip_name} ...")

        pkg_str = pip_name if version is None else f"{pip_name}=={version}"

        try:
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                pkg_str,
                "--target",
                LIB_DIR,
                "--upgrade",
                "--no-warn-script-location",
            ]

            if wheel_dir:
                command.extend(["--no-index", "--find-links", wheel_dir])

            subprocess.check_call(command)
            log(f"✔ Installed {pkg_str}")
        except Exception as exc:  # pragma: no cover - runtime guard
            log(f"!! Failed: {exc}")
            if optional:
                log("Skipping optional torch")
                continue
            sys.exit(1)

    print()


def safe_import(name: str):  # type: ignore[override]
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - runtime guard
        log(f"Import error: {name} → {exc}")
        return None


def bootstrap(
    *, install_deps: bool = False, wheel_dir: Optional[str] = None, clear_after: bool = False
) -> None:
    log("Bootstrapping...")
    if install_deps:
        install_packages_local(wheel_dir=wheel_dir)
    else:
        log("Skipping dependency installation (use --install-deps to provision packages).")

    global yfinance, pd, np, ta_mod, pta, skl
    global requests_mod, tabulate_mod, torch_mod, plt_mod

    yfinance = safe_import("yfinance")
    pd = safe_import("pandas")
    np = safe_import("numpy")
    ta_mod = safe_import("ta")
    pta = safe_import("pandas_ta")
    skl = safe_import("sklearn")
    requests_mod = safe_import("requests")
    tabulate_mod = safe_import("tabulate")
    torch_mod = safe_import("torch")
    plt_mod = safe_import("matplotlib.pyplot")

    if yfinance is None or pd is None or np is None:
        print(
            "Critical imports failed. Run with --install-deps (and optionally --wheel-dir) "
            "to provision dependencies. Exiting."
        )
        sys.exit(1)

    log("Bootstrap complete.\n")
    if clear_after:
        clear_screen()


# =============================================================
# MEMORY SYSTEM
# =============================================================

HEADER = (
    "timestamp,ticker,period,horizon_bars,engine,predicted_price," "last_close," "target_time,actual_price,abs_error,pct_error\n"
)


def prompt(message: str, default: Optional[str] = None) -> str:
    try:
        value = input(message)
    except EOFError:
        value = ""
    value = value.strip()
    if value:
        return value
    return default or ""


def is_interactive() -> bool:
    return sys.stdin.isatty()


def safe_int(value: Optional[str], fallback: int) -> int:
    try:
        return int(str(value))
    except Exception:
        return fallback


def maybe_pause(pause: bool, message: str = "Press Enter to return to the menu...") -> None:
    if not pause:
        return
    try:
        input(message)
    except EOFError:
        pass


def finalize_section(pause: bool, clear: bool = False) -> None:
    maybe_pause(pause)
    if clear:
        clear_screen()


def ensure_memory_file() -> None:
    if not os.path.exists(MEM_FILE):
        with open(MEM_FILE, "w", encoding="utf-8") as handle:
            handle.write(HEADER)


def safe_float(value: Optional[str], fallback: float) -> float:
    try:
        return float(str(value))
    except Exception:
        return fallback


def normalize_ticker_symbol(value: Optional[str]) -> Optional[str]:
    ticker = str(value or "").strip().upper()
    if not ticker or len(ticker) > 10:
        return None
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=")
    if any(ch not in allowed_chars for ch in ticker):
        return None
    return ticker


def load_external_ticker_universe() -> List[str]:
    """Load additional tickers from a CSV file or API response when available."""

    candidates: List[str] = []
    seen = set()

    def add_candidate(raw: Optional[str]) -> None:
        ticker = normalize_ticker_symbol(raw)
        if ticker and ticker not in seen:
            candidates.append(ticker)
            seen.add(ticker)

    for ticker in TRADING212_TICKERS:
        add_candidate(ticker)

    csv_path = DEFAULT_TICKER_CSV
    if csv_path and not os.path.isabs(csv_path):
        csv_path = os.path.join(BASE_DIR, csv_path)

    if csv_path and os.path.exists(csv_path):
        try:
            if pd is not None:
                df = pd.read_csv(csv_path)
                column = None
                for candidate_col in ("ticker", "symbol", "Ticker", "Symbol"):
                    if candidate_col in df.columns:
                        column = candidate_col
                        break
                if column is None and not df.empty:
                    column = df.columns[0]
                if column:
                    for value in df[column].dropna().unique():
                        add_candidate(str(value))
            else:
                with open(csv_path, "r", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        if not row:
                            continue
                        add_candidate(row[0])
        except Exception as exc:
            log(f"Unable to load ticker CSV at {csv_path}: {exc}")

    api_url = TICKER_UNIVERSE_API
    if api_url and requests_mod is not None:
        try:
            response = requests_mod.get(api_url, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                if payload and isinstance(payload[0], dict):
                    for item in payload:
                        add_candidate(item.get("symbol") or item.get("ticker") or item.get("code"))
                else:
                    for item in payload:
                        add_candidate(str(item))
            elif isinstance(payload, dict):
                for key in ("tickers", "symbols"):
                    if key in payload and isinstance(payload[key], list):
                        for item in payload[key]:
                            add_candidate(item.get("symbol") if isinstance(item, dict) else str(item))
        except Exception as exc:
            log(f"Unable to load ticker universe from API: {exc}")

    return candidates


def discover_tracked_tickers() -> List[str]:
    ensure_memory_file()
    tracked: List[str] = []
    try:
        df = pd.read_csv(MEM_FILE)
        tracked = [
            normalize_ticker_symbol(t)
            for t in df["ticker"].dropna().unique()
            if isinstance(t, str)
        ]
        tracked = [t for t in tracked if t]
    except Exception:
        tracked = []

    external = load_external_ticker_universe()

    combined: List[str] = []
    seen = set()

    for source in (tracked, external, DEFAULT_TICKERS):
        for ticker in source:
            ticker_norm = normalize_ticker_symbol(ticker)
            if ticker_norm and ticker_norm not in seen:
                combined.append(ticker_norm)
                seen.add(ticker_norm)

    return combined


def select_ticker_with_history(
    primary: str, period_choice: str, bars_back: Optional[int] = None
):
    """Return the first ticker that produces price history.

    This is primarily used for quiet/automated workflows so we can gracefully
    fall back to another tracked symbol (or one of the defaults) when a ticker
    has been delisted or otherwise has no data. The primary ticker is excluded
    from retries because the caller will already have tried it.
    """

    seen = {primary.strip().upper()}
    candidates = discover_tracked_tickers() + DEFAULT_TICKERS

    for candidate in candidates:
        candidate = str(candidate).strip().upper()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        df = fetch_history(candidate, period_choice, bars_back=bars_back)
        if df is not None and not df.empty:
            return candidate, df

    return primary, None


def default_horizon_for_period(period_choice: str) -> int:
    period_choice = (period_choice or "day").lower()
    if period_choice == "hour":
        return 6
    if period_choice == "week":
        return 1
    if period_choice in ("month", "quarter"):
        return 1
    return 3


def horizon_days_for_period(period_choice: str, horizon: int) -> int:
    normalized = (period_choice or "day").lower()
    if normalized == "hour":
        return max(1, math.ceil(horizon / 6))
    if normalized == "day":
        return max(1, horizon)
    if normalized == "week":
        return max(1, horizon * 7)
    if normalized == "month":
        return max(1, horizon * 30)
    if normalized == "quarter":
        return max(1, horizon * 90)
    return max(1, horizon)


def append_prediction_record(
    ticker: str,
    period: str,
    horizon: int,
    engine: str,
    predicted_price: float,
    last_close: float,
    target_time: datetime,
) -> None:
    ensure_memory_file()
    line = (
        f"{datetime.now().isoformat()},{ticker},{period},{horizon},{engine},"
        f"{predicted_price:.4f},{last_close:.4f},{target_time.isoformat()},,,\n"
    )
    with open(MEM_FILE, "a", encoding="utf-8") as handle:
        handle.write(line)


def _fetch_actual_price(ticker: str, target_time: datetime) -> Optional[float]:
    if yfinance is None:
        return None
    start = target_time - timedelta(hours=6)
    end = target_time + timedelta(hours=6)
    try:
        df = yfinance.Ticker(ticker).history(start=start, end=end, interval="1h")
        if df.empty:
            df = yfinance.Ticker(ticker).history(start=start.date(), end=end.date())
        if df.empty:
            return None
        idx = (df.index - target_time).abs().argmin()  # type: ignore[attr-defined]
        return float(df.iloc[idx]["Close"])
    except Exception:
        return None


def update_memory_accuracy_for_ticker(
    ticker: str,
    period: Optional[str] = None,
    return_update_flag: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], bool]:
    ensure_memory_file()
    try:
        df = pd.read_csv(MEM_FILE)
    except Exception:
        return {}

    if df.empty:
        return {}

    df_ticker = df[df["ticker"] == ticker].copy()
    if period:
        df_ticker = df_ticker[df_ticker["period"] == period]
    if df_ticker.empty:
        return ({}, False) if return_update_flag else {}

    updated = False
    now = datetime.now()
    for idx, row in df_ticker[df_ticker["actual_price"].isna()].iterrows():
        try:
            target_time = datetime.fromisoformat(str(row["target_time"]))
        except Exception:
            continue
        if target_time > now:
            continue
        actual = _fetch_actual_price(ticker, target_time)
        if actual is None:
            continue
        predicted = float(row["predicted_price"])
        abs_error = abs(actual - predicted)
        pct_error = abs_error / actual * 100 if actual else 0.0
        df.loc[idx, "actual_price"] = actual
        df.loc[idx, "abs_error"] = abs_error
        df.loc[idx, "pct_error"] = pct_error
        updated = True

    if updated:
        df.to_csv(MEM_FILE, index=False)
        df_ticker = df[df["ticker"] == ticker].copy()

    grouped = df_ticker.dropna(subset=["pct_error"])
    if grouped.empty:
        return ({}, updated) if return_update_flag else {}

    accuracy = (
        grouped.groupby("engine")["pct_error"].mean().to_dict()
    )  # type: ignore[return-value]

    if return_update_flag:
        return accuracy, updated

    return accuracy


# =============================================================
# DATA / INDICATORS
# =============================================================

INTERVAL_MAP = {
    "hour": ("60d", "1h"),
    "day": ("10y", "1d"),
    "week": ("max", "1wk"),
    "month": ("max", "1mo"),
    "quarter": ("max", "3mo"),
}

LONG_PERIODS = {"month", "quarter"}


def default_bars_for_period(period_choice: str) -> Optional[int]:
    normalized = (period_choice or "day").lower()
    if normalized in LONG_PERIODS:
        return None

    # Pull the full dataset for predictions so models see as much context as the
    # API provides for the chosen interval instead of trimming recent bars.
    return None


def fetch_history(ticker: str, period_choice: str, bars_back: Optional[int] = None):
    if yfinance is None:
        raise RuntimeError("yfinance missing after bootstrap")

    period_choice = period_choice.lower()
    period, interval = INTERVAL_MAP.get(period_choice, INTERVAL_MAP["day"])
    bars_limit = default_bars_for_period(period_choice) if bars_back is None else bars_back

    try:
        ticker_client = yfinance.Ticker(ticker)
        data = ticker_client.history(period=period, interval=interval)
        if data is None or data.empty:
            log(
                "No data returned; running diagnostics to identify possible interval or ticker issues."
            )
            data = _diagnose_and_retry_history(
                ticker_client, ticker, period, interval, bars_limit
            )
            if data is None or data.empty:
                return None
        if bars_limit is not None and len(data) > bars_limit:
            data = data.tail(bars_limit)
        return data
    except Exception as exc:
        log(f"Failed to fetch data for {ticker}: {exc}")
        return None


def _diagnose_and_retry_history(
    ticker_client,
    ticker: str,
    period: str,
    interval: str,
    bars_limit: Optional[int],
):
    """Attempt to determine why no data was returned and retry with safer defaults."""

    hints: List[str] = []

    try:
        fast_info = getattr(ticker_client, "fast_info", None)
        if fast_info:
            currency = fast_info.get("currency") if isinstance(fast_info, dict) else None
            last_price = fast_info.get("lastPrice") if isinstance(fast_info, dict) else None
            hints.append(
                f"fast_info present (currency={currency}, lastPrice={last_price})"
            )
        else:
            hints.append("fast_info empty or unavailable")
    except Exception as exc:
        hints.append(f"fast_info error: {exc}")

    fallback_requests = [
        ("1y", "1d"),
        ("6mo", "1d"),
        ("3mo", "1d"),
        ("1mo", "1d"),
        ("1y", "1wk"),
    ]

    for fb_period, fb_interval in fallback_requests:
        try:
            retry = ticker_client.history(period=fb_period, interval=fb_interval)
            if retry is not None and not retry.empty:
                hints.append(
                    f"Recovered with fallback period={fb_period}, interval={fb_interval}"
                )
                if bars_limit is not None and len(retry) > bars_limit:
                    retry = retry.tail(bars_limit)
                log(
                    "Diagnostics: "
                    + "; ".join(hints)
                    + f"; primary request period={period}, interval={interval}"
                )
                return retry
        except Exception as exc:
            hints.append(f"fallback {fb_period}/{fb_interval} failed: {exc}")

    log(
        "Diagnostics: "
        + "; ".join(hints)
        + f"; unable to fetch data for {ticker} with period={period}, interval={interval}"
    )
    return None


def compute_indicators(df):  # type: ignore[override]
    if pd is None or np is None:
        return df

    df = df.copy()
    close = df["Close"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["RSI14"] = _compute_rsi(close, 14)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["ATR14"] = _compute_atr(df, 14)
    df["OBV"] = _compute_obv(df)
    df["Dist_SMA20"] = (close - df["SMA20"]) / df["SMA20"]
    df["Dist_SMA50"] = (close - df["SMA50"]) / df["SMA50"]
    df["Volatility20"] = close.pct_change().rolling(20).std()
    return df


def _compute_rsi(series, window: int):  # type: ignore[override]
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_atr(df, window: int):  # type: ignore[override]
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _compute_obv(df):  # type: ignore[override]
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def build_feature_frame(df):
    df = df.copy().dropna()
    cols = [
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA12",
        "EMA26",
        "RSI14",
        "MACD",
        "MACD_SIGNAL",
        "ATR14",
        "OBV",
        "Dist_SMA20",
        "Dist_SMA50",
        "Volatility20",
        "Close",
    ]
    return df[cols]


def build_synthetic_history(rows: int = 256, freq: str = "H"):
    """Generate a deterministic-ish synthetic price series for offline tests."""

    if pd is None or np is None:
        raise RuntimeError("pandas/numpy required for synthetic history generation")

    index = pd.date_range(end=datetime.now(), periods=rows, freq=freq)

    base_price = np.cumsum(np.random.normal(0, 1, rows)) + 150
    open_price = base_price + np.random.normal(0, 0.5, rows)
    high_price = base_price + np.abs(np.random.normal(0.8, 0.3, rows))
    low_price = base_price - np.abs(np.random.normal(0.8, 0.3, rows))
    volume = np.random.randint(100_000, 500_000, size=rows)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": base_price,
            "Volume": volume,
        },
        index=index,
    )


# =============================================================
# NEWS SIGNALS
# =============================================================

POSITIVE_KEYWORDS = [
    "beat",
    "surge",
    "record",
    "upgrade",
    "outperform",
    "growth",
    "strong",
]

NEGATIVE_KEYWORDS = [
    "miss",
    "downgrade",
    "lawsuit",
    "recall",
    "probe",
    "slump",
    "weak",
]


def _score_headline(title: str) -> int:
    title_lower = title.lower()
    score = 0
    for keyword in POSITIVE_KEYWORDS:
        if keyword in title_lower:
            score += 1
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in title_lower:
            score -= 1
    return score


def collect_news_context(
    ticker: str, horizon_days: int = 7, headline_limit: int = 10
) -> Dict[str, object]:
    if yfinance is None:
        return {
            "sentiment": 0,
            "headline_count": 0,
            "sources": [],
            "upcoming_events": [],
        }

    context: Dict[str, object] = {
        "sentiment": 0,
        "headline_count": 0,
        "sources": [],
        "upcoming_events": [],
    }

    try:
        ticker_obj = yfinance.Ticker(ticker)
        news_items = ticker_obj.news or []
    except Exception:
        news_items = []
        ticker_obj = None

    for item in news_items[:headline_limit]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        publisher = item.get("publisher") or "unknown"
        link = item.get("link") or item.get("url") or ""
        ts = item.get("providerPublishTime")
        published = None
        if isinstance(ts, (int, float)):
            try:
                published = datetime.fromtimestamp(ts)
            except Exception:
                published = None
        score_delta = _score_headline(title)
        context["sentiment"] = int(context.get("sentiment", 0)) + score_delta
        context["headline_count"] = int(context.get("headline_count", 0)) + 1
        sources = context.get("sources", [])
        if isinstance(sources, list):
            sources.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published": published,
                    "sentiment": score_delta,
                }
            )
        context["sources"] = sources

    if ticker_obj is not None:
        try:
            calendar = ticker_obj.get_calendar()
            if hasattr(calendar, "index") and not calendar.empty:
                for event_key in ["Earnings Date", "Earnings Date*"]:
                    if event_key in calendar.index:
                        raw = calendar.loc[event_key]
                        if hasattr(raw, "__iter__"):
                            for value in raw:
                                event_date = None
                                if hasattr(value, "to_pydatetime"):
                                    event_date = value.to_pydatetime()
                                elif isinstance(value, datetime):
                                    event_date = value
                                if not event_date:
                                    continue
                                if datetime.now() <= event_date <= datetime.now() + timedelta(days=horizon_days):
                                    context["upcoming_events"].append(
                                        f"Earnings on {event_date.date()}"
                                    )
                        break
        except Exception:
            pass

    return context


def _score_publisher_reliability(publisher: str) -> float:
    trusted = {
        "reuters",
        "bloomberg",
        "financial times",
        "wall street journal",
        "wsj",
        "cnbc",
    }
    low_trust = {"reddit", "seekingalpha", "yahoo finance rumor", "unknown"}

    norm = publisher.lower()
    if norm in trusted:
        return 0.85
    if norm in low_trust:
        return 0.35
    return 0.55


def evaluate_news_quality(news_context: Dict[str, object]) -> Dict[str, float]:
    """Summarize reliability/value of news for downstream learning models."""

    headline_count = int(news_context.get("headline_count", 0) or 0)
    sentiment = int(news_context.get("sentiment", 0) or 0)
    sources = news_context.get("sources", []) if isinstance(news_context, dict) else []
    upcoming_events = news_context.get("upcoming_events", [])

    if not isinstance(sources, list):
        sources = []

    if headline_count == 0:
        return {
            "reliability": 0.5,
            "value": 0.0,
            "sentiment_strength": 0.0,
        }

    unique_publishers = {
        (src.get("publisher") or "unknown").lower() for src in sources if isinstance(src, dict)
    }
    recency_bonus = 0.0
    for src in sources:
        published = src.get("published")
        if hasattr(published, "timestamp"):
            age_days = max(0.0, (datetime.now() - published).total_seconds() / 86400)
            recency_bonus += max(0.0, 1.5 - age_days * 0.25)

    reliability_scores = [
        _score_publisher_reliability((src.get("publisher") or "unknown"))
        for src in sources
        if isinstance(src, dict)
    ]
    avg_reliability = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.5
    reliability = max(0.2, min(1.0, avg_reliability + 0.05 * len(unique_publishers) + 0.05 * recency_bonus))

    sentiment_strength = min(3.0, abs(sentiment) / max(1.0, headline_count))
    event_bonus = 0.3 * len(upcoming_events) if isinstance(upcoming_events, list) else 0.0
    diversity_bonus = 0.1 * len(unique_publishers)
    value = max(
        0.0,
        min(1.0, 0.2 + 0.2 * sentiment_strength + event_bonus + diversity_bonus),
    )

    return {
        "reliability": float(reliability),
        "value": float(value),
        "sentiment_strength": float(sentiment_strength),
    }


# =============================================================
# FORECAST ENGINES
# =============================================================

@dataclass
class EngineResult:
    engine: str
    prediction: float
    confidence: float
    comment: str = ""
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    band_percent: Optional[float] = None
    band_probability: Optional[float] = None


FEATURE_COLUMNS = [
    "SMA20",
    "SMA50",
    "SMA200",
    "EMA12",
    "EMA26",
    "RSI14",
    "MACD",
    "MACD_SIGNAL",
    "ATR14",
    "OBV",
    "Dist_SMA20",
    "Dist_SMA50",
    "Volatility20",
]

# Tunable weight for how strongly the learning engine leans on news-derived
# signals. Increase NEWS_INFLUENCE_WEIGHT to make sentiment/value adjustments
# more pronounced; decrease to make the model more price/technical driven.
NEWS_INFLUENCE_WEIGHT = float(os.environ.get("NEWS_INFLUENCE_WEIGHT", "0.35"))


def technical_engine(df) -> Optional[EngineResult]:  # type: ignore[override]
    if df is None or df.empty:
        return None
    latest = df.iloc[-1]
    score = 0.0
    comment_bits = []

    if latest["Close"] > latest.get("SMA20", latest["Close"]):
        score += 0.6
        comment_bits.append(">SMA20")
    else:
        score -= 0.6
        comment_bits.append("<SMA20")

    if latest.get("RSI14", 50) < 30:
        score += 0.8
        comment_bits.append("RSI oversold")
    elif latest.get("RSI14", 50) > 70:
        score -= 0.8
        comment_bits.append("RSI overbought")

    macd = latest.get("MACD", 0) - latest.get("MACD_SIGNAL", 0)
    score += math.tanh(macd)
    comment_bits.append(f"MACD {macd:+.2f}")

    dist = latest.get("Dist_SMA20", 0)
    score -= dist * 2.5  # mean reversion

    atr = latest.get("ATR14", 0)
    volatility_penalty = min(1.0, (atr / latest["Close"]) * 10)
    score -= volatility_penalty * 0.5

    prediction = latest["Close"] * (1 + score * 0.01)
    confidence = max(1.0, min(9.5, 5 + score))

    return EngineResult(
        engine="Technical",
        prediction=float(prediction),
        confidence=float(confidence),
        comment=", ".join(comment_bits),
    )


def analyst_consensus_engine(ticker: str, last_close: float) -> Optional[EngineResult]:  # type: ignore[override]
    if yfinance is None:
        return None

    def _coerce(value) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    def _probe_quote_summary() -> Dict:
        """Try direct Yahoo Finance quoteSummary endpoints to recover from 404s.

        yfinance occasionally ships outdated endpoints. When that happens we
        opportunistically scan known base URLs (query2/query1/fc) for a working
        quoteSummary response so the caller still gets analyst targets without
        needing a manual code update.
        """

        if requests_mod is None:
            return {}

        modules = ["financialData", "defaultKeyStatistics", "price"]
        base_urls = [
            "https://query2.finance.yahoo.com",
            "https://query1.finance.yahoo.com",
            "https://fc.yahoo.com",
        ]

        for base in base_urls:
            try:
                response = requests_mod.get(
                    f"{base}/v7/finance/quoteSummary/{ticker}",
                    params={"modules": ",".join(modules)},
                    timeout=8,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json() or {}
                result = (
                    payload.get("quoteSummary", {}).get("result") or []
                )
                if not result:
                    continue
                merged: Dict = {"_source_url": response.url}
                for module_payload in result[0].values():
                    if isinstance(module_payload, dict):
                        merged.update(module_payload)
                return merged
            except Exception:
                continue

        return {}

    def _is_404_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 404:
            return True
        status = getattr(exc, "status", None)
        if status == 404:
            return True
        return "404" in str(exc)

    def _fetch_info_safely() -> Dict:
        try:
            ticker_obj = yfinance.Ticker(ticker)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                return ticker_obj.get_info()
        except Exception as exc:
            if _is_404_error(exc):
                return {}
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    return yfinance.Ticker(ticker).info
            except Exception as exc_inner:
                if _is_404_error(exc_inner):
                    return {}
                return _probe_quote_summary()

    ticker_key = ticker.upper()
    if ticker_key not in _INFO_CACHE:
        _INFO_CACHE[ticker_key] = _fetch_info_safely() or _probe_quote_summary()

    info = _INFO_CACHE.get(ticker_key, {})
    if not info:
        return None

    target_mean = _coerce(info.get("targetMeanPrice"))
    target_high = _coerce(info.get("targetHighPrice"))
    target_low = _coerce(info.get("targetLowPrice"))
    analyst_count = safe_int(info.get("numberOfAnalystOpinions"), 0)

    fallback_price = _coerce(info.get("currentPrice")) or _coerce(
        info.get("regularMarketPrice")
    )
    if fallback_price is None:
        try:
            fast_info = yfinance.Ticker(ticker).fast_info  # type: ignore[attr-defined]
            if isinstance(fast_info, dict):
                fallback_price = _coerce(
                    fast_info.get("last_price")
                    or fast_info.get("lastPrice")
                    or fast_info.get("regularMarketPrice")
                )
            else:
                fallback_price = _coerce(
                    getattr(fast_info, "last_price", None)
                    or getattr(fast_info, "lastPrice", None)
                    or getattr(fast_info, "regularMarketPrice", None)
                )
        except Exception:
            fallback_price = None
    prediction = target_mean or (
        (target_high and target_low and (target_high + target_low) / 2)
    ) or fallback_price or last_close

    spread = 0.0
    if prediction and target_high and target_low:
        spread = (target_high - target_low) / prediction

    confidence = 5 + math.log1p(max(analyst_count, 1)) - spread * 2
    confidence = max(3.0, min(9.5, confidence))

    comments = []
    if analyst_count:
        comments.append(f"{analyst_count} analysts")
    if target_high and target_low:
        comments.append(f"range {target_low:.2f}-{target_high:.2f}")
    if target_mean:
        comments.append(f"consensus {target_mean:.2f}")

    return EngineResult(
        engine="AnalystConsensus",
        prediction=prediction,
        confidence=confidence,
        comment=", ".join(comments) or "Analyst targets",
        range_low=target_low,
        range_high=target_high,
    )


def random_forest_engine(df) -> Optional[EngineResult]:  # type: ignore[override]
    if skl is None:
        return None

    df_feat = build_feature_frame(df)
    if len(df_feat) < 80:
        return None

    target = df_feat["Close"].shift(-1).dropna()
    features = df_feat.loc[target.index, FEATURE_COLUMNS]

    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.metrics import mean_absolute_error  # type: ignore

    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]

    if len(X_train) < 50 or len(X_test) < 10:
        return None

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    last_features = features.iloc[[-1]]
    next_price = float(model.predict(last_features)[0])

    confidence = max(3.0, 10 - (mae / (y_test.mean() or 1)) * 50)

    return EngineResult(
        engine="RandomForest",
        prediction=next_price,
        confidence=confidence,
        comment=f"MAE {mae:.2f}",
    )


def neural_network_engine(df, news_context: Optional[Dict[str, object]] = None) -> Optional[EngineResult]:  # type: ignore[override]
    df_feat = build_feature_frame(df)
    if len(df_feat) < 120:
        return None

    target = df_feat["Close"].shift(-1).dropna()
    features = df_feat.loc[target.index, FEATURE_COLUMNS]

    base_result: Optional[EngineResult]
    if torch_mod is not None:
        base_result = _torch_lstm_engine(features, target)
    else:
        base_result = _mlp_engine(features, target)

    if base_result is None:
        return None

    news_quality = evaluate_news_quality(news_context or {})
    sentiment = int((news_context or {}).get("sentiment", 0) or 0)
    headline_count = int((news_context or {}).get("headline_count", 0) or 0)
    direction = math.tanh(sentiment / max(1.0, headline_count or 1)) if headline_count else 0.0

    influence = NEWS_INFLUENCE_WEIGHT * news_quality["value"] * news_quality["reliability"]
    news_modifier = direction * news_quality["sentiment_strength"] * influence
    adjusted_prediction = base_result.prediction * (1 + news_modifier * 0.05)

    confidence_boost = news_quality["reliability"] * news_quality["value"] * 1.5
    adjusted_confidence = max(
        1.0,
        min(10.0, base_result.confidence + news_modifier + confidence_boost - (1 - news_quality["reliability"]))
    )

    news_comment = (
        f"news adj {news_modifier:+.2f} (rel {news_quality['reliability']:.2f}, "
        f"value {news_quality['value']:.2f}, sentiment {sentiment:+d})"
    )
    base_comment = base_result.comment.strip()
    combined_comment = f"{base_comment}; {news_comment}" if base_comment else news_comment

    return EngineResult(
        engine=base_result.engine,
        prediction=float(adjusted_prediction),
        confidence=float(adjusted_confidence),
        comment=combined_comment,
    )


def _torch_lstm_engine(features, target):  # type: ignore[override]
    import numpy as _np

    tensor = torch_mod.tensor
    nn = torch_mod.nn

    seq_len = 10
    X_list = []
    y_list = []
    feat_values = features.values
    for idx in range(len(features) - seq_len):
        X_list.append(feat_values[idx : idx + seq_len])
        y_list.append(target.iloc[idx + seq_len])

    if len(X_list) < 50:
        return None

    X = tensor(_np.array(X_list), dtype=torch_mod.float32)
    y = tensor(_np.array(y_list), dtype=torch_mod.float32).unsqueeze(-1)

    dataset = torch_mod.utils.data.TensorDataset(X, y)
    loader = torch_mod.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    class MiniLSTM(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):  # type: ignore[override]
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = MiniLSTM(features.shape[1])
    criterion = nn.L1Loss()
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(10):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch_mod.no_grad():
        last_seq = tensor(feat_values[-seq_len:], dtype=torch_mod.float32).unsqueeze(0)
        next_price = float(model(last_seq).item())

    confidence = 6.5

    return EngineResult(
        engine="TorchLSTM",
        prediction=next_price,
        confidence=confidence,
        comment="Trained 10 epochs",
    )


def _mlp_engine(features, target):  # type: ignore[override]
    if skl is None:
        return None
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.metrics import mean_absolute_error  # type: ignore

    X_train = features.iloc[:-20]
    y_train = target.iloc[:-20]
    X_test = features.iloc[-20:]
    y_test = target.iloc[-20:]

    if len(X_train) < 60:
        return None

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    next_price = float(model.predict(features.iloc[[-1]])[0])
    confidence = max(2.0, 8 - mae / (y_test.mean() or 1) * 50)

    return EngineResult(
        engine="MLPRegressor",
        prediction=next_price,
        confidence=confidence,
        comment=f"MAE {mae:.2f}",
    )


def news_sentiment_engine(
    ticker: str, df, news_context: Dict[str, object]
) -> Optional[EngineResult]:  # type: ignore[override]
    if df is None or df.empty:
        return None

    last_close = float(df.iloc[-1]["Close"])
    headline_count = int(news_context.get("headline_count", 0) or 0)
    sentiment = int(news_context.get("sentiment", 0) or 0)
    upcoming_events = news_context.get("upcoming_events", [])

    if headline_count == 0 and not upcoming_events:
        return EngineResult(
            engine="NewsSentiment",
            prediction=last_close,
            confidence=4.0,
            comment="No recent headlines found",
        )

    sentiment_adjustment = max(-3.0, min(3.0, float(sentiment)))
    event_risk = 0.5 * len(upcoming_events) if isinstance(upcoming_events, list) else 0.0
    prediction = last_close * (1 + sentiment_adjustment * 0.004)
    confidence = max(1.0, min(10.0, 5 + sentiment_adjustment - event_risk))

    event_note = ""
    if upcoming_events:
        readable_events = "; ".join(str(evt) for evt in upcoming_events)
        event_note = f"; {readable_events}"

    return EngineResult(
        engine="NewsSentiment",
        prediction=float(prediction),
        confidence=float(confidence),
        comment=f"Net sentiment {sentiment:+} from {headline_count} headlines{event_note}",
    )


# =============================================================
# CONSOLIDATION / REPORTING
# =============================================================


def estimate_band_probability(volatility: float, horizon: int, band_percent: float) -> float:
    if volatility <= 0:
        return 1.0

    horizon = max(1, horizon)
    sigma = volatility * math.sqrt(horizon)
    band_fraction = band_percent / 100.0
    if sigma == 0:
        return 1.0

    z = band_fraction / (sigma * math.sqrt(2))
    prob = math.erf(z)
    return max(0.0, min(1.0, prob))


def evaluate_confidence_satisfaction(
    results: Iterable[EngineResult], threshold: float
) -> Tuple[List[EngineResult], List[str]]:
    satisfied: List[EngineResult] = []
    reasons: List[str] = []

    for res in results:
        if res.confidence >= threshold:
            satisfied.append(res)
            continue

        detail = res.comment.strip() if res.comment else "Signals were mixed or insufficient to support a higher confidence score."
        reasons.append(
            (
                f"{res.engine} is below the {threshold:.1f}/10 requirement with "
                f"{res.confidence:.1f}/10 – {detail}"
            )
        )

    return satisfied, reasons


def consolidate_predictions(
    ticker: str,
    period: str,
    horizon: int,
    df,
    engines: Iterable[EngineResult],
    band_percent: float = DEFAULT_BAND_PERCENT,
) -> List[EngineResult]:
    results = [engine for engine in engines if engine is not None]
    last_close = float(df.iloc[-1]["Close"])
    target_time = datetime.now() + timedelta(hours=horizon if period == "hour" else horizon * 24)

    returns = df["Close"].pct_change().dropna()
    recent_vol = float(returns.tail(120).std()) if not returns.empty else 0.0
    band_prob = estimate_band_probability(recent_vol, horizon, band_percent)
    spread = band_percent / 100.0

    for engine in results:
        if engine.range_low is None or engine.range_high is None:
            engine.range_low = engine.prediction * (1 - spread)
            engine.range_high = engine.prediction * (1 + spread)
        if engine.band_percent is None:
            engine.band_percent = band_percent
        if engine.band_probability is None:
            engine.band_probability = band_prob
        append_prediction_record(
            ticker=ticker,
            period=period,
            horizon=horizon,
            engine=engine.engine,
            predicted_price=engine.prediction,
            last_close=last_close,
            target_time=target_time,
        )

    return results


def format_prediction_table(results: List[EngineResult], last_close: float) -> str:
    if not tabulate_mod:
        lines = []
        for res in results:
            low = res.range_low if res.range_low is not None else res.prediction
            high = res.range_high if res.range_high is not None else res.prediction
            band_width = res.band_percent if res.band_percent is not None else DEFAULT_BAND_PERCENT
            prob = res.band_probability * 100 if res.band_probability is not None else None
            prob_str = f"{prob:.1f}% @ ±{band_width:.1f}%" if prob is not None else "-"
            lines.append(
                f"{res.engine}: {res.prediction:.2f} ({res.confidence:.1f}/10) "
                f"range {low:.2f}-{high:.2f} prob {prob_str} - {res.comment}"
            )
        return "\n".join(lines)
    rows = []
    for res in results:
        delta = ((res.prediction / last_close) - 1) * 100 if last_close else 0.0
        range_str = "-"
        if res.range_low is not None and res.range_high is not None:
            range_str = f"{res.range_low:.2f}–{res.range_high:.2f}"
        band_width = res.band_percent if res.band_percent is not None else DEFAULT_BAND_PERCENT
        prob_str = "-"
        if res.band_probability is not None:
            prob_str = f"{res.band_probability * 100:.1f}% @ ±{band_width:.1f}%"
        rows.append([
            res.engine,
            f"{res.prediction:.2f}",
            range_str,
            prob_str,
            f"{delta:+.2f}%",
            f"{res.confidence:.1f}/10",
            res.comment,
        ])
    return tabulate_mod.tabulate(
        rows,
        headers=["Engine", "Target", "Range", "Prob", "Δ vs Close", "Confidence", "Notes"],
        tablefmt="github",
    )


# =============================================================
# CLI ACTIONS
# =============================================================

def run_sma_report(
    ticker: Optional[str] = None,
    period_choice: Optional[str] = None,
    pause: bool = True,
) -> None:
    print("\n=== SMA 20 + 50 Report ===\n")
    if pd is None:
        print("pandas unavailable; run bootstrap first.")
        return
    ticker = (
        ticker
        or guided_prompt(
            "Ticker symbol",
            "Use a single stock/ETF ticker in uppercase letters.",
            "AAPL",
        )
    ).strip().upper()
    if not ticker:
        print("Ticker required.")
        return
    period_choice = (
        period_choice
        or guided_prompt(
            "Period",
            "Choose one of: hour, day, week, month, or quarter to set candle duration.",
            "day",
            "day",
        )
    ).strip().lower() or "day"
    df = fetch_history(ticker, period_choice, bars_back=default_bars_for_period(period_choice))
    if df is None or df.empty:
        print("No data downloaded.")
        return
    df = compute_indicators(df)
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest

    def _pct_diff(value: float, reference: float) -> Optional[float]:
        if reference is None or reference == 0 or pd.isna(reference):
            return None
        return ((value - reference) / reference) * 100

    def _fmt(value: Optional[float], decimals: int = 2) -> str:
        if value is None or pd.isna(value):
            return "-"
        return f"{value:.{decimals}f}"

    close = float(latest["Close"])
    sma20 = float(latest.get("SMA20", float("nan")))
    sma50 = float(latest.get("SMA50", float("nan")))
    sma200 = float(latest.get("SMA200", float("nan")))
    rsi14 = float(latest.get("RSI14", float("nan")))
    macd = float(latest.get("MACD", float("nan")))
    atr14 = float(latest.get("ATR14", float("nan")))

    pct_to_close = [
        ["SMA20", _pct_diff(close, sma20)],
        ["SMA50", _pct_diff(close, sma50)],
        ["SMA200", _pct_diff(close, sma200)],
    ]

    rows = [
        ["Close", _fmt(close), "-", "-"],
        ["SMA20", _fmt(sma20), _fmt(pct_to_close[0][1]), _fmt(sma20 - float(previous.get("SMA20", sma20)))],
        ["SMA50", _fmt(sma50), _fmt(pct_to_close[1][1]), _fmt(sma50 - float(previous.get("SMA50", sma50)))],
        ["SMA200", _fmt(sma200), _fmt(pct_to_close[2][1]), _fmt(sma200 - float(previous.get("SMA200", sma200)))],
        ["RSI14", _fmt(rsi14), "-", "-"],
        ["MACD", _fmt(macd), "-", _fmt(macd - float(previous.get("MACD", macd)))],
        ["ATR14", _fmt(atr14), _fmt(_pct_diff(atr14, close)), "-"],
    ]

    headers = ["Metric", "Value", "% vs Close", "1-bar Δ"]
    if tabulate_mod:
        print(tabulate_mod.tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(headers)
        for row in rows:
            print(" | ".join(f"{cell}" for cell in row))

    insights: List[str] = []

    stacked = [sma20, sma50, sma200]
    if all(pd.notna(val) for val in stacked):
        if sma20 > sma50 > sma200:
            insights.append("MAs stacked bullish (20 > 50 > 200) – trend support.")
        elif sma20 < sma50 < sma200:
            insights.append("MAs stacked bearish (20 < 50 < 200) – downward pressure.")
        else:
            insights.append("Mixed MA stack – watch for upcoming crossovers.")

    if pd.notna(macd) and pd.notna(latest.get("MACD_SIGNAL")):
        macd_signal = float(latest["MACD_SIGNAL"])
        macd_state = "above" if macd >= macd_signal else "below"
        insights.append(f"MACD is {macd_state} signal (momentum gauge).")

    if pd.notna(rsi14):
        if rsi14 >= 70:
            insights.append(f"RSI at {rsi14:.1f} → overbought risk zone.")
        elif rsi14 <= 30:
            insights.append(f"RSI at {rsi14:.1f} → oversold potential rebound.")

    recent_span = df.tail(60)
    if not recent_span.empty:
        swing_high = float(recent_span["High"].max())
        swing_low = float(recent_span["Low"].min())
        if swing_high and close:
            dist_high = _pct_diff(close, swing_high)
            insights.append(f"Next resistance near {swing_high:.2f} ({_fmt(dist_high)}% from close).")
        if swing_low and close:
            dist_low = _pct_diff(close, swing_low)
            insights.append(f"Nearest support near {swing_low:.2f} ({_fmt(dist_low)}% from close).")

    if pd.notna(atr14) and atr14 > 0:
        atr_pct = _pct_diff(atr14, close)
        insights.append(
            f"ATR14 suggests ~{atr14:.2f} daily move ({_fmt(atr_pct)}% of price) – set stops/targets accordingly."
        )

    print()
    print_panel("Snapshot", insights or ["No additional signals available."])
    maybe_pause(pause)


def run_plot(
    ticker: Optional[str] = None,
    period_choice: Optional[str] = None,
    output_path: Optional[str] = None,
    bars_back: Optional[int] = None,
) -> None:
    print("\n=== Plot Price + SMA ===\n")
    if plt_mod is None:
        print("Matplotlib not installed.")
        return
    ticker = (
        ticker
        or guided_prompt(
            "Ticker symbol",
            "Provide a single ticker in uppercase letters for plotting.",
            "MSFT",
        )
    ).strip().upper()
    period_choice = (
        period_choice
        or guided_prompt(
            "Period",
            "Select hour, day, week, month, or quarter to match your plotting cadence.",
            "week",
            "day",
        )
    ).strip().lower() or "day"
    df = fetch_history(
        ticker,
        period_choice,
        bars_back=bars_back if bars_back is not None else default_bars_for_period(period_choice),
    )
    if df is None or df.empty:
        print("No data downloaded.")
        return
    df = compute_indicators(df)
    plt_mod.figure(figsize=(10, 5))
    plt_mod.plot(df.index, df["Close"], label="Close")
    plt_mod.plot(df.index, df["SMA20"], label="SMA20")
    plt_mod.plot(df.index, df["SMA50"], label="SMA50")
    plt_mod.title(f"{ticker} - Price vs SMA")
    plt_mod.grid(True)
    plt_mod.legend()
    plt_mod.tight_layout()
    if output_path:
        plt_mod.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt_mod.show()


def run_csv_download(
    ticker: Optional[str] = None,
    period_choice: Optional[str] = None,
    bars_override: Optional[int] = None,
    output_path: Optional[str] = None,
    pause: bool = True,
) -> None:
    print("\n=== Download Historical CSV ===\n")
    ticker = (
        ticker
        or guided_prompt(
            "Ticker symbol",
            "Enter one uppercase ticker to export its history.",
            "SPY",
        )
    ).strip().upper()
    if not ticker:
        print("Ticker required.")
        return
    period_choice = (
        period_choice
        or guided_prompt(
            "Period",
            "Pick hour, day, week, month, or quarter candles for the CSV output.",
            "hour",
            "day",
        )
    ).strip().lower() or "day"
    df = fetch_history(
        ticker,
        period_choice,
        bars_back=bars_override if bars_override is not None else default_bars_for_period(period_choice),
    )
    if df is None or df.empty:
        print("No data downloaded.")
        return
    out_path = output_path or os.path.join(BASE_DIR, f"{ticker}_{period_choice}_history.csv")
    df.to_csv(out_path)
    print(f"Saved to {out_path}")
    maybe_pause(pause)


def run_forecast_workflow(
    ticker: Optional[str] = None,
    period_choice: Optional[str] = None,
    horizon: Optional[int] = None,
    band: Optional[float] = None,
    pause: bool = True,
    quiet: bool = False,
) -> None:
    try:
        if not quiet:
            print("\n=== Forecast (Technical + ML + NN) ===\n")

        interactive = is_interactive()

        if ticker is None:
            if quiet or not interactive:
                ticker = discover_tracked_tickers()[0]
                log(f"Auto-filled ticker: {ticker}")
            else:
                ticker = guided_prompt(
                    "Ticker symbol",
                    "Specify one uppercase ticker to run the combined forecast.",
                    discover_tracked_tickers()[0],
                )
            ticker = ticker.strip().upper()
        if not ticker:
            print("Ticker required.")
            return

        if period_choice is None:
            if quiet or not interactive:
                period_choice = "day"
            else:
                period_choice = guided_prompt(
                    "Period",
                    "Choose the candle size: hour, day, week, month, or quarter.",
                    "hour",
                    "day",
                )
        period_choice = period_choice.strip().lower() or "day"

        horizon = (
            horizon
            if horizon is not None
            else default_horizon_for_period(period_choice)
        )
        horizon = max(1, horizon)

        band = (
            band
            if band is not None
            else safe_float(
                DEFAULT_BAND_PERCENT
                if quiet or not interactive
                else guided_prompt(
                    "+/- band for probability (%)",
                    "Set the percentage band to estimate hit probability around each target.",
                    f"{DEFAULT_BAND_PERCENT:.1f}",
                    f"{DEFAULT_BAND_PERCENT:.1f}",
                ),
                DEFAULT_BAND_PERCENT,
            )
        )
        band = max(0.5, band)
        df = fetch_history(
            ticker,
            period_choice,
            bars_back=default_bars_for_period(period_choice),
        )
        if (df is None or df.empty) and (quiet or not interactive):
            original = ticker
            ticker, df = select_ticker_with_history(
                ticker, period_choice, bars_back=default_bars_for_period(period_choice)
            )
            if df is not None and not df.empty and ticker != original:
                log(
                    f"Auto-switched to {ticker} after {original} returned no price data."
                )
        if df is None or df.empty:
            print("No data downloaded.")
            return

        df = compute_indicators(df)
        last_close = float(df["Close"].iloc[-1])
        horizon_days = horizon_days_for_period(period_choice, horizon)
        news_context = collect_news_context(ticker, horizon_days=horizon_days)
        history_accuracy = update_memory_accuracy_for_ticker(ticker)
        if history_accuracy and not quiet:
            print("Historical MAPE by engine (lower is better):")
            for engine, mape in history_accuracy.items():
                print(f"  - {engine}: {mape:.2f}%")
            print()

        results = consolidate_predictions(
            ticker,
            period_choice,
            horizon,
            df,
            filter(
                None,
                [
                    technical_engine(df),
                    analyst_consensus_engine(ticker, last_close),
                    random_forest_engine(df),
                    neural_network_engine(df, news_context),
                    news_sentiment_engine(ticker, df, news_context),
                ],
            ),
            band_percent=band,
        )

        if not results:
            print("No engine produced a forecast.")
            return

        satisfied, low_confidence_reasons = evaluate_confidence_satisfaction(
            results, MIN_CONFIDENCE_SATISFACTION
        )

        if quiet:
            log(
                f"Recorded {len(results)} forecasts for {ticker} ({period_choice}) with horizon {horizon} bars."
            )
            if satisfied:
                log(
                    "Confidence satisfaction reached for: "
                    + ", ".join(res.engine for res in satisfied)
                )
            else:
                log(
                    f"No engines met the {MIN_CONFIDENCE_SATISFACTION:.1f}/10 confidence satisfaction threshold."
                )

            for reason in low_confidence_reasons:
                log(f"Low confidence: {reason}")

        else:
            print(format_prediction_table(results, last_close))
            print(
                f"Probabilities reflect the chance of landing within ±{band:.1f}% of each target using recent volatility."
            )
            print()

            print(
                f"Satisfaction threshold: {MIN_CONFIDENCE_SATISFACTION:.1f}/10 (only predictions at or above this are considered reliable)."
            )
            if satisfied:
                print(
                    "Engines meeting the satisfaction bar: "
                    + ", ".join(res.engine for res in satisfied)
                )
            else:
                print("No engines met the satisfaction bar this run.")

            if low_confidence_reasons:
                print("Confidence shortfalls and reasons:")
                for reason in low_confidence_reasons:
                    print(f"  - {reason}")
            print()

            sources = news_context.get("sources", []) if isinstance(news_context, dict) else []
            if sources:
                print("News sources factored into the forecast:")
                for src in sources:
                    published = src.get("published")
                    ts = published.strftime("%Y-%m-%d %H:%M") if hasattr(published, "strftime") else "n/a"
                    sentiment_note = f"sentiment {src.get('sentiment', 0):+d}"
                    link = src.get("link") or ""
                    link_note = f" → {link}" if link else ""
                    print(f"  - {src.get('publisher', 'unknown')} ({ts}): {src.get('title', '').strip()} [{sentiment_note}]{link_note}")
            else:
                print("No recent headlines were available for news weighting.")
            print()
    finally:
        finalize_section(pause, clear=False)


def run_training_workflow(
    pause: bool = True,
    loop: bool = False,
    periods: Optional[List[str]] = None,
    sleep_interval: int = 60,
) -> None:
    track_periods = periods or list(INTERVAL_MAP.keys())
    stop_requested = False
    ticker_universe = discover_tracked_tickers()
    default_combos = [(ticker, period) for ticker in ticker_universe for period in track_periods]

    def handle_shutdown(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        log(f"Signal {signum} received; finishing current training pass before exit...")

    def execute_training_pass() -> Tuple[List[Tuple[str, str]], bool]:
        print("\n=== Training / Self-Evaluation ===\n")
        ensure_memory_file()
        try:
            df = pd.read_csv(MEM_FILE)
        except Exception as exc:
            print(f"Could not load memory file: {exc}")
            return sorted(default_combos), False

        combos: List[Tuple[str, str]] = []
        raw_combos = (
            df[["ticker", "period"]]
            .dropna()
            .drop_duplicates()
            .apply(lambda row: (str(row["ticker"]).strip(), str(row["period"]).strip()), axis=1)
            .tolist()
        )

        for ticker_raw, period_raw in raw_combos:
            ticker = normalize_ticker_symbol(ticker_raw)
            period_choice = (period_raw or "").strip().lower()
            if ticker and period_choice in INTERVAL_MAP:
                combos.append((ticker, period_choice))

        if not combos:
            combos = list(default_combos)
        else:
            combos_set = {(ticker, period) for ticker, period in combos}
            combos_set.update(default_combos)
            combos = sorted(combos_set)

        updates_made = False
        if df.empty:
            print("No prediction history found. Running live forecasts for tracked tickers.")
        else:
            completed = int(df["actual_price"].notna().sum())
            pending = int(len(df) - completed)

            print(
                "Found {rows} prediction records across {tickers} tracked ticker/period pairs.".format(
                    rows=len(df), tickers=len(combos)
                )
            )
            print(f"Completed predictions with actual prices: {completed}")
            print(f"Pending predictions awaiting actual prices: {pending}")

            per_combo_accuracy: Dict[Tuple[str, str], Dict[str, float]] = {}
            global_accuracy: Dict[str, List[float]] = {}

            for ticker, period_choice in combos:
                accuracy_result = update_memory_accuracy_for_ticker(
                    ticker,
                    period=period_choice,
                    return_update_flag=True,
                )
                if isinstance(accuracy_result, tuple):
                    accuracy, updated = accuracy_result
                    updates_made = updates_made or updated
                else:
                    accuracy = accuracy_result
                if not accuracy:
                    continue
                per_combo_accuracy[(ticker, period_choice)] = accuracy
                for engine, mape in accuracy.items():
                    global_accuracy.setdefault(engine, []).append(mape)

            if per_combo_accuracy:
                print("\nUpdated MAPE (by ticker, period, and engine):")
                for (ticker, period_choice), scores in sorted(per_combo_accuracy.items()):
                    parts = ", ".join(f"{engine}: {mape:.2f}%" for engine, mape in scores.items())
                    print(f"  - {ticker} [{period_choice}]: {parts}")
            else:
                print("\nNo tickers have evaluable targets yet; waiting for target times to elapse.")

            if global_accuracy:
                print("\nAverage MAPE across all tracked tickers:")
                for engine, values in global_accuracy.items():
                    avg_mape = sum(values) / len(values)
                    print(f"  - {engine}: {avg_mape:.2f}% (n={len(values)})")

            pending_by_combo = {
                (ticker, period_choice): int(
                    df[
                        (df["ticker"] == ticker)
                        & (df["period"] == period_choice)
                        & (df["actual_price"].isna())
                    ].shape[0]
                )
                for ticker, period_choice in combos
            }
            if pending_by_combo:
                print("\nPending prediction targets still waiting for actual prices:")
                for (ticker, period_choice), pending_count in sorted(pending_by_combo.items()):
                    print(f"  - {ticker} [{period_choice}]: {pending_count} pending")

        return combos, updates_made

    def run_live_forecasts(combos: List[Tuple[str, str]]) -> None:
        print("\n=== Live forecast refresh (auto-filled tickers) ===\n")
        batch_size = max(1, safe_int(os.environ.get("TRAINING_BATCH_SIZE"), 25))
        cooldown = max(0, safe_int(os.environ.get("TRAINING_BATCH_COOLDOWN"), 5))

        for start in range(0, len(combos), batch_size):
            batch = combos[start : start + batch_size]
            batch_number = (start // batch_size) + 1
            print(f"Processing batch {batch_number} with {len(batch)} ticker/period pairs...")

            for ticker, period in batch:
                if stop_requested:
                    break
                ticker_norm = normalize_ticker_symbol(ticker)
                period_choice = (period or "").strip().lower()
                if not ticker_norm or period_choice not in INTERVAL_MAP:
                    log(f"Skipping invalid combo: ticker={ticker}, period={period}")
                    continue
                try:
                    run_forecast_workflow(
                        ticker=ticker_norm,
                        period_choice=period_choice,
                        horizon=default_horizon_for_period(period_choice),
                        band=DEFAULT_BAND_PERCENT,
                        pause=False,
                        quiet=True,
                    )
                except Exception as exc:
                    log(f"Live forecast failed for {ticker_norm} ({period_choice}): {exc}")

            if stop_requested:
                break

            if start + batch_size < len(combos) and cooldown:
                log(
                    "Cooling down for {seconds}s before the next batch to respect data provider limits.".format(
                        seconds=cooldown
                    )
                )
                try:
                    time.sleep(cooldown)
                except KeyboardInterrupt:
                    log("Cooldown interrupted; stopping training loop after current batch.")
                    break

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        while True:
            if stop_requested:
                log("Stopping training loop before starting the next pass.")
                break
            combos, updates = execute_training_pass()
            if combos:
                run_live_forecasts(combos)
            if not loop or stop_requested:
                if stop_requested:
                    log("Stopping training loop after completing the current pass.")
                break
            if updates:
                log("New actual prices were captured; refreshing metrics before the next loop pass...")
            log(
                "Training loop active; sleeping {seconds} seconds before the next refresh...".format(
                    seconds=max(1, sleep_interval)
                )
            )
            try:
                time.sleep(max(1, sleep_interval))
            except KeyboardInterrupt:
                stop_requested = True
                log("Keyboard interrupt received; preparing to exit training loop.")
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        finalize_section(pause and not loop, clear=False)


def run_review_workflow(
    pause: bool = True,
    ticker: Optional[str] = None,
    period: Optional[str] = None,
    limit: int = 20,
    export_path: Optional[str] = None,
) -> None:
    print("\n=== Prediction History Review ===\n")
    ensure_memory_file()
    if pd is None:
        print("pandas is required to review prediction history.")
        finalize_section(pause, clear=False)
        return

    try:
        df = pd.read_csv(MEM_FILE)
    except Exception as exc:
        print(f"Could not load memory file: {exc}")
        finalize_section(pause, clear=False)
        return

    if df.empty:
        print("No prediction history found yet.")
        finalize_section(pause, clear=False)
        return

    view = df.copy()
    if ticker:
        ticker_upper = ticker.strip().upper()
        view = view[view["ticker"].astype(str).str.upper() == ticker_upper]
    if period:
        period_lower = period.strip().lower()
        view = view[view["period"].astype(str).str.lower() == period_lower]

    view["timestamp_parsed"] = pd.to_datetime(view["timestamp"], errors="coerce")
    view = view.sort_values(by="timestamp_parsed", ascending=False)
    if limit > 0:
        view = view.head(limit)

    if view.empty:
        scope = []
        if ticker:
            scope.append(f"ticker {ticker.strip().upper()}")
        if period:
            scope.append(f"period {period.strip().lower()}")
        scope_note = " for " + " and ".join(scope) if scope else ""
        print(f"No matching prediction records found{scope_note}.")
        finalize_section(pause, clear=False)
        return

    columns = [
        "timestamp",
        "ticker",
        "period",
        "horizon_bars",
        "engine",
        "predicted_price",
        "actual_price",
        "abs_error",
        "pct_error",
        "target_time",
    ]
    display_df = view[columns]

    if tabulate_mod:
        print(
            tabulate_mod.tabulate(
                display_df,
                headers="keys",
                tablefmt="github",
                floatfmt=".4f",
            )
        )
    else:
        print(display_df.to_string(index=False))

    if export_path:
        try:
            display_df.to_csv(export_path, index=False)
            print(f"\nExported {len(display_df)} records to {export_path}")
        except Exception as exc:
            print(f"\nCould not export to {export_path}: {exc}")

    finalize_section(pause, clear=False)


def run_help(pause: bool = True) -> None:
    print(
        """
=== HELP ===
This refactored build downloads market data via yfinance, computes a stable
set of technical indicators with pandas and produces four forecasts:

1. Technical: rules that use SMA, RSI, MACD and volatility.
2. AnalystConsensus: aggregates published analyst targets for an external
   view.
3. RandomForest: supervised model trained on indicators.
4. Neural Network: PyTorch LSTM when torch is installed, otherwise an sklearn
   MLP fallback.
5. NewsSentiment: headline tone and nearby earnings events weighted into the
   forecast, with the factored sources listed below the table.

All predictions are logged to ./memory/predictions_log.csv together with
horizon information.  On each run the tool attempts to fetch the actual price
for previous targets so we accumulate real-world error metrics over time.
Forecasts now also report a probability of closing within a configurable +/-
band around each target based on recent volatility.

Review mode: --action review reads memory/predictions_log.csv without running
new forecasts. Combine it with --ticker/--period filters, --review-limit to
cap the number of rows, and --review-export to save the filtered slice.

Use the menu self-test option or --action selftest to run the engines against
a synthetic dataset without requiring network connectivity.

Automation: --action train runs a single self-evaluation cycle by default.
Pass --auto-train to keep it running in a loop, and adjust --train-interval
to control how many seconds the workflow sleeps between cycles. The loop
listens for SIGINT/SIGTERM so it can stop gracefully in unattended mode.

Continuous auto-evaluation: --action auto-eval now accepts --auto-eval-loop
and --eval-interval to keep cycling through the full ticker universe. Use
--max-tickers if you want to cap the universe per pass when dealing with very
large symbol lists while still collecting hundreds of thousands of samples
over time.
        """
    )
    maybe_pause(pause)


def run_diagnostics(pause: bool = True) -> None:
    print(f"\n=== System Diagnostics for Market Monitoring & ML (build {VERSION}) ===\n")
    dependency_states = [
        ("yfinance", yfinance is not None),
        ("pandas", pd is not None),
        ("numpy", np is not None),
        ("ta", ta_mod is not None),
        ("pandas_ta", pta is not None),
        ("sklearn", skl is not None),
        ("tabulate", tabulate_mod is not None),
        ("matplotlib", plt_mod is not None),
        ("torch", torch_mod is not None),
    ]
    rows = [[name, "OK" if ok else "missing"] for name, ok in dependency_states]
    if tabulate_mod:
        print(tabulate_mod.tabulate(rows, headers=["Dependency", "Status"], tablefmt="github"))
    else:
        for name, status in rows:
            print(f"{name:>12}: {status}")

    ensure_memory_file()
    mem_size = os.path.getsize(MEM_FILE)
    print(f"\nMemory file: {MEM_FILE} ({mem_size} bytes)")

    fetch_note = "skipped (yfinance missing)"
    test_df = None
    if yfinance is not None:
        test_df = fetch_history("SPY", "day", bars_back=60)
        if test_df is None or test_df.empty:
            fetch_note = "no data returned (network/firewall?)"
        else:
            fetch_note = f"{len(test_df)} bars downloaded"
    print(f"Data fetch test: {fetch_note}")

    if test_df is not None and not test_df.empty:
        try:
            compute_indicators(test_df.tail(20))
            print("Indicator sanity check: OK")
        except Exception as exc:
            print(f"Indicator sanity check failed: {exc}")

    accuracy_probe = update_memory_accuracy_for_ticker("SPY")
    if accuracy_probe:
        print("Historical accuracy sample (SPY):")
        for engine, mape in accuracy_probe.items():
            print(f"  - {engine}: {mape:.2f}% MAPE")
    else:
        print("Historical accuracy sample: no records yet")

    maybe_pause(pause)


def run_self_test(pause: bool = True) -> None:
    print(f"\n=== Offline Self-Test (build {VERSION}) ===\n")

    if pd is None or np is None:
        print("pandas and numpy are required for the self-test.")
        return

    synthetic_df = build_synthetic_history()
    df = compute_indicators(synthetic_df)
    last_close = float(df["Close"].iloc[-1])

    offline_news = {
        "sentiment": 0,
        "headline_count": 0,
        "sources": [],
        "upcoming_events": [],
    }

    engine_outputs = list(
        filter(
            None,
            [
                technical_engine(df),
                random_forest_engine(df),
                neural_network_engine(df, offline_news),
                news_sentiment_engine("TEST", df, offline_news),
            ],
        )
    )

    if not engine_outputs:
        print("Engines could not run against the synthetic dataset.")
        return

    results = consolidate_predictions(
        "TEST",
        "hour",
        5,
        df,
        engine_outputs,
        band_percent=DEFAULT_BAND_PERCENT,
    )

    print(format_prediction_table(results, last_close))
    maybe_pause(pause)


def run_auto_evaluation_workflow(
    review_limit: int = 10,
    loop: bool = True,
    sleep_interval: int = 300,
    max_tickers: Optional[int] = None,
) -> None:
    """Cycle through the full ticker universe in a loop by default."""

    next_eligible: Dict[Tuple[str, str], datetime] = {}

    def target_time_for(period_choice: str, horizon: int) -> datetime:
        normalized = (period_choice or "day").lower()
        hours = horizon if normalized == "hour" else horizon * 24
        return datetime.now() + timedelta(hours=hours)

    def refresh_next_eligible_from_memory() -> None:
        if pd is None or not os.path.exists(MEM_FILE):
            return

        try:
            df = pd.read_csv(MEM_FILE)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"Could not refresh active horizons from memory: {exc}")
            return

        now = datetime.now()
        pending = df[(df["actual_price"].isna()) & (df["target_time"].notna())]

        for _, row in pending.iterrows():
            ticker = normalize_ticker_symbol(str(row.get("ticker", "")))
            period_choice = str(row.get("period", "")).strip().lower()
            if not ticker or period_choice not in INTERVAL_MAP:
                continue
            try:
                target_time = datetime.fromisoformat(str(row["target_time"]))
            except Exception:
                continue
            if target_time <= now:
                continue
            combo = (ticker, period_choice)
            existing = next_eligible.get(combo)
            if existing is None or target_time > existing:
                next_eligible[combo] = target_time

    def execute_pass() -> Tuple[bool, Optional[datetime]]:
        print("\n=== Auto Ticker Self-Evaluation ===\n")

        tracked_tickers = discover_tracked_tickers()
        if max_tickers is not None:
            tracked_tickers = tracked_tickers[:max_tickers]

        if not tracked_tickers:
            print("No tickers available to evaluate.")
            return False, None

        random.shuffle(tracked_tickers)
        track_periods = list(INTERVAL_MAP.keys())
        combos = [(ticker, period) for ticker in tracked_tickers for period in track_periods]

        refresh_next_eligible_from_memory()

        ready_combos: List[Tuple[str, str]] = []
        soonest_next: Optional[datetime] = None
        now = datetime.now()
        for ticker, period_choice in combos:
            history = fetch_history(
                ticker,
                period_choice,
                bars_back=default_bars_for_period(period_choice),
            )
            if history is None or history.empty:
                log(f"No history available for {ticker} ({period_choice}); skipping.")
                continue
            combo = (ticker, period_choice)
            ready_at = next_eligible.get(combo)
            if ready_at and ready_at > now:
                log(
                    "Skipping {ticker} ({period}) until {time} due to active horizon.".format(
                        ticker=ticker, period=period_choice, time=ready_at.isoformat()
                    )
                )
                if soonest_next is None or ready_at < soonest_next:
                    soonest_next = ready_at
                continue
            log(
                "History ready for {ticker} ({period}); {bars} bars downloaded.".format(
                    ticker=ticker, period=period_choice, bars=len(history)
                )
            )
            ready_combos.append((ticker, period_choice))

        if not ready_combos:
            print("Could not fetch history for any tracked ticker/period pair.")
            return False, soonest_next

        for ticker, period_choice in ready_combos:
            horizon = default_horizon_for_period(period_choice)
            run_forecast_workflow(
                ticker=ticker,
                period_choice=period_choice,
                horizon=horizon,
                pause=False,
                quiet=True,
            )
            next_eligible[(ticker, period_choice)] = target_time_for(period_choice, horizon)

        run_training_workflow(pause=False, loop=False, periods=track_periods)
        run_review_workflow(pause=False, limit=review_limit)
        return True, None

    while True:
        work_done, next_ready = execute_pass()
        if not loop:
            break
        sleep_for = max(1, sleep_interval)
        soonest_active = min(
            (ts for ts in next_eligible.values() if ts > datetime.now()),
            default=None,
        )
        if soonest_active:
            sleep_for = min(
                sleep_for,
                max(1, int((soonest_active - datetime.now()).total_seconds())),
            )
        if not work_done:
            reason = (
                f"next eligible at {next_ready.isoformat()}" if next_ready else "waiting for data"
            )
            log(
                "No tickers processed on this pass ({reason}); retrying after {seconds} seconds.".format(
                    reason=reason, seconds=sleep_for
                )
            )
        log(
            "Auto evaluation loop sleeping for {seconds} seconds before restarting...".format(
                seconds=sleep_for
            )
        )
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            log("Auto evaluation loop interrupted; exiting after current pass.")
            break


# =============================================================
# MENU
# =============================================================

def main_menu():
    while True:
        print(f"\n{divider('=')}\nStock Analyzer {VERSION} — Main Menu\n{divider('-')}")
        print_panel(
            "Select the workflow you want to run:",
            [
                "1 → SMA 20 + 50 snapshot with current indicators",
                "2 → Plot price with SMA overlays (save or display)",
                "3 → Auto Ticker Self-Evaluation",
                "4 → Help and usage guidance",
                "5 → Diagnostics and environment checks",
                "6 → Offline self-test using synthetic data",
                "7 → Exit the analyzer",
            ],
        )
        print("Enter the menu number (1-7).")
        choice = input("Choice [1-7]: ").strip()
        if choice == "1":
            run_sma_report()
        elif choice == "2":
            run_plot()
        elif choice == "3":
            run_auto_evaluation_workflow()
        elif choice == "4":
            run_help()
        elif choice == "5":
            run_diagnostics()
        elif choice == "6":
            run_self_test()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.\n")


# =============================================================
# ENTRYPOINT
# =============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock Analyzer V8")
    parser.add_argument(
        "--action",
        choices=[
            "menu",
            "sma",
            "plot",
            "csv",
            "forecast",
            "auto-eval",
            "review",
            "train",
            "help",
            "diagnostics",
            "selftest",
        ],
        default="menu",
        help="Workflow to run (default: interactive menu)",
    )
    parser.add_argument(
        "--auto-train",
        action="store_true",
        help="Run the training workflow in a continuous loop instead of once.",
    )
    parser.add_argument(
        "--train-interval",
        type=int,
        default=60,
        help="Seconds to sleep between training loop iterations when --auto-train is set.",
    )
    parser.add_argument(
        "--auto-eval-loop",
        action="store_true",
        default=None,
        help="(Deprecated) Continuously run the auto-evaluation workflow instead of a single pass.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=300,
        help="Seconds to sleep between auto-eval loop iterations when --auto-eval-loop is set.",
    )
    parser.add_argument(
        "--eval-once",
        action="store_true",
        help="Run the auto-evaluation workflow only once (loop by default).",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        help="Limit the number of tickers processed per auto-eval pass (use all by default).",
    )
    parser.add_argument("--ticker", help="Ticker symbol for automated runs.")
    parser.add_argument(
        "--period",
        choices=sorted(INTERVAL_MAP.keys()),
        help="Data period: hour, day, week, month, or quarter.",
    )
    parser.add_argument("--horizon", type=int, help="Forecast horizon in bars.")
    parser.add_argument("--bars", type=int, help="Override number of historical bars to fetch.")
    parser.add_argument("--csv-path", dest="csv_path", help="Destination path for CSV downloads.")
    parser.add_argument(
        "--plot-path",
        dest="plot_path",
        help="Save plots to this file instead of showing a window.",
    )
    parser.add_argument(
        "--band",
        type=float,
        help="Set the +/- band (in %) used to estimate hit probability around predictions.",
    )
    parser.add_argument(
        "--review-limit",
        dest="review_limit",
        type=int,
        default=20,
        help="Maximum number of prediction records to show during review (default: 20).",
    )
    parser.add_argument(
        "--review-export",
        dest="review_export",
        help="Optional path to export filtered prediction history when using --action review.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install/update dependencies before running (skipped by default).",
    )
    parser.add_argument(
        "--wheel-dir",
        dest="wheel_dir",
        help="Path to a local wheel/cache directory to use with --install-deps.",
    )
    return parser


def dispatch_cli_action(args) -> None:
    action = args.action
    if action == "menu":
        main_menu()
        return

    if action == "sma":
        run_sma_report(ticker=args.ticker, period_choice=args.period, pause=False)
    elif action == "plot":
        run_plot(
            ticker=args.ticker,
            period_choice=args.period,
            output_path=args.plot_path,
            bars_back=args.bars,
        )
    elif action == "csv":
        run_csv_download(
            ticker=args.ticker,
            period_choice=args.period,
            bars_override=args.bars,
            output_path=args.csv_path,
            pause=False,
        )
    elif action == "forecast":
        run_forecast_workflow(
            ticker=args.ticker,
            period_choice=args.period,
            horizon=args.horizon,
            band=args.band,
            pause=False,
        )
    elif action == "auto-eval":
        run_auto_evaluation_workflow(
            review_limit=args.review_limit,
            loop=args.auto_eval_loop if args.auto_eval_loop is not None else not args.eval_once,
            sleep_interval=args.eval_interval,
            max_tickers=args.max_tickers,
        )
    elif action == "review":
        run_review_workflow(
            pause=False,
            ticker=args.ticker,
            period=args.period,
            limit=args.review_limit,
            export_path=args.review_export,
        )
    elif action == "train":
        run_training_workflow(
            pause=False,
            loop=args.auto_train,
            sleep_interval=args.train_interval,
        )
    elif action == "help":
        run_help(pause=False)
    elif action == "diagnostics":
        run_diagnostics(pause=False)
    elif action == "selftest":
        run_self_test(pause=False)
    else:  # pragma: no cover - argparse should guard this
        raise SystemExit(f"Unknown action: {action}")


def main(argv: Optional[List[str]] = None) -> None:
    print_banner()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bootstrap(install_deps=args.install_deps, wheel_dir=args.wheel_dir)
    dispatch_cli_action(args)


if __name__ == "__main__":
    main()
