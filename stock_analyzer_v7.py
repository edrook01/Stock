#!/usr/bin/env python3
"""Stock Analyzer V7 – Predictive (Complete Build).

This release folds in all improvements that landed after the previous merge
and ships as a single, self-contained file. The deterministic bootstrapper is
still here, but the CLI has been expanded with argument-based automation, an
offline self-test harness, and a diagnostics workflow so the tool is easier to
debug on fresh machines.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import subprocess
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

VERSION = "V7"
print(f"\n=== Stock Analyzer {VERSION} - Predictive (Complete Build) ===\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(BASE_DIR, "libs")
MEM_DIR = os.path.join(BASE_DIR, "memory")

os.makedirs(LIB_DIR, exist_ok=True)
os.makedirs(MEM_DIR, exist_ok=True)
sys.path.insert(0, LIB_DIR)

MEM_FILE = os.path.join(MEM_DIR, "predictions_log.csv")
DEFAULT_BAND_PERCENT = 5.0


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


def install_packages_local() -> None:
    log("Checking dependencies...")
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
            subprocess.check_call(
                [
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
            )
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


def bootstrap() -> None:
    log("Bootstrapping...")
    install_packages_local()

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
        print("Critical imports failed. Exiting.")
        sys.exit(1)

    log("Bootstrap complete.\n")
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


def finalize_section(pause: bool) -> None:
    maybe_pause(pause)
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


def update_memory_accuracy_for_ticker(ticker: str) -> Dict[str, float]:
    ensure_memory_file()
    try:
        df = pd.read_csv(MEM_FILE)
    except Exception:
        return {}

    if df.empty:
        return {}

    df_ticker = df[df["ticker"] == ticker].copy()
    if df_ticker.empty:
        return {}

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
        return {}

    return (
        grouped.groupby("engine")["pct_error"].mean().to_dict()
    )  # type: ignore[return-value]


# =============================================================
# DATA / INDICATORS
# =============================================================

INTERVAL_MAP = {
    "hour": ("60d", "1h"),
    "day": ("5y", "1d"),
    "week": ("10y", "1wk"),
}


def fetch_history(ticker: str, period_choice: str, bars_back: int = 300):
    if yfinance is None:
        raise RuntimeError("yfinance missing after bootstrap")

    period_choice = period_choice.lower()
    period, interval = INTERVAL_MAP.get(period_choice, INTERVAL_MAP["day"])

    try:
        ticker_client = yfinance.Ticker(ticker)
        data = ticker_client.history(period=period, interval=interval)
        if data is None or data.empty:
            return None
        if len(data) > bars_back:
            data = data.tail(bars_back)
        return data
    except Exception as exc:
        log(f"Failed to fetch data for {ticker}: {exc}")
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

    info = None
    try:
        info = yfinance.Ticker(ticker).get_info()
    except Exception:
        try:
            info = yfinance.Ticker(ticker).info
        except Exception:
            return None

    if not info:
        return None

    target_mean = _coerce(info.get("targetMeanPrice"))
    target_high = _coerce(info.get("targetHighPrice"))
    target_low = _coerce(info.get("targetLowPrice"))
    analyst_count = safe_int(info.get("numberOfAnalystOpinions"), 0)

    fallback_price = _coerce(info.get("currentPrice")) or _coerce(
        info.get("regularMarketPrice")
    )
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


def neural_network_engine(df) -> Optional[EngineResult]:  # type: ignore[override]
    df_feat = build_feature_frame(df)
    if len(df_feat) < 120:
        return None

    target = df_feat["Close"].shift(-1).dropna()
    features = df_feat.loc[target.index, FEATURE_COLUMNS]

    if torch_mod is not None:
        return _torch_lstm_engine(features, target)
    return _mlp_engine(features, target)


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
            "Choose one of: hour, day, or week to set candle duration.",
            "day",
            "day",
        )
    ).strip().lower() or "day"
    df = fetch_history(ticker, period_choice, bars_back=400)
    if df is None or df.empty:
        print("No data downloaded.")
        return
    df = compute_indicators(df)
    latest = df.iloc[-1]
    rows = [
        ["Close", f"{latest['Close']:.2f}"],
        ["SMA20", f"{latest['SMA20']:.2f}"],
        ["SMA50", f"{latest['SMA50']:.2f}"],
        ["SMA200", f"{latest['SMA200']:.2f}"],
        ["RSI14", f"{latest['RSI14']:.2f}"],
        ["MACD", f"{latest['MACD']:.2f}"],
        ["ATR14", f"{latest['ATR14']:.2f}"],
    ]
    if tabulate_mod:
        print(tabulate_mod.tabulate(rows, tablefmt="github"))
    else:
        for key, value in rows:
            print(f"{key:>8}: {value}")
    print()
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
            "Select hour, day, or week to match your plotting cadence.",
            "week",
            "day",
        )
    ).strip().lower() or "day"
    df = fetch_history(ticker, period_choice, bars_back=bars_back or 300)
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
            "Pick hour, day, or week candles for the CSV output.",
            "hour",
            "day",
        )
    ).strip().lower() or "day"
    df = fetch_history(ticker, period_choice, bars_back=bars_override or 2000)
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
) -> None:
    try:
        print("\n=== Forecast (Technical + ML + NN) ===\n")
        ticker = (
            ticker
            or guided_prompt(
                "Ticker symbol",
                "Specify one uppercase ticker to run the combined forecast.",
                "NVDA",
            )
        ).strip().upper()
        if not ticker:
            print("Ticker required.")
            return
        period_choice = (
            period_choice
            or guided_prompt(
                "Period",
                "Choose the candle size: hour, day, or week.",
                "hour",
                "day",
            )
        ).strip().lower() or "day"
        horizon = (
            horizon
            if horizon is not None
            else safe_int(
                guided_prompt(
                    "Forecast horizon in bars",
                    "Enter a positive integer showing how many candles to project.",
                    "5",
                    "5",
                ),
                5,
            )
        )
        horizon = max(1, horizon)
        band = (
            band
            if band is not None
            else safe_float(
                guided_prompt(
                    "+/- band for probability (%)",
                    "Set the percentage band to estimate hit probability around each target.",
                    f"{DEFAULT_BAND_PERCENT:.1f}",
                    f"{DEFAULT_BAND_PERCENT:.1f}",
                ),
                DEFAULT_BAND_PERCENT,
            )
        )
        band = max(0.5, band)
        df = fetch_history(ticker, period_choice, bars_back=600)
        if df is None or df.empty:
            print("No data downloaded.")
            return

        df = compute_indicators(df)
        last_close = float(df["Close"].iloc[-1])
        horizon_days = max(1, math.ceil(horizon / 6)) if period_choice == "hour" else max(1, horizon)
        news_context = collect_news_context(ticker, horizon_days=horizon_days)
        history_accuracy = update_memory_accuracy_for_ticker(ticker)
        if history_accuracy:
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
                    neural_network_engine(df),
                    news_sentiment_engine(ticker, df, news_context),
                ],
            ),
            band_percent=band,
        )

        if not results:
            print("No engine produced a forecast.")
            return

        print(format_prediction_table(results, last_close))
        print(
            f"Probabilities reflect the chance of landing within ±{band:.1f}% of each target using recent volatility."
        )
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
        finalize_section(pause)


def run_training_workflow(pause: bool = True, loop: bool = False) -> None:
    def execute_training_pass() -> None:
        print("\n=== Training / Self-Evaluation ===\n")
        ensure_memory_file()
        try:
            df = pd.read_csv(MEM_FILE)
        except Exception as exc:
            print(f"Could not load memory file: {exc}")
            return

        if df.empty:
            print("No prediction history found. Run forecasts to generate training data.")
            return

        tickers = sorted(df["ticker"].dropna().unique())
        completed = int(df["actual_price"].notna().sum())
        pending = int(len(df) - completed)

        print(
            "Found {rows} prediction records across {tickers} tracked tickers.".format(
                rows=len(df), tickers=len(tickers)
            )
        )
        print(f"Completed predictions with actual prices: {completed}")
        print(f"Pending predictions awaiting actual prices: {pending}")

        per_ticker_accuracy: Dict[str, Dict[str, float]] = {}
        global_accuracy: Dict[str, List[float]] = {}

        for ticker in tickers:
            accuracy = update_memory_accuracy_for_ticker(ticker)
            if not accuracy:
                continue
            per_ticker_accuracy[ticker] = accuracy
            for engine, mape in accuracy.items():
                global_accuracy.setdefault(engine, []).append(mape)

        if per_ticker_accuracy:
            print("\nUpdated MAPE (by ticker and engine):")
            for ticker, scores in per_ticker_accuracy.items():
                parts = ", ".join(f"{engine}: {mape:.2f}%" for engine, mape in scores.items())
                print(f"  - {ticker}: {parts}")
        else:
            print("\nNo tickers have evaluable targets yet; waiting for target times to elapse.")

        if global_accuracy:
            print("\nAverage MAPE across all tracked tickers:")
            for engine, values in global_accuracy.items():
                avg_mape = sum(values) / len(values)
                print(f"  - {engine}: {avg_mape:.2f}% (n={len(values)})")

    try:
        while True:
            execute_training_pass()
            if not loop:
                break

            print(
                "\nTraining loop controls: [Enter] rerun training, [P] forecast now, [M] return to menu, [Q] quit"
            )
            selection = input("Selection: ").strip().lower()
            if selection in {"m", "menu"}:
                break
            if selection in {"q", "quit"}:
                sys.exit(0)
            if selection in {"p", "predict", "forecast"}:
                run_forecast_workflow(pause=False)
                continue
            # Any other input (including Enter) simply reruns the training pass
    finally:
        finalize_section(pause and not loop)


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

Use the menu self-test option or --action selftest to run the engines against
a synthetic dataset without requiring network connectivity.
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
                neural_network_engine(df),
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


# =============================================================
# MENU
# =============================================================

def main_menu():
    while True:
        print(
            f"""
=== Stock Analyzer {VERSION} ===
1. SMA 20 + 50 Report
2. Plot Price + SMA
3. Download Historical CSV
4. Forecast (Technical + ML + NN)
5. Training loop / self-eval
6. Help
7. System Diagnostics (Monitoring/ML)
8. Offline self-test (synthetic data)
9. Exit
"""
            f"\n{divider('=')}\nStock Analyzer {VERSION} — Main Menu\n{divider('-')}"
        )
        print_panel(
            "Select the workflow you want to run:",
            [
                "1 → SMA 20 + 50 snapshot with current indicators",
                "2 → Plot price with SMA overlays (save or display)",
                "3 → Download historical candles as a CSV export",
                "4 → Forecast price movement (technical + ML + NN)",
                "5 → Train from stored predictions for self-evaluation",
                "6 → Help and usage guidance",
                "7 → Diagnostics and environment checks",
                "8 → Offline self-test using synthetic data",
                "9 → Exit the analyzer",
            ],
        )
        print("Enter the menu number (1-9) as a single digit.")
        choice = input("Choice [1-9]: ").strip()
        if choice == "1":
            run_sma_report()
        elif choice == "2":
            run_plot()
        elif choice == "3":
            run_csv_download()
        elif choice == "4":
            run_forecast_workflow()
        elif choice == "5":
            run_training_workflow(loop=True)
        elif choice == "6":
            run_help()
        elif choice == "7":
            run_diagnostics()
        elif choice == "8":
            run_self_test()
        elif choice == "9":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.\n")


# =============================================================
# ENTRYPOINT
# =============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock Analyzer V7")
    parser.add_argument(
        "--action",
        choices=[
            "menu",
            "sma",
            "plot",
            "csv",
            "forecast",
            "train",
            "help",
            "diagnostics",
            "selftest",
        ],
        default="menu",
        help="Workflow to run (default: interactive menu)",
    )
    parser.add_argument("--ticker", help="Ticker symbol for automated runs.")
    parser.add_argument(
        "--period",
        choices=sorted(INTERVAL_MAP.keys()),
        help="Data period: hour, day or week.",
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
    elif action == "train":
        run_training_workflow(pause=False)
    elif action == "help":
        run_help(pause=False)
    elif action == "diagnostics":
        run_diagnostics(pause=False)
    elif action == "selftest":
        run_self_test(pause=False)
    else:  # pragma: no cover - argparse should guard this
        raise SystemExit(f"Unknown action: {action}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bootstrap()
    dispatch_cli_action(args)


if __name__ == "__main__":
    main()
