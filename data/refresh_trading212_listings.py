#!/usr/bin/env python3
"""Refresh the bundled ticker universe.

This script downloads the S&P 500 constituents, merges them with a curated list
of ETFs and any provided Trading212 export, and writes the combined list to
``data/trading212_listings.csv``. If the online source is unavailable, it falls
back to an embedded S&P 500 snapshot so you can refresh offline without losing
coverage.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.request import urlopen

# Core ETF list that should always be present even if the Trading212 export is
# missing a few of them.
ETF_ROWS = [
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSEARCA"},
    {"ticker": "VOO", "name": "Vanguard S&P 500 ETF", "exchange": "NYSEARCA"},
    {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSEARCA"},
    {"ticker": "VXUS", "name": "Vanguard Total International Stock ETF", "exchange": "NASDAQ"},
    {"ticker": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "exchange": "NYSEARCA"},
    {"ticker": "VT", "name": "Vanguard Total World Stock ETF", "exchange": "NYSEARCA"},
    {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "exchange": "NYSEARCA"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
    {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSEARCA"},
    {"ticker": "EFA", "name": "iShares MSCI EAFE ETF", "exchange": "NYSEARCA"},
    {"ticker": "EEM", "name": "iShares MSCI Emerging Markets ETF", "exchange": "NYSEARCA"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "exchange": "NYSEARCA"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "exchange": "NASDAQ"},
    {"ticker": "XLK", "name": "Technology Select Sector SPDR Fund", "exchange": "NYSEARCA"},
    {"ticker": "XLF", "name": "Financial Select Sector SPDR Fund", "exchange": "NYSEARCA"},
    {"ticker": "XLE", "name": "Energy Select Sector SPDR Fund", "exchange": "NYSEARCA"},
    {"ticker": "IEMG", "name": "iShares Core MSCI Emerging Markets ETF", "exchange": "NYSEARCA"},
    {"ticker": "SCHD", "name": "Schwab US Dividend Equity ETF", "exchange": "NYSEARCA"},
    {"ticker": "VUG", "name": "Vanguard Growth ETF", "exchange": "NYSEARCA"},
    {"ticker": "VTV", "name": "Vanguard Value ETF", "exchange": "NYSEARCA"},
]

S_AND_P_500_CSV = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

STATIC_SP500_TICKERS = [
    'A', 'AAPL', 'ABBV', 'ABC', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK',
    'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALLE', 'ALNY', 'AMAT',
    'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD',
    'APH', 'APTV', 'ARE', 'ATO', 'ATR', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA',
    'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK',
    'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB',
    'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW',
    'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
    'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS',
    'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CTXS', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS',
    'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ',
    'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX',
    'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR',
    'ETSY', 'EVRG', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FB', 'FBHS', 'FCX', 'FDS',
    'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLS', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRC',
    'FRT', 'FTI', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC',
    'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES',
    'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
    'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG',
    'IQV', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM',
    'K', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L', 'LDOS', 'LEN', 'LH', 'LHX',
    'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LULU', 'LVS', 'LW', 'LYB', 'LYV',
    'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM',
    'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA',
    'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE',
    'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
    'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA', 'PAYC',
    'PAYX', 'PCAR', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PM',
    'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD',
    'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK',
    'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEDG', 'SEE', 'SHW', 'SIVB',
    'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ',
    'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC',
    'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT',
    'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI',
    'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB',
    'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION',
    'ZTS',
]


def normalize_ticker(value: Optional[str]) -> Optional[str]:
    ticker = (value or "").strip().upper()
    if not ticker:
        return None
    return ticker


def load_trading212_export(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            ticker = None
            for key in ("ticker", "Ticker", "symbol", "Symbol"):
                if key in row:
                    ticker = normalize_ticker(row.get(key))
                    break
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "name": row.get("name")
                    or row.get("Name")
                    or row.get("description")
                    or row.get("Description")
                    or "",
                    "exchange": row.get("exchange") or row.get("Exchange") or "",
                }
            )
        return rows


def fetch_sp500(existing_output: Optional[Path] = None) -> List[dict]:
    try:
        with urlopen(S_AND_P_500_CSV, timeout=30) as response:
            content = response.read().decode("utf-8")
        buffer = io.StringIO(content)
        reader = csv.DictReader(buffer)
        rows = []
        for row in reader:
            ticker = normalize_ticker(row.get("Symbol"))
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "name": row.get("Name", ""),
                    "exchange": row.get("Exchange", ""),
                }
            )
        if rows:
            return rows
    except Exception as exc:  # pragma: no cover - best effort network fetch
        print(f"Failed to download S&P 500 list ({exc}); using embedded snapshot.")

    # Offline fallback: use the embedded ticker set
    return [
        {"ticker": ticker, "name": ticker, "exchange": "S&P500"}
        for ticker in STATIC_SP500_TICKERS
    ]


def unique_rows(rows: Iterable[dict]) -> List[dict]:
    seen = set()
    deduped: List[dict] = []
    for row in rows:
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        deduped.append(
            {
                "ticker": ticker,
                "name": row.get("name", ""),
                "exchange": row.get("exchange", ""),
            }
        )
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh ticker universe CSV")
    parser.add_argument(
        "--trading212-export",
        type=Path,
        default=None,
        help="Optional path to a Trading212 CSV export to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("trading212_listings.csv"),
        help="Destination CSV file",
    )
    args = parser.parse_args()

    rows: List[dict] = []
    rows.extend(ETF_ROWS)
    rows.extend(fetch_sp500(existing_output=args.output))

    if args.trading212_export:
        rows.extend(load_trading212_export(args.trading212_export))

    rows = unique_rows(rows)
    rows.sort(key=lambda row: row["ticker"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "name", "exchange"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tickers to {args.output}")


if __name__ == "__main__":
    main()
