# Stock Analyzer V8

This build bundles the market data workflows, forecasting engines, and the
self-training loop into a single executable script.

## Quick start

```bash
python3 stock_analyzer_v8.py --action menu
```

## Refreshing the ticker universe

The default ticker universe comes from `data/trading212_listings.csv`, which
now bundles the full S&P 500 along with a curated ETF set (Vanguard, SPDR, and
other popular funds). Use the helper script to regenerate the file when new
symbols are added or when you want to merge your own Trading212 export:

```bash
python3 data/refresh_trading212_listings.py \
  --trading212-export path/to/your_trading212_export.csv \
  --output data/trading212_listings.csv
```

The script downloads the latest S&P 500 membership; if the network is blocked
it falls back to the embedded snapshot contained in the repository so coverage
is never lost. The app will read this file automatically via the
`TICKER_UNIVERSE_CSV` environment variable (defaults to `data/trading212_listings.csv`).

## Automated training

The training workflow can run once or continuously:

- Run a single pass (default):
  ```bash
  python3 stock_analyzer_v8.py --action train
  ```
- Enable continuous mode with a configurable pause between passes:
  ```bash
  python3 stock_analyzer_v8.py --action train --auto-train --train-interval 120
  ```

When `--auto-train` is set, the loop sleeps for the given interval (seconds)
before starting the next pass. It listens for `SIGINT`/`SIGTERM` so it can shut
down cleanly and stop before launching the next cycle, preventing partially
written state in unattended mode.
