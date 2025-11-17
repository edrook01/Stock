# Stock Analyzer V7

This build bundles the market data workflows, forecasting engines, and the
self-training loop into a single executable script.

## Quick start

```bash
python3 stock_analyzer_v7.py --action menu
```

## Automated training

The training workflow can run once or continuously:

- Run a single pass (default):
  ```bash
  python3 stock_analyzer_v7.py --action train
  ```
- Enable continuous mode with a configurable pause between passes:
  ```bash
  python3 stock_analyzer_v7.py --action train --auto-train --train-interval 120
  ```

When `--auto-train` is set, the loop sleeps for the given interval (seconds)
before starting the next pass. It listens for `SIGINT`/`SIGTERM` so it can shut
down cleanly and stop before launching the next cycle, preventing partially
written state in unattended mode.
