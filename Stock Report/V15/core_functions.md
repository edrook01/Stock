# Core Function Contracts (Do Not Refactor Without Authorization)

These three capabilities are locked interfaces. Treat them as living contracts: extend implementations only via the extension points they expose and never rewrite them wholesale without explicit approval from the project owner.

**Global constraints**
- Preserve existing model weights when updating any learner.
- Validate every new training sample; reject or quarantine corrupt data.
- Never retrain the unified model stack from scratch unless the owner explicitly authorizes a full reset.
- Any file explicitly labeled `core_contracts.md` is strictly read-only—do not edit it, and ensure your changes comply with the requirements it declares.

## 1. Manual User Data Analysis
- Accepts an ad-hoc ticker (or basket) entered by the human operator.
- Runs the full ML prediction pipeline on-demand and returns a current prediction bundle across all tracked intervals (1m, 5m, 10m, 15m, 1h, 1d, 1mo, 3mo, 1y).
- Generates a written, auditable summary that explains how the conclusion was reached, explicitly citing technical indicators, news signals, and market-sentiment inputs that influenced the outcome.
- Reports interval target price, target high, target low, elapsed high/low/close (once interval completes), confidence rating, and the three accuracy metrics after the interval elapses.

## 2. Autonomous Trading
- Runs end-to-end automation: signal generation → trade execution → position monitoring → outcome tracking.
- Must execute trades, manage stops/targets, and log every action with timestamps and identifiers that can be reconciled with broker logs.
- Keeps the CLI available while also exposing its status to the Streamlit UI when that layer ships.
- Tracks trade outcomes so constant learning can ingest them without losing historical weights.

## 3. Constant Learning
- Continuously produces fresh predictions for every supported interval (1m, 5m, 10m, 15m, 1h, 1d, 1mo, 3mo, 1y) and automatically back-fills any interval whose prior prediction has elapsed.
- Every prediction includes the contract data fields: interval target price/high/low, elapsed high/low/close when available, current confidence rating, plus three post-interval accuracy ratings.
- Evaluates elapsed predictions automatically, refills each interval with a new prediction, and records the evaluation results for downstream analytics.
- Treat the ticker universe as the full Trading212 listed market (see `data/tickers.txt`) so the constant learner continuously covers every available instrument, not just a hand-picked subset.
- Improves accuracy incrementally: reuse prior weights, validate incoming samples, and avoid full retrains unless the owner says otherwise.

Failure to comply with this document breaks the core API and is not allowed without written approval. Keep this file synced with any future `core_contracts.md` references and treat both as source-of-truth guardrails.

