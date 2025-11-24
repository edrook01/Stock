# Core Function Contracts (V15)

These three capabilities are *locked contracts/APIs*. Do not refactor or replace them without explicit authorization from the project owner.

## 1. Manual User Data Analysis
- Must accept manual ticker input and run the unified ML inference pipeline.
- Must produce interval-based predictions plus an explanatory summary (technical, news, sentiment factors).
- Must preserve previous model weights between runs.
- Must validate new samples before incorporating them into any learning process.
- Must never retrain the model from scratch during manual analysis mode.

## 2. Autonomous Trading
- Must run the end-to-end automation stack (data fetch, signal, trade execution, outcome tracking).
- Must execute trades only after the Manual Analysis contract has produced a valid prediction payload.
- Must log actions, broker confirmations, and any deviations for post-trade evaluation.
- Must reuse existing broker state and credentials; no destructive resets mid-session.
- Must maintain prediction-to-trade traceability for later audits.

## 3. Constant Learning
- Must continuously generate predictions for 1m, 5m, 10m, 15m, 1h, 1d, 1mo, 3mo, and 1y timeframes.
- Must track elapsed predictions, record actual high/low/close outcomes, and automatically refill each interval once it elapses.
- Must output a confidence score plus three accuracy ratings (high/low/close) after each interval completes.
- Must draw its ticker universe from the entire Trading212 market listing stored in `data/tickers.txt`, ensuring every listed instrument receives active coverage.
- Must preserve previous weights, validate any new samples, and avoid full retrains; learning is strictly incremental.
- Must only update models once Autonomous Trading and Manual Analysis artifacts have been archived for that interval.

---

**Core Contract Safeguard:** Do not modify any file marked as `core_contracts.md`. All code across the repo must comply with the constraints in that file. Keep this warning in place to alert new agents.

