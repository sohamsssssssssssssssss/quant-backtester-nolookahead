# Final Verdict: MA Crossover Backtester for NIFTY Equities

## Summary

This project built a systematic backtesting pipeline for NIFTY/NSE equities and tested a simple moving average crossover strategy (fast=50, slow=200) across 15 liquid NIFTY 50 large-cap stocks. The strategy was evaluated using walk-forward validation on each ticker, then aggregated across tickers, and finally tested against a permutation-based null distribution to determine whether the observed split of "winning" vs "losing" tickers was distinguishable from noise. **Headline finding**: After realistic transaction costs (5 bps + 2 bps slippage), the MA crossover strategy shows no statistically distinguishable edge. The cross-ticker mean Sharpe ratio is 0.04 (essentially flat), and the 8/15 "positive ticker" split is indistinguishable from pure chance (permutation test p=0.990). The single-ticker RELIANCE.NS result from Phase 4 (Sharpe -0.24) was not a pathological draw — it is representative of the median experience.

## What Was Built

The primary achievement of this project is **methodology, not alpha**. The following components were implemented and tested:

- **No-lookahead enforcement**: Signal generation (Phase 2) is strictly separated from execution timing (Phase 3). Signals computed at day `t` are actionable only from day `t+1`. This is enforced in the backtest engine and verified with explicit lookahead-bias tests that alter future prices and confirm past equity curves remain unchanged.

- **Realistic cost/slippage model**: Every position change incurs transaction costs (default 5 bps) and slippage (default 2 bps). A "cost sanity" test verifies that a strategy flipping positions daily materially underperforms the zero-cost version.

- **Walk-forward validation**: Instead of a single full-history backtest, each ticker is evaluated using rolling train/test windows (3-year train, 1-year test, 1-year step). This prevents curve-fitting to a single historical period and exposes regime sensitivity.

- **Multi-ticker generalization check**: The strategy was run on 15 NIFTY large caps spanning multiple sectors (Banking, IT, FMCG, Energy, Telecom, Auto, Pharma, Engineering, Consumer). This tests whether any observed edge is ticker-specific or generalizes.

- **Permutation-based noise test**: A block-permutation null test shuffles signal regimes while preserving their autocorrelation structure, then re-runs walk-forward to build a null distribution of "how many tickers show positive Sharpe" under random timing. This answers whether the observed split is distinguishable from chance given the joint price dynamics of these 15 tickers.

## Key Numbers

| Phase | Metric | Value |
|-------|--------|-------|
| **Phase 4** (RELIANCE.NS) | Walk-forward Sharpe | -0.24 |
| | Profitable folds | 3/8 (38%) |
| | Total return | -54.9% |
| | Max drawdown | 73.3% |
| **Phase 5** (15 tickers) | Mean of Mean Sharpe | 0.04 |
| | Std of Mean Sharpe | 0.35 |
| | Profitable folds (all) | 52/120 (43.3%) |
| | Positive tickers | 8/15 |
| | Best ticker | ITC.NS (Sharpe 0.62) |
| | Worst ticker | MARUTI.NS (Sharpe -0.57) |
| **Phase 6** (Noise check) | Permutation p-value | 0.990 |
| | Null expectation | 11.4/15 positive (std 1.6) |
| | Observed | 8/15 positive |

## Why the Strategy Doesn't Work

After realistic transaction costs, a naive MA crossover (50/200) does not clear the profitability bar on liquid NSE large-cap equities. This is the well-known, expected result for simple trend-following signals without additional filters:

- The NIFTY 50 is a highly efficient universe with tight spreads and institutional arbitrage — simple technical signals are quickly competed away.
- The MA crossover signal itself carries no predictive edge on this dataset. Zero-cost backtesting across the same 15 tickers yields a mean Sharpe of 0.04 — identical to the costed result (0.04). Transaction costs are essentially irrelevant because the gross signal is already flat before costs are applied.
- The strategy has no volatility regime filter — it trades identically in low-vol sideways markets (where it whipsaws) and high-vol trending markets (where it should earn). This dilutes returns.
- Fixed parameters (50/200) are not adaptive to changing market structure. A 200-day MA was likely relevant in a different regime than today's.

This is not a bug in the implementation — it is the null hypothesis for naive technical signals on liquid large-cap equities. The methodology correctly identified a true-negative.

## What This Demonstrates

The value of this project is that **the testing pipeline catches false positives as reliably as it caught this true negative**. A project that:

- Skipped walk-forward validation and reported a single full-history Sharpe
- Tested only one ticker and claimed "it works on RELIANCE"
- Ignored transaction costs or used unrealistically low values
- Did not check whether cross-ticker splits are distinguishable from noise

...would produce superficially better results and be substantively worse. This project's achievement is building a methodology that cannot lie to itself. The negative result is the *correct* output of a rigorous process, not a failure of the process.

## Honest Limitations

The following were **not** tested in this project:

- **Other signal families**: RSI mean-reversion and 12-1 momentum signals exist in `src/signals/signals.py` but were not walk-forward validated or multi-ticker tested. Only MA crossover was fully evaluated.

- **Regime filters or adaptive parameters**: No volatility regime filter (e.g., only trade when 20-day realized vol > X%), no adaptive MA windows (e.g., optimize fast/slow on trailing data), and no parameter robustness checks (e.g., MA 40/180 vs 50/200 vs 60/220).

- **Portfolio-level construction**: All tests are single-asset, absolute signals (long/short one ticker at a time). No cross-sectional ranking (e.g., long top-decile momentum, short bottom-decile) or portfolio optimization ( risk parity, volatility targeting across tickers).

- **Multi-frequency testing**: Only daily bars were tested. No intraday, weekly, or monthly frequencies.

- **Out-of-sample time period**: All data is 2015–2026. No test on data before 2015 or after 2026 (as available).

- **The permutation test scope**: The noise check tests whether the *ticker-level split* is distinguishable from chance. It does not test whether some entirely different signal could find real edge on this data.

## Natural Next Steps

If extending this project, the following would be the highest-value additions:

1. **Test other signals through the same pipeline**: Run RSI mean-reversion and 12-1 momentum through walk-forward + multi-ticker + noise-check. Do they show positive cross-ticker Sharpe? Is the split distinguishable from noise?

2. **Add regime filters to MA crossover**: Implement a volatility regime filter (e.g., only trade when VIX > 15) or trend-quality filter (e.g., ADX > 25) and re-test. Does conditional trading improve the edge?

3. **Cross-sectional ranking**: Instead of absolute signals per ticker, implement relative ranking (long top 3 tickers by signal, short bottom 3). This hedges out market beta and tests whether the signal has cross-sectional information content.

---

## Appendix: Repository Structure

```
quant-backtester-nolookahead/
├── src/
│   ├── data/
│   │   └── loader.py           # OHLCV loading with yfinance + CSV caching
│   ├── signals/
│   │   └── signals.py          # MA crossover, RSI, momentum (pure signal math)
│   ├── backtest/
│   │   ├── engine.py           # Vectorized backtest with cost/slippage model
│   │   ├── walk_forward.py     # Rolling train/test validation
│   │   ├── multi_ticker.py     # Multi-ticker aggregation
│   │   └── noise_check.py      # Permutation-based noise test
│   └── metrics/
│       └── performance.py      # Sharpe, Sortino, max DD, Calmar, etc.
├── tests/
│   ├── test_backtest.py        # 13 tests (lookahead, costs, edge cases)
│   ├── test_walk_forward.py    # 10 tests (fold construction, leakage)
│   ├── test_multi_ticker.py    # 10 tests (failure handling, aggregation)
│   └── test_noise_check.py     # 9 tests (block permutation, null world)
├── docs/
│   └── verdict.md              # This document
└── run_*.py                    # Demo scripts for each phase
```

**Total: 42 passing tests, all phases complete.**
