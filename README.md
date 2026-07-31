# quant-backtester-nolookahead

A rigorously-tested backtesting pipeline for NIFTY/NSE equities, built to demonstrate **methodology** — no-lookahead enforcement, realistic transaction costs, walk-forward validation, multi-ticker generalization, and statistical noise-checking — rather than to chase a high Sharpe ratio.

The core idea: **a pipeline that cannot lie to itself.** A negative result found honestly beats a fake positive result found by accident or overfitting, every time. This repo tests several well-known trading signals on the Indian equity market and reports what actually happened, including the failures.

---

## What this project actually shows

Every strategy tested here was evaluated the same way:

1. Build the signal with strict no-lookahead guarantees (nothing uses information not yet public on that date).
2. Backtest with realistic transaction costs, not zero-cost assumptions.
3. Validate out-of-sample via walk-forward folds, not a single in-sample fit.
4. Run a permutation-based noise-check (shuffle the null hypothesis hundreds/thousands of times) to see if the result is distinguishable from random chance.

| Strategy | Universe | Walk-Forward Sharpe | Noise-Check p-value | Verdict |
|---|---|---|---|---|
| MA Crossover (50/200), single ticker | RELIANCE.NS | -0.24 | — | Not profitable |
| MA Crossover (50/200), basket | 15 NIFTY large caps | 0.04 (flat) | 0.990 (n=500) | Indistinguishable from noise |
| Cross-Sectional Momentum (12-1, long/short) | 15 NIFTY large caps | -0.42 | 0.135 (n=2000) | Indistinguishable from noise |
| Earnings Momentum / PEAD | 15 NIFTY large caps | -0.068 (OOS) vs +0.564 (full-sample) | pending full-n confirmation | Full-sample result did not survive out-of-sample validation — textbook overfitting example |

**None of the tested strategies produced a statistically significant, out-of-sample edge.** That is treated here as a real finding, not a failure of the project. The value of the repo is the methodology that caught it.

---

## Repo structure

```
src/
  data/
    loader.py                        # OHLCV data loading, no-lookahead-safe
  signals/
    signals.py                       # MA crossover, RSI, momentum
    cross_sectional_momentum.py      # Cross-sectional ranking (12-1 momentum)
    earnings_momentum.py             # PEAD / earnings-surprise signal
  backtest/
    engine.py                        # Core backtest engine + cost model
    walk_forward.py                  # Single-ticker walk-forward validation
    multi_ticker.py                  # Multi-ticker walk-forward
    cross_sectional_engine.py        # Long/short portfolio engine
    cross_sectional_walk_forward.py  # Walk-forward for cross-sectional strategies
    noise_check.py                   # Permutation-based null hypothesis test
  metrics/
    performance.py                   # Sharpe, drawdown, returns, etc.

tests/                                # pytest suite covering lookahead safety,
                                       # signal correctness, engine behavior,
                                       # and noise-check validity

docs/
  hypotheses/                        # Pre-registered hypotheses, committed
                                       # BEFORE each backtest is run
  verdict.md                         # Final, honest write-up of all results

results/                              # Committed raw permutation values and
                                       # backtest outputs (not /tmp — durable,
                                       # reproducible from the repo)
```

---

## Methodology notes (why this repo is built the way it is)

**No-lookahead enforcement.** Every signal is checked to confirm it only uses data that would have been publicly available at the time of the trading decision — including earnings data, which uses as-reported (not restated) figures keyed to actual announcement dates.

**Realistic costs.** All backtests include transaction costs (basis points per trade), not just gross returns. Where cost sensitivity matters, zero-cost and with-cost runs are compared directly.

**Walk-forward, not single-fit.** Every strategy is validated across multiple out-of-sample folds. A strategy that looks good full-sample but fails walk-forward (see: Earnings Momentum above) is reported as a failure, not smoothed over — that discrepancy is itself the point of doing walk-forward validation.

**Permutation noise-checks.** For each strategy, the observed test statistic (Sharpe, ticker-positive count, etc.) is compared against a null distribution built by permuting labels/regimes hundreds or thousands of times. Results are only treated as final once the permutation count is large enough that the p-value has visibly stabilized — partial or timed-out runs are never reported as final.

**Pre-registration.** Before testing a new hypothesis, the hypothesis, universe, and expected mechanism are written to `docs/hypotheses/` and committed *before* any backtest is run, so the reasoning can't be quietly adjusted after seeing the result.

**Multiple-testing awareness.** With more than one strategy tested, the probability of a false positive rises. `docs/verdict.md` states this explicitly rather than reporting each test in isolation.

---

## What this project deliberately does NOT do

- Does not set a target Sharpe ratio and optimize toward it — that's how curve-fitting happens, and it defeats the purpose of having a noise-check pipeline at all.
- Does not report a p-value from a partial or timed-out permutation run as final.
- Does not let a full-sample or marginally-significant result slip into the verdict as a trading recommendation without out-of-sample confirmation.

---

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Running a backtest

```bash
python run_noise_check.py            # MA crossover noise-check
python run_cross_sectional.py        # Cross-sectional momentum backtest
python run_earnings_momentum.py      # Earnings momentum / PEAD backtest
```

Permutation noise-checks run in chunked batches (to avoid timeouts on long-running processes) and append raw results to `results/*.json`, which are committed to the repo so every reported p-value is reproducible from version-controlled data, not just from a summary.

---

## Status

Core methodology (data pipeline, signal library, backtest engine, walk-forward validation, noise-checking) is complete and tested. `docs/verdict.md` is being finalized as the last strategy's noise-check is confirmed at a stable permutation count. See `docs/verdict.md` for the full, current write-up of results.
