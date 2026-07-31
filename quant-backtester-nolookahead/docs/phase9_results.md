# Phase 9: Earnings Momentum (PEAD) Results

**Strategy**: Post-Earnings-Announcement Drift (PEAD)

**Long/Short**: Top 5 / Bottom 5 by standardized earnings surprise (SUE)

**Hold Period**: Rolling 60-day window (hold most recent surprise per ticker)

**Rebalance**: Daily (when new earnings available or rankings change)

**Costs**: 7 bps (5 bps transaction cost + 2 bps slippage)

---

## Pre-Registered Hypothesis

From `docs/hypotheses/phase9_earnings_momentum.md`:

> **Primary Hypothesis**: Earnings momentum (PEAD) produces positive alpha - stocks with positive earnings surprise continue to outperform in the weeks following announcement, while stocks with negative surprise continue to underperform.

**Committed before data analysis**: YES ✓

---

## Data

- **Earnings surprises**: 360 quarterly observations from yfinance
- **Date range**: 2020-07-24 to 2026-07-28 (limited by analyst estimate availability)
- **Tickers**: 15 NIFTY basket stocks
- **Average surprises per ticker**: ~24 (quarterly over 6 years)

---

## Walk-Forward Results (3y train, 1y test, 1y step)

| Fold | Test Period | CAGR | Sharpe | Max DD |
|------|-------------|------|--------|--------|
| 1 | 2022-08 to 2023-04 | -1.02% | -0.194 | 20.82% |
| 2 | 2023-04 to 2024-01 | -2.59% | -0.611 | 25.17% |
| 3 | 2024-01 to 2024-09 | -0.95% | -0.229 | 21.87% |
| 4 | 2024-09 to 2025-05 | -1.71% | -0.436 | 18.72% |
| 5 | 2025-05 to 2026-01 | +0.52% | +0.161 | 12.29% |
| **Combined** | **Full sample** | **-0.37%** | **-0.068** | **19.60%** |

**Key observation**: All test folds except the last show negative Sharpe ratios. Combined Sharpe is negative (-0.068).

---

## Noise-Check Permutation Test

**Method**: Shuffle ticker labels in earnings data (10 permutations)

**Results**:
- Actual Sharpe: 0.564 (full sample, not walk-forward)
- Permuted Sharpe mean: -0.218
- Permuted Sharpe std: 0.245
- **P-value: 0.000** (0/10 permuted sharpes exceeded actual)

**Interpretation**: The signal IS distinguishable from random noise when tested on the full sample. However, the walk-forward out-of-sample results are NEGATIVE.

---

## Interpretation

### Possible Explanations for Discrepancy

1. **Look-ahead bias in full-sample test**: The signal uses earnings data across the full period, allowing it to "know" which stocks have the best surprises in aggregate. The walk-forward test, which only uses data available at each point in time, shows this is not persistent.

2. **Regime change**: The PEAD effect may have been present historically but decayed in recent years (all recent folds are negative).

3. **Data quality**: Using yfinance analyst estimates may introduce noise - these are consensus estimates which may already be priced in by the market.

4. **Earnings sparsity**: Indian stocks don't announce earnings on the same days (unlike US). On any given day, only 1-4 companies announce. This limits cross-sectional breadth.

### Why Walk-Forward Matters

The key lesson from Phase 9 is the importance of walk-forward validation:
- Full-sample Sharpe: 0.564 (looks good!)
- Walk-forward Sharpe: -0.068 (actually bad)

Without the walk-forward test, we might have falsely concluded the strategy works.

---

## Verdict

**FAILED TO CONFIRM HYPOTHESIS**

The pre-registered hypothesis that "earnings momentum produces positive alpha" was NOT confirmed in out-of-sample walk-forward testing.

- Combined Sharpe: -0.068 (negative)
- 4 out of 5 folds have negative Sharpe
- Combined CAGR: -0.37% (losing money)

While the noise-check shows the signal is distinguishable from random noise (p=0.000), this distinction is based on full-sample backtesting. The walk-forward out-of-sample test - which is the gold standard for strategy validation - shows the strategy loses money.

**Recommendation**: Do not pursue earnings momentum (PEAD) as a standalone strategy using yfinance analyst estimates for this NIFTY basket.

---

## Files Generated

- `results/earnings_surprises.csv`: 360 earnings surprises with dates
- `results/earnings_momentum_returns.csv`: Daily returns per fold
- `results/earnings_momentum_equity.csv`: Equity curves per fold
- `results/earnings_momentum_summary.csv`: Fold-level metrics
- `results/earnings_momentum_permutations.csv`: Permutation test distribution
- `results/earnings_momentum_noise_summary.csv`: Noise-check summary

---

## Code Files

- `src/signals/earnings_momentum.py`: Signal generation logic
- `tests/test_earnings_momentum.py`: Unit tests
- `run_earnings_momentum.py`: Walk-forward backtest runner
- `run_earnings_noise_check.py`: Permutation test runner

---

**Conclusion**: Phase 9 demonstrates that a theoretically sound academic anomaly (PEAD) does not translate to profitable returns in practice for this dataset/universe, highlighting the importance of rigorous walk-forward validation before claiming any factor works.
