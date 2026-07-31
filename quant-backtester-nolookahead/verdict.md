# Verdict: Systematic Factor Testing Results (Phases 5-9)

## Overview

This document summarizes 5 distinct systematic trading factors tested with strict walk-forward validation and no-lookahead controls.

---

## Summary Table

| Phase | Strategy | Combined Sharpe | Combined CAGR | Max DD | Verdict |
|-------|----------|----------------|---------------|--------|---------|
| 5 | Single-Asset MA Crossover | 0.15 - 0.30 | 2-5% | 15-25% | ❌ Not robust |
| 6 | Multi-Ticker MA Crossover | 0.00 - 0.15 | ~0% | 20-30% | ❌ Failed |
| 7 | Cross-Sectional Momentum | -0.05 - 0.20 | 0-3% | 15-25% | ⚠️ Mixed |
| 8 | Sector Rotation | -0.10 - 0.15 | -2-2% | 20-35% | ❌ Failed |
| 9 | **Earnings Momentum (PEAD)** | **-0.07** | **-0.4%** | **20%** | **❌ Failed** |

---

## Phase 9: Earnings Momentum (PEAD) — Detailed Results

**Status**: COMPLETE ❌

**Pre-registered hypothesis**: Earnings momentum produces positive alpha — stocks with positive earnings surprise continue to outperform in the weeks following announcement.

**Committed before analysis**: YES ✓

### Strategy Construction

- **Signal**: Standardized unexpected earnings (SUE) = (Actual - Estimate) / |Estimate|
- **Portfolio**: Long top-5 / Short bottom-5 by SUE
- **Hold period**: Rolling 60-day window (hold most recent surprise)
- **Rebalance**: Daily
- **Costs**: 7 bps round-trip

### Data

- **Source**: yfinance analyst EPS estimates
- **Observations**: 360 quarterly surprises (2020-07 to 2026-07)
- **Universe**: 15 NIFTY basket stocks
- **Coverage**: ~24 observations per ticker

### Walk-Forward Results

| Fold | Test Period | CAGR | Sharpe |
|------|-------------|------|--------|
| 1 | 2022-08 to 2023-04 | -1.02% | -0.194 |
| 2 | 2023-04 to 2024-01 | -2.59% | -0.611 |
| 3 | 2024-01 to 2024-09 | -0.95% | -0.229 |
| 4 | 2024-09 to 2025-05 | -1.71% | -0.436 |
| 5 | 2025-05 to 2026-01 | +0.52% | +0.161 |
| **Combined** | **All periods** | **-0.37%** | **-0.068** |

### Noise-Check Permutation Test

- **Actual Sharpe**: 0.564 (full-sample, in-sample!)
- **Permuted Sharpe**: mean=-0.218, std=0.245
- **P-value**: 0.000 (0/10)
- **Interpretation**: Statistically significant full-sample, but **does not persist out-of-sample**

### Why This Matters

The PEAD anomaly is one of the most robust findings in academic finance — yet it fails here. Key lessons:

1. **Full-sample vs Walk-forward divergence**: Full-sample Sharpe (+0.564) vs walk-forward (-0.068) shows the strategy overfits. Without walk-forward validation, we'd have a false positive.

2. **Data quality matters**: Using yfinance analyst estimates (consensus) vs proprietary databases may introduce noise. The market may already price in consensus surprises.

3. **Cross-sectional sparsity**: Indian stocks don't announce earnings on synchronized dates. Only 1-4 stocks announce per day, limiting true cross-sectional breadth.

4. **Academic anomalies decay**: PEAD was discovered in the 1980s. By 2020s, the anomaly may have been arbitraged away, especially in liquid large-cap indices like NIFTY.

### Verdict

**FAILED TO CONFIRM HYPOTHESIS**

The pre-registered hypothesis that earnings momentum produces positive alpha was NOT confirmed in out-of-sample walk-forward testing:
- 4 of 5 folds negative Sharpe
- Combined walk-forward Sharpe: -0.068
- Combined walk-forward CAGR: -0.37%

---

## Overall Verdict (Phases 5-9)

**None of the 5 tested factors produced robust, persistent alpha in walk-forward out-of-sample testing.**

### Pattern Observed

- **In-sample**: Most strategies show reasonable or positive Sharpe
- **Out-of-sample (walk-forward)**: Performance degrades, often to negative
- **Noise-check**: Many pass permutation tests, suggesting signal is "real" but not persistent

### Interpretation

This pattern suggests:
1. **Market efficiency**: The NIFTY 50 constituent stocks are efficiently priced — simple factors don't consistently extract alpha
2. **Factor decay**: Academic anomalies discovered decades ago may not persist in modern markets
3. **Importance of OOS testing**: Without strict walk-forward validation, factors appear to work in-sample but fail in practice
4. **Data quality**: Free/public data (yfinance) may not capture the edge that proprietary datasets provide

### Recommendations

For future research:
1. **Try newer/alternative factors**: Quality, low-vol, ESG, short-term reversal
2. **Improve data quality**: Use proprietary fundamental data, alternative data
3. **Explore shorter horizons**: Daily/weekly mean reversion may still work
4. **Consider transaction cost optimization**: Smart execution, limit orders
5. **Multi-factor combinations**: Combining weak factors may produce robust signals

---

## Files

- `docs/hypotheses/`: Pre-registered hypotheses for each phase
- `docs/phase5_results.md` through `docs/phase9_results.md`: Complete results per phase
- `results/`: All backtest outputs, metrics, and permutation tests

**Last updated**: 2026-01-31
