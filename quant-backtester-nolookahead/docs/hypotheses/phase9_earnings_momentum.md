# Phase 9 Pre-Registration: Earnings Momentum (Post-Earnings-Announcement Drift)

## Hypothesis

**Post-Earnings-Announcement Drift (PEAD)**: Stocks with positive earnings surprises tend to continue outperforming in the weeks following the earnings announcement, while stocks with negative earnings surprises tend to continue underperforming.

**Economic Rationale**: Investor underreaction to earnings news. When companies report earnings that differ from market expectations, prices do not fully incorporate the information immediately. Instead, adjustment occurs gradually over subsequent weeks as investors process the news and revise their beliefs. This behavioral explanation (anchoring, conservatism bias) is well-documented in academic literature.

**Prediction**: A strategy that goes long stocks with large positive earnings surprises and short stocks with large negative earnings surprises will earn positive risk-adjusted returns after transaction costs, on a liquid Indian equity universe.

## Universe

**Primary**: 15-ticker NIFTY 50 large-cap basket (same as Phases 5-8):
- RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, SBIN.NS, BHARTIARTL.NS, KOTAKBANK.NS, LT.NS, AXISBANK.NS, MARUTI.NS, SUNPHARMA.NS, TITAN.NS

**Expansion (if data available)**: NIFTY Midcap 50 constituents (up to 35 additional tickers, subject to data availability and liquidity screening).

## Date Range

**Start**: 2015-01-01 (earnings announcements from this date forward)
**End**: Latest available data (expected ~2026-03, subject to yfinance/NSE data availability)

**Minimum data requirement**: At least 20 quarterly earnings announcements per ticker over the sample period for reliable signal construction.

## Earnings Surprise Definition

**Primary measure**: Standardized Unexpected Earnings (SUE)
```
SUE = (EPS_actual - EPS_estimate) / |EPS_estimate|
```

**Fallback (if analyst estimates unavailable)**: Year-over-Year EPS growth surprise
```
Surprise = (EPS_current_quarter - EPS_same_quarter_prior_year) / |EPS_same_quarter_prior_year|
```

**Decision rule**: Use analyst estimates if available from yfinance for ≥10 tickers; otherwise use YoY surprise proxy for all tickers uniformly.

## Signal Construction

**Ranking**: At each earnings announcement date, rank all stocks by their earnings surprise magnitude (signed, not absolute).

**Portfolio**: Long top quintile (or top 5 if universe < 25 stocks), short bottom quintile (or bottom 5). Equal weight within each leg. Dollar-neutral construction.

**Holding period**: 21 trading days (1 month) post-announcement, then rebalance at next announcement cohort.

**Rebalancing**: Monthly, aligned with earnings announcement clusters (most Indian companies report within 2-3 weeks of quarter end).

## Backtest Parameters

| Parameter | Value |
|-----------|-------|
| Cost (bps) | 5 |
| Slippage (bps) | 2 |
| Walk-forward | 3y train, 1y test, 1y step |
| Lookahead protection | Signal at day T uses only earnings announced on or before day T |
| Warm-up period | 4 quarters (to establish YoY baseline) |

## Success Criteria

**Statistical significance**: Noise-check p-value < 0.05 (two-tailed, permutation-based null test with n=500 permutations minimum).

**Economic significance**: 
- Post-cost Sharpe ratio > 0.5 on the primary 15-ticker universe
- Maximum drawdown < 40%
- Positive returns in ≥60% of walk-forward folds

**Interpretation**:
- If p < 0.05 AND Sharpe > 0.5: Evidence consistent with PEAD anomaly in Indian equities
- If p ≥ 0.05 OR Sharpe ≤ 0.5: No reliable edge detectable after costs on this universe

## Pre-Registration Timestamp

**File created**: [To be filled by git commit timestamp]
**Backtest run**: [To be filled after pre-registration commit]

---

## Notes

1. **Data quality caveat**: Earnings data from free sources (yfinance) may have gaps, restatements, or timezone inconsistencies. Will document any data quality issues openly rather than silently filling gaps.

2. **Estimate availability**: Analyst EPS estimates may not be available for all tickers in the universe, especially for earlier years (2015-2018). The YoY fallback ensures coverage but changes interpretation from "vs. market expectations" to "vs. company's own history."

3. **Announcement timing**: Indian companies typically announce earnings after market close or before market open. Treating all announcements as "known at open of next trading day" for conservative no-lookahead enforcement.

4. **No multiple testing**: This is a single pre-registered hypothesis. Will not try alternative surprise definitions, holding periods, or portfolio constructions and report only the best result.
