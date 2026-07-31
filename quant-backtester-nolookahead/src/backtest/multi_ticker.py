"""
Multi-ticker walk-forward validation.

This module extends walk-forward validation across a basket of tickers to answer:
is a strategy broken, or was a single ticker just a bad path?

Key Principles
--------------
1. NO CROSS-TICKER LEAKAGE: Each ticker is processed independently.

2. NO CHERRY-PICKING: All tickers are reported, including the ugly results.

3. EXPLICIT LOGGING: Tickers that fail to load or have insufficient data
   are explicitly logged, not silently dropped.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Any
import logging
import sys

from src.data.loader import load_ohlcv, clean_ohlcv
from src.backtest.walk_forward import run_walk_forward, aggregate_walk_forward_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Default NIFTY 50 large-cap basket (sector diversified)
DEFAULT_NIFTY_BASKET = [
    "RELIANCE.NS",      # Energy/Petrochemicals
    "TCS.NS",           # IT Services
    "HDFCBANK.NS",      # Banking
    "INFY.NS",          # IT Services
    "ICICIBANK.NS",     # Banking
    "HINDUNILVR.NS",    # FMCG
    "ITC.NS",           # FMCG/Hotels
    "SBIN.NS",          # Banking (PSU)
    "BHARTIARTL.NS",    # Telecom
    "KOTAKBANK.NS",     # Banking
    "LT.NS",            # Engineering/Construction
    "AXISBANK.NS",      # Banking
    "MARUTI.NS",        # Automobiles
    "SUNPHARMA.NS",     # Pharmaceuticals
    "TITAN.NS",         # Consumer Discretionary
]


def run_multi_ticker_walk_forward(
    tickers: List[str],
    signal_fn: Callable,
    engine_kwargs: Optional[Dict] = None,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    skip_insufficient_data: bool = True,
) -> pd.DataFrame:
    """
    Run walk-forward backtest across multiple tickers.
    
    For each ticker:
    1. Load data via load_ohlcv (Phase 1)
    2. Run run_walk_forward (Phase 4)
    3. Collect results with ticker identifier
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols in yfinance format (e.g., "RELIANCE.NS")
    signal_fn : Callable
        Signal function (e.g., moving_average_crossover)
    engine_kwargs : dict, optional
        Kwargs for BacktestEngine
    train_years : int
        Training window length in years
    test_years : int
        Test window length in years
    step_years : int
        Step size between folds in years
    skip_insufficient_data : bool
        If True, continue with remaining tickers if one fails to load
        If False, raise on first error
    
    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        - ticker, fold, train_start, train_end, test_start, test_end
        - sharpe_ratio, max_drawdown, calmar_ratio, win_rate
        - total_return, trades, test_returns, test_equity_curve
    """
    if engine_kwargs is None:
        engine_kwargs = {}
    
    all_results = []
    failed_tickers = []
    insufficient_data_tickers = []
    
    for ticker_idx, ticker in enumerate(tickers):
        logger.info(f"[{ticker_idx + 1}/{len(tickers)}] Processing {ticker}...")
        
        try:
            # Load data
            prices_df = load_ohlcv(ticker, start="2015-01-01")
            prices = prices_df["Close"]
            
            # Check minimum data requirement
            min_days_needed = int((train_years + test_years) * 252)
            if len(prices) < min_days_needed:
                msg = (
                    f"{ticker}: Insufficient data for walk-forward. "
                    f"Have {len(prices)} days, need {min_days_needed} days "
                    f"(train={train_years}y + test={test_years}y). "
                    f"Excluding from results."
                )
                logger.warning(msg)
                insufficient_data_tickers.append({
                    "ticker": ticker,
                    "reason": f"Insufficient data: {len(prices)} days (need {min_days_needed})",
                    "data_start": str(prices.index.min()),
                    "data_end": str(prices.index.max()),
                })
                continue
            
            logger.info(f"  {ticker}: {len(prices)} days ({prices.index.min().date()} to {prices.index.max().date()})")
            
            # Run walk-forward
            ticker_results = run_walk_forward(
                prices,
                signal_fn,
                engine_kwargs=engine_kwargs,
                train_years=train_years,
                test_years=test_years,
                step_years=step_years,
            )
            
            if len(ticker_results) == 0:
                logger.warning(f"  {ticker}: No folds generated (signal may be all NaN)")
                insufficient_data_tickers.append({
                    "ticker": ticker,
                    "reason": "No folds generated",
                    "data_start": str(prices.index.min()),
                    "data_end": str(prices.index.max()),
                })
                continue
            
            # Add ticker column
            ticker_results["ticker"] = ticker
            
            all_results.append(ticker_results)
            logger.info(f"  {ticker}: {len(ticker_results)} folds completed")
            
        except Exception as e:
            error_msg = f"{ticker}: Failed to process - {str(e)}"
            logger.error(error_msg)
            failed_tickers.append({
                "ticker": ticker,
                "error": str(e),
            })
            
            if not skip_insufficient_data:
                raise
    
    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MULTI-TICKER RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tickers attempted: {len(tickers)}")
    logger.info(f"Successfully processed:  {len(all_results)}")
    logger.info(f"Failed to load:          {len(failed_tickers)}")
    logger.info(f"Insufficient data:       {len(insufficient_data_tickers)}")
    
    if failed_tickers:
        logger.info("")
        logger.info("Failed tickers:")
        for ft in failed_tickers:
            logger.info(f"  - {ft['ticker']}: {ft['error']}")
    
    if insufficient_data_tickers:
        logger.info("")
        logger.info("Insufficient data tickers:")
        for it in insufficient_data_tickers:
            logger.info(f"  - {it['ticker']}: {it['reason']}")
    
    logger.info("=" * 60)
    
    if len(all_results) == 0:
        return pd.DataFrame()
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns: ticker first, then fold, then the rest
    base_cols = ["ticker", "fold", "train_start", "train_end", "test_start", "test_end",
                 "sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate",
                 "total_return", "trades"]
    
    # Ensure all base columns exist
    for col in base_cols:
        if col not in combined_results.columns:
            combined_results[col] = np.nan
    
    return combined_results[base_cols + ["test_returns", "test_equity_curve"]]


def aggregate_across_tickers(results: pd.DataFrame) -> Dict:
    """
    Aggregate multi-ticker walk-forward results.
    
    Computes:
    1. Per-ticker summary: mean Sharpe, mean max_dd, % profitable folds
    2. Cross-ticker summary: mean of means, std across tickers, % profitable overall
    3. Explicit flagging: is failure consistent or concentrated?
    
    Parameters
    ----------
    results : pd.DataFrame
        Output from run_multi_ticker_walk_forward
    
    Returns
    -------
    dict
        Aggregated results with:
        - per_ticker_summary: DataFrame with per-ticker metrics
        - cross_ticker_summary: Dict with overall metrics
        - best_ticker: str (by mean Sharpe)
        - worst_ticker: str (by mean Sharpe)
        - failure_pattern: str ('consistent' or 'concentrated')
        - all_ticker_fold_results: DataFrame (for detailed inspection)
    """
    if results.empty:
        return {
            "per_ticker_summary": pd.DataFrame(),
            "cross_ticker_summary": {},
            "best_ticker": None,
            "worst_ticker": None,
            "failure_pattern": "unknown",
            "all_ticker_fold_results": pd.DataFrame(),
        }
    
    # Per-ticker aggregation
    per_ticker_list = []
    
    for ticker in results["ticker"].unique():
        ticker_data = results[results["ticker"] == ticker]
        
        mean_sharpe = ticker_data["sharpe_ratio"].mean()
        mean_max_dd = ticker_data["max_drawdown"].mean()
        mean_calmar = ticker_data["calmar_ratio"].mean()
        mean_return = ticker_data["total_return"].mean()
        profitable_folds = (ticker_data["total_return"] > 0).sum()
        total_folds = len(ticker_data)
        pct_profitable = profitable_folds / total_folds if total_folds > 0 else np.nan
        sharpe_std = ticker_data["sharpe_ratio"].std() if len(ticker_data) > 1 else np.nan
        total_trades = ticker_data["trades"].sum()
        
        per_ticker_list.append({
            "ticker": ticker,
            "mean_sharpe": mean_sharpe,
            "mean_max_dd": mean_max_dd,
            "mean_calmar": mean_calmar,
            "mean_return": mean_return,
            "pct_profitable_folds": pct_profitable,
            "total_folds": total_folds,
            "profitable_folds": profitable_folds,
            "sharpe_std": sharpe_std,
            "total_trades": total_trades,
        })
    
    per_ticker_summary = pd.DataFrame(per_ticker_list)
    per_ticker_summary = per_ticker_summary.sort_values("mean_sharpe", ascending=False)
    
    # Cross-ticker summary
    all_folds = results.dropna(subset=["sharpe_ratio"])
    total_folds_all = len(all_folds)
    profitable_folds_all = (all_folds["total_return"] > 0).sum()
    pct_profitable_all = profitable_folds_all / total_folds_all if total_folds_all > 0 else np.nan
    
    # Mean of means
    mean_of_mean_sharpe = per_ticker_summary["mean_sharpe"].mean()
    mean_of_mean_max_dd = per_ticker_summary["mean_max_dd"].mean()
    mean_of_mean_calmar = per_ticker_summary["mean_calmar"].mean()
    mean_of_mean_return = per_ticker_summary["mean_return"].mean()
    
    # Std across tickers (measures consistency)
    std_of_mean_sharpe = per_ticker_summary["mean_sharpe"].std()
    std_of_mean_max_dd = per_ticker_summary["mean_max_dd"].std()
    
    # Best and worst performers
    best_ticker = per_ticker_summary.iloc[0]["ticker"]  # Already sorted by Sharpe desc
    worst_ticker = per_ticker_summary.iloc[-1]["ticker"]
    
    # Determine failure pattern
    # Consistent: most tickers have negative Sharpe
    # Concentrated: high variance, some good some bad
    negative_sharpe_tickers = (per_ticker_summary["mean_sharpe"] < 0).sum()
    pct_negative = negative_sharpe_tickers / len(per_ticker_summary)
    
    # High variance threshold: std > 0.5 is "variable"
    high_variance = std_of_mean_sharpe > 0.5
    
    if pct_negative > 0.7:
        failure_pattern = "consistent"
        failure_description = (
            f"FAILURE IS CONSISTENT: {pct_negative*100:.0f}% of tickers have negative mean Sharpe. "
            f"This is a strategy problem, not a single-ticker draw issue."
        )
    elif high_variance and pct_negative < 0.5:
        failure_pattern = "concentrated"
        failure_description = (
            f"FAILURE IS CONCENTRATED: High variance across tickers (std={std_of_mean_sharpe:.2f}). "
            f"Only {pct_negative*100:.0f}% of tickers negative. "
            f"Strategy may work in specific regimes/universes."
        )
    elif high_variance and pct_negative >= 0.5:
        failure_pattern = "mixed_but_mostly_bad"
        failure_description = (
            f"MIXED BUT MOSTLY BAD: High variance (std={std_of_mean_sharpe:.2f}) but "
            f"{pct_negative*100:.0f}% of tickers negative. "
            f"Strategy has regime dependency but currently unfavorable."
        )
    else:
        failure_pattern = "inconclusive"
        failure_description = (
            f"INCONCLUSIVE: {pct_negative*100:.0f}% negative, low variance. "
            f"More data needed."
        )
    
    cross_ticker_summary = {
        "total_tickers": len(per_ticker_summary),
        "total_folds_all": total_folds_all,
        "profitable_folds_all": int(profitable_folds_all),
        "pct_profitable_all": pct_profitable_all,
        "mean_of_mean_sharpe": mean_of_mean_sharpe,
        "mean_of_mean_max_dd": mean_of_mean_max_dd,
        "mean_of_mean_calmar": mean_of_mean_calmar,
        "mean_of_mean_return": mean_of_mean_return,
        "std_of_mean_sharpe": std_of_mean_sharpe,
        "std_of_mean_max_dd": std_of_mean_max_dd,
        "best_ticker": best_ticker,
        "worst_ticker": worst_ticker,
        "failure_pattern": failure_pattern,
        "failure_description": failure_description,
    }
    
    return {
        "per_ticker_summary": per_ticker_summary,
        "cross_ticker_summary": cross_ticker_summary,
        "best_ticker": best_ticker,
        "worst_ticker": worst_ticker,
        "failure_pattern": failure_pattern,
        "all_ticker_fold_results": results,
    }
