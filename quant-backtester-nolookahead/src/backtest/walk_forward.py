"""
Walk-forward validation for systematic strategies.

This module implements rolling train/test validation to ensure strategy
performance is not curve-fit to a single historical period.

Key Principles
--------------
1. ZERO LEAKAGE ACROSS FOLDS: A fold's test period must never influence
   signal calculation for earlier folds.

2. REPORT EVERY FOLD: Individual fold results are reported, not just averages.
   Variance across folds is a feature, not a bug - it reveals regime sensitivity.

3. NO PARAMETER RE-OPTIMIZATION: This phase uses fixed parameters across all
   folds. Per-fold optimization would defeat the purpose of out-of-sample testing.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from datetime import timedelta

from src.backtest.engine import BacktestEngine, BacktestResult
from src.metrics.performance import (
    sharpe_ratio, sortino_ratio, max_drawdown, 
    calmar_ratio, win_rate, summary
)


def walk_forward_split(
    index: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate rolling walk-forward windows.
    
    For a given date range, generates a sequence of (train_start, train_end, test_end)
    tuples defining non-overlapping train/test splits that roll forward in time.
    
    The train period is used for signal computation (the signal function sees
    data from train_start up to each point in time). The test period is where
    performance is evaluated out-of-sample.
    
    IMPORTANT: Test periods do not overlap with future train periods to prevent
    any lookahead leakage across folds.
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        The full date range of available data
    train_years : int
        Length of training window in years (default: 3)
    test_years : int
        Length of test window in years (default: 1)
    step_years : int
        How far to roll forward between folds (default: 1)
    
    Returns
    -------
    List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]
        List of (train_start, train_end, test_end) tuples for each fold
    
    Example
    -------
    For data spanning 2010-2020 with train_years=3, test_years=1, step_years=1:
    - Fold 1: train=2010-2012, test=2013
    - Fold 2: train=2011-2013, test=2014
    - Fold 3: train=2012-2014, test=2015
    - etc.
    """
    if len(index) == 0:
        return []
    
    index = pd.DatetimeIndex(index).sort_values()
    full_start = index.min()
    full_end = index.max()
    
    train_days = int(train_years * 252)
    test_days = int(test_years * 252)
    step_days = int(step_years * 252)
    
    # Minimum data needed for first fold
    min_data_needed = train_days + test_days
    
    if len(index) < min_data_needed:
        raise ValueError(
            f"Insufficient data for walk-forward split. "
            f"Need at least {min_data_needed} days (train={train_days} + test={test_days}), "
            f"but only have {len(index)} days."
        )
    
    folds = []
    
    # First fold starts at the beginning
    current_train_start = 0
    current_train_end = train_days
    current_test_end = train_days + test_days
    
    while current_test_end <= len(index):
        train_start = index[current_train_start]
        train_end = index[min(current_train_end, len(index) - 1)]
        test_end = index[min(current_test_end - 1, len(index) - 1)]
        
        folds.append((train_start, train_end, test_end))
        
        # Roll forward
        current_train_start += step_days
        current_train_end += step_days
        current_test_end += step_days
        
        # Safety: ensure we don't go backwards
        if current_train_start >= current_train_end:
            break
    
    return folds


def run_walk_forward(
    prices: pd.Series,
    signal_fn: Callable[[pd.Series], pd.Series],
    engine_kwargs: Optional[Dict] = None,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> pd.DataFrame:
    """
    Run walk-forward backtest across all folds.
    
    For each fold:
    1. Extract the train+test window from prices
    2. Compute signal using signal_fn on the window (signal sees data up to each point)
    3. Run BacktestEngine on the test period only
    4. Collect performance metrics
    
    IMPORTANT: The signal function is always called on data available up to that point.
    It does not see future prices beyond the current fold's test_end.
    
    Parameters
    ----------
    prices : pd.Series
        Full price series
    signal_fn : Callable
        Signal function (e.g., moving_average_crossover from src/signals/signals.py)
        Must accept a pd.Series of prices and return a pd.Series of signals
    engine_kwargs : dict, optional
        Additional kwargs for BacktestEngine (e.g., cost_bps, slippage_bps)
    train_years : int
        Training window length in years
    test_years : int
        Test window length in years
    step_years : int
        Step size between folds in years
    
    Returns
    -------
    pd.DataFrame
        One row per fold with columns:
        - fold: fold number (0-indexed)
        - train_start, train_end, test_start, test_end: dates
        - sharpe_ratio, max_drawdown, calmar_ratio, win_rate: metrics
        - total_return: cumulative return over test period
        - trades: number of trades
        - test_returns: raw returns series (object column)
        - test_equity_curve: equity curve series (object column)
    """
    if engine_kwargs is None:
        engine_kwargs = {}
    
    # Generate fold windows
    folds = walk_forward_split(
        prices.index,
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
    )
    
    results = []
    
    for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
        # Find the test_start (day after train_end)
        # Get the index position of train_end
        train_end_idx = prices.index.get_loc(train_end)
        if train_end_idx + 1 >= len(prices):
            continue
        test_start = prices.index[train_end_idx + 1]
        
        # Extract the data window for this fold
        # Signal function will see data from train_start onwards
        # But we only evaluate performance on test_start to test_end
        fold_mask = (prices.index >= train_start) & (prices.index <= test_end)
        fold_prices = prices.loc[fold_mask].copy()
        
        if len(fold_prices) == 0:
            continue
        
        # Compute signal on the fold data
        # The signal function only sees prices within this fold window
        signal = signal_fn(fold_prices)
        
        if signal.isna().all() or signal.count() == 0:
            # Signal is all NaN - skip this fold
            results.append({
                "fold": fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "sharpe_ratio": np.nan,
                "max_drawdown": np.nan,
                "calmar_ratio": np.nan,
                "win_rate": np.nan,
                "total_return": np.nan,
                "trades": 0,
                "test_returns": pd.Series(dtype=float),
                "test_equity_curve": pd.Series(dtype=float),
            })
            continue
        
        # Run backtest on the test period only
        # But we need to include train data for signal warm-up
        engine = BacktestEngine(**engine_kwargs)
        backtest_result = engine.run(fold_prices, signal)
        
        # Extract test period results
        test_mask = (backtest_result.returns.index >= test_start) & (
            backtest_result.returns.index <= test_end
        )
        test_returns = backtest_result.returns.loc[test_mask]
        test_equity = backtest_result.equity_curve.loc[test_mask]
        test_positions = backtest_result.positions.loc[test_mask]
        
        if len(test_returns) == 0 or test_equity.iloc[0] == 0:
            continue
        
        # Compute metrics for this fold
        fold_metrics = summary(test_returns, test_equity, test_positions)
        
        results.append({
            "fold": fold_idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "sharpe_ratio": fold_metrics.get("sharpe_ratio", np.nan),
            "max_drawdown": fold_metrics.get("max_drawdown", np.nan),
            "calmar_ratio": fold_metrics.get("calmar_ratio", np.nan),
            "win_rate": fold_metrics.get("win_rate", np.nan),
            "total_return": fold_metrics.get("total_return", np.nan),
            "trades": backtest_result.trades,
            "test_returns": test_returns,
            "test_equity_curve": test_equity,
        })
    
    return pd.DataFrame(results)


def aggregate_walk_forward_results(
    results: pd.DataFrame,
) -> Dict:
    """
    Aggregate walk-forward results into a single out-of-sample equity curve.
    
    Combines all fold test-period returns into one continuous series, then
    computes overall metrics. Also reports fold-level consistency metrics.
    
    Parameters
    ----------
    results : pd.DataFrame
        Output from run_walk_forward()
    
    Returns
    -------
    dict
        Aggregated results with:
        - combined_equity_curve: pd.Series of continuous out-of-sample equity
        - combined_returns: pd.Series of all test-period returns concatenated
        - overall_sharpe: Sharpe ratio on combined returns
        - overall_max_drawdown: Max drawdown on combined equity
        - overall_calmar: Calmar ratio on combined series
        - overall_total_return: Total return across all folds
        - profitable_folds: Number of folds with positive return
        - total_folds: Total number of folds
        - fold_sharpe_std: Standard deviation of per-fold Sharpe ratios
        - fold_returns: List of per-fold total returns
        - fold_metrics: DataFrame with per-fold metrics
    """
    if results.empty:
        return {
            "combined_equity_curve": pd.Series(dtype=float),
            "combined_returns": pd.Series(dtype=float),
            "overall_sharpe": np.nan,
            "overall_max_drawdown": np.nan,
            "overall_calmar": np.nan,
            "overall_total_return": np.nan,
            "profitable_folds": 0,
            "total_folds": 0,
            "fold_sharpe_std": np.nan,
            "fold_returns": [],
            "fold_metrics": pd.DataFrame(),
        }
    
    # Concatenate all test-period returns
    all_returns_list = []
    all_equity_list = []
    
    for _, row in results.iterrows():
        if isinstance(row["test_returns"], pd.Series) and len(row["test_returns"]) > 0:
            all_returns_list.append(row["test_returns"])
        if isinstance(row["test_equity_curve"], pd.Series) and len(row["test_equity_curve"]) > 0:
            all_equity_list.append(row["test_equity_curve"])
    
    if len(all_returns_list) == 0:
        return {
            "combined_equity_curve": pd.Series(dtype=float),
            "combined_returns": pd.Series(dtype=float),
            "overall_sharpe": np.nan,
            "overall_max_drawdown": np.nan,
            "overall_calmar": np.nan,
            "overall_total_return": np.nan,
            "profitable_folds": 0,
            "total_folds": 0,
            "fold_sharpe_std": np.nan,
            "fold_returns": [],
            "fold_metrics": pd.DataFrame(),
        }
    
    # Concatenate returns (they're already non-overlapping by construction)
    combined_returns = pd.concat(all_returns_list)
    combined_returns = combined_returns.sort_index()
    
    # Build combined equity curve
    # Start each fold's equity from where the previous one ended
    combined_equity = []
    current_equity = 1.0
    
    for _, row in results.iterrows():
        if isinstance(row["test_equity_curve"], pd.Series) and len(row["test_equity_curve"]) > 0:
            fold_equity = row["test_equity_curve"]
            # Normalize to start from current_equity
            if fold_equity.iloc[0] != 0:
                scale = current_equity / fold_equity.iloc[0]
            else:
                scale = current_equity
            scaled_equity = fold_equity * scale
            combined_equity.append(scaled_equity)
            current_equity = scaled_equity.iloc[-1]
    
    if len(combined_equity) > 0:
        combined_equity_curve = pd.concat(combined_equity)
        combined_equity_curve = combined_equity_curve.sort_index()
    else:
        combined_equity_curve = pd.Series(dtype=float)
    
    # Compute overall metrics
    overall_sharpe = sharpe_ratio(combined_returns)
    overall_mdd = max_drawdown(combined_equity_curve) if len(combined_equity_curve) > 0 else np.nan
    overall_calmar = calmar_ratio(combined_returns, combined_equity_curve) if len(combined_equity_curve) > 0 else np.nan
    overall_total_return = combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0] - 1 if len(combined_equity_curve) > 0 and combined_equity_curve.iloc[0] != 0 else np.nan
    
    # Fold-level consistency metrics
    valid_sharpes = results["sharpe_ratio"].dropna()
    fold_sharpe_std = valid_sharpes.std() if len(valid_sharpes) > 1 else np.nan
    
    profitable_folds = (results["total_return"] > 0).sum()
    total_folds = len(results)
    
    fold_returns = results["total_return"].tolist()
    
    # Extract per-fold metrics for summary
    fold_metrics = results[["fold", "train_start", "train_end", "test_start", "test_end",
                            "sharpe_ratio", "max_drawdown", "calmar_ratio", 
                            "win_rate", "total_return", "trades"]].copy()
    
    return {
        "combined_equity_curve": combined_equity_curve,
        "combined_returns": combined_returns,
        "overall_sharpe": overall_sharpe,
        "overall_max_drawdown": overall_mdd,
        "overall_calmar": overall_calmar,
        "overall_total_return": overall_total_return,
        "profitable_folds": int(profitable_folds),
        "total_folds": int(total_folds),
        "fold_sharpe_std": float(fold_sharpe_std) if not np.isnan(fold_sharpe_std) else np.nan,
        "fold_returns": fold_returns,
        "fold_metrics": fold_metrics,
    }
