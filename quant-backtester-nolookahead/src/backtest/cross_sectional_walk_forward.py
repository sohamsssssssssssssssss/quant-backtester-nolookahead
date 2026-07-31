"""
Walk-forward validation for cross-sectional portfolio strategies.

Adapts the single-asset walk_forward module to portfolio-level strategies.
Same 3yr train / 1yr test / 1yr step structure, but applied to the whole
15-ticker portfolio at once since this is inherently a cross-sectional strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
import logging

from src.backtest.cross_sectional_engine import CrossSectionalBacktestEngine, CrossSectionalResult
from src.signals.cross_sectional_momentum import rank_momentum
from src.metrics.performance import sharpe_ratio, max_drawdown, calmar_ratio, summary

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CrossSectionalFoldResult:
    """Results for a single walk-forward fold."""
    
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_return: float
    turnover: float
    trades: int
    returns: pd.Series
    equity_curve: pd.Series


def run_cross_sectional_walk_forward(
    prices_dict: Dict[str, pd.Series],
    top_n: int = 5,
    bottom_n: int = 5,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> pd.DataFrame:
    """
    Run walk-forward backtest for cross-sectional momentum strategy.
    
    Parameters
    ----------
    prices_dict : Dict[str, pd.Series]
        Dictionary mapping ticker strings to price Series
    top_n : int
        Number of top-ranked tickers to go long
    bottom_n : int
        Number of bottom-ranked tickers to go short
    cost_bps : float
        Transaction cost in basis points
    slippage_bps : float
        Slippage in basis points
    train_years : int
        Training window length (used for momentum calculation lookback)
    test_years : int
        Test window length
    step_years : int
        Step size between folds
    
    Returns
    -------
    pd.DataFrame
        One row per fold with fold metrics
    """
    # Generate fold windows based on the common date range
    # Get common index
    all_dates = None
    for ticker, prices in prices_dict.items():
        if all_dates is None:
            all_dates = prices.index
        else:
            all_dates = all_dates.intersection(prices.index)
    
    if all_dates is None or len(all_dates) == 0:
        raise ValueError("No common dates across tickers")
    
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    folds = _generate_folds(all_dates, train_years, test_years, step_years)
    
    if len(folds) == 0:
        logger.warning("No folds could be generated with the given parameters")
        return pd.DataFrame()
    
    results = []
    
    for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
        logger.info(f"Processing fold {fold_idx}: {train_start.date()} to {test_end.date()}")
        
        # Find test_start
        train_end_idx = all_dates.get_loc(train_end)
        if train_end_idx + 1 >= len(all_dates):
            continue
        test_start = all_dates[train_end_idx + 1]
        
        # Filter prices to fold window (including train for momentum warm-up)
        fold_mask = (all_dates >= train_start) & (all_dates <= test_end)
        fold_dates = all_dates[fold_mask]
        
        prices_fold = {
            ticker: prices.loc[prices.index.intersection(fold_dates)]
            for ticker, prices in prices_dict.items()
        }
        
        # Compute cross-sectional momentum ranks
        ranks_df = rank_momentum(prices_fold)
        
        # Run backtest
        engine = CrossSectionalBacktestEngine(
            top_n=top_n,
            bottom_n=bottom_n,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )
        
        try:
            backtest_result = engine.run(prices_fold, ranks_df)
        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")
            continue
        
        if len(backtest_result.returns) == 0 or backtest_result.equity_curve.iloc[0] == 0:
            logger.warning(f"Fold {fold_idx} produced empty or invalid results")
            continue
        
        # Extract test period results
        test_mask = (backtest_result.returns.index >= test_start) & (
            backtest_result.returns.index <= test_end
        )
        test_returns = backtest_result.returns.loc[test_mask]
        test_equity = backtest_result.equity_curve.loc[test_mask]
        
        if len(test_returns) == 0:
            continue
        
        # Compute metrics
        fold_metrics = summary(test_returns, test_equity)
        
        results.append({
            "fold": fold_idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "sharpe_ratio": fold_metrics.get("sharpe_ratio", np.nan),
            "max_drawdown": fold_metrics.get("max_drawdown", np.nan),
            "calmar_ratio": fold_metrics.get("calmar_ratio", np.nan),
            "total_return": fold_metrics.get("total_return", np.nan),
            "turnover": backtest_result.turnover,
            "trades": backtest_result.trades,
            "test_returns": test_returns,
            "test_equity_curve": test_equity,
        })
    
    return pd.DataFrame(results)


def _generate_folds(
    dates: pd.DatetimeIndex,
    train_years: int,
    test_years: int,
    step_years: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate rolling walk-forward windows.
    
    Returns list of (train_start, train_end, test_end) tuples.
    """
    if len(dates) == 0:
        return []
    
    train_days = int(train_years * 252)
    test_days = int(test_years * 252)
    step_days = int(step_years * 252)
    
    min_data_needed = train_days + test_days
    
    if len(dates) < min_data_needed:
        raise ValueError(
            f"Insufficient data for walk-forward split. "
            f"Need at least {min_data_needed} days, have {len(dates)}."
        )
    
    folds = []
    current_train_start = 0
    current_train_end = train_days
    current_test_end = train_days + test_days
    
    while current_test_end <= len(dates):
        train_start = dates[current_train_start]
        train_end = dates[min(current_train_end, len(dates) - 1)]
        test_end = dates[min(current_test_end - 1, len(dates) - 1)]
        
        folds.append((train_start, train_end, test_end))
        
        current_train_start += step_days
        current_train_end += step_days
        current_test_end += step_days
        
        if current_train_start >= current_train_end:
            break
    
    return folds


def aggregate_cross_sectional_results(
    results: pd.DataFrame,
) -> dict:
    """
    Aggregate cross-sectional walk-forward results.
    
    Combines all fold test-period returns into one continuous series.
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
        }
    
    combined_returns = pd.concat(all_returns_list)
    combined_returns = combined_returns.sort_index()
    
    # Build combined equity curve
    combined_equity = []
    current_equity = 1.0
    
    for _, row in results.iterrows():
        if isinstance(row["test_equity_curve"], pd.Series) and len(row["test_equity_curve"]) > 0:
            fold_equity = row["test_equity_curve"]
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
    
    profitable_folds = (results["total_return"] > 0).sum()
    total_folds = len(results)
    
    return {
        "combined_equity_curve": combined_equity_curve,
        "combined_returns": combined_returns,
        "overall_sharpe": overall_sharpe,
        "overall_max_drawdown": overall_mdd,
        "overall_calmar": overall_calmar,
        "overall_total_return": overall_total_return,
        "profitable_folds": int(profitable_folds),
        "total_folds": int(total_folds),
    }
