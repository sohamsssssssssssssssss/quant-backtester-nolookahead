"""
Performance metrics for evaluating backtest results.

All metrics assume 252 trading days per year for annualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Sharpe Ratio = (mean excess return) / (std of excess returns) * sqrt(252)
    
    Assumes 252 trading days per year for annualization.
    
    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns
    risk_free_rate : float
        Annual risk-free rate (default: 0.0)
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    
    # Annualize
    return sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sortino ratio.
    
    Sortino Ratio = (mean excess return) / (downside deviation) * sqrt(252)
    
    Uses only negative returns for computing deviation (downside risk).
    Assumes 252 trading days per year for annualization.
    
    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns
    risk_free_rate : float
        Annual risk-free rate (default: 0.0)
    
    Returns
    -------
    float
        Annualized Sortino ratio
    """
    if returns.empty:
        return 0.0
    
    excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    
    mean_excess = excess_returns.mean()
    
    # Downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        # No negative returns - infinite sortino, but return a large number
        return np.inf
    
    downside_std = negative_returns.std()
    
    if downside_std == 0:
        return 0.0
    
    sortino = mean_excess / downside_std
    
    # Annualize
    return sortino * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown from equity curve.
    
    Maximum drawdown is the largest peak-to-trough decline,
    expressed as a positive fraction (e.g., 0.20 = 20% drawdown).
    
    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve (net asset value)
    
    Returns
    -------
    float
        Maximum drawdown as a positive fraction
    """
    if equity_curve.empty:
        return 0.0
    
    # Running maximum
    running_max = equity_curve.cummax()
    
    # Drawdown at each point
    drawdown = (running_max - equity_curve) / running_max
    
    # Handle division by zero
    drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)
    
    return float(drawdown.max())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    
    Calmar Ratio = (annualized return) / (max drawdown)
    
    Assumes 252 trading days per year for annualization.
    
    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns
    equity_curve : pd.Series
        Cumulative equity curve
    
    Returns
    -------
    float
        Calmar ratio (annualized return / max drawdown)
    """
    if equity_curve.empty or returns.empty:
        return 0.0
    
    # Annualized return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if equity_curve.iloc[0] != 0 else 0
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    
    if n_years <= 0:
        return 0.0
    
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Max drawdown
    mdd = max_drawdown(equity_curve)
    
    if mdd == 0 or mdd == 0.0:
        return 0.0
    
    return annualized_return / mdd


def win_rate(returns: pd.Series) -> float:
    """
    Compute win rate (proportion of positive returns).
    
    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns
    
    Returns
    -------
    float
        Win rate (0 to 1)
    """
    if returns.empty:
        return 0.0
    
    positive_returns = (returns > 0).sum()
    total_returns = len(returns)
    
    return positive_returns / total_returns if total_returns > 0 else 0.0


def turnover(positions: pd.Series) -> float:
    """
    Compute total turnover (sum of absolute position changes).
    
    Parameters
    ----------
    positions : pd.Series
        Position series
    
    Returns
    -------
    float
        Total turnover
    """
    if positions.empty:
        return 0.0
    
    return float(positions.diff().abs().fillna(0).sum())


def summary(returns: pd.Series, equity_curve: pd.Series, positions: Optional[pd.Series] = None) -> Dict:
    """
    Compute summary of all performance metrics.
    
    Assumes 252 trading days per year for all annualization.
    
    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns
    equity_curve : pd.Series
        Cumulative equity curve
    positions : pd.Series, optional
        Position series (for turnover calculation)
    
    Returns
    -------
    dict
        Dictionary with all metrics:
        - sharpe_ratio: Annualized Sharpe ratio
        - sortino_ratio: Annualized Sortino ratio
        - max_drawdown: Maximum drawdown (positive fraction)
        - calmar_ratio: Calmar ratio
        - win_rate: Proportion of positive returns
        - total_return: Total cumulative return
        - annualized_return: Annualized return
        - turnover: Total turnover (if positions provided)
    """
    result = {
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(returns, equity_curve),
        "win_rate": win_rate(returns),
        "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 0 and equity_curve.iloc[0] != 0 else 0.0,
    }
    
    # Annualized return
    if len(returns) > 0:
        n_years = len(returns) / TRADING_DAYS_PER_YEAR
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if equity_curve.iloc[0] != 0 else 0
        if n_years > 0:
            result["annualized_return"] = (1 + total_return) ** (1 / n_years) - 1
        else:
            result["annualized_return"] = 0.0
    else:
        result["annualized_return"] = 0.0
    
    if positions is not None:
        result["turnover"] = turnover(positions)
    
    return result
