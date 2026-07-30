"""
Signal generation module for systematic strategies.

This module provides pure signal generation functions. Signals are computed
as-of each day's close using only data available up to and including that day.

IMPORTANT: This module does NOT handle execution timing. A signal computed at
day t is actionable from day t+1 onward. The backtest engine (Phase 3) is
responsible for shifting signals forward one bar before computing returns.
This separation ensures no lookahead bias and keeps concerns modular.

Signal Convention
-----------------
- 1: Long signal
- -1: Short signal  
- 0: Neutral / no position / insufficient data
"""

import pandas as pd
import numpy as np
from typing import Optional


def moving_average_crossover(
    prices: pd.Series, fast: int = 50, slow: int = 200
) -> pd.Series:
    """
    Moving average crossover signal.
    
    Returns 1 when fast MA is above slow MA (bullish),
    -1 when fast MA is below slow MA (bearish),
    0 during warm-up period when slow MA doesn't have enough data.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically closing prices)
    fast : int
        Fast moving average window (default: 50)
    slow : int
        Slow moving average window (default: 200)
    
    Returns
    -------
    pd.Series
        Signal series with values {-1, 0, 1}
    """
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()
    
    # Initialize signal with zeros
    signal = pd.Series(0, index=prices.index, dtype=int)
    
    # Long when fast MA > slow MA
    signal[fast_ma > slow_ma] = 1
    
    # Short when fast MA < slow MA
    signal[fast_ma < slow_ma] = -1
    
    # Zero during warm-up (when slow_ma is NaN)
    signal[slow_ma.isna()] = 0
    
    return signal


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing method.
    
    Uses exponential weighted moving average with alpha=1/period for
    smoothing gains and losses, as per Wilder's original methodology.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically closing prices)
    period : int
        RSI lookback period (default: 14)
    
    Returns
    -------
    pd.Series
        RSI values in range [0, 100]
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Wilder's smoothing: EWMA with alpha = 1/period
    alpha = 1.0 / period
    
    avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate RS and RSI
    # Avoid division by zero
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_values = 100 - (100 / (1 + rs))
    
    # Handle edge case where losses are zero (RSI = 100)
    rsi_values[avg_losses == 0] = 100
    
    return rsi_values


def rsi_mean_reversion(
    prices: pd.Series, period: int = 14, oversold: float = 30, overbought: float = 70
) -> pd.Series:
    """
    RSI mean-reversion signal.
    
    Returns 1 when RSI is below oversold threshold (buy signal),
    -1 when RSI is above overbought threshold (sell signal),
    0 otherwise.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically closing prices)
    period : int
        RSI lookback period (default: 14)
    oversold : float
        RSI threshold for oversold condition (default: 30)
    overbought : float
        RSI threshold for overbought condition (default: 70)
    
    Returns
    -------
    pd.Series
        Signal series with values {-1, 0, 1}
    """
    rsi_values = rsi(prices, period)
    
    # Initialize signal with zeros
    signal = pd.Series(0, index=prices.index, dtype=int)
    
    # Long when RSI < oversold
    signal[rsi_values < oversold] = 1
    
    # Short when RSI > overbought
    signal[rsi_values > overbought] = -1
    
    return signal


def momentum_12_1(prices: pd.Series) -> pd.Series:
    """
    Jegadeesh-Titman 12-1 month momentum signal.
    
    Computes the cumulative return over the trailing 12 months,
    skipping the most recent month to avoid short-term reversal
    contamination.
    
    For daily data:
    - 12 months ≈ 252 trading days
    - 1 month ≈ 21 trading days
    - Total lookback: 252 + 21 = 273 days
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically closing prices)
    
    Returns
    -------
    pd.Series
        Momentum signal (12-month return skipping last month)
        NaN until sufficient history exists
    """
    skip_days = 21  # ~1 month
    lookback_days = 252 + skip_days  # 12 months + 1 month skip
    
    # Compute returns: (price_t / price_t-273) - 1
    # Shift prices to get the price 273 days ago
    prices_lagged = prices.shift(skip_days)
    prices_lagged = prices_lagged.shift(252)
    
    momentum = (prices / prices_lagged) - 1
    
    return momentum
