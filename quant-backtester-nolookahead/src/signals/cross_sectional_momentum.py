"""
Cross-sectional momentum signal generation.

This module ranks tickers by their 12-1 momentum at each point in time,
producing a cross-sectional ranking suitable for long/short portfolio construction.

The momentum_12_1 logic is reused from src/signals/signals.py (Jegadeesh-Titman 12-1 month).
"""

import pandas as pd
import numpy as np
from typing import Dict

from src.signals.signals import momentum_12_1


def rank_momentum(
    prices_dict: Dict[str, pd.Series],
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.DataFrame:
    """
    Rank all tickers by momentum_12_1 at each date.
    
    For each date, computes momentum_12_1 for each ticker, then ranks them
    cross-sectionally (1 = lowest momentum, N = highest momentum where N is
    the number of tickers with valid momentum on that date).
    
    Parameters
    ----------
    prices_dict : Dict[str, pd.Series]
        Dictionary mapping ticker strings to price Series
    lookback_days : int
        Lookback days for momentum (default: 252 ~ 12 months)
    skip_days : int
        Skip days to avoid short-term reversal (default: 21 ~ 1 month)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index, tickers as columns,
        values = cross-sectional rank of momentum (1 = lowest, N = highest)
        NaN where momentum is undefined (insufficient history)
    """
    # Compute momentum for each ticker
    momentum_dict = {}
    
    for ticker, prices in prices_dict.items():
        # Use the existing momentum_12_1 function
        momentum = momentum_12_1(prices)
        momentum_dict[ticker] = momentum
    
    # Combine into single dataframe
    momentum_df = pd.DataFrame(momentum_dict)
    
    # Compute cross-sectional ranks at each date
    # Rank from 1 (lowest momentum) to N (highest momentum)
    ranks_df = momentum_df.rank(axis=1, method='average', na_option='keep')
    
    return ranks_df


def get_top_bottom_groups(
    ranks_df: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
) -> tuple:
    """
    Identify top and bottom momentum tickers at each date.
    
    Parameters
    ----------
    ranks_df : pd.DataFrame
        Output from rank_momentum (ranks per ticker per date)
    top_n : int
        Number of top-ranked tickers to select (default: 5)
    bottom_n : int
        Number of bottom-ranked tickers to select (default: 5)
    
    Returns
    -------
    tuple
        (top_dates_df, bottom_dates_df)
        Each is a DataFrame with boolean values: True if ticker is in top/bottom group
    """
    n_tickers = len(ranks_df.columns)
    
    # Top group: tickers with rank >= (n_tickers - top_n + 1)
    # Bottom group: tickers with rank <= bottom_n
    top_threshold = n_tickers - top_n + 0.5  # Use 0.5 to handle ties cleanly
    bottom_threshold = bottom_n + 0.5
    
    top_group = ranks_df.apply(
        lambda row: row >= top_threshold, axis=1, result_type='broadcast'
    )
    bottom_group = ranks_df.apply(
        lambda row: row <= bottom_threshold, axis=1, result_type='broadcast'
    )
    
    # Handle NaN values (should remain False)
    top_group = top_group.fillna(False)
    bottom_group = bottom_group.fillna(False)
    
    return top_group, bottom_group
