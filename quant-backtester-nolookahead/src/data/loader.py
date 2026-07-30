"""
Data loader module for OHLCV data ingestion.

This module provides functions to load and clean daily OHLCV data from yfinance,
with local caching to avoid repeated network requests.

IMPORTANT: Cache invalidation is manual. If you need fresh data, you must manually
delete the cached CSV file from data/raw/. Stale cache is the user's responsibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_ohlcv(ticker: str, start: str = "2015-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """
    Load daily OHLCV data for a given ticker.
    
    Pulls data via yfinance and caches the result to data/raw/{ticker}.csv.
    If the cache file exists, loads from disk instead of re-hitting the network.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol in yfinance format (e.g., "RELIANCE.NS" for NSE stocks)
    start : str, optional
        Start date in YYYY-MM-DD format (default: "2015-01-01")
    end : str, optional
        End date in YYYY-MM-DD format (default: None, which uses latest available)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
    
    Raises
    ------
    ValueError
        If the ticker returns no data (likely a typo or delisted security)
    
    Notes
    -----
    - Cache invalidation is MANUAL. Delete data/raw/{ticker}.csv to refresh data.
    - No automatic refresh logic is implemented.
    """
    cache_path = Path("data/raw") / f"{ticker}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if cache exists
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    
    # Fetch from yfinance
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end, interval="1d")
    
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check if the ticker symbol is correct (should be in yfinance format, "
            "e.g., 'RELIANCE.NS' for NSE stocks) or if the security is delisted."
        )
    
    # Standardize column names
    df = df.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })
    
    # Ensure correct column order and types
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]
    
    # Cache to disk
    df.to_csv(cache_path)
    
    return df


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data by handling gaps (missing trading days).
    
    This function explicitly handles gaps rather than silently dropping them:
    - Short gaps (<=2 consecutive missing days): Forward-filled
    - Long gaps (>2 consecutive missing days): Flagged and dropped
    
    The choice to forward-fill short gaps is based on the assumption that
    short gaps are typically exchange holidays or data vendor issues where
    prices didn't materially change. Long gaps may indicate delistings,
    suspensions, or data corruption and are removed to avoid spurious signals.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex
    
    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame with gaps handled
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Identify gaps by computing day differences
    date_diff = df.index.to_series().diff()
    
    # Business day is typically 1 day (calendar days between trading days)
    # A gap of >3 calendar days means >2 business days missing
    gap_days = date_diff.dt.days
    
    # Mark rows after gaps > 2 days (i.e., gap_days > 3 means 3+ business days missing)
    # We use >3 because weekend is 2-3 days, so 3 days is normal Fri->Mon
    long_gap_mask = gap_days > 3
    
    if long_gap_mask.any():
        # Get indices of rows following long gaps
        long_gap_indices = df.index[long_gap_mask]
        print(
            f"Warning: Detected {long_gap_mask.sum()} long gap(s) (>2 business days). "
            f"Rows after long gaps will be dropped. Gap dates: {long_gap_indices.tolist()}"
        )
        # Drop rows that follow long gaps
        df = df[~long_gap_mask]
    
    # Forward-fill short gaps (remaining NaNs after dropping long gaps)
    df = df.ffill()
    
    # Drop any remaining NaNs at the very beginning (can't fill those)
    df = df.dropna()
    
    return df
