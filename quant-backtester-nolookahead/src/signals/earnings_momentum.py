"""
Earnings momentum signal (Post-Earnings-Announcement Drift).

This signal ranks stocks by their earnings surprise at each announcement date
and generates a long/short portfolio betting on continued drift.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_earnings_surprises() -> pd.DataFrame:
    """
    Load earnings surprise data from CSV.
    
    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, estimate, actual, surprise
    """
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent / 'results' / 'earnings_surprises.csv',
        Path(__file__).parent.parent / 'results' / 'earnings_surprises.csv',
        Path('/workspace/quant-backtester-nolookahead/results/earnings_surprises.csv'),
    ]
    
    for csv_path in possible_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['date'])
            df = df.sort_values(['ticker', 'date'])
            return df
    
    raise FileNotFoundError(
        f"earnings_surprises.csv not found. Searched: {possible_paths}"
    )


def rank_earnings_surprises(earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank stocks by earnings surprise at each announcement date.
    
    For each unique date, ranks stocks by their surprise value.
    Higher surprise = higher rank.
    
    Parameters
    ----------
    earnings_df : pd.DataFrame
        Earnings data with columns: ticker, date, surprise
    
    Returns
    -------
    pd.DataFrame
        Ranked data with columns: date, ticker, surprise, rank
    """
    ranked = earnings_df.copy()
    
    # Rank at each date (higher surprise = higher rank)
    ranked['rank'] = ranked.groupby('date')['surprise'].rank(method='average', ascending=True)
    
    return ranked[['date', 'ticker', 'surprise', 'rank']]


def get_earnings_portfolio(
    earnings_df: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct long/short portfolio from earnings surprises.
    
    Long: top_n stocks with highest surprise
    Short: bottom_n stocks with lowest surprise
    
    Parameters
    ----------
    earnings_df : pd.DataFrame
        Ranked earnings data
    top_n : int
        Number of stocks to go long
    bottom_n : int
        Number of stocks to go short
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Long portfolio weights and short portfolio weights
        Each has columns: date, ticker, weight
    """
    ranked = rank_earnings_surprises(earnings_df)
    
    long_positions = []
    short_positions = []
    
    for date in ranked['date'].unique():
        date_data = ranked[ranked['date'] == date].copy()
        
        if len(date_data) < (top_n + bottom_n):
            continue
        
        # Sort by rank descending (highest surprise first)
        date_data = date_data.sort_values('rank', ascending=False)
        
        # Long: top N
        long_tickers = date_data.head(top_n)
        long_tickers = long_tickers[['date', 'ticker']].copy()
        long_tickers['weight'] = 1.0 / top_n
        long_positions.append(long_tickers)
        
        # Short: bottom N
        short_tickers = date_data.tail(bottom_n)
        short_tickers = short_tickers[['date', 'ticker']].copy()
        short_tickers['weight'] = -1.0 / bottom_n
        short_positions.append(short_tickers)
    
    long_df = pd.concat(long_positions, ignore_index=True) if long_positions else pd.DataFrame()
    short_df = pd.concat(short_positions, ignore_index=True) if short_positions else pd.DataFrame()
    
    return long_df, short_df


def earnings_momentum_signal(
    prices_dict: dict[str, pd.Series],
    earnings_df: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
    hold_days: int = 21,
) -> pd.DataFrame:
    """
    Generate earnings momentum signal for backtest.
    
    Uses a rolling window approach: at each date, considers the most recent
    earnings surprise for each ticker (within the past hold_days) and forms
    a long/short portfolio based on cross-sectional ranking.
    
    Parameters
    ----------
    prices_dict : dict[str, pd.Series]
        Price series for each ticker
    earnings_df : pd.DataFrame
        Earnings surprise data with columns: ticker, date, surprise
    top_n : int
        Number of stocks to long
    bottom_n : int
        Number of stocks to short
    hold_days : int
        Lookback window for considering recent earnings announcements
    
    Returns
    -------
    pd.DataFrame
        Signal DataFrame (dates x tickers) with position weights
    """
    tickers = list(prices_dict.keys())
    
    # Get common dates
    all_dates = None
    for prices in prices_dict.values():
        if all_dates is None:
            all_dates = set(prices.index)
        else:
            all_dates = all_dates.intersection(set(prices.index))
    
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    
    # Initialize signal DataFrame
    signal_df = pd.DataFrame(0.0, index=all_dates, columns=tickers)
    
    # Filter earnings data to only include tickers in prices_dict
    earnings_filtered = earnings_df[earnings_df['ticker'].isin(tickers)].copy()
    
    # Determine timezone from signal index
    signal_tz = all_dates.tz
    
    # Ensure earnings dates are timezone-aware
    if earnings_filtered['date'].dt.tz is None:
        earnings_filtered['date'] = earnings_filtered['date'].dt.tz_localize(signal_tz)
    
    # For each date, find the most recent earnings surprise for each ticker
    # and use it to rank tickers cross-sectionally
    for i, signal_date in enumerate(all_dates):
        # Look back up to hold_days to find recent earnings
        lookback_start = signal_date - pd.Timedelta(days=hold_days * 2)
        
        # Get announcements in the lookback window
        recent_earnings = earnings_filtered[
            (earnings_filtered['date'] >= lookback_start) & 
            (earnings_filtered['date'] <= signal_date)
        ].copy()
        
        if len(recent_earnings) == 0:
            continue
        
        # Get most recent surprise per ticker
        # Sort by date descending and take first occurrence per ticker
        recent_earnings = recent_earnings.sort_values('date', ascending=False)
        latest_surprises = recent_earnings.groupby('ticker').first().reset_index()
        
        # Need at least top_n + bottom_n valid surprises to form portfolio
        if len(latest_surprises) < (top_n + bottom_n):
            continue
        
        # Rank by surprise (highest surprise = highest rank)
        latest_surprises['rank'] = latest_surprises['surprise'].rank(method='average', ascending=True)
        
        # Sort by rank descending
        latest_surprises = latest_surprises.sort_values('rank', ascending=False)
        
        # Long: top N
        long_tickers = latest_surprises.head(top_n)
        
        # Short: bottom N
        short_tickers = latest_surprises.tail(bottom_n)
        
        # Set positions
        long_weight = 1.0 / top_n
        short_weight = -1.0 / bottom_n
        
        for _, row in long_tickers.iterrows():
            ticker = row['ticker']
            if ticker in signal_df.columns:
                signal_df.loc[signal_date, ticker] = long_weight
        
        for _, row in short_tickers.iterrows():
            ticker = row['ticker']
            if ticker in signal_df.columns:
                signal_df.loc[signal_date, ticker] = short_weight
    
    return signal_df
