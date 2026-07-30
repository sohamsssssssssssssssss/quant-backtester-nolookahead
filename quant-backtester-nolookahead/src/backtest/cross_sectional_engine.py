"""
Cross-sectional backtest engine for long/short portfolio strategies.

This engine extends the single-asset BacktestEngine to handle a portfolio of tickers
with cross-sectional ranking (long top-N, short bottom-N, dollar-neutral).

Key features:
- Monthly rebalancing (standard for cross-sectional momentum)
- Dollar-neutral: long leg notional = short leg notional
- Equal weight within each leg
- Same cost/slippage model as single-asset engine (applied to both legs)
- No lookahead: ranking at month-end t uses data through t, portfolio held from t+1
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from backtest.engine import BacktestEngine


@dataclass
class CrossSectionalResult:
    """Results container for cross-sectional backtest output."""
    
    returns: pd.Series  # Portfolio daily returns after costs
    equity_curve: pd.Series  # Cumulative equity curve
    long_leg_returns: pd.Series  # Long leg daily returns
    short_leg_returns: pd.Series  # Short leg daily returns
    positions: pd.DataFrame  # Position weights per ticker per day
    rebalance_dates: pd.DatetimeIndex  # Dates when portfolio was rebalanced
    turnover: float  # Total turnover (sum of absolute position changes)
    trades: int  # Number of rebalance trades


class CrossSectionalBacktestEngine:
    """
    Cross-sectional backtest engine for long/short portfolio strategies.
    
    Parameters
    ----------
    top_n : int
        Number of top-ranked tickers to go long (default: 5)
    bottom_n : int
        Number of bottom-ranked tickers to go short (default: 5)
    cost_bps : float
        Transaction cost in basis points (default: 5 bps)
    slippage_bps : float
        Slippage in basis points (default: 2 bps)
    rebalance_freq : str
        Rebalancing frequency: 'M' for month-end (default)
    """
    
    def __init__(
        self,
        top_n: int = 5,
        bottom_n: int = 5,
        cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        rebalance_freq: str = 'M',
    ):
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.rebalance_freq = rebalance_freq
        
        # Total cost per trade in decimal
        self.total_cost_per_trade = (cost_bps + slippage_bps) / 10000.0
    
    def run(
        self,
        prices_dict: Dict[str, pd.Series],
        ranks_df: pd.DataFrame,
    ) -> CrossSectionalResult:
        """
        Run cross-sectional backtest.
        
        Parameters
        ----------
        prices_dict : Dict[str, pd.Series]
            Dictionary mapping ticker strings to price Series
        ranks_df : pd.DataFrame
            Output from rank_momentum (ranks per ticker per date)
        
        Returns
        -------
        CrossSectionalResult
            Container with portfolio returns, equity curve, positions, etc.
        """
        # Get common index across all prices
        all_dates = None
        for ticker, prices in prices_dict.items():
            if all_dates is None:
                all_dates = prices.index
            else:
                all_dates = all_dates.intersection(prices.index)
        
        if all_dates is None or len(all_dates) == 0:
            raise ValueError("No common dates across tickers")
        
        # Filter ranks to common dates
        ranks_df = ranks_df.loc[ranks_df.index.intersection(all_dates)]
        
        # Filter prices to common dates
        prices_dict = {
            ticker: prices.loc[prices.index.intersection(all_dates)]
            for ticker, prices in prices_dict.items()
        }
        
        # Get rebalance dates (month-end by default)
        rebalance_dates = self._get_rebalance_dates(all_dates)
        
        # Initialize position weights dataframe
        tickers = list(prices_dict.keys())
        n_tickers = len(tickers)
        positions_df = pd.DataFrame(0.0, index=all_dates, columns=tickers)
        
        # Build positions at each rebalance date
        for rebal_idx, rebal_date in enumerate(rebalance_dates):
            # Find rank date (the rebalance date itself)
            rank_date = rebal_date
            
            if rank_date not in ranks_df.index:
                continue
            
            # Get ranks at this date
            ranks_at_date = ranks_df.loc[rank_date]
            
            # Identify top and bottom groups
            # Top: highest ranks (top_n tickers)
            # Bottom: lowest ranks (bottom_n tickers)
            valid_ranks = ranks_at_date.dropna()
            n_valid = len(valid_ranks)
            
            if n_valid < (self.top_n + self.bottom_n):
                # Not enough valid tickers
                continue
            
            # Sort by rank descending
            sorted_ranks = valid_ranks.sort_values(ascending=False)
            
            # Top N tickers (long)
            top_tickers = sorted_ranks.head(self.top_n).index.tolist()
            
            # Bottom N tickers (short)
            bottom_tickers = sorted_ranks.tail(self.bottom_n).index.tolist()
            
            # Equal weight within each leg
            # Dollar neutral: long weight = +1/top_n, short weight = -1/bottom_n
            long_weight = 1.0 / self.top_n
            short_weight = -1.0 / self.bottom_n
            
            # Set positions from rebalance date to just before next rebalance
            start_idx = all_dates.get_loc(rebal_date)
            if rebal_idx + 1 < len(rebalance_dates):
                next_rebal_date = rebalance_dates[rebal_idx + 1]
                end_idx = all_dates.get_loc(next_rebal_date)
            else:
                end_idx = len(all_dates)
            
            # CRITICAL: Positions are set starting the NEXT day after ranking
            # (no lookahead - signal at day t is actionable from t+1)
            if start_idx + 1 < end_idx:
                for ticker in top_tickers:
                    positions_df.iloc[start_idx + 1:end_idx, positions_df.columns.get_loc(ticker)] = long_weight
                for ticker in bottom_tickers:
                    positions_df.iloc[start_idx + 1:end_idx, positions_df.columns.get_loc(ticker)] = short_weight
        
        # Compute returns
        # Portfolio return = sum of (position_weight * individual_return) for all tickers
        individual_returns = {}
        for ticker, prices in prices_dict.items():
            individual_returns[ticker] = prices.pct_change()
        
        returns_df = pd.DataFrame(individual_returns)
        
        # Gross portfolio returns
        gross_returns = (positions_df * returns_df).sum(axis=1)
        
        # Compute transaction costs
        # Costs are charged at each rebalance when positions change
        costs = self._compute_costs(positions_df, rebalance_dates)
        
        # Net returns
        net_returns = gross_returns - costs
        
        # Handle NaNs
        net_returns = net_returns.fillna(0)
        costs = costs.fillna(0)
        
        # Equity curve
        equity_curve = (1 + net_returns).cumprod()
        
        # Long and short leg returns (for analysis)
        long_mask = positions_df > 0
        short_mask = positions_df < 0
        
        long_leg_returns = (positions_df.where(long_mask, 0) * returns_df).sum(axis=1)
        short_leg_returns = (positions_df.where(short_mask, 0) * returns_df).sum(axis=1)
        
        # Turnover
        turnover = positions_df.diff().abs().sum().sum()
        
        # Count trades (rebalance events with position changes)
        position_changes = positions_df.diff().abs().sum(axis=1)
        trades = (position_changes > 0).sum()
        
        return CrossSectionalResult(
            returns=net_returns,
            equity_curve=equity_curve,
            long_leg_returns=long_leg_returns,
            short_leg_returns=short_leg_returns,
            positions=positions_df,
            rebalance_dates=pd.DatetimeIndex(rebalance_dates),
            turnover=turnover,
            trades=trades,
        )
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Get rebalance dates from the given date range.
        
        By default, uses month-end dates (last business day of each month).
        """
        # Find month-end dates: last business day of each month
        dates_df = pd.DataFrame({'date': dates})
        dates_df['year'] = dates_df['date'].dt.year
        dates_df['month'] = dates_df['date'].dt.month
        dates_df['is_month_end'] = dates_df.groupby(['year', 'month']).cumcount(ascending=False) == 0
        rebalance_mask = dates_df['is_month_end'].values
        
        # Return as DatetimeIndex using boolean mask (preserves timezone)
        return dates[rebalance_mask]
    
    def _compute_costs(
        self,
        positions_df: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Compute transaction costs.
        
        Costs are charged only at rebalance dates when positions change.
        Cost is applied to the absolute position change (turnover).
        """
        # Position turnover
        turnover = positions_df.diff().abs()
        
        # Cost only at rebalance dates
        # At each rebalance, we charge cost on all position changes
        costs = pd.Series(0.0, index=positions_df.index)
        
        for rebal_date in rebalance_dates:
            if rebal_date in turnover.index:
                # Sum of absolute position changes across all tickers
                rebal_turnover = turnover.loc[rebal_date].sum()
                costs.loc[rebal_date] = rebal_turnover * self.total_cost_per_trade
        
        return costs
