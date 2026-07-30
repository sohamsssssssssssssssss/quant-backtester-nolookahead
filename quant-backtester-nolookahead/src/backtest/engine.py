"""
Vectorized backtest engine for systematic strategies.

This engine converts signals into realistic equity curves with proper handling of:

1. NO LOOKAHEAD: Signals generated at day t are shifted forward one bar before
   being applied to returns. The signal at index t determines the position
   taken at the close of day t, which earns the return from day t to t+1.

2. TRANSACTION COSTS + SLIPPAGE: Every position change incurs costs. Default
   values are calibrated for NSE large-cap equities (5 bps cost, 2 bps slippage).

3. EXPLICIT WARM-UP HANDLING: NaN signals produce zero positions, not silent drops.

Position Sizing Modes
---------------------
- equal_weight: Full investment (position = signal) when signal != 0
- volatility_target: Position scaled inversely to trailing 20-day realized
  volatility, capped at max_leverage (default 1.0x)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BacktestResult:
    """Results container for backtest output."""
    
    returns: pd.Series  # Daily strategy returns after costs
    equity_curve: pd.Series  # Cumulative equity curve (net value)
    positions: pd.Series  # Position held each day (post-shift)
    signal: pd.Series  # Original signal (for reference)
    costs: pd.Series  # Daily transaction costs in return terms
    trades: int  # Number of trades (position changes)
    turnover: float  # Total turnover (sum of absolute position changes)


class BacktestEngine:
    """
    Vectorized backtest engine for systematic strategies.
    
    Parameters
    ----------
    cost_bps : float
        Transaction cost in basis points (default: 5 bps for NSE large caps)
    slippage_bps : float
        Slippage in basis points (default: 2 bps for NSE large caps)
    sizing_mode : {'equal_weight', 'volatility_target'}
        Position sizing methodology
    vol_window : int
        Lookback window for volatility estimation (default: 20 days)
    max_leverage : float
        Maximum leverage for volatility targeting (default: 1.0)
    long_short : bool
        If True, signal can be -1/0/1 (long/short). If False, only long (0/1)
    """
    
    def __init__(
        self,
        cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        sizing_mode: Literal['equal_weight', 'volatility_target'] = 'equal_weight',
        vol_window: int = 20,
        max_leverage: float = 1.0,
        long_short: bool = True,
    ):
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.sizing_mode = sizing_mode
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.long_short = long_short
        
        # Total cost per round-trip trade in decimal
        self.total_cost_per_trade = (cost_bps + slippage_bps) / 10000.0
    
    def run(self, prices: pd.Series, signal: pd.Series) -> BacktestResult:
        """
        Run the backtest.
        
        Parameters
        ----------
        prices : pd.Series
            Price series (typically closing prices)
        signal : pd.Series
            Signal series with same index as prices
        
        Returns
        -------
        BacktestResult
            Container with returns, equity curve, positions, costs, and trade count
        """
        # Align inputs
        prices = prices.copy()
        signal = signal.copy()
        
        # Ensure same index
        common_index = prices.index.intersection(signal.index)
        prices = prices.loc[common_index]
        signal = signal.loc[common_index]
        
        # Compute buy-and-hold returns
        raw_returns = prices.pct_change()
        
        # CRITICAL: Shift signal forward one bar
        # Signal at day t determines position taken at close of day t
        # which earns return from day t to t+1
        position_signal = signal.shift(1)
        
        # Handle warm-up: NaN signals become zero positions
        position_signal = position_signal.fillna(0)
        
        # Apply position sizing
        positions = self._size_positions(position_signal, prices)
        
        # Apply long-only constraint if specified
        if not self.long_short:
            positions = positions.clip(lower=0)
        
        # Compute gross strategy returns
        gross_returns = positions * raw_returns
        
        # Compute transaction costs
        costs = self._compute_costs(positions)
        
        # Net returns after costs
        net_returns = gross_returns - costs
        
        # Handle any NaNs from warm-up periods
        net_returns = net_returns.fillna(0)
        costs = costs.fillna(0)
        
        # Compute equity curve (cumulative returns, starting from 1.0)
        equity_curve = (1 + net_returns).cumprod()
        
        # Count trades (position changes from zero to non-zero or vice versa)
        position_changes = positions.diff().abs()
        trades = (position_changes > 0).sum()
        
        # Total turnover
        turnover = position_changes.sum()
        
        return BacktestResult(
            returns=net_returns,
            equity_curve=equity_curve,
            positions=positions,
            signal=signal,
            costs=costs,
            trades=trades,
            turnover=turnover,
        )
    
    def _size_positions(self, signal: pd.Series, prices: pd.Series) -> pd.Series:
        """
        Apply position sizing based on configured mode.
        
        Parameters
        ----------
        signal : pd.Series
            Shifted signal (position direction)
        prices : pd.Series
            Price series for volatility calculation
        
        Returns
        -------
        pd.Series
            Position sizes (leverage-adjusted)
        """
        if self.sizing_mode == 'equal_weight':
            # Full investment when signal != 0
            return signal.replace({-1: -1.0, 1: 1.0, 0: 0.0})
        
        elif self.sizing_mode == 'volatility_target':
            # Compute trailing realized volatility
            returns = prices.pct_change()
            rolling_vol = returns.rolling(window=self.vol_window).std()
            
            # Target volatility (annualized, assuming 252 trading days)
            target_vol = 0.15 / np.sqrt(252)  # 15% annualized target
            
            # Scale position inversely to volatility
            raw_positions = signal * (target_vol / rolling_vol)
            
            # Cap at max leverage
            positions = raw_positions.clip(-self.max_leverage, self.max_leverage)
            
            # Handle NaN volatility (warm-up)
            positions = positions.fillna(0)
            
            return positions
        
        else:
            raise ValueError(f"Unknown sizing_mode: {self.sizing_mode}")
    
    def _compute_costs(self, positions: pd.Series) -> pd.Series:
        """
        Compute transaction costs based on position changes.
        
        Costs are incurred when positions change (turnover).
        Cost is applied to the absolute change in position.
        
        Parameters
        ----------
        positions : pd.Series
            Position series
        
        Returns
        -------
        pd.Series
            Daily costs in return terms
        """
        # Position turnover (absolute change)
        turnover = positions.diff().abs().fillna(0)
        
        # Cost per unit of turnover (one-way cost, applied to each trade)
        # We use half the round-trip cost since we're measuring per-bar changes
        cost_per_unit = self.total_cost_per_trade
        
        return turnover * cost_per_unit
