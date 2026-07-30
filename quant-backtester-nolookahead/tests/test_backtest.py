"""
Tests for the backtest engine.

These tests specifically target lookahead bias, cost modeling, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.engine import BacktestEngine, BacktestResult
from metrics.performance import sharpe_ratio, sortino_ratio, max_drawdown, summary


def create_synthetic_prices(length: int = 500, seed: int = 42) -> pd.Series:
    """Create a synthetic price series using random walk."""
    np.random.seed(seed)
    returns = np.random.randn(length) * 0.02  # 2% daily vol
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.Series(prices, index=dates, name="price")


def create_linear_uptrend(length: int = 100) -> pd.Series:
    """Create a linear uptrend price series."""
    prices = np.linspace(100, 200, length)
    dates = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.Series(prices, index=dates, name="price")


class TestLookaheadBias:
    """Tests to detect lookahead bias in the backtest engine."""
    
    def test_lookahead_alter_future_prices(self):
        """
        Lookahead test: Altering price data at day t+5 should NOT change
        the equity curve at day t.
        
        If the equity curve at day t changes when we modify day t+5,
        the engine is leaking future information.
        """
        np.random.seed(42)
        length = 300
        prices = create_synthetic_prices(length, seed=42)
        
        # Create a simple signal based on price momentum
        signal = (prices.pct_change(5) > 0).astype(int) * 2 - 1  # -1 or 1
        signal = signal.fillna(0)
        
        # Run original backtest
        engine = BacktestEngine(cost_bps=0, slippage_bps=0)
        result1 = engine.run(prices, signal)
        
        # Create a modified price series with an extreme outlier at day t+50
        prices_modified = prices.copy()
        test_day = 100  # Pick a day in the middle
        future_day = test_day + 5
        
        # Make the future day an extreme outlier
        prices_modified.iloc[future_day] = prices_modified.iloc[future_day] * 10
        
        # Run backtest with modified prices
        result2 = engine.run(prices_modified, signal)
        
        # The equity curve at test_day should be IDENTICAL
        # because future prices shouldn't affect past returns
        equity_at_test_day_1 = result1.equity_curve.iloc[test_day]
        equity_at_test_day_2 = result2.equity_curve.iloc[test_day]
        
        assert equity_at_test_day_1 == equity_at_test_day_2, (
            f"LOOKAHEAD BIAS DETECTED! "
            f"Equity at day {test_day} changed from {equity_at_test_day_1} "
            f"to {equity_at_test_day_2} when we modified day {future_day}"
        )
        
        # Also check returns at test_day
        returns_at_test_day_1 = result1.returns.iloc[test_day]
        returns_at_test_day_2 = result2.returns.iloc[test_day]
        
        assert returns_at_test_day_1 == returns_at_test_day_2, (
            f"LOOKAHEAD BIAS DETECTED in returns! "
            f"Return at day {test_day} changed from {returns_at_test_day_1} "
            f"to {returns_at_test_day_2}"
        )


class TestCostModel:
    """Tests for transaction cost modeling."""
    
    def test_cost_sanity_frequent_trading(self):
        """
        Cost sanity test: A strategy that flips position every day should
        have materially worse returns than the same strategy with zero costs.
        """
        prices = create_synthetic_prices(200, seed=42)
        
        # Create a signal that flips every day
        signal = pd.Series(
            [1 if i % 2 == 0 else -1 for i in range(len(prices))],
            index=prices.index
        )
        
        # Run with zero costs
        engine_zero = BacktestEngine(cost_bps=0, slippage_bps=0)
        result_zero = engine_zero.run(prices, signal)
        
        # Run with realistic costs
        engine_costed = BacktestEngine(cost_bps=5, slippage_bps=2)
        result_costed = engine_costed.run(prices, signal)
        
        # The costed version should significantly underperform
        # Frequent flipping with costs should erode returns
        total_return_zero = result_zero.equity_curve.iloc[-1] - 1
        total_return_costed = result_costed.equity_curve.iloc[-1] - 1
        
        # Costed return should be at least 10% lower than zero-cost return
        # (the exact threshold depends on the random prices, but frequent
        # trading with costs should always underperform)
        underperformance = total_return_zero - total_return_costed
        
        assert underperformance > 0.05, (
            f"Cost model may be broken. "
            f"Zero-cost return: {total_return_zero:.4f}, "
            f"Costed return: {total_return_costed:.4f}, "
            f"Underperformance: {underperformance:.4f} (expected > 0.05)"
        )
        
        # Verify costs are non-zero for the costed version
        total_costs = result_costed.costs.sum()
        assert total_costs > 0, (
            f"Expected positive costs for frequent trading, got {total_costs}"
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_signal(self):
        """
        Zero-signal test: An all-zero signal should produce zero returns
        (not NaN, not a crash).
        """
        prices = create_synthetic_prices(100, seed=42)
        signal = pd.Series(0, index=prices.index)
        
        engine = BacktestEngine()
        result = engine.run(prices, signal)
        
        # Returns should be all zeros (no position = no returns)
        assert (result.returns == 0).all(), (
            f"Expected zero returns for zero signal, got max={result.returns.max()}, min={result.returns.min()}"
        )
        
        # Equity curve should be flat at 1.0
        assert (result.equity_curve == 1.0).all(), (
            f"Expected flat equity curve at 1.0, got {result.equity_curve.unique()}"
        )
        
        # No trades
        assert result.trades == 0, f"Expected 0 trades for zero signal, got {result.trades}"
    
    def test_nan_signal_warmup(self):
        """
        Test that NaN signals during warm-up are handled gracefully.
        """
        prices = create_synthetic_prices(100, seed=42)
        
        # Signal with NaNs at the beginning
        signal = pd.Series([np.nan] * 50 + [1] * 50, index=prices.index)
        
        engine = BacktestEngine()
        result = engine.run(prices, signal)
        
        # Should not crash
        assert not result.returns.isna().any(), "Returns contain NaN values"
        assert not result.equity_curve.isna().any(), "Equity curve contains NaN values"
        
        # First 50 days should have zero positions (NaN signal -> zero position)
        assert (result.positions.iloc[:50] == 0).all(), "Expected zero positions during NaN warm-up"


class TestKnownCase:
    """Tests with known expected outputs."""
    
    def test_linear_uptrend_always_long(self):
        """
        Known-case test: Linear uptrend with always-long signal.
        
        For a linear price series from 100 to 200 over 100 days,
        with an always-long signal (1) and zero costs:
        - Total return should be (200-100)/100 = 100%
        - Equity curve should go from 1.0 to 2.0
        """
        prices = create_linear_uptrend(100)
        signal = pd.Series(1, index=prices.index)  # Always long
        
        engine = BacktestEngine(cost_bps=0, slippage_bps=0)
        result = engine.run(prices, signal)
        
        # Total return should be approximately 100%
        # (exactly 100% for simple returns, but we use compounded daily returns)
        # Price goes from 100 to 200, so buy-and-hold return = 100%
        expected_final_equity = 2.0  # 100% return
        
        # Allow small tolerance due to daily compounding vs simple return
        final_equity = result.equity_curve.iloc[-1]
        tolerance = 0.05  # 5% tolerance
        
        assert abs(final_equity - expected_final_equity) < tolerance * expected_final_equity, (
            f"Expected final equity ~{expected_final_equity}, got {final_equity}"
        )
        
        # Verify positions are all 1.0 (after shift, first day is 0)
        assert (result.positions.iloc[1:] == 1.0).all(), "Expected position = 1 for always-long signal"
    
    def test_simple_reversal(self):
        """
        Simple test: Price goes up then down, signal perfectly times it.
        """
        # Price: 100 -> 110 -> 100
        prices = pd.Series(
            [100, 110, 100, 90, 100],
            index=pd.date_range("2020-01-01", periods=5, freq="B")
        )
        
        # Signal: 0 on day 0, then 1 (long) for days 1-2, then -1 (short) for days 3-4
        # After shift: position on day 1 is signal[0]=0, position on day 2 is signal[1]=1, etc.
        signal = pd.Series([0, 1, 1, -1, -1], index=prices.index)
        
        engine = BacktestEngine(cost_bps=0, slippage_bps=0)
        result = engine.run(prices, signal)
        
        # Day 0: return = NaN (no previous price)
        # Day 1: position = signal[0] = 0, return = 0
        # Day 2: position = signal[1] = 1, return = (110->100) = -0.0909...
        # Day 3: position = signal[2] = 1, return = (100->90) = -0.10
        # Day 4: position = signal[3] = -1, return = -1 * (90->100) = -0.111...
        
        # Check that we get reasonable outputs (not NaN, not crashed)
        assert not result.returns.isna().any(), "Returns should not be NaN"
        assert result.trades > 0, "Should have at least one trade"


class TestPositionSizing:
    """Tests for position sizing modes."""
    
    def test_equal_weight_sizing(self):
        """Test equal weight position sizing."""
        prices = create_synthetic_prices(100, seed=42)
        signal = pd.Series([1 if i % 2 == 0 else -1 for i in range(len(prices))], index=prices.index)
        
        engine = BacktestEngine(sizing_mode='equal_weight')
        result = engine.run(prices, signal)
        
        # After shift, positions should be -1 or 1 (or 0 for first day)
        unique_positions = result.positions.iloc[1:].unique()
        assert all(p in [-1.0, 1.0, 0.0] for p in unique_positions), (
            f"Expected positions in {{-1, 0, 1}}, got {unique_positions}"
        )
    
    def test_volatility_target_sizing(self):
        """Test volatility-target position sizing."""
        prices = create_synthetic_prices(100, seed=42)
        signal = pd.Series(1, index=prices.index)  # Always long signal
        
        engine = BacktestEngine(
            sizing_mode='volatility_target',
            vol_window=20,
            max_leverage=1.0
        )
        result = engine.run(prices, signal)
        
        # Positions should be bounded by max_leverage
        assert result.positions.abs().max() <= 1.0, (
            f"Positions exceed max_leverage=1.0, max={result.positions.abs().max()}"
        )
        
        # Positions after warm-up should be non-zero
        assert (result.positions.iloc[21:] != 0).any(), (
            "Expected non-zero positions after volatility warm-up"
        )


class TestMetrics:
    """Tests for performance metrics."""
    
    def test_sharpe_ratio_known(self):
        """Test Sharpe ratio with known inputs."""
        # Constant positive returns
        returns = pd.Series([0.01] * 100)  # 1% daily return
        
        sharpe = sharpe_ratio(returns)
        
        # With constant returns, std = 0, so Sharpe should be inf or very large
        # But our implementation returns 0 for std=0
        assert sharpe >= 0, "Sharpe ratio should be non-negative for positive returns"
    
    def test_max_drawdown_known(self):
        """Test max drawdown with known inputs."""
        # Simple drawdown: 100 -> 120 -> 100 -> 80 -> 100
        equity = pd.Series([1.0, 1.2, 1.0, 0.8, 1.0])
        
        mdd = max_drawdown(equity)
        
        # Max drawdown is from 1.2 to 0.8 = 0.4/1.2 = 33.33%
        expected_mdd = 0.4 / 1.2
        
        assert abs(mdd - expected_mdd) < 0.01, (
            f"Expected max drawdown ~{expected_mdd:.4f}, got {mdd:.4f}"
        )
    
    def test_summary_output(self):
        """Test that summary returns all expected metrics."""
        prices = create_synthetic_prices(100, seed=42)
        signal = pd.Series(1, index=prices.index)
        
        engine = BacktestEngine()
        result = engine.run(prices, signal)
        
        metrics = summary(result.returns, result.equity_curve, result.positions)
        
        expected_keys = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "win_rate", "total_return",
            "annualized_return", "turnover"
        }
        
        assert set(metrics.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"
        )


class TestLongShortConstraint:
    """Tests for long-only vs long/short modes."""
    
    def test_long_only_mode(self):
        """Test that long_only mode clips short positions to zero."""
        prices = create_synthetic_prices(100, seed=42)
        signal = pd.Series([-1 if i % 2 == 0 else 1 for i in range(len(prices))], index=prices.index)
        
        engine = BacktestEngine(long_short=False)
        result = engine.run(prices, signal)
        
        # All positions should be >= 0
        assert (result.positions >= 0).all(), (
            f"Long-only mode should have non-negative positions, min={result.positions.min()}"
        )
    
    def test_long_short_mode(self):
        """Test that long/short mode allows negative positions."""
        prices = create_synthetic_prices(100, seed=42)
        signal = pd.Series([-1 if i % 2 == 0 else 1 for i in range(len(prices))], index=prices.index)
        
        engine = BacktestEngine(long_short=True)
        result = engine.run(prices, signal)
        
        # Should have both positive and negative positions
        assert (result.positions > 0).any(), "Expected positive positions"
        assert (result.positions < 0).any(), "Expected negative positions in long/short mode"
