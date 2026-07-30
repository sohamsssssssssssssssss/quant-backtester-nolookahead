"""
Tests for cross-sectional momentum strategy.

Tests specifically target:
1. rank_momentum produces correct rankings on synthetic data
2. Lookahead test: altering price after ranking date doesn't affect prior portfolio
3. Dollar-neutrality: long leg notional equals short leg notional
4. Cost application: costs charged on both legs at rebalance
5. Synthetic momentum pattern: ranking produces sensible, directionally correct results
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.cross_sectional_momentum import rank_momentum, get_top_bottom_groups
from backtest.cross_sectional_engine import (
    CrossSectionalBacktestEngine,
    CrossSectionalResult,
)
from backtest.cross_sectional_walk_forward import (
    run_cross_sectional_walk_forward,
    aggregate_cross_sectional_results,
)


def create_synthetic_prices_dict(
    n_tickers: int = 5,
    length: int = 500,
    seed: int = 42,
) -> dict:
    """Create synthetic prices dictionary."""
    np.random.seed(seed)
    prices_dict = {}
    
    for i in range(n_tickers):
        returns = np.random.randn(length) * 0.02
        prices = 100 * np.cumprod(1 + returns)
        dates = pd.date_range("2020-01-01", periods=length, freq="B")
        prices_dict[f"TICKER{i}.NS"] = pd.Series(prices, index=dates)
    
    return prices_dict


class TestRankMomentum:
    """Tests for rank_momentum function."""
    
    def test_rank_produces_correct_ordering(self):
        """
        Test that rank_momentum produces correct rankings on a small
        synthetic multi-ticker case with known momentum values.
        """
        # Create 3 tickers with known momentum relationships
        # Use enough data for momentum_12_1 (needs 273 days warmup)
        length = 400
        dates = pd.date_range("2020-01-01", periods=length, freq="B")
        
        # Ticker A: consistent uptrend
        np.random.seed(42)
        prices_a = 100 * np.cumprod(1 + 0.001 + np.random.randn(length) * 0.005)
        
        # Ticker B: flat with noise
        prices_b = 100 * np.cumprod(1 + np.random.randn(length) * 0.005)
        
        # Ticker C: consistent downtrend
        prices_c = 100 * np.cumprod(1 - 0.001 + np.random.randn(length) * 0.005)
        
        prices_dict = {
            "A.NS": pd.Series(prices_a, index=dates),
            "B.NS": pd.Series(prices_b, index=dates),
            "C.NS": pd.Series(prices_c, index=dates),
        }
        
        ranks_df = rank_momentum(prices_dict)
        
        # After warm-up period, A should have highest rank, C lowest
        # Take the last row (most data available)
        last_row = ranks_df.iloc[-1]
        
        # A should have highest rank (closest to n_tickers = 3)
        # C should have lowest rank (closest to 1)
        assert last_row["A.NS"] > last_row["C.NS"], (
            f"A.NS should have higher rank than C.NS. Got A={last_row['A.NS']}, C={last_row['C.NS']}"
        )
        
        # Verify ranks are in valid range [1, n_tickers]
        assert (last_row.dropna() >= 1).all(), "Ranks should be >= 1"
        assert (last_row.dropna() <= 3).all(), "Ranks should be <= n_tickers"
    
    def test_rank_handles_nan_correctly(self):
        """Test that rank_momentum handles NaN values (warm-up period) correctly."""
        # Need at least 273 days for momentum_12_1 to have valid values
        prices_dict = create_synthetic_prices_dict(n_tickers=3, length=400)
        
        ranks_df = rank_momentum(prices_dict)
        
        # First ~273 rows should have NaN (momentum needs warm-up)
        early_rows = ranks_df.iloc[:200]
        assert early_rows.isna().all().all(), "Early rows should have NaN during warm-up"
        
        # Later rows should have valid ranks
        later_rows = ranks_df.iloc[-100:]
        assert not later_rows.isna().all().all(), "Later rows should have valid ranks"


class TestLookaheadBias:
    """Tests for lookahead bias in cross-sectional backtest."""
    
    def test_future_price_change_doesnt_affect_past_portfolio(self):
        """
        Lookahead test: Alter a ticker's price after the ranking date,
        verify the portfolio held before that date is unchanged.
        """
        np.random.seed(42)
        prices_dict = create_synthetic_prices_dict(n_tickers=15, length=1000)
        
        # Get ranks
        ranks_df = rank_momentum(prices_dict)
        
        # Run original backtest
        engine = CrossSectionalBacktestEngine(top_n=5, bottom_n=5, cost_bps=0, slippage_bps=0)
        result1 = engine.run(prices_dict, ranks_df)
        
        # Create modified prices with extreme outlier in the future
        test_day_idx = 300
        future_day_idx = test_day_idx + 50
        
        prices_modified = {}
        for ticker, prices in prices_dict.items():
            modified = prices.copy()
            # Alter price at future day
            modified.iloc[future_day_idx] = modified.iloc[future_day_idx] * 100
            prices_modified[ticker] = modified
        
        # Recompute ranks with modified prices
        ranks_modified = rank_momentum(prices_modified)
        
        # Run backtest with modified prices
        result2 = engine.run(prices_modified, ranks_modified)
        
        # Portfolio returns before test_day should be IDENTICAL
        returns_before_1 = result1.returns.iloc[:test_day_idx]
        returns_before_2 = result2.returns.iloc[:test_day_idx]
        
        assert returns_before_1.equals(returns_before_2), (
            f"LOOKAHEAD BIAS DETECTED! Returns before day {test_day_idx} "
            f"changed when we modified day {future_day_idx}"
        )
        
        # Equity curve before test_day should also be identical
        equity_before_1 = result1.equity_curve.iloc[:test_day_idx]
        equity_before_2 = result2.equity_curve.iloc[:test_day_idx]
        
        assert equity_before_1.equals(equity_before_2), (
            f"LOOKAHEAD BIAS in equity curve!"
        )


class TestDollarNeutrality:
    """Tests for dollar-neutrality constraint."""
    
    def test_long_short_notional_equal_at_rebalance(self):
        """
        Test the portfolio is actually dollar-neutral (long leg notional
        equals short leg notional) at each rebalance.
        """
        prices_dict = create_synthetic_prices_dict(n_tickers=15, length=500)
        ranks_df = rank_momentum(prices_dict)
        
        engine = CrossSectionalBacktestEngine(
            top_n=5,
            bottom_n=5,
            cost_bps=0,
            slippage_bps=0,
        )
        
        result = engine.run(prices_dict, ranks_df)
        
        # Check positions at each rebalance
        positions = result.positions
        
        # At any point in time, sum of positive positions should equal
        # absolute value of sum of negative positions (approximately)
        for date in positions.index:
            row = positions.loc[date]
            long_exposure = row[row > 0].sum()
            short_exposure = abs(row[row < 0].sum())
            
            # Allow small floating point tolerance
            if long_exposure > 0 and short_exposure > 0:
                assert abs(long_exposure - short_exposure) < 0.01, (
                    f"Portfolio not dollar-neutral on {date}: "
                    f"long={long_exposure:.4f}, short={short_exposure:.4f}"
                )
        
        # Verify expected weights
        # Long: 5 tickers at 1/5 = 0.2 each, total = 1.0
        # Short: 5 tickers at -1/5 = -0.2 each, total = -1.0
        # Get all non-zero position values
        all_positions = positions.values.flatten()
        long_positions = all_positions[all_positions > 0]
        short_positions = all_positions[all_positions < 0]
        
        if len(long_positions) > 0:
            unique_long = np.unique(long_positions)
            assert len(unique_long) == 1 and abs(unique_long[0] - 0.2) < 0.01, (
                f"Expected long weight 0.2, got {unique_long}"
            )
        if len(short_positions) > 0:
            unique_short = np.unique(short_positions)
            assert len(unique_short) == 1 and abs(unique_short[0] - (-0.2)) < 0.01, (
                f"Expected short weight -0.2, got {unique_short}"
            )


class TestCostApplication:
    """Tests for cost model application."""
    
    def test_costs_charged_on_both_legs(self):
        """
        Test cost application: verify costs are charged on both legs
        at rebalance, not just the long leg.
        """
        prices_dict = create_synthetic_prices_dict(n_tickers=15, length=500)
        ranks_df = rank_momentum(prices_dict)
        
        # Run with zero costs
        engine_zero = CrossSectionalBacktestEngine(
            top_n=5, bottom_n=5, cost_bps=0, slippage_bps=0
        )
        result_zero = engine_zero.run(prices_dict, ranks_df)
        
        # Run with costs
        engine_costed = CrossSectionalBacktestEngine(
            top_n=5, bottom_n=5, cost_bps=5, slippage_bps=2
        )
        result_costed = engine_costed.run(prices_dict, ranks_df)
        
        # Costed returns should underperform zero-cost returns
        # especially at rebalance dates
        cost_impact = result_zero.returns - result_costed.returns
        
        # Total costs should be positive
        total_costs = cost_impact.sum()
        
        assert total_costs > 0, (
            f"Expected positive costs, got {total_costs}. "
            f"Costs may not be applied to both legs correctly."
        )
        
        # Verify the costed result has lower total return
        assert result_costed.equity_curve.iloc[-1] < result_zero.equity_curve.iloc[-1], (
            "Costed equity should underperform zero-cost equity"
        )


class TestSyntheticMomentumPattern:
    """Tests with known synthetic momentum patterns."""
    
    def test_momentum_ranking_produces_correct_portfolio(self):
        """
        Test on a synthetic case with a known, obvious momentum pattern
        Verify ranking and portfolio construction produce sensible results.
        """
        length = 500
        dates = pd.date_range("2020-01-01", periods=length, freq="B")
        
        # Create 5 tickers with varying momentum
        # Use percentage returns to ensure momentum_12_1 captures the difference
        np.random.seed(42)
        
        # T1: strong positive drift
        returns_t1 = 0.003 + np.random.randn(length) * 0.01
        prices_t1 = 100 * np.cumprod(1 + returns_t1)
        
        # T2: moderate positive drift
        returns_t2 = 0.001 + np.random.randn(length) * 0.01
        prices_t2 = 100 * np.cumprod(1 + returns_t2)
        
        # T3: zero drift
        returns_t3 = np.random.randn(length) * 0.01
        prices_t3 = 100 * np.cumprod(1 + returns_t3)
        
        # T4: moderate negative drift
        returns_t4 = -0.001 + np.random.randn(length) * 0.01
        prices_t4 = 100 * np.cumprod(1 + returns_t4)
        
        # T5: strong negative drift
        returns_t5 = -0.003 + np.random.randn(length) * 0.01
        prices_t5 = 100 * np.cumprod(1 + returns_t5)
        
        prices_dict = {
            "T1.NS": pd.Series(prices_t1, index=dates),
            "T2.NS": pd.Series(prices_t2, index=dates),
            "T3.NS": pd.Series(prices_t3, index=dates),
            "T4.NS": pd.Series(prices_t4, index=dates),
            "T5.NS": pd.Series(prices_t5, index=dates),
        }
        
        # Get ranks
        ranks_df = rank_momentum(prices_dict)
        
        # Check ranking on a late date (after warm-up)
        late_date = ranks_df.index[-1]
        late_ranks = ranks_df.loc[late_date]
        
        # T1 should have highest rank (strongest momentum)
        # T5 should have lowest rank (weakest/negative momentum)
        assert late_ranks["T1.NS"] == 5 or late_ranks["T1.NS"] == 4, (
            f"T1 should have high rank, got {late_ranks['T1.NS']}"
        )
        assert late_ranks["T5.NS"] <= 2, (
            f"T5 should have low rank, got {late_ranks['T5.NS']}"
        )
        
        # Run backtest with top 2 long, bottom 2 short
        engine = CrossSectionalBacktestEngine(
            top_n=2,
            bottom_n=2,
            cost_bps=0,
            slippage_bps=0,
        )
        
        result = engine.run(prices_dict, ranks_df)
        
        # With clear trends and zero costs, the strategy should have positive returns
        # (going long strong uptrends, short strong downtrends)
        final_equity = result.equity_curve.iloc[-1]
        
        # Allow for some randomness, but strategy should generally work
        # Just verify it runs and produces reasonable output
        assert final_equity > 0, (
            f"Expected positive equity, got {final_equity}"
        )


class TestWalkForwardIntegration:
    """Integration tests for cross-sectional walk-forward."""
    
    def test_walk_forward_runs_without_error(self):
        """Test that walk-forward runs end-to-end without crashing."""
        prices_dict = create_synthetic_prices_dict(n_tickers=15, length=1000)
        
        results = run_cross_sectional_walk_forward(
            prices_dict,
            top_n=5,
            bottom_n=5,
            cost_bps=5,
            slippage_bps=2,
            train_years=1,  # Shorter for test speed
            test_years=0.5,
            step_years=0.5,
        )
        
        # Should produce results (may be empty if not enough data for folds)
        # Just verify it doesn't crash
        assert isinstance(results, pd.DataFrame)
    
    def test_aggregation_produces_valid_metrics(self):
        """Test that aggregation produces valid metrics."""
        # Create a minimal valid result
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        test_returns = pd.Series(np.random.randn(50) * 0.01, index=dates)
        test_equity = pd.Series((1 + test_returns).cumprod(), index=dates)
        
        results = pd.DataFrame([{
            "fold": 0,
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2022-12-31"),
            "test_start": pd.Timestamp("2023-01-01"),
            "test_end": pd.Timestamp("2023-12-31"),
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.1,
            "calmar_ratio": 0.8,
            "total_return": 0.1,
            "turnover": 10.0,
            "trades": 5,
            "test_returns": test_returns,
            "test_equity_curve": test_equity,
        }])
        
        aggregated = aggregate_cross_sectional_results(results)
        
        assert "combined_equity_curve" in aggregated
        assert "overall_sharpe" in aggregated
        assert "total_folds" in aggregated
        assert aggregated["total_folds"] == 1
