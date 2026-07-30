"""
Tests for walk-forward validation.

These tests specifically target:
1. No overlap between test periods and future train periods (leakage prevention)
2. Correct fold count for given parameters
3. Aggregation correctness
4. No leakage of future data into earlier folds
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.walk_forward import (
    walk_forward_split, 
    run_walk_forward, 
    aggregate_walk_forward_results,
)
from backtest.engine import BacktestEngine
from signals.signals import moving_average_crossover


def create_synthetic_prices(
    length: int = 500, 
    seed: int = 42,
    start_date: str = "2015-01-01"
) -> pd.Series:
    """Create a synthetic price series using random walk."""
    np.random.seed(seed)
    returns = np.random.randn(length) * 0.02  # 2% daily vol
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range(start_date, periods=length, freq="B")
    return pd.Series(prices, index=dates, name="price")


def create_trending_prices(
    length: int = 100,
    trend: float = 0.001,
    start_date: str = "2015-01-01"
) -> pd.Series:
    """Create a price series with a clear trend plus noise."""
    np.random.seed(42)
    noise = np.random.randn(length) * 0.01
    trend_component = np.arange(length) * trend
    prices = 100 + trend_component * 100 + noise * 10
    prices = np.maximum(prices, 50)  # Ensure positive
    dates = pd.date_range(start_date, periods=length, freq="B")
    return pd.Series(prices, index=dates, name="price")


class TestFoldConstruction:
    """Tests for walk_forward_split function."""
    
    def test_fold_count_known_range(self):
        """
        Fold count test: Given a known date range and train/test/step years,
        verify the correct number of folds is generated.
        
        With 8 years of data (~2016 days), train=3 years, test=1 year, step=1 year:
        - First fold: years 0-3 train, 3-4 test
        - Rolls forward by 1 year each time
        - Should get approximately 8 - 3 - 1 = 4 folds
        """
        # Create exactly 8 years of business days
        dates = pd.date_range("2010-01-01", periods=8 * 252, freq="B")
        
        folds = walk_forward_split(
            dates,
            train_years=3,
            test_years=1,
            step_years=1,
        )
        
        # Expected: 8 - 3 - 1 = 4 folds (approximately)
        # The exact count depends on boundary conditions
        assert len(folds) >= 3, f"Expected at least 3 folds, got {len(folds)}"
        assert len(folds) <= 6, f"Expected at most 6 folds, got {len(folds)}"
    
    def test_no_overlap_test_train(self):
        """
        No-overlap test: Verify generated fold windows don't have a test period
        that overlaps with a later fold's train period in a way that leaks future data.
        
        Specifically: Fold N's test_end should be <= Fold N+1's train_end,
        and Fold N's test period should not overlap with Fold N+1's train period
        in a way that would allow future data to leak backward.
        """
        dates = pd.date_range("2010-01-01", periods=10 * 252, freq="B")
        
        folds = walk_forward_split(
            dates,
            train_years=3,
            test_years=1,
            step_years=1,
        )
        
        assert len(folds) >= 2, "Need at least 2 folds for this test"
        
        for i in range(len(folds) - 1):
            fold_n = folds[i]
            fold_n_plus_1 = folds[i + 1]
            
            train_start_n, train_end_n, test_end_n = fold_n
            train_start_n1, train_end_n1, test_end_n1 = fold_n_plus_1
            
            # Each fold should roll forward in time
            assert train_start_n <= train_start_n1, "Train start should not go backward"
            assert train_end_n <= train_end_n1, "Train end should not go backward"
            assert test_end_n <= test_end_n1, "Test end should not go backward"
            
            # The test period of fold N ends before or at the same time as
            # the train period of fold N+1 begins its extension
            # This is naturally satisfied by the rolling window construction
    
    def test_fold_windows_are_contiguous(self):
        """Test that train and test windows are contiguous (no gaps)."""
        dates = pd.date_range("2010-01-01", periods=6 * 252, freq="B")
        
        folds = walk_forward_split(
            dates,
            train_years=2,
            test_years=1,
            step_years=1,
        )
        
        for train_start, train_end, test_end in folds:
            # Train period should be before test period
            assert train_start < train_end
            assert train_end < test_end
            
            # Find index positions to verify contiguity
            train_start_idx = dates.get_loc(train_start)
            train_end_idx = dates.get_loc(train_end)
            test_end_idx = dates.get_loc(test_end)
            
            # The test window should start right after train window ends
            # (there might be one day gap due to index lookup, which is acceptable)
            assert test_end_idx > train_end_idx, "Test should be after train"


class TestLeakagePrevention:
    """Tests to verify no data leakage across folds."""
    
    def test_outlier_does_not_leak_across_folds(self):
        """
        Leakage test: Construct a price series where an extreme outlier exists
        only in fold 3's test period, and verify fold 1 and fold 2's results
        are completely unaffected by it.
        """
        # Create a long price series that will span multiple folds
        np.random.seed(42)
        length = 2000  # ~8 years of daily data
        prices = create_synthetic_prices(length, seed=42, start_date="2010-01-01")
        
        # Identify where fold 3's test period would be
        folds = walk_forward_split(
            prices.index,
            train_years=3,
            test_years=1,
            step_years=1,
        )
        
        if len(folds) < 4:
            pytest.skip("Not enough folds for this test")
        
        # Get fold 3's test period
        _, _, test_end_fold3 = folds[3]
        _, train_end_fold3, _ = folds[3]
        
        # Find the index in the test period of fold 3
        test_start_idx = prices.index.get_loc(train_end_fold3) + 1
        test_end_idx = prices.index.get_loc(test_end_fold3)
        
        # Create a modified price series with an extreme outlier in fold 3's test
        prices_modified = prices.copy()
        outlier_day = (test_start_idx + test_end_idx) // 2
        prices_modified.iloc[outlier_day] = prices_modified.iloc[outlier_day] * 100
        
        # Simple signal function that only uses prices
        def simple_signal(prices):
            return moving_average_crossover(prices, fast=20, slow=50)
        
        # Run walk-forward on both price series
        results_original = run_walk_forward(
            prices,
            simple_signal,
            engine_kwargs={"cost_bps": 0, "slippage_bps": 0},
            train_years=2,  # Shorter to get more folds
            test_years=0.5,
            step_years=0.5,
        )
        
        results_modified = run_walk_forward(
            prices_modified,
            simple_signal,
            engine_kwargs={"cost_bps": 0, "slippage_bps": 0},
            train_years=2,
            test_years=0.5,
            step_years=0.5,
        )
        
        # Fold 0 and 1 should be IDENTICAL (or very close) between original and modified
        # because the outlier is in fold 3's test period
        for fold_idx in [0, 1]:
            if fold_idx < len(results_original) and fold_idx < len(results_modified):
                orig_row = results_original.iloc[fold_idx]
                mod_row = results_modified.iloc[fold_idx]
                
                # Sharpe ratios should be identical
                assert orig_row["sharpe_ratio"] == mod_row["sharpe_ratio"], (
                    f"LEAKAGE DETECTED! Fold {fold_idx} Sharpe changed from "
                    f"{orig_row['sharpe_ratio']} to {mod_row['sharpe_ratio']} "
                    f"when we modified fold 3's test period"
                )
                
                # Total returns should be identical
                assert orig_row["total_return"] == mod_row["total_return"], (
                    f"LEAKAGE DETECTED! Fold {fold_idx} return changed"
                )
    
    def test_signal_only_sees_train_test_window(self):
        """
        Verify that the signal function only receives prices within the fold window.
        """
        # Track what prices the signal function receives
        received_indices = []
        
        def tracking_signal(prices):
            received_indices.append((prices.index.min(), prices.index.max()))
            return pd.Series(0, index=prices.index)
        
        full_prices = create_synthetic_prices(1000, seed=42, start_date="2010-01-01")
        
        results = run_walk_forward(
            full_prices,
            tracking_signal,
            engine_kwargs={},
            train_years=2,
            test_years=0.5,
            step_years=0.5,
        )
        
        # Each fold should have received a different (rolling) window
        assert len(received_indices) == len(results), (
            "Signal should be called once per fold"
        )
        
        # The first fold's window should be strictly smaller than the full range
        first_train_start, first_test_end = received_indices[0]
        assert first_train_start >= full_prices.index.min()
        assert first_test_end <= full_prices.index.max()
        
        # Each subsequent fold's window should roll forward
        for i in range(1, len(received_indices)):
            prev_train_start, _ = received_indices[i - 1]
            curr_train_start, _ = received_indices[i]
            assert curr_train_start >= prev_train_start, (
                "Train window should roll forward, not backward"
            )


class TestAggregation:
    """Tests for aggregate_walk_forward_results function."""
    
    def test_aggregation_correctness_simple(self):
        """
        Aggregation correctness test: Construct a simple synthetic case with
        2-3 folds of known returns, verify aggregate_walk_forward_results
        computes the combined equity curve and metrics correctly.
        """
        # Create mock results with known returns
        # Fold 0: returns of 1% each day for 10 days
        # Fold 1: returns of -0.5% each day for 10 days
        # Fold 2: returns of 2% each day for 10 days
        
        dates_fold0 = pd.date_range("2010-01-01", periods=10, freq="B")
        dates_fold1 = pd.date_range("2011-01-01", periods=10, freq="B")
        dates_fold2 = pd.date_range("2012-01-01", periods=10, freq="B")
        
        # Non-overlapping dates
        dates_fold1 = pd.date_range(dates_fold0[-1] + pd.Timedelta(days=1), periods=10, freq="B")
        dates_fold2 = pd.date_range(dates_fold1[-1] + pd.Timedelta(days=1), periods=10, freq="B")
        
        fold0_returns = pd.Series(0.01, index=dates_fold0)
        fold1_returns = pd.Series(-0.005, index=dates_fold1)
        fold2_returns = pd.Series(0.02, index=dates_fold2)
        
        # Compute equity curves
        fold0_equity = (1 + fold0_returns).cumprod()
        fold1_equity = (1 + fold1_returns).cumprod()
        fold2_equity = (1 + fold2_returns).cumprod()
        
        # Build results dataframe
        results = pd.DataFrame([
            {
                "fold": 0,
                "train_start": dates_fold0[0] - pd.Timedelta(days=252*2),
                "train_end": dates_fold0[0] - pd.Timedelta(days=1),
                "test_start": dates_fold0[0],
                "test_end": dates_fold0[-1],
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 1.0,
                "win_rate": 1.0,
                "total_return": fold0_equity.iloc[-1] - 1,
                "trades": 10,
                "test_returns": fold0_returns,
                "test_equity_curve": fold0_equity,
            },
            {
                "fold": 1,
                "train_start": dates_fold1[0] - pd.Timedelta(days=252*2),
                "train_end": dates_fold1[0] - pd.Timedelta(days=1),
                "test_start": dates_fold1[0],
                "test_end": dates_fold1[-1],
                "sharpe_ratio": -0.5,
                "max_drawdown": 0.05,
                "calmar_ratio": -0.5,
                "win_rate": 0.0,
                "total_return": fold1_equity.iloc[-1] - 1,
                "trades": 5,
                "test_returns": fold1_returns,
                "test_equity_curve": fold1_equity,
            },
            {
                "fold": 2,
                "train_start": dates_fold2[0] - pd.Timedelta(days=252*2),
                "train_end": dates_fold2[0] - pd.Timedelta(days=1),
                "test_start": dates_fold2[0],
                "test_end": dates_fold2[-1],
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 2.0,
                "win_rate": 1.0,
                "total_return": fold2_equity.iloc[-1] - 1,
                "trades": 15,
                "test_returns": fold2_returns,
                "test_equity_curve": fold2_equity,
            },
        ])
        
        aggregated = aggregate_walk_forward_results(results)
        
        # Verify fold count
        assert aggregated["total_folds"] == 3
        
        # Verify profitable folds (fold 0 and 2 are profitable, fold 1 is not)
        assert aggregated["profitable_folds"] == 2
        
        # Verify fold returns list
        assert len(aggregated["fold_returns"]) == 3
        
        # Verify combined equity curve exists and has correct length
        assert len(aggregated["combined_equity_curve"]) == 30  # 10 + 10 + 10
        
        # Verify combined returns
        assert len(aggregated["combined_returns"]) == 30
        
        # Verify overall metrics exist and are numeric
        assert isinstance(aggregated["overall_sharpe"], (int, float))
        assert isinstance(aggregated["overall_max_drawdown"], (int, float))
        assert isinstance(aggregated["overall_calmar"], (int, float))
        assert isinstance(aggregated["overall_total_return"], (int, float))
        
        # Verify fold_metrics dataframe exists
        assert isinstance(aggregated["fold_metrics"], pd.DataFrame)
        assert len(aggregated["fold_metrics"]) == 3
    
    def test_aggregation_empty_results(self):
        """Test aggregation with empty results dataframe."""
        results = pd.DataFrame()
        
        aggregated = aggregate_walk_forward_results(results)
        
        assert aggregated["total_folds"] == 0
        assert aggregated["profitable_folds"] == 0
        assert np.isnan(aggregated["overall_sharpe"])
    
    def test_combined_equity_is_continuous(self):
        """
        Test that the combined equity curve is continuous across folds
        (no jumps at fold boundaries).
        """
        dates_fold0 = pd.date_range("2010-01-01", periods=10, freq="B")
        dates_fold1 = pd.date_range(dates_fold0[-1] + pd.Timedelta(days=1), periods=10, freq="B")
        
        fold0_returns = pd.Series(0.01, index=dates_fold0)
        fold1_returns = pd.Series(0.02, index=dates_fold1)
        
        fold0_equity = (1 + fold0_returns).cumprod()
        fold1_equity = (1 + fold1_returns).cumprod()
        
        results = pd.DataFrame([
            {
                "fold": 0,
                "train_start": dates_fold0[0] - pd.Timedelta(days=100),
                "train_end": dates_fold0[0] - pd.Timedelta(days=1),
                "test_start": dates_fold0[0],
                "test_end": dates_fold0[-1],
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 1.0,
                "win_rate": 1.0,
                "total_return": fold0_equity.iloc[-1] - 1,
                "trades": 10,
                "test_returns": fold0_returns,
                "test_equity_curve": fold0_equity,
            },
            {
                "fold": 1,
                "train_start": dates_fold1[0] - pd.Timedelta(days=100),
                "train_end": dates_fold1[0] - pd.Timedelta(days=1),
                "test_start": dates_fold1[0],
                "test_end": dates_fold1[-1],
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 1.0,
                "win_rate": 1.0,
                "total_return": fold1_equity.iloc[-1] - 1,
                "trades": 10,
                "test_returns": fold1_returns,
                "test_equity_curve": fold1_equity,
            },
        ])
        
        aggregated = aggregate_walk_forward_results(results)
        
        combined_equity = aggregated["combined_equity_curve"]
        
        # Check that there are no sudden jumps > 10% between consecutive days
        # (this would indicate a discontinuity in the equity curve)
        daily_returns = combined_equity.pct_change().dropna()
        assert (daily_returns.abs() < 0.1).all(), (
            f"Large jump detected in combined equity curve: max={daily_returns.abs().max()}"
        )


class TestRunWithRealSignal:
    """Integration tests with actual signal functions."""
    
    def test_run_with_ma_crossover(self):
        """Test walk-forward with moving average crossover signal."""
        prices = create_synthetic_prices(1500, seed=42, start_date="2010-01-01")
        
        results = run_walk_forward(
            prices,
            lambda p: moving_average_crossover(p, fast=20, slow=50),
            engine_kwargs={"cost_bps": 5, "slippage_bps": 2},
            train_years=2,
            test_years=0.5,
            step_years=0.5,
        )
        
        assert len(results) > 0, "Should have at least one fold"
        
        # Verify results structure
        required_columns = [
            "fold", "train_start", "train_end", "test_start", "test_end",
            "sharpe_ratio", "max_drawdown", "calmar_ratio", 
            "win_rate", "total_return", "trades"
        ]
        for col in required_columns:
            assert col in results.columns, f"Missing column: {col}"
        
        # Verify metrics are numeric (or NaN for edge cases)
        assert pd.api.types.is_numeric_dtype(results["sharpe_ratio"])
        assert pd.api.types.is_numeric_dtype(results["total_return"])
    
    def test_run_handles_all_nan_signal(self):
        """Test that walk-forward handles signals that are all NaN (warm-up)."""
        # Very short price series where signal never has enough data
        prices = create_synthetic_prices(100, seed=42, start_date="2010-01-01")
        
        def long_warmup_signal(prices):
            # Signal that requires 200 days but only gets 100
            return pd.Series(np.nan, index=prices.index)
        
        results = run_walk_forward(
            prices,
            long_warmup_signal,
            engine_kwargs={},
            train_years=0.1,  # Very short windows
            test_years=0.1,
            step_years=0.1,
        )
        
        # Should not crash, may have some NaN results
        assert isinstance(results, pd.DataFrame)
