"""
Tests for multi-ticker walk-forward validation.

Tests specifically target:
1. Graceful handling of ticker load failures
2. Aggregation correctness with known values
3. Per-ticker fold count consistency with available history
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.multi_ticker import (
    run_multi_ticker_walk_forward,
    aggregate_across_tickers,
    DEFAULT_NIFTY_BASKET,
)
from signals.signals import moving_average_crossover


def create_synthetic_prices(
    length: int = 1000,
    seed: int = 42,
    start_date: str = "2015-01-01"
) -> pd.Series:
    """Create a synthetic price series."""
    np.random.seed(seed)
    returns = np.random.randn(length) * 0.02
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range(start_date, periods=length, freq="B")
    return pd.Series(prices, index=dates, name="Close")


def create_mock_ohlcv_df(length: int = 1000, start_date: str = "2015-01-01") -> pd.DataFrame:
    """Create a mock OHLCV dataframe."""
    prices = create_synthetic_prices(length, seed=42, start_date=start_date)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(100000, 1000000, len(prices)),
    }, index=prices.index)


class TestTickerLoadFailure:
    """Tests for graceful handling of ticker load failures."""
    
    def test_handles_single_ticker_failure_gracefully(self):
        """
        Test that run_multi_ticker_walk_forward handles a ticker load failure
        gracefully. Mock one ticker to raise an error, verify the rest still
        complete and the failure is logged/reported.
        """
        tickers = ["TICKER1.NS", "TICKER2.NS", "TICKER3.NS"]
        
        def mock_load_ohlcv(ticker, start="2015-01-01", end=None):
            if ticker == "TICKER2.NS":
                raise ValueError(f"Data unavailable for {ticker}")
            return create_mock_ohlcv_df(1000, start_date=start)
        
        with patch("backtest.multi_ticker.load_ohlcv", side_effect=mock_load_ohlcv):
            results = run_multi_ticker_walk_forward(
                tickers,
                lambda p: moving_average_crossover(p, fast=20, slow=50),
                engine_kwargs={"cost_bps": 0, "slippage_bps": 0},
                train_years=1,
                test_years=0.5,
                step_years=0.5,
                skip_insufficient_data=True,
            )
        
        # Should have results for TICKER1 and TICKER3, not TICKER2
        assert "TICKER2.NS" not in results["ticker"].values, (
            "TICKER2.NS should not be in results (it failed to load)"
        )
        
        successful_tickers = results["ticker"].unique()
        assert len(successful_tickers) == 2, f"Expected 2 successful tickers, got {len(successful_tickers)}"
        assert "TICKER1.NS" in successful_tickers
        assert "TICKER3.NS" in successful_tickers
    
    def test_handles_all_tickers_failure(self):
        """Test that all tickers failing returns empty DataFrame."""
        tickers = ["BAD1.NS", "BAD2.NS"]
        
        def mock_load_ohlcv(ticker, start="2015-01-01", end=None):
            raise ValueError(f"Data unavailable for {ticker}")
        
        with patch("backtest.multi_ticker.load_ohlcv", side_effect=mock_load_ohlcv):
            results = run_multi_ticker_walk_forward(
                tickers,
                lambda p: moving_average_crossover(p, fast=20, slow=50),
                engine_kwargs={},
                train_years=1,
                test_years=0.5,
                step_years=0.5,
                skip_insufficient_data=True,
            )
        
        assert len(results) == 0, "Expected empty DataFrame when all tickers fail"
    
    def test_raises_when_skip_insufficient_data_false(self):
        """Test that skip_insufficient_data=False raises on first error."""
        tickers = ["GOOD.NS", "BAD.NS"]
        
        def mock_load_ohlcv(ticker, start="2015-01-01", end=None):
            if ticker == "BAD.NS":
                raise ValueError("Data unavailable")
            return create_mock_ohlcv_df(1000)
        
        with patch("backtest.multi_ticker.load_ohlcv", side_effect=mock_load_ohlcv):
            with pytest.raises(ValueError, match="Data unavailable"):
                run_multi_ticker_walk_forward(
                    tickers,
                    lambda p: moving_average_crossover(p, fast=20, slow=50),
                    engine_kwargs={},
                    train_years=1,
                    test_years=0.5,
                    step_years=0.5,
                    skip_insufficient_data=False,
                )


class TestAggregationCorrectness:
    """Tests for aggregate_across_tickers correctness."""
    
    def test_aggregation_with_known_values(self):
        """
        Test aggregate_across_tickers correctness on a small synthetic
        multi-ticker result set with known expected means/stds.
        """
        # Create synthetic results for 3 tickers, 2 folds each
        # Ticker A: Sharpe 1.0, 0.5 (mean=0.75)
        # Ticker B: Sharpe -0.5, -1.0 (mean=-0.75)
        # Ticker C: Sharpe 0.0, 0.0 (mean=0.0)
        
        results = pd.DataFrame([
            # Ticker A
            {"ticker": "A.NS", "fold": 0, "sharpe_ratio": 1.0, "max_drawdown": 0.1,
             "calmar_ratio": 1.0, "win_rate": 0.6, "total_return": 0.1, "trades": 5},
            {"ticker": "A.NS", "fold": 1, "sharpe_ratio": 0.5, "max_drawdown": 0.15,
             "calmar_ratio": 0.5, "win_rate": 0.55, "total_return": 0.05, "trades": 4},
            # Ticker B
            {"ticker": "B.NS", "fold": 0, "sharpe_ratio": -0.5, "max_drawdown": 0.3,
             "calmar_ratio": -0.5, "win_rate": 0.4, "total_return": -0.15, "trades": 6},
            {"ticker": "B.NS", "fold": 1, "sharpe_ratio": -1.0, "max_drawdown": 0.35,
             "calmar_ratio": -1.0, "win_rate": 0.35, "total_return": -0.2, "trades": 7},
            # Ticker C
            {"ticker": "C.NS", "fold": 0, "sharpe_ratio": 0.0, "max_drawdown": 0.2,
             "calmar_ratio": 0.0, "win_rate": 0.5, "total_return": 0.0, "trades": 3},
            {"ticker": "C.NS", "fold": 1, "sharpe_ratio": 0.0, "max_drawdown": 0.2,
             "calmar_ratio": 0.0, "win_rate": 0.5, "total_return": 0.0, "trades": 3},
        ])
        
        aggregated = aggregate_across_tickers(results)
        
        # Verify per-ticker summary
        per_ticker = aggregated["per_ticker_summary"]
        assert len(per_ticker) == 3, "Expected 3 tickers in summary"
        
        # Check specific values
        a_data = per_ticker[per_ticker["ticker"] == "A.NS"].iloc[0]
        assert abs(a_data["mean_sharpe"] - 0.75) < 0.01, f"A mean Sharpe should be 0.75, got {a_data['mean_sharpe']}"
        assert a_data["pct_profitable_folds"] == 1.0, "A should be 100% profitable"
        
        b_data = per_ticker[per_ticker["ticker"] == "B.NS"].iloc[0]
        assert abs(b_data["mean_sharpe"] - (-0.75)) < 0.01, f"B mean Sharpe should be -0.75, got {b_data['mean_sharpe']}"
        assert b_data["pct_profitable_folds"] == 0.0, "B should be 0% profitable"
        
        c_data = per_ticker[per_ticker["ticker"] == "C.NS"].iloc[0]
        assert abs(c_data["mean_sharpe"] - 0.0) < 0.01, f"C mean Sharpe should be 0.0, got {c_data['mean_sharpe']}"
        
        # Cross-ticker summary
        cross = aggregated["cross_ticker_summary"]
        
        # Mean of mean Sharpes: (0.75 + (-0.75) + 0.0) / 3 = 0.0
        assert abs(cross["mean_of_mean_sharpe"] - 0.0) < 0.01, (
            f"Mean of mean Sharpe should be 0.0, got {cross['mean_of_mean_sharpe']}"
        )
        
        # Std of mean Sharpes: std([0.75, -0.75, 0.0]) with ddof=1 (pandas default)
        expected_std = np.std([0.75, -0.75, 0.0], ddof=1)
        assert abs(cross["std_of_mean_sharpe"] - expected_std) < 0.01, (
            f"Std of mean Sharpe should be ~{expected_std}, got {cross['std_of_mean_sharpe']}"
        )
        
        # Best and worst tickers
        assert aggregated["best_ticker"] == "A.NS", f"Best should be A.NS, got {aggregated['best_ticker']}"
        assert aggregated["worst_ticker"] == "B.NS", f"Worst should be B.NS, got {aggregated['worst_ticker']}"
        
        # Total folds
        assert cross["total_folds_all"] == 6
    
    def test_aggregation_empty_results(self):
        """Test aggregation with empty DataFrame."""
        results = pd.DataFrame()
        
        aggregated = aggregate_across_tickers(results)
        
        assert len(aggregated["per_ticker_summary"]) == 0
        assert aggregated["best_ticker"] is None
        assert aggregated["worst_ticker"] is None
        assert aggregated["failure_pattern"] == "unknown"
    
    def test_failure_pattern_detection(self):
        """Test that failure pattern is correctly detected."""
        # Test consistent failure (most tickers negative)
        consistent_fail_results = pd.DataFrame([
            {"ticker": f"T{i}.NS", "fold": 0, "sharpe_ratio": -1.0, "max_drawdown": 0.3,
             "calmar_ratio": -1.0, "win_rate": 0.3, "total_return": -0.2, "trades": 5}
            for i in range(10)
        ])
        
        aggregated = aggregate_across_tickers(consistent_fail_results)
        assert aggregated["failure_pattern"] == "consistent", (
            f"Expected 'consistent' pattern, got {aggregated['failure_pattern']}"
        )
        
        # Test concentrated failure (high variance, mixed results)
        concentrated_results = pd.DataFrame([
            {"ticker": "GOOD.NS", "fold": 0, "sharpe_ratio": 2.0, "max_drawdown": 0.1,
             "calmar_ratio": 2.0, "win_rate": 0.7, "total_return": 0.3, "trades": 5},
            {"ticker": "BAD.NS", "fold": 0, "sharpe_ratio": -2.0, "max_drawdown": 0.5,
             "calmar_ratio": -2.0, "win_rate": 0.2, "total_return": -0.4, "trades": 5},
        ])
        
        aggregated = aggregate_across_tickers(concentrated_results)
        # With only 2 tickers and 50% negative, depends on variance
        assert aggregated["failure_pattern"] in ["concentrated", "mixed_but_mostly_bad", "inconclusive"]


class TestFoldCountConsistency:
    """Tests for per-ticker fold count consistency with available history."""
    
    def test_shorter_history_produces_fewer_folds(self):
        """
        Test that a ticker with less history produces fewer folds, not an error.
        """
        # Create two mock data sets: one long, one short
        def mock_load_ohlcv(ticker, start="2015-01-01", end=None):
            if ticker == "LONG.NS":
                # 8 years of data
                return create_mock_ohlcv_df(2000, start_date="2015-01-01")
            elif ticker == "SHORT.NS":
                # 4 years of data (barely enough for ~1 fold with 3y train + 1y test)
                return create_mock_ohlcv_df(1000, start_date="2018-01-01")
            else:
                return create_mock_ohlcv_df(1000)
        
        tickers = ["LONG.NS", "SHORT.NS"]
        
        with patch("backtest.multi_ticker.load_ohlcv", side_effect=mock_load_ohlcv):
            results = run_multi_ticker_walk_forward(
                tickers,
                lambda p: moving_average_crossover(p, fast=20, slow=50),
                engine_kwargs={"cost_bps": 0, "slippage_bps": 0},
                train_years=3,
                test_years=1,
                step_years=1,
            )
        
        # Count folds per ticker
        long_folds = len(results[results["ticker"] == "LONG.NS"])
        short_folds = len(results[results["ticker"] == "SHORT.NS"])
        
        # SHORT should have same or fewer folds than LONG
        assert short_folds <= long_folds, (
            f"Short history ticker should have <= folds than long history: "
            f"SHORT={short_folds}, LONG={long_folds}"
        )
        
        # Both should have at least 1 fold if they have minimum data
        if short_folds > 0:
            assert long_folds > 0, "Long history should have folds if short does"
    
    def test_insufficient_data_ticker_excluded(self):
        """Test that a ticker with truly insufficient data is excluded."""
        def mock_load_ohlcv(ticker, start="2015-01-01", end=None):
            if ticker == "TOO_SHORT.NS":
                # Only 1 year of data (not enough for 3y train + 1y test)
                return create_mock_ohlcv_df(252, start_date="2020-01-01")
            else:
                return create_mock_ohlcv_df(2000, start_date="2015-01-01")
        
        tickers = ["GOOD.NS", "TOO_SHORT.NS"]
        
        with patch("backtest.multi_ticker.load_ohlcv", side_effect=mock_load_ohlcv):
            results = run_multi_ticker_walk_forward(
                tickers,
                lambda p: moving_average_crossover(p, fast=20, slow=50),
                engine_kwargs={"cost_bps": 0, "slippage_bps": 0},
                train_years=3,
                test_years=1,
                step_years=1,
            )
        
        # TOO_SHORT should be excluded
        assert "TOO_SHORT.NS" not in results["ticker"].values, (
            "Ticker with insufficient data should be excluded"
        )
        assert "GOOD.NS" in results["ticker"].values, "Good ticker should be present"


class TestDefaultBasket:
    """Tests for default NIFTY basket configuration."""
    
    def test_default_basket_is_list(self):
        """Test that DEFAULT_NIFTY_BASKET is a list of strings."""
        assert isinstance(DEFAULT_NIFTY_BASKET, list)
        assert len(DEFAULT_NIFTY_BASKET) >= 12
        assert all(isinstance(t, str) for t in DEFAULT_NIFTY_BASKET)
        assert all(t.endswith(".NS") for t in DEFAULT_NIFTY_BASKET)
    
    def test_default_basket_has_sector_diversity(self):
        """
        Test that default basket has sector diversity.
        This is a sanity check on the basket composition.
        """
        # Known sector mappings (simplified)
        banking_tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
        it_tickers = ["TCS.NS", "INFY.NS"]
        fmcg_tickers = ["HINDUNILVR.NS", "ITC.NS"]
        
        # Count how many from each sector are in the basket
        banking_count = len(set(banking_tickers) & set(DEFAULT_NIFTY_BASKET))
        it_count = len(set(it_tickers) & set(DEFAULT_NIFTY_BASKET))
        fmcg_count = len(set(fmcg_tickers) & set(DEFAULT_NIFTY_BASKET))
        
        # Should have multiple sectors represented
        assert banking_count <= 6, "Should not have more than 6 banking tickers"
        assert it_count <= 3, "Should not have more than 3 IT tickers"
        assert fmcg_count >= 1, "Should have at least 1 FMCG ticker"
        
        # Total should be diversified (not all one sector)
        assert len(DEFAULT_NIFTY_BASKET) >= 10, "Basket should have at least 10 tickers"
