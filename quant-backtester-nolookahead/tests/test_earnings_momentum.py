"""Tests for earnings momentum signal."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from signals.earnings_momentum import (
    load_earnings_surprises,
    rank_earnings_surprises,
    get_earnings_portfolio,
    earnings_momentum_signal,
)


class TestLoadEarningsSurprises:
    """Test earnings data loading."""
    
    def test_load_returns_dataframe(self):
        """Loading returns a DataFrame with expected columns."""
        df = load_earnings_surprises()
        
        assert isinstance(df, pd.DataFrame)
        assert 'ticker' in df.columns
        assert 'date' in df.columns
        assert 'surprise' in df.columns
        assert len(df) > 0
    
    def test_all_tickers_have_data(self):
        """All 15 NIFTY tickers have earnings surprise data."""
        df = load_earnings_surprises()
        
        tickers = df['ticker'].unique()
        assert len(tickers) == 15  # All 15 tickers
    
    def test_surprise_values_reasonable(self):
        """Surprise values are in reasonable range."""
        df = load_earnings_surprises()
        
        # Most surprises should be within -1 to +1 (100% miss/beat)
        # Extreme values possible but rare
        extreme = df[(df['surprise'] < -2) | (df['surprise'] > 2)]
        assert len(extreme) < len(df) * 0.05  # Less than 5% extreme


class TestRankEarningsSurprises:
    """Test surprise ranking logic."""
    
    def test_rank_produces_correct_ordering(self):
        """Higher surprise = higher rank."""
        test_data = pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D'],
            'date': pd.to_datetime(['2023-01-01'] * 4),
            'surprise': [0.1, -0.2, 0.5, -0.1],
        })
        
        ranked = rank_earnings_surprises(test_data)
        
        # C should have highest rank (4), B should have lowest (1)
        c_rank = ranked[ranked['ticker'] == 'C']['rank'].values[0]
        b_rank = ranked[ranked['ticker'] == 'B']['rank'].values[0]
        
        assert c_rank > b_rank
    
    def test_handles_multiple_dates(self):
        """Ranking works independently for each date."""
        test_data = pd.DataFrame({
            'ticker': ['A', 'B', 'A', 'B'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
            'surprise': [0.5, -0.5, -0.3, 0.3],
        })
        
        ranked = rank_earnings_surprises(test_data)
        
        # On date 1, A should have higher rank
        date1 = ranked[ranked['date'] == '2023-01-01']
        assert date1[date1['ticker'] == 'A']['rank'].values[0] > \
               date1[date1['ticker'] == 'B']['rank'].values[0]
        
        # On date 2, B should have higher rank
        date2 = ranked[ranked['date'] == '2023-01-02']
        assert date2[date2['ticker'] == 'B']['rank'].values[0] > \
               date2[date2['ticker'] == 'A']['rank'].values[0]


class TestGetEarningsPortfolio:
    """Test portfolio construction."""
    
    def test_long_short_weights_sum_to_one(self):
        """Long leg sums to +1, short leg sums to -1."""
        test_data = pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D', 'E'],
            'date': pd.to_datetime(['2023-01-01'] * 5),
            'surprise': [0.5, 0.3, 0.0, -0.3, -0.5],
        })
        
        long_df, short_df = get_earnings_portfolio(test_data, top_n=2, bottom_n=2)
        
        assert abs(long_df['weight'].sum() - 1.0) < 0.001
        assert abs(short_df['weight'].sum() - (-1.0)) < 0.001
    
    def test_correct_stocks_selected(self):
        """Long picks highest surprise, short picks lowest."""
        test_data = pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D', 'E'],
            'date': pd.to_datetime(['2023-01-01'] * 5),
            'surprise': [0.5, 0.3, 0.0, -0.3, -0.5],
        })
        
        long_df, short_df = get_earnings_portfolio(test_data, top_n=2, bottom_n=2)
        
        # Long should be A and B (highest surprises)
        long_tickers = set(long_df['ticker'].tolist())
        assert long_tickers == {'A', 'B'}
        
        # Short should be D and E (lowest surprises)
        short_tickers = set(short_df['ticker'].tolist())
        assert short_tickers == {'D', 'E'}


class TestLookaheadBias:
    """Test that signal does not use future data."""
    
    def test_signal_only_starts_at_announcement(self):
        """Signal weight appears only on or after announcement date."""
        test_data = pd.DataFrame({
            'ticker': ['A', 'B'],
            'date': pd.to_datetime(['2023-06-01', '2023-06-01']).tz_localize('UTC'),
            'surprise': [0.5, -0.5],
        })
        
        prices_dict = {
            'A': pd.Series(100, index=pd.date_range('2023-01-01', '2023-12-31', tz='UTC')),
            'B': pd.Series(100, index=pd.date_range('2023-01-01', '2023-12-31', tz='UTC')),
        }
        
        signal = earnings_momentum_signal(
            prices_dict, test_data,
            top_n=1, bottom_n=1, hold_days=21
        )
        
        # Before announcement, signal should be zero
        pre_mask = signal.index.normalize().tz_localize(None) < pd.Timestamp('2023-06-01')
        pre_announcement = signal[pre_mask]
        assert (pre_announcement == 0).all().all()
        
        # On or after announcement date within hold period, signal should be non-zero
        post_mask = (signal.index.normalize().tz_localize(None) >= pd.Timestamp('2023-06-01')) & \
                    (signal.index.normalize().tz_localize(None) <= pd.Timestamp('2023-06-21'))
        post_ann = signal[post_mask]
        assert post_ann.abs().sum().sum() > 0


class TestWalkForwardIntegration:
    """Test integration with walk-forward backtest."""
    
    def test_signal_aligns_with_price_dates(self):
        """Signal index matches price index exactly."""
        earnings_df = load_earnings_surprises()
        
        from data.loader import load_ohlcv
        prices_dict = {}
        for ticker in earnings_df['ticker'].unique()[:5]:
            prices = load_ohlcv(ticker, start='2020-01-01')
            prices_dict[ticker] = prices['Close']
        
        signal = earnings_momentum_signal(prices_dict, earnings_df, top_n=2, bottom_n=2)
        
        # Signal index should be subset of price index
        for ticker, prices in prices_dict.items():
            signal_idx = signal.index.tz_localize(None)
            price_idx = prices.index.tz_localize(None)
            assert set(signal_idx).issubset(set(price_idx))
    
    def test_signal_has_valid_weights(self):
        """Signal weights are valid (long=positive, short=negative, or zero)."""
        earnings_df = load_earnings_surprises()
        
        from data.loader import load_ohlcv
        prices_dict = {}
        for ticker in earnings_df['ticker'].unique()[:5]:
            prices = load_ohlcv(ticker, start='2020-01-01')
            prices_dict[ticker] = prices['Close']
        
        signal = earnings_momentum_signal(prices_dict, earnings_df, top_n=2, bottom_n=2)
        
        # All values should be in valid range [-1, 1]
        assert (signal >= -1.0).all().all()
        assert (signal <= 1.0).all().all()
