"""Run earnings momentum strategy walk-forward backtest."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.loader import load_ohlcv
from src.signals.earnings_momentum import load_earnings_surprises, earnings_momentum_signal
from src.backtest.engine import BacktestEngine, BacktestResult
from src.metrics.performance import sharpe_ratio, max_drawdown, calmar_ratio, summary


def run_earnings_momentum_walk_forward(
    prices_dict: dict,
    earnings_df: pd.DataFrame,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    cost_bps: float = 7.0,
    capital: float = 1_000_000,
):
    """
    Run walk-forward backtest for earnings momentum strategy.
    
    Parameters
    ----------
    prices_dict : dict
        Dictionary mapping tickers to price Series
    earnings_df : pd.DataFrame
        Earnings surprises with columns: ticker, date, surprise
    train_years : int
        Training window years (for parameter optimization, if any)
    test_years : int
        Test window years
    step_years : int
        Step between folds in years
    cost_bps : float
        Transaction cost + slippage in bps
    capital : float
        Starting capital
    
    Returns
    -------
    dict
        Walk-forward results with folds and aggregated metrics
    """
    # Get date range
    all_dates = None
    for prices in prices_dict.values():
        if all_dates is None:
            all_dates = set(prices.index)
        else:
            all_dates = all_dates.intersection(set(prices.index))
    
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    
    # Ensure earnings dates are timezone-aware for comparison
    earnings_df = earnings_df.copy()
    if earnings_df['date'].dt.tz is None:
        # Use the timezone from prices (typically UTC+05:30 for NSE)
        earnings_df['date'] = earnings_df['date'].dt.tz_localize(all_dates.tz)
    
    # Adjust min_date to start from when we have earnings data
    min_earnings_date = earnings_df['date'].min()
    min_date = max(all_dates.min(), min_earnings_date)
    max_date = all_dates.max()
    
    # Generate walk-forward folds
    trade_days_per_year = 252
    train_days = train_years * trade_days_per_year
    test_days = test_years * trade_days_per_year
    step_days = step_years * trade_days_per_year
    
    folds = []
    current_start = min_date
    
    while current_start + pd.Timedelta(days=train_days + test_days) <= max_date:
        train_end = current_start + pd.Timedelta(days=train_days)
        test_end = current_start + pd.Timedelta(days=train_days + test_days)
        
        # Find nearest trading dates
        train_end_idx = all_dates.searchsorted(train_end)
        if train_end_idx >= len(all_dates):
            break
        train_end = all_dates[train_end_idx - 1]
        
        test_end_idx = all_dates.searchsorted(test_end)
        if test_end_idx >= len(all_dates):
            test_end_idx = len(all_dates)
        test_end = all_dates[test_end_idx - 1]
        
        # Get test period data
        test_prices_dict = {}
        for ticker, prices in prices_dict.items():
            test_prices_dict[ticker] = prices.loc[:test_end]
        
        # Filter earnings data to only include announcements before or during test period
        test_earnings = earnings_df[
            (earnings_df['date'] >= current_start) & 
            (earnings_df['date'] <= test_end)
        ].copy()
        
        # Also include earnings before test period (for carry-over positions)
        # Look back up to hold_days (21 days) before train start
        hold_days = 21
        early_cutoff = current_start - pd.Timedelta(days=hold_days * 2)
        all_test_earnings = earnings_df[
            (earnings_df['date'] >= early_cutoff) & 
            (earnings_df['date'] <= test_end)
        ].copy()
        
        # Generate signal
        signal_df = earnings_momentum_signal(
            test_prices_dict,
            all_test_earnings,
            top_n=5,
            bottom_n=5,
            hold_days=21,
        )
        
        # Get returns for each ticker and aggregate
        tickers = list(prices_dict.keys())
        n_tickers = len(tickers)
        
        # Compute individual returns
        individual_returns = {}
        for ticker in tickers:
            individual_returns[ticker] = test_prices_dict[ticker].loc[signal_df.index].pct_change()
        
        returns_df = pd.DataFrame(individual_returns, index=signal_df.index)
        
        # Portfolio returns = weighted sum of individual returns
        gross_returns = (signal_df * returns_df).sum(axis=1)
        
        # Compute transaction costs
        costs = signal_df.diff().abs().sum(axis=1) * (cost_bps / 10000)
        
        # Net returns
        net_returns = gross_returns - costs
        net_returns = net_returns.fillna(0)
        
        # Equity curve
        equity_curve = (1 + net_returns).cumprod() * capital
        
        # Calculate metrics
        metrics = summary(net_returns, equity_curve)
        
        folds.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_period': f'{current_start.strftime("%Y-%m-%d")} to {train_end.strftime("%Y-%m-%d")}',
            'test_period': f'{train_end.strftime("%Y-%m-%d")} to {test_end.strftime("%Y-%m-%d")}',
            'returns': net_returns,
            'equity_curve': equity_curve,
            'signal': signal_df,
            'sharpe': metrics.get('sharpe_ratio', 0),
            'cagr': metrics.get('annualized_return', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'calmar': metrics.get('calmar_ratio', 0),
            'volatility': metrics.get('volatility', 0),
        })
        
        # Move to next fold
        current_start = current_start + pd.Timedelta(days=step_days)
    
    # Aggregate results
    if len(folds) == 0:
        return {'folds': [], 'combined_cagr': 0, 'combined_sharpe': 0, 'combined_max_drawdown': 0}
    
    # Combine all returns
    all_returns = pd.concat([f['returns'] for f in folds])
    all_returns_grouped = all_returns.groupby(level=0)
    combined_returns = all_returns_grouped.mean()
    
    # Combined metrics
    combined_equity = (1 + combined_returns).cumprod()
    combined_metrics = summary(combined_returns, combined_equity)
    combined_cagr = combined_metrics.get('annualized_return', 0)
    combined_sharpe = combined_metrics.get('sharpe_ratio', 0)
    combined_max_dd = combined_metrics.get('max_drawdown', 0)
    
    return {
        'folds': folds,
        'combined_cagr': combined_cagr,
        'combined_sharpe': combined_sharpe,
        'combined_max_drawdown': combined_max_dd,
        'combined_returns': combined_returns,
        'combined_equity': combined_equity,
    }


def main():
    print("=" * 80)
    print("PHASE 9: EARNINGS MOMENTUM (POST-EARNINGS-ANNOUNCEMENT DRIFT)")
    print("=" * 80)
    print()
    
    # Load earnings surprises
    print("Loading earnings surprises...")
    earnings_df = load_earnings_surprises()
    print(f"  Total surprises: {len(earnings_df)}")
    print(f"  Date range: {earnings_df['date'].min().strftime('%Y-%m-%d')} to {earnings_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Tickers: {earnings_df['ticker'].nunique()}")
    print()
    
    # Load prices
    print("Loading price data...")
    tickers = earnings_df['ticker'].unique()
    prices_dict = {}
    for ticker in tickers:
        prices = load_ohlcv(ticker, start='2015-01-01')
        prices_dict[ticker] = prices['Close']
    print(f"  Loaded {len(tickers)} tickers")
    print(f"  Price range: {min(p.index.min() for p in prices_dict.values()).strftime('%Y-%m-%d')} to {max(p.index.max() for p in prices_dict.values()).strftime('%Y-%m-%d')}")
    print()
    
    # Run walk-forward
    print("Running walk-forward backtest (train=3y, test=1y, step=1y)...")
    print("(This tests if earnings momentum predictive power persists out-of-sample)")
    print()
    
    results = run_earnings_momentum_walk_forward(
        prices_dict=prices_dict,
        earnings_df=earnings_df,
        train_years=3,
        test_years=1,
        step_years=1,
        cost_bps=7,  # 5 bps cost + 2 bps slippage
        capital=1_000_000,
    )
    
    if len(results['folds']) == 0:
        print("ERROR: No folds generated. Check date ranges.")
        return
    
    print("=" * 80)
    print("WALK-FORWARD RESULTS")
    print("=" * 80)
    print()
    
    for i, fold in enumerate(results['folds']):
        print(f"Fold {i+1}:")
        print(f"  Test Period: {fold['test_period']}")
        print(f"  CAGR: {fold['cagr']*100:.2f}%")
        print(f"  Sharpe: {fold['sharpe']:.3f}")
        print(f"  Max DD: {fold['max_drawdown']*100:.2f}%")
        print(f"  Calmar: {fold['calmar']:.3f}")
        print()
    
    print("-" * 80)
    print("COMBINED RESULTS")
    print("-" * 80)
    print(f"  Combined CAGR: {results['combined_cagr']*100:.2f}%")
    print(f"  Combined Sharpe: {results['combined_sharpe']:.3f}")
    print(f"  Combined Max DD: {results['combined_max_drawdown']*100:.2f}%")
    print()
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save fold-level returns
    all_returns = []
    for i, fold in enumerate(results['folds']):
        fold_returns = fold['returns'].copy()
        fold_returns.name = f'fold_{i+1}'
        all_returns.append(fold_returns)
    
    combined_returns_df = pd.concat(all_returns, axis=1)
    combined_returns_df.to_csv(results_dir / "earnings_momentum_returns.csv")
    print(f"Saved returns to: results/earnings_momentum_returns.csv")
    
    # Save equity curves
    equity_curves = []
    for i, fold in enumerate(results['folds']):
        eq = fold['equity_curve'].copy()
        eq.name = f'fold_{i+1}'
        equity_curves.append(eq)
    
    combined_equity_df = pd.concat(equity_curves, axis=1)
    combined_equity_df.to_csv(results_dir / "earnings_momentum_equity.csv")
    print(f"Saved equity curves to: results/earnings_momentum_equity.csv")
    
    # Save summary
    summary_data = {
        'metric': ['combined_cagr', 'combined_sharpe', 'combined_max_drawdown', 'num_folds'],
        'value': [
            results['combined_cagr'],
            results['combined_sharpe'],
            results['combined_max_drawdown'],
            len(results['folds'])
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "earnings_momentum_summary.csv", index=False)
    print(f"Saved summary to: results/earnings_momentum_summary.csv")
    print()
    
    print("=" * 80)
    print("PHASE 9 COMPLETE")
    print("=" * 80)
    print()
    print("Next step: Run noise-check permutation test to assess statistical significance")
    print()
    
    return results


if __name__ == "__main__":
    results = main()
