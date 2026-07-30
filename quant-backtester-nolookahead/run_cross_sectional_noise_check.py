"""Noise check for cross-sectional momentum strategy using rank permutation."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loader import load_ohlcv
from backtest.cross_sectional_walk_forward import (
    run_cross_sectional_walk_forward,
    aggregate_cross_sectional_results,
)
from backtest.multi_ticker import DEFAULT_NIFTY_BASKET


def permute_ranks_at_each_date(ranks_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Randomly permute ranks at each date to destroy any signal.
    
    For each date, shuffle the rank values across tickers. This destroys
    the cross-sectional ranking signal while preserving the rank structure
    (i.e., each date still has ranks 1 to N across tickers).
    
    Parameters
    ----------
    ranks_df : pd.DataFrame
        Original ranks DataFrame (dates x tickers)
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    pd.DataFrame
        Permuted ranks with same index and columns
    """
    permuted_ranks = ranks_df.copy()
    
    for date in ranks_df.index:
        valid_mask = ranks_df.loc[date].notna()
        valid_tickers = valid_mask[valid_mask].index.tolist()
        
        if len(valid_tickers) > 1:
            # Get the ranks for valid tickers
            ranks_at_date = ranks_df.loc[date, valid_tickers].values.copy()
            
            # Randomly permute the ranks
            rng.shuffle(ranks_at_date)
            
            # Assign back
            permuted_ranks.loc[date, valid_tickers] = ranks_at_date
    
    return permuted_ranks


def run_cross_sectional_noise_check(
    prices_dict: Dict[str, pd.Series],
    top_n: int = 5,
    bottom_n: int = 5,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    n_permutations: int = 500,
) -> Tuple[float, float, List[float]]:
    """
    Run noise check for cross-sectional momentum strategy.
    
    Permutes the cross-sectional ranks at each date to destroy the signal,
    then re-runs the backtest to build a null distribution of Sharpe ratios.
    
    Parameters
    ----------
    prices_dict : Dict[str, pd.Series]
        Price data for each ticker
    top_n : int
        Number of top-ranked tickers to go long
    bottom_n : int
        Number of bottom-ranked tickers to go short
    cost_bps : float
        Transaction cost in basis points
    slippage_bps : float
        Slippage in basis points
    train_years : int
        Training period in years
    test_years : int
        Test period in years
    step_years : int
        Step between folds in years
    n_permutations : int
        Number of permutations for noise check
    
    Returns
    -------
    Tuple[float, float, List[float]]
        Actual Sharpe ratio, p-value, list of permuted Sharpe ratios
    """
    print("=" * 80)
    print("CROSS-SECTIONAL MOMENTUM: NOISE CHECK (RANK PERMUTATION)")
    print("=" * 80)
    print()
    
    # Step 1: Run actual strategy
    print("Step 1: Running actual strategy...")
    actual_results = run_cross_sectional_walk_forward(
        prices_dict,
        top_n=top_n,
        bottom_n=bottom_n,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
    )
    
    if len(actual_results) == 0:
        raise ValueError("No results from actual strategy")
    
    actual_agg = aggregate_cross_sectional_results(actual_results)
    actual_sharpe = actual_agg['overall_sharpe']
    actual_max_dd = actual_agg['overall_max_drawdown']
    actual_return = actual_agg['overall_total_return']
    
    print(f"  Actual Sharpe Ratio:    {actual_sharpe:.3f}")
    print(f"  Actual Max Drawdown:    {actual_max_dd:.1%}")
    print(f"  Actual Total Return:    {actual_return:.1%}")
    print()
    
    # Step 2: Run permutations
    print(f"Step 2: Running {n_permutations} rank permutations...")
    print()
    
    from signals.cross_sectional_momentum import rank_momentum
    from backtest.cross_sectional_engine import CrossSectionalBacktestEngine
    
    # Get common index
    all_dates = None
    for ticker, prices in prices_dict.items():
        if all_dates is None:
            all_dates = prices.index
        else:
            all_dates = all_dates.intersection(prices.index)
    
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    folds = _generate_folds(all_dates, train_years, test_years, step_years)
    
    rng = np.random.default_rng(42)
    permuted_sharpes = []
    
    for perm_idx in range(n_permutations):
        all_permuted_returns = []
        
        for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
            # Find test_start
            train_end_idx = all_dates.get_loc(train_end)
            if train_end_idx + 1 >= len(all_dates):
                continue
            test_start = all_dates[train_end_idx + 1]
            
            # Filter prices to fold window
            fold_mask = (all_dates >= train_start) & (all_dates <= test_end)
            fold_dates = all_dates[fold_mask]
            
            prices_fold = {
                ticker: prices.loc[prices.index.intersection(fold_dates)]
                for ticker, prices in prices_dict.items()
            }
            
            # Compute ranks and PERMUTE them
            ranks_df = rank_momentum(prices_fold)
            permuted_ranks = permute_ranks_at_each_date(ranks_df, rng)
            
            # Run backtest with permuted ranks
            engine = CrossSectionalBacktestEngine(
                top_n=top_n,
                bottom_n=bottom_n,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
            )
            
            try:
                result = engine.run(prices_fold, permuted_ranks)
                # Extract test period returns
                test_mask = (result.returns.index >= test_start) & (result.returns.index <= test_end)
                test_returns = result.returns.loc[test_mask]
                all_permuted_returns.extend(test_returns.dropna().tolist())
            except Exception:
                continue
        
        if len(all_permuted_returns) > 0:
            perm_returns = pd.Series(all_permuted_returns)
            perm_sharpe = (perm_returns.mean() / perm_returns.std()) * np.sqrt(252)
        else:
            perm_sharpe = 0.0
        
        permuted_sharpes.append(perm_sharpe)
        
        if (perm_idx + 1) % 100 == 0:
            print(f"  Permutation {perm_idx + 1}/{n_permutations}...")
    
    print()
    
    # Step 3: Compute p-value
    permuted_sharpes = np.array(permuted_sharpes)
    
    if actual_sharpe >= 0:
        # One-sided test: how often does noise produce Sharpe >= actual?
        p_value = (permuted_sharpes >= actual_sharpe).mean()
    else:
        # For negative Sharpe, test how often noise produces Sharpe <= actual
        p_value = (permuted_sharpes <= actual_sharpe).mean()
    
    print("=" * 80)
    print("NOISE CHECK RESULTS")
    print("=" * 80)
    print()
    print(f"Actual Sharpe Ratio:     {actual_sharpe:.3f}")
    print(f"Permuted Sharpe (mean):  {permuted_sharpes.mean():.3f}")
    print(f"Permuted Sharpe (std):   {permuted_sharpes.std():.3f}")
    print(f"Permuted Sharpe (min):   {permuted_sharpes.min():.3f}")
    print(f"Permuted Sharpe (max):   {permuted_sharpes.max():.3f}")
    print()
    print(f"P-VALUE:                 {p_value:.4f}")
    print()
    
    if p_value < 0.01:
        print("INTERPRETATION: Strategy performance is statistically significant (p < 0.01)")
        print("                Unlikely to be explained by random noise alone.")
    elif p_value < 0.05:
        print("INTERPRETATION: Strategy performance is moderately significant (p < 0.05)")
        print("                Some evidence against pure noise explanation.")
    else:
        print("INTERPRETATION: Strategy performance is NOT statistically significant (p >= 0.05)")
        print("                Performance could plausibly arise from random noise.")
    
    print()
    print("=" * 80)
    
    return actual_sharpe, p_value, permuted_sharpes.tolist()


def _generate_folds(
    all_dates: pd.DatetimeIndex,
    train_years: int,
    test_years: int,
    step_years: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward folds."""
    folds = []
    min_dates = all_dates.min()
    max_dates = all_dates.max()
    
    train_days = int(train_years * 252)
    test_days = int(test_years * 252)
    step_days = int(step_years * 252)
    
    # Start from earliest date that allows full train+test window
    current_start = min_dates
    
    while True:
        # Find train_end index
        try:
            # Use business day offset
            train_end = current_start + pd.Timedelta(days=train_days)
            if train_end >= max_dates:
                break
            
            # Find actual closest date in index
            train_end_idx = all_dates.searchsorted(train_end)
            if train_end_idx >= len(all_dates):
                break
            train_end = all_dates[train_end_idx - 1]
            
            # Find test_end
            test_end_target = train_end + pd.Timedelta(days=test_days)
            test_end_idx = all_dates.searchsorted(test_end_target)
            if test_end_idx >= len(all_dates):
                break
            test_end = all_dates[test_end_idx - 1]
            
            folds.append((current_start, train_end, test_end))
            
            # Move start forward
            current_start = current_start + pd.Timedelta(days=step_days)
            if current_start >= max_dates:
                break
                
        except Exception:
            break
    
    return folds


def main():
    # Load price data for all 15 tickers
    print("Loading price data...")
    prices_dict = {}
    
    for ticker in DEFAULT_NIFTY_BASKET:
        try:
            prices_df = load_ohlcv(ticker, start="2015-01-01")
            prices_dict[ticker] = prices_df["Close"]
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")
    
    if len(prices_dict) == 0:
        print("ERROR: No tickers loaded")
        return
    
    print(f"Loaded {len(prices_dict)} tickers")
    print()
    
    # Run noise check with 500 permutations
    actual_sharpe, p_value, perm_sharpes = run_cross_sectional_noise_check(
        prices_dict,
        top_n=5,
        bottom_n=5,
        cost_bps=5,
        slippage_bps=2,
        train_years=3,
        test_years=1,
        step_years=1,
        n_permutations=500,
    )


if __name__ == "__main__":
    main()
