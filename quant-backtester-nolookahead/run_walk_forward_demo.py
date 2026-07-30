"""Run walk-forward backtest on real data."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loader import load_ohlcv, clean_ohlcv
from signals.signals import moving_average_crossover
from backtest.walk_forward import run_walk_forward, aggregate_walk_forward_results

def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION: RELIANCE.NS (NSE)")
    print("Strategy: Moving Average Crossover (fast=50, slow=200)")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    prices_df = load_ohlcv("RELIANCE.NS", start="2015-01-01")
    prices = prices_df["Close"]
    print(f"Loaded {len(prices)} days of data from {prices.index.min().date()} to {prices.index.max().date()}")
    print()
    
    # Run walk-forward
    print("Running walk-forward backtest (train=3y, test=1y, step=1y)...")
    print()
    
    results = run_walk_forward(
        prices,
        lambda p: moving_average_crossover(p, fast=50, slow=200),
        engine_kwargs={"cost_bps": 5, "slippage_bps": 2},
        train_years=3,
        test_years=1,
        step_years=1,
    )
    
    # Display fold-by-fold results
    print("-" * 70)
    print("FOLD-BY-FOLD RESULTS")
    print("-" * 70)
    
    if len(results) == 0:
        print("No folds generated (insufficient data)")
        return
    
    # Display each fold
    display_cols = ["fold", "test_start", "test_end", "sharpe_ratio", 
                    "max_drawdown", "calmar_ratio", "win_rate", 
                    "total_return", "trades"]
    
    display_df = results[display_cols].copy()
    display_df["test_start"] = display_df["test_start"].dt.strftime("%Y-%m-%d")
    display_df["test_end"] = display_df["test_end"].dt.strftime("%Y-%m-%d")
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["calmar_ratio"] = display_df["calmar_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["total_return"] = display_df["total_return"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    
    print(display_df.to_string(index=False))
    print()
    
    # Aggregate results
    print("-" * 70)
    print("AGGREGATED OUT-OF-SAMPLE RESULTS")
    print("-" * 70)
    
    aggregated = aggregate_walk_forward_results(results)
    
    print(f"Total folds:              {aggregated['total_folds']}")
    print(f"Profitable folds:         {aggregated['profitable_folds']} ({aggregated['profitable_folds']/aggregated['total_folds']*100:.0f}%)")
    print()
    print(f"Overall Sharpe Ratio:     {aggregated['overall_sharpe']:.2f}")
    print(f"Overall Max Drawdown:     {aggregated['overall_max_drawdown']:.1%}")
    print(f"Overall Calmar Ratio:     {aggregated['overall_calmar']:.2f}")
    print(f"Overall Total Return:     {aggregated['overall_total_return']:.1%}")
    print()
    print(f"Fold Sharpe Std Dev:      {aggregated['fold_sharpe_std']:.2f}")
    print(f"Fold Returns:             {[f'{r:.1%}' if pd.notna(r) else 'NaN' for r in aggregated['fold_returns']]}")
    print()
    
    # Show combined equity curve summary
    if len(aggregated['combined_equity_curve']) > 0:
        combined_equity = aggregated['combined_equity_curve']
        print("Combined Out-of-Sample Equity Curve:")
        print(f"  Start:                  1.00")
        print(f"  End:                    {combined_equity.iloc[-1]:.2f}")
        print(f"  Peak:                   {combined_equity.max():.2f}")
        print(f"  Trough:                 {combined_equity.min():.2f}")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
