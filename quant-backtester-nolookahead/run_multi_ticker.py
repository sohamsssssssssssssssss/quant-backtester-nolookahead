"""Run multi-ticker walk-forward backtest on NIFTY 50 basket."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from signals.signals import moving_average_crossover
from backtest.multi_ticker import (
    run_multi_ticker_walk_forward,
    aggregate_across_tickers,
    DEFAULT_NIFTY_BASKET,
)

def main():
    print("=" * 80)
    print("MULTI-TICKER WALK-FORWARD VALIDATION: NIFTY 50 BASKET")
    print("Strategy: Moving Average Crossover (fast=50, slow=200)")
    print("=" * 80)
    print()
    
    # Use default NIFTY basket
    tickers = DEFAULT_NIFTY_BASKET
    print(f"Ticker basket ({len(tickers)} stocks):")
    for i, t in enumerate(tickers):
        print(f"  {i+1}. {t}")
    print()
    
    # Run multi-ticker walk-forward
    print("Running walk-forward backtest (train=3y, test=1y, step=1y)...")
    print("(This may take a few minutes as it downloads data for each ticker)")
    print()
    
    results = run_multi_ticker_walk_forward(
        tickers,
        lambda p: moving_average_crossover(p, fast=50, slow=200),
        engine_kwargs={"cost_bps": 5, "slippage_bps": 2},
        train_years=3,
        test_years=1,
        step_years=1,
    )
    
    if len(results) == 0:
        print("ERROR: No results generated")
        return
    
    print()
    print("=" * 80)
    print("PER-TICKER, PER-FOLD RESULTS")
    print("=" * 80)
    
    # Display full results table
    display_cols = ["ticker", "fold", "test_start", "test_end", "sharpe_ratio",
                    "max_drawdown", "calmar_ratio", "win_rate", "total_return", "trades"]
    
    display_df = results[display_cols].copy()
    
    # Convert to string format for display
    for col in ["test_start", "test_end"]:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
        else:
            display_df[col] = display_df[col].apply(lambda x: str(x)[:10] if pd.notna(x) else "NaN")
    
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["calmar_ratio"] = display_df["calmar_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["total_return"] = display_df["total_return"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    
    print(display_df.to_string(index=False))
    print()
    
    # Aggregate across tickers
    print("=" * 80)
    print("CROSS-TICKER AGGREGATE SUMMARY")
    print("=" * 80)
    
    aggregated = aggregate_across_tickers(results)
    
    per_ticker = aggregated["per_ticker_summary"]
    cross = aggregated["cross_ticker_summary"]
    
    # Per-ticker summary (sorted by Sharpe)
    print()
    print("Per-Ticker Summary (sorted by mean Sharpe):")
    print("-" * 80)
    
    per_ticker_display = per_ticker[
        ["ticker", "mean_sharpe", "mean_max_dd", "mean_calmar", 
         "pct_profitable_folds", "total_folds"]
    ].copy()
    per_ticker_display["mean_sharpe"] = per_ticker_display["mean_sharpe"].apply(lambda x: f"{x:.2f}")
    per_ticker_display["mean_max_dd"] = per_ticker_display["mean_max_dd"].apply(lambda x: f"{x:.1%}")
    per_ticker_display["mean_calmar"] = per_ticker_display["mean_calmar"].apply(lambda x: f"{x:.2f}")
    per_ticker_display["pct_profitable_folds"] = per_ticker_display["pct_profitable_folds"].apply(lambda x: f"{x:.0%}")
    
    print(per_ticker_display.to_string(index=False))
    print()
    
    # Cross-ticker summary
    print("Cross-Ticker Aggregate:")
    print("-" * 80)
    print(f"  Total tickers analyzed:       {cross['total_tickers']}")
    print(f"  Total folds (all tickers):    {cross['total_folds_all']}")
    print(f"  Profitable folds:             {cross['profitable_folds_all']} ({cross['pct_profitable_all']*100:.1f}%)")
    print()
    print(f"  Mean of Mean Sharpe:          {cross['mean_of_mean_sharpe']:.2f}")
    print(f"  Std of Mean Sharpe:           {cross['std_of_mean_sharpe']:.2f}")
    print(f"  Mean of Mean Max DD:          {cross['mean_of_mean_max_dd']:.1%}")
    print(f"  Mean of Mean Calmar:          {cross['mean_of_mean_calmar']:.2f}")
    print(f"  Mean of Mean Total Return:    {cross['mean_of_mean_return']:.1%}")
    print()
    print(f"  Best ticker (by Sharpe):      {cross['best_ticker']}")
    print(f"  Worst ticker (by Sharpe):     {cross['worst_ticker']}")
    print()
    print(f"  FAILURE PATTERN:              {cross['failure_pattern'].upper()}")
    print()
    print(f"  ASSESSMENT:")
    print(f"  {cross['failure_description']}")
    print()
    print("=" * 80)
    
    # Return the key conclusion for programmatic use
    return aggregated

if __name__ == "__main__":
    main()
