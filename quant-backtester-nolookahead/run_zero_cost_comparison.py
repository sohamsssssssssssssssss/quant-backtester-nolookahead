"""Run zero-cost walk-forward on 15-ticker basket for comparison."""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loader import load_ohlcv
from signals.signals import moving_average_crossover
from backtest.multi_ticker import run_multi_ticker_walk_forward, aggregate_across_tickers, DEFAULT_NIFTY_BASKET

def main():
    print("=" * 80)
    print("ZERO-COST BASELINE: MA Crossover 50/200 (15 NIFTY tickers)")
    print("Parameters: cost_bps=0, slippage_bps=0")
    print("=" * 80)
    print()
    
    # Load price data for all 15 tickers
    prices_dict = {}
    for ticker in DEFAULT_NIFTY_BASKET:
        try:
            prices_df = load_ohlcv(ticker, start="2015-01-01")
            prices_dict[ticker] = prices_df["Close"]
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")
    
    print(f"Loaded {len(prices_dict)} tickers")
    print()
    
    # Run walk-forward with ZERO costs
    results = run_multi_ticker_walk_forward(
        prices_dict,
        lambda p: moving_average_crossover(p, fast=50, slow=200),
        engine_kwargs={"cost_bps": 0, "slippage_bps": 0},  # ZERO COST
        train_years=3,
        test_years=1,
        step_years=1,
    )
    
    # Aggregate results
    aggregated = aggregate_across_tickers(results)
    
    print("=" * 80)
    print("ZERO-COST RESULTS (Gross Returns)")
    print("=" * 80)
    print()
    
    print("Per-Ticker Summary (sorted by mean Sharpe):")
    print("-" * 80)
    
    per_ticker = aggregated["per_ticker_summary"]
    display_df = per_ticker[["ticker", "mean_sharpe", "mean_max_dd", "mean_calmar", 
                             "pct_profitable_folds", "total_folds"]].copy()
    display_df["mean_sharpe"] = display_df["mean_sharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["mean_max_dd"] = display_df["mean_max_dd"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["mean_calmar"] = display_df["mean_calmar"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["pct_profitable_folds"] = display_df["pct_profitable_folds"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "NaN")
    
    print(display_df.to_string(index=False))
    print()
    
    print("Cross-Ticker Aggregate:")
    print("-" * 80)
    cross = aggregated["cross_ticker_summary"]
    print(f"  Total tickers:                {cross['total_tickers']}")
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
    print("  Profitable tickers:           {}/{}".format(
        len([t for t, r in aggregated["real_results"].items() if r.get("mean_sharpe", 0) > 0]),
        len(aggregated["real_results"])
    ))
    print("=" * 80)
    
    # Print comparison summary
    print()
    print("COMPARISON: Zero-Cost vs With-Cost (Phase 5)")
    print("-" * 80)
    print(f"  Zero-cost Mean Sharpe:  {cross['mean_of_mean_sharpe']:.2f}")
    print(f"  With-cost Mean Sharpe:  0.04")
    print(f"  Difference:             {cross['mean_of_mean_sharpe'] - 0.04:.2f}")
    print()
    print(f"  Zero-cost profitable folds: {cross['pct_profitable_all']*100:.1f}%")
    print(f"  With-cost profitable folds: 43.3%")
    print("=" * 80)
    
    return aggregated

if __name__ == "__main__":
    main()
