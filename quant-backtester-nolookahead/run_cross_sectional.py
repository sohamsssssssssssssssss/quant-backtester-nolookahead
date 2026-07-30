"""Run cross-sectional momentum walk-forward on 15-ticker basket."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loader import load_ohlcv
from backtest.cross_sectional_walk_forward import (
    run_cross_sectional_walk_forward,
    aggregate_cross_sectional_results,
)
from backtest.multi_ticker import DEFAULT_NIFTY_BASKET

def main():
    print("=" * 80)
    print("CROSS-SECTIONAL MOMENTUM WALK-FORWARD: 15 NIFTY TICKERS")
    print("Strategy: Long top 5, Short bottom 5 by 12-1 momentum")
    print("Rebalance: Monthly, Dollar-neutral")
    print("=" * 80)
    print()
    
    # Load price data for all 15 tickers
    print("Loading price data...")
    prices_dict = {}
    
    for ticker in DEFAULT_NIFTY_BASKET:
        try:
            prices_df = load_ohlcv(ticker, start="2015-01-01")
            prices_dict[ticker] = prices_df["Close"]
            print(f"  {ticker}: {len(prices_dict[ticker])} days")
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")
    
    if len(prices_dict) == 0:
        print("ERROR: No tickers loaded")
        return
    
    print(f"\nLoaded {len(prices_dict)} tickers")
    print()
    
    # Run walk-forward
    print("Running walk-forward (train=3y, test=1y, step=1y)...")
    print()
    
    results = run_cross_sectional_walk_forward(
        prices_dict,
        top_n=5,
        bottom_n=5,
        cost_bps=5,
        slippage_bps=2,
        train_years=3,
        test_years=1,
        step_years=1,
    )
    
    if len(results) == 0:
        print("ERROR: No folds generated")
        return
    
    print("=" * 80)
    print("FOLD-BY-FOLD RESULTS")
    print("=" * 80)
    
    display_cols = ["fold", "test_start", "test_end", "sharpe_ratio",
                    "max_drawdown", "calmar_ratio", "total_return", "turnover", "trades"]
    
    display_df = results[display_cols].copy()
    for col in ["test_start", "test_end"]:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
        else:
            display_df[col] = display_df[col].apply(lambda x: str(x)[:10] if pd.notna(x) else "NaN")
    
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["calmar_ratio"] = display_df["calmar_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    display_df["total_return"] = display_df["total_return"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    display_df["turnover"] = display_df["turnover"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
    
    print(display_df.to_string(index=False))
    print()
    
    # Aggregate results
    print("=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    
    aggregated = aggregate_cross_sectional_results(results)
    
    print(f"Total folds:              {aggregated['total_folds']}")
    print(f"Profitable folds:         {aggregated['profitable_folds']} ({aggregated['profitable_folds']/aggregated['total_folds']*100:.0f}%)")
    print()
    print(f"Overall Sharpe Ratio:     {aggregated['overall_sharpe']:.2f}")
    print(f"Overall Max Drawdown:     {aggregated['overall_max_drawdown']:.1%}")
    print(f"Overall Calmar Ratio:     {aggregated['overall_calmar']:.2f}")
    print(f"Overall Total Return:     {aggregated['overall_total_return']:.1%}")
    print()
    
    if len(aggregated['combined_equity_curve']) > 0:
        combined_equity = aggregated['combined_equity_curve']
        print("Combined Equity Curve:")
        print(f"  Start:  1.00")
        print(f"  End:    {combined_equity.iloc[-1]:.2f}")
        print(f"  Peak:   {combined_equity.max():.2f}")
        print(f"  Trough: {combined_equity.min():.2f}")
    
    print()
    print("=" * 80)
    
    return aggregated

if __name__ == "__main__":
    main()
