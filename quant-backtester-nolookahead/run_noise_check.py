"""Run noise-check permutation test on 15-ticker basket."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loader import load_ohlcv
from signals.signals import moving_average_crossover
from backtest.noise_check import permutation_null_test, structural_comparison, SECTOR_MAP
from backtest.multi_ticker import DEFAULT_NIFTY_BASKET

def main():
    print("=" * 80)
    print("NOISE-CHECK PERMUTATION TEST: 15 NIFTY TICKERS")
    print("Question: Is 7/15 positive distinguishable from chance?")
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
    
    # Run permutation null test
    print("=" * 80)
    print("PART A: PERMUTATION NULL TEST")
    print("=" * 80)
    print()
    
    results = permutation_null_test(
        prices_dict,
        lambda p: moving_average_crossover(p, fast=50, slow=200),
        engine_kwargs={"cost_bps": 5, "slippage_bps": 2},
        n_permutations=500,
        train_years=3,
        test_years=1,
        step_years=1,
        seed=42,
    )
    
    print()
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print(f"Observed: {results['observed_positive_count']}/15 tickers with positive mean Sharpe")
    print(f"Null expectation: {results['null_mean']:.1f}/15 (std={results['null_std']:.1f})")
    print(f"P-value: {results['p_value']:.3f}")
    print()
    
    # Direct answer
    p_value = results["p_value"]
    
    if p_value >= 0.10:
        print("DIRECT ANSWER:")
        print("-" * 80)
        print("This split is CONSISTENT WITH PURE NOISE given only 15 tickers.")
        print(f"A p-value of {p_value:.3f} means that observing 7+ positive tickers out of 15")
        print("happens frequently under random signal timing with these joint price dynamics.")
        print("There is no statistical evidence that the 7 'winning' tickers reflect something")
        print("real about the strategy — this is what chance produces.")
        print()
        print("NO FURTHER STRUCTURAL ANALYSIS IS WARRANTED.")
        print("Any post-hoc sector/story explanation would be apophenia — pattern-seeking")
        print("in randomness. The honest answer is: we got unlucky with RELIANCE in Phase 4,")
        print("but the cross-ticker result is essentially flat (mean Sharpe 0.04), and the")
        print("7/15 split is noise.")
        print("-" * 80)
    else:
        print("DIRECT ANSWER:")
        print("-" * 80)
        print(f"The split is UNLIKELY TO BE PURE NOISE (p={p_value:.3f}).")
        print("There is statistical evidence that some tickers genuinely work better with")
        print("this strategy than others — beyond what chance would produce.")
        print()
        print("Proceeding to Part B: Structural Comparison...")
        print("-" * 80)
        print()
        
        # Part B: Structural comparison
        print("=" * 80)
        print("PART B: STRUCTURAL COMPARISON")
        print("=" * 80)
        print()
        
        # Build ticker results DataFrame
        ticker_results_list = []
        for ticker in prices_dict.keys():
            ticker_results_list.append({
                "ticker": ticker,
                "mean_sharpe": results["real_results"].get(ticker, {}).get("mean_sharpe", np.nan),
            })
        
        ticker_results_df = pd.DataFrame(ticker_results_list)
        
        # Run structural comparison
        struct_df = structural_comparison(ticker_results_df)
        
        print(struct_df.to_string(index=False))
        print()
        print()
        print("CAVEAT (restated): With n=15, this table is DESCRIPTIVE ONLY.")
        print("It is hypothesis-generating, NOT statistically confirmatory.")
        print("No correlation coefficient or regression is computed — that would be")
        print("a second layer of overfitting on top of the first.")
        print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()
