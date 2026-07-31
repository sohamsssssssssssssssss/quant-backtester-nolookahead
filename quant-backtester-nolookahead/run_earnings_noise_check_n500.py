"""N=500 permutation noise-check for earnings momentum strategy."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_ohlcv
from src.signals.earnings_momentum import load_earnings_surprises, earnings_momentum_signal

print("="*80)
print("EARNINGS MOMENTUM: N=500 PERMUTATION NOISE-CHECK")
print("="*80)
print()

# Load data
print("Loading data...")
earnings_df = load_earnings_surprises()
tickers = earnings_df['ticker'].unique()
prices_dict = {}
for ticker in tickers:
    prices = load_ohlcv(ticker, start='2020-01-01')
    prices_dict[ticker] = prices['Close']
print(f"  Tickers: {len(tickers)}")
print(f"  Earnings surprises: {len(earnings_df)}")
print()

# Generate actual signal
print("Generating actual signal...")
actual_signal = earnings_momentum_signal(prices_dict, earnings_df, top_n=5, bottom_n=5, hold_days=60).fillna(0)

# Calculate returns
all_dates = actual_signal.index
individual_returns = {}
for ticker in prices_dict.keys():
    individual_returns[ticker] = prices_dict[ticker].loc[all_dates].pct_change()
returns_df = pd.DataFrame(individual_returns, index=all_dates)

gross_returns = (actual_signal * returns_df).sum(axis=1)
costs = actual_signal.diff().abs().sum(axis=1) * (7 / 10000)
actual_returns = (gross_returns - costs).fillna(0)

# Calculate actual Sharpe
actual_sharpe = actual_returns.mean() / actual_returns.std() * np.sqrt(252) if actual_returns.std() > 0 else 0
print(f"Actual Sharpe: {actual_sharpe:.3f}")
print()

# Permutation test with n=500 in batches
n_total = 500
batch_size = 100
n_batches = n_total // batch_size

print(f"Running {n_total} permutations in {n_batches} batches of {batch_size}...")
print()

all_perm_sharpes = []
batch_results = []

for batch_num in range(n_batches):
    batch_start = batch_num * batch_size + 1
    batch_end = (batch_num + 1) * batch_size
    
    batch_sharpes = []
    
    for _ in range(batch_size):
        # Shuffle ticker labels in earnings data
        perm_earnings = earnings_df.copy()
        perm_earnings['ticker'] = np.random.permutation(perm_earnings['ticker'].values)
        
        # Generate signal with shuffled tickers
        perm_signal = earnings_momentum_signal(prices_dict, perm_earnings, top_n=5, bottom_n=5, hold_days=60).fillna(0)
        
        # Calculate returns
        gross_perm = (perm_signal * returns_df).sum(axis=1)
        costs_perm = perm_signal.diff().abs().sum(axis=1) * (7 / 10000)
        perm_ret = (gross_perm - costs_perm).fillna(0)
        
        # Calculate Sharpe
        perm_sharpe = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
        batch_sharpes.append(perm_sharpe)
    
    all_perm_sharpes.extend(batch_sharpes)
    
    # Calculate current p-value trajectory
    current_pvalue = np.mean([s >= actual_sharpe for s in all_perm_sharpes])
    current_mean = np.mean(all_perm_sharpes)
    current_std = np.std(all_perm_sharpes)
    count_above = sum(s >= actual_sharpe for s in all_perm_sharpes)
    
    batch_results.append({
        'batch': batch_num + 1,
        'permutations_processed': batch_end,
        'cumulative_mean_sharpe': current_mean,
        'cumulative_std_sharpe': current_std,
        'cumulative_pvalue': current_pvalue,
        'count_above_actual': count_above
    })
    
    print(f"Batch {batch_num + 1}/{n_batches} complete")
    print(f"  Permutations so far: {len(all_perm_sharpes)}")
    print(f"  Permuted Sharpe: mean={current_mean:.3f}, std={current_std:.3f}")
    print(f"  P-value trajectory: {current_pvalue:.3f} ({count_above}/{len(all_perm_sharpes)} above actual)")
    print()

# Final results
final_pvalue = np.mean([s >= actual_sharpe for s in all_perm_sharpes])
final_mean = np.mean(all_perm_sharpes)
final_std = np.std(all_perm_sharpes)
count_above = sum(s >= actual_sharpe for s in all_perm_sharpes)

print("="*80)
print("FINAL RESULTS")
print("="*80)
print()
print(f"Actual Sharpe:           {actual_sharpe:.3f}")
print(f"Permuted Sharpe (n=500): mean={final_mean:.3f}, std={final_std:.3f}")
print(f"P-value (n=500):         {final_pvalue:.3f} ({count_above}/500 above actual)")
print()

if final_pvalue < 0.05:
    print("RESULT: Statistically significant (p < 0.05)")
    print("  The earnings momentum signal is distinguishable from random noise.")
else:
    print("RESULT: NOT statistically significant (p >= 0.05)")
    print("  The earnings momentum signal is indistinguishable from random noise.")
print()

# Save results
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

# Save full permutation distribution
perm_df = pd.DataFrame({
    'permutation': range(1, n_total + 1),
    'sharpe': all_perm_sharpes,
    'batch': ([1]*batch_size + [2]*batch_size + [3]*batch_size + [4]*batch_size + [5]*batch_size)
})
perm_df.to_csv(results_dir / "earnings_momentum_permutations_n500.csv", index=False)
print(f"Saved full permutation results: results/earnings_momentum_permutations_n500.csv")

# Save batch summary
batch_summary_df = pd.DataFrame(batch_results)
batch_summary_df.to_csv(results_dir / "earnings_momentum_noise_batch_trajectory.csv", index=False)
print(f"Saved batch trajectory: results/earnings_momentum_noise_batch_trajectory.csv")

# Save summary
summary_data = {
    'metric': ['actual_sharpe', 'perm_mean', 'perm_std', 'p_value', 'n_permutations'],
    'value': [actual_sharpe, final_mean, final_std, final_pvalue, n_total]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(results_dir / "earnings_momentum_noise_summary.csv", index=False)
print(f"Saved summary: results/earnings_momentum_noise_summary.csv")

print()
print("="*80)
print("NOISE-CHECK COMPLETE")
print("="*80)
