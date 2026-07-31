"""Run noise-check permutation test for earnings momentum."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.loader import load_ohlcv
from src.signals.earnings_momentum import load_earnings_surprises, earnings_momentum_signal

print("="*80)
print("PHASE 9: EARNINGS MOMENTUM NOISE-CHECK PERMUTATION TEST")
print("="*80)
print()

# Load data
earnings_df = load_earnings_surprises()
tickers = earnings_df['ticker'].unique()
print(f"Tickers: {len(tickers)}")
print(f"Earnings surprises: {len(earnings_df)}")

prices_dict = {}
for ticker in tickers:
    prices = load_ohlcv(ticker, start='2020-01-01')
    prices_dict[ticker] = prices['Close']
print(f"Price data loaded: {len(prices_dict)} tickers")
print()

# Generate actual signal
print("Generating actual signal...")
actual_signal = earnings_momentum_signal(prices_dict, earnings_df, top_n=5, bottom_n=5, hold_days=60).fillna(0)

# Calculate actual returns
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

# Permutation test
n_permutations = 10
print(f"\nRunning {n_permutations} permutations (this may take a few minutes)...")

perm_sharpes = []
for i in range(n_permutations):
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
    perm_sharpes.append(perm_sharpe)
    
    if (i + 1) % 20 == 0:
        print(f"  Completed {i+1}/{n_permutations} permutations...")

# P-value
p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
perm_mean = np.mean(perm_sharpes)
perm_std = np.std(perm_sharpes)

print()
print("="*80)
print("NOISE-CHECK RESULTS")
print("="*80)
print()
print(f"Actual Sharpe:      {actual_sharpe:.3f}")
print(f"Permuted Sharpe:    mean={perm_mean:.3f}, std={perm_std:.3f}")
print(f"P-value:            {p_value:.3f}")
print()

if p_value < 0.05:
    print("RESULT: Statistically significant (p < 0.05)")
    print("  The signal is distinguishable from random noise.")
else:
    print("RESULT: NOT statistically significant (p >= 0.05)")
    print("  The signal is indistinguishable from random noise.")

print()

# Save results
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

# Save permutation distribution
perm_df = pd.DataFrame({
    'permutation': range(1, n_permutations + 1),
    'sharpe': perm_sharpes,
})
perm_df.to_csv(results_dir / "earnings_momentum_permutations.csv", index=False)
print(f"Saved permutation results to: results/earnings_momentum_permutations.csv")

# Save summary
summary_data = {
    'metric': ['actual_sharpe', 'perm_mean', 'perm_std', 'p_value'],
    'value': [actual_sharpe, perm_mean, perm_std, p_value]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(results_dir / "earnings_momentum_noise_summary.csv", index=False)
print(f"Saved summary to: results/earnings_momentum_noise_summary.csv")

print()
print("="*80)
print("PHASE 9 COMPLETE")
print("="*80)
