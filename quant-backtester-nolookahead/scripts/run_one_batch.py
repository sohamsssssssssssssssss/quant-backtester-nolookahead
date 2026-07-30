#!/usr/bin/env python3
"""Run ONE batch of 100 permutations - fastest possible chunk."""
import sys, json
sys.path.insert(0, '/workspace/quant-backtester-nolookahead/src')
import pandas as pd, numpy as np
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

from data.loader import load_ohlcv
from signals.cross_sectional_momentum import rank_momentum
from backtest.cross_sectional_engine import CrossSectionalBacktestEngine
from backtest.multi_ticker import DEFAULT_NIFTY_BASKET

RESULTS_FILE = '/workspace/quant-backtester-nolookahead/results/cross_sectional_permutation_raw.json'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    args = parser.parse_args()
    
    prices_dict = {t: load_ohlcv(t, start='2015-01-01')['Close'] for t in DEFAULT_NIFTY_BASKET}
    ranks_df = rank_momentum(prices_dict)
    engine = CrossSectionalBacktestEngine(top_n=5, bottom_n=5, cost_bps=5, slippage_bps=2)
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        actual_sharpe = all_results['actual_sharpe']
    except:
        result = engine.run(prices_dict, ranks_df)
        actual_returns = result.returns.dropna()
        actual_sharpe = (actual_returns.mean() / actual_returns.std()) * np.sqrt(252)
        all_results = {'actual_sharpe': float(actual_sharpe), 'batches': [], 'metadata': {'target': 2000}}
    
    seed = 42 + args.batch - 1
    rng = np.random.default_rng(seed)
    batch_sharpes = []
    
    for i in range(100):
        perm_ranks = ranks_df.copy()
        for date in ranks_df.index:
            valid = ranks_df.loc[date].dropna()
            if len(valid) > 1:
                vals = valid.values.copy()
                rng.shuffle(vals)
                perm_ranks.loc[date, valid.index] = vals
        try:
            perm_result = engine.run(prices_dict, perm_ranks)
            perm_ret = perm_result.returns.dropna()
            perm_sh = (perm_ret.mean() / perm_ret.std()) * np.sqrt(252) if len(perm_ret) > 0 and perm_ret.std() > 0 else 0.0
            batch_sharpes.append(float(perm_sh))
        except:
            batch_sharpes.append(None)
    
    all_results['batches'].append({'batch_num': args.batch, 'seed': seed, 'raw_sharpes': batch_sharpes, 'valid_count': len([s for s in batch_sharpes if s is not None])})
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    all_s = [s for b in all_results['batches'] for s in b['raw_sharpes'] if s is not None]
    n = len(all_s)
    count_beat = sum(1 for s in all_s if s <= actual_sharpe) if actual_sharpe < 0 else sum(1 for s in all_s if s >= actual_sharpe)
    print(f"Batch {args.batch}/20: n={n}, p={count_beat/n:.6f} ({count_beat}/{n})")

if __name__ == '__main__':
    main()
