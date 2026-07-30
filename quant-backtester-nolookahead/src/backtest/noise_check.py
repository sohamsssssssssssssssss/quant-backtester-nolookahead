"""
Noise-check validation for multi-ticker walk-forward results.

This module answers: is the observed split of "7/15 tickers positive" distinguishable
from pure noise, or is it what you'd expect from chance alone given only 15 tickers?

Key Principles
--------------
1. PERMUTATION-BASED NULL: Rather than assuming a theoretical distribution (e.g.,
   binomial), we use block-permutation of signals to preserve realistic regime
   lengths while destroying actual timing edges.

2. BLOCK SHUFFLING, NOT DAY SHUFFLING: Individual day shuffling destroys signal
   autocorrelation and creates unrealistically weak nulls. Block shuffling preserves
   regime structure while breaking the price-signal relationship.

3. STRUCTURAL COMPARISON IS DESCRIPTIVE ONLY: With n=15, any feature correlation
   is hypothesis-generating, not statistically confirmatory. No regression, no
   correlation coefficients — just present the raw table.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from collections import Counter
import logging

from backtest.walk_forward import run_walk_forward
from signals.signals import moving_average_crossover

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Sector mapping for NIFTY 15 basket (hardcoded, pre-specified)
SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Engineering",
    "AXISBANK.NS": "Banking",
    "MARUTI.NS": "Auto",
    "SUNPHARMA.NS": "Pharma",
    "TITAN.NS": "Consumer",
}


def _extract_signal_blocks(signal: pd.Series) -> List[np.ndarray]:
    """
    Extract contiguous blocks of identical signal values.
    
    The MA crossover signal has regime structure: once the signal flips from
    -1 to 0 to 1, it tends to stay there for multiple days. We identify these
    contiguous blocks for block-permutation.
    
    Parameters
    ----------
    signal : pd.Series
        Signal series with values {-1, 0, 1}
    
    Returns
    -------
    List[np.ndarray]
        List of blocks, where each block is an array of signal values
        (all identical within the block)
    """
    if len(signal) == 0:
        return []
    
    signal_values = signal.values
    blocks = []
    
    current_block = [signal_values[0]]
    for i in range(1, len(signal_values)):
        if signal_values[i] == signal_values[i - 1]:
            current_block.append(signal_values[i])
        else:
            blocks.append(np.array(current_block))
            current_block = [signal_values[i]]
    
    # Don't forget the last block
    if len(current_block) > 0:
        blocks.append(np.array(current_block))
    
    return blocks


def _permute_blocks(blocks: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Randomly permute the order of blocks, preserving internal block structure.
    
    This destroys the timing relationship between signal and returns while
    preserving the autocorrelation structure (regime lengths) of the signal.
    
    Parameters
    ----------
    blocks : List[np.ndarray]
        List of signal blocks
    rng : np.random.Generator
        Random number generator for reproducibility
    
    Returns
    -------
    np.ndarray
        Permuted signal as a flat array
    """
    if len(blocks) == 0:
        return np.array([])
    
    # Shuffle block order
    block_indices = rng.permutation(len(blocks))
    shuffled_blocks = [blocks[i] for i in block_indices]
    
    # Concatenate back into flat array
    return np.concatenate(shuffled_blocks)


def _block_permute_signal(
    signal: pd.Series, 
    original_index: pd.Index,
    rng: np.random.Generator
) -> pd.Series:
    """
    Block-permute a signal and reconstruct with original index.
    
    Parameters
    ----------
    signal : pd.Series
        Original signal series
    original_index : pd.Index
        Original datetime index to preserve
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    pd.Series
        Block-permuted signal with same index as original
    """
    blocks = _extract_signal_blocks(signal)
    
    if len(blocks) == 0:
        return pd.Series(np.nan, index=original_index)
    
    permuted_values = _permute_blocks(blocks, rng)
    
    # Ensure length matches (should always match by construction)
    if len(permuted_values) != len(original_index):
        # Trim or pad if needed (shouldn't happen, but safety check)
        if len(permuted_values) > len(original_index):
            permuted_values = permuted_values[:len(original_index)]
        else:
            permuted_values = np.pad(
                permuted_values, 
                (0, len(original_index) - len(permuted_values)),
                mode='edge'
            )
    
    return pd.Series(permuted_values, index=original_index)


def permutation_null_test(
    prices_dict: Dict[str, pd.Series],
    signal_fn: Callable,
    engine_kwargs: Optional[Dict] = None,
    n_permutations: int = 500,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    seed: int = 42,
) -> Dict:
    """
    Permutation-based null test for multi-ticker walk-forward results.
    
    For each ticker:
    1. Compute the real signal on real prices
    2. Extract contiguous signal blocks (regimes)
    3. For each permutation: block-shuffle the signal, run walk-forward
    
    This builds a null distribution of "how many tickers show positive mean Sharpe"
    under random-but-realistic signal timing, then compares the observed count.
    
    Parameters
    ----------
    prices_dict : Dict[str, pd.Series]
        Dictionary mapping ticker strings to price Series
    signal_fn : Callable
        Signal function (e.g., moving_average_crossover)
    engine_kwargs : dict, optional
        Kwargs for BacktestEngine
    n_permutations : int
        Number of permutation iterations (default: 500)
    train_years : int
        Training window for walk-forward
    test_years : int
        Test window for walk-forward
    step_years : int
        Step size for walk-forward
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Results containing:
        - observed_positive_count: Number of tickers with positive mean Sharpe
        - null_distribution: Array of positive counts from permutations
        - p_value: Fraction of permutations with >= observed positive count
        - null_mean: Mean of null distribution
        - null_std: Std of null distribution
        - observed_percentile: Percentile of observed count in null
        - permutation_details: List of per-permutation results
    """
    if engine_kwargs is None:
        engine_kwargs = {}
    
    rng = np.random.default_rng(seed)
    tickers = list(prices_dict.keys())
    n_tickers = len(tickers)
    
    logger.info(f"Running permutation null test with {n_permutations} permutations...")
    logger.info(f"Tickers: {tickers}")
    
    # Step 1: Compute REAL signals and real walk-forward results
    logger.info("Step 1: Computing real walk-forward results...")
    real_signals = {}
    real_results = {}
    
    for ticker, prices in prices_dict.items():
        signal = signal_fn(prices)
        real_signals[ticker] = signal
        
        # Run walk-forward on real signal
        wf_results = run_walk_forward(
            prices,
            lambda p: signal_fn(p),  # Use the same signal
            engine_kwargs=engine_kwargs,
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
        )
        
        if len(wf_results) > 0:
            mean_sharpe = wf_results["sharpe_ratio"].mean()
            real_results[ticker] = {
                "mean_sharpe": mean_sharpe,
                "wf_results": wf_results,
                "positive": mean_sharpe > 0,
            }
        else:
            real_results[ticker] = {
                "mean_sharpe": np.nan,
                "wf_results": wf_results,
                "positive": False,
            }
    
    # Count observed positive tickers
    observed_positive_count = sum(
        1 for r in real_results.values() if r["positive"]
    )
    logger.info(f"Observed: {observed_positive_count}/{n_tickers} tickers with positive mean Sharpe")
    
    # Step 2: Run permutations
    logger.info(f"Step 2: Running {n_permutations} permutations...")
    null_distribution = []
    permutation_details = []
    
    for perm_idx in range(n_permutations):
        positive_count = 0
        perm_results = {}
        
        for ticker, prices in prices_dict.items():
            real_signal = real_signals[ticker]
            
            # Block-permute the signal
            permuted_signal = _block_permute_signal(
                real_signal, 
                prices.index, 
                rng
            )
            
            # Create a signal function that returns the permuted signal
            def make_perm_signal_fn(ps):
                return lambda p: ps
            
            # Run walk-forward with permuted signal
            wf_results = run_walk_forward(
                prices,
                make_perm_signal_fn(permuted_signal),
                engine_kwargs=engine_kwargs,
                train_years=train_years,
                test_years=test_years,
                step_years=step_years,
            )
            
            if len(wf_results) > 0:
                mean_sharpe = wf_results["sharpe_ratio"].mean()
                perm_results[ticker] = {
                    "mean_sharpe": mean_sharpe,
                    "positive": mean_sharpe > 0,
                }
                if mean_sharpe > 0:
                    positive_count += 1
            else:
                perm_results[ticker] = {
                    "mean_sharpe": np.nan,
                    "positive": False,
                }
        
        null_distribution.append(positive_count)
        permutation_details.append({
            "permutation": perm_idx,
            "positive_count": positive_count,
            "tickers_positive": [t for t, r in perm_results.items() if r["positive"]],
        })
        
        # Progress logging
        if (perm_idx + 1) % 100 == 0:
            logger.info(f"  Completed {perm_idx + 1}/{n_permutations} permutations")
    
    null_distribution = np.array(null_distribution)
    
    # Step 3: Compute p-value and statistics
    # One-tailed test: P(X >= observed) under null
    p_value = np.mean(null_distribution >= observed_positive_count)
    null_mean = null_distribution.mean()
    null_std = null_distribution.std()
    
    # Observed percentile in null distribution
    observed_percentile = np.mean(null_distribution < observed_positive_count) * 100
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("PERMUTATION NULL TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Observed positive tickers: {observed_positive_count}/{n_tickers}")
    logger.info(f"Null distribution mean:    {null_mean:.1f} (std={null_std:.1f})")
    logger.info(f"P-value (one-tailed):      {p_value:.3f}")
    logger.info(f"Observed percentile:       {observed_percentile:.1f}%")
    logger.info("")
    
    # Interpretation
    if p_value < 0.05:
        interpretation = (
            f"SIGNIFICANT at 5% level: Observing {observed_positive_count}/15 positive "
            f"tickers is unlikely under pure noise (p={p_value:.3f}). "
            f"This suggests the split may reflect real structural differences."
        )
    elif p_value < 0.10:
        interpretation = (
            f"MARGINALLY SIGNIFICANT at 10% level: p={p_value:.3f}. "
            f"Borderline evidence against pure noise — proceed with structural analysis "
            f"but interpret cautiously."
        )
    else:
        interpretation = (
            f"NOT SIGNIFICANT: p={p_value:.3f}. "
            f"Observing {observed_positive_count}/15 positive tickers is consistent with "
            f"pure noise given the joint price dynamics of these 15 tickers. "
            f"No further structural explanation is needed — this split is what chance produces."
        )
    
    logger.info(f"INTERPRETATION: {interpretation}")
    logger.info("=" * 60)
    
    return {
        "observed_positive_count": observed_positive_count,
        "n_tickers": n_tickers,
        "null_distribution": null_distribution,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "observed_percentile": observed_percentile,
        "permutation_details": permutation_details,
        "real_results": real_results,
        "interpretation": interpretation,
    }


def structural_comparison(tickers_with_results: pd.DataFrame) -> pd.DataFrame:
    """
    Structural comparison table for tickers.
    
    For each ticker, computes pre-specified features:
    - Sector (hardcoded mapping)
    - Trailing return autocorrelation (lag-1 to lag-5)
    - Realized volatility (annualized)
    
    These are presented alongside mean Sharpe from walk-forward results.
    
    IMPORTANT: With n=15, this is DESCRIPTIVE ONLY — hypothesis-generating,
    not statistically confirmatory. No correlation coefficients, no regression.
    Just present the raw table and let patterns (or lack thereof) be visible.
    
    Parameters
    ----------
    tickers_with_results : pd.DataFrame
        DataFrame with columns:
        - ticker: Ticker symbol
        - mean_sharpe: Mean Sharpe from walk-forward
    
    Returns
    -------
    pd.DataFrame
        Structural comparison table with columns:
        - ticker, mean_sharpe, sector, autocorr_lag1-5, realized_vol
    """
    if tickers_with_results.empty:
        return pd.DataFrame()
    
    results_list = []
    
    for _, row in tickers_with_results.iterrows():
        ticker = row["ticker"]
        mean_sharpe = row["mean_sharpe"]
        
        # Sector
        sector = SECTOR_MAP.get(ticker, "Unknown")
        
        # Get price data to compute features
        try:
            from data.loader import load_ohlcv
            prices_df = load_ohlcv(ticker, start="2015-01-01")
            prices = prices_df["Close"]
            
            # Returns
            returns = prices.pct_change().dropna()
            
            # Autocorrelations (lag 1-5)
            autocorrs = {}
            for lag in range(1, 6):
                if len(returns) > lag:
                    autocorrs[f"autocorr_lag{lag}"] = returns.autocorr(lag)
                else:
                    autocorrs[f"autocorr_lag{lag}"] = np.nan
            
            # Realized volatility (annualized)
            realized_vol = returns.std() * np.sqrt(252)
            
        except Exception as e:
            logger.warning(f"Could not compute features for {ticker}: {e}")
            autocorrs = {f"autocorr_lag{lag}": np.nan for lag in range(1, 6)}
            realized_vol = np.nan
        
        results_list.append({
            "ticker": ticker,
            "mean_sharpe": mean_sharpe,
            "sector": sector,
            "realized_vol": realized_vol,
            **autocorrs,
        })
    
    result_df = pd.DataFrame(results_list)
    result_df = result_df.sort_values("mean_sharpe", ascending=False)
    
    # Print caveat
    logger.info("")
    logger.info("=" * 80)
    logger.info("STRUCTURAL COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("CAVEAT: With n=15, this is DESCRIPTIVE ONLY — hypothesis-generating,")
    logger.info("not statistically confirmatory. No correlation coefficients or regression")
    logger.info("are computed — that would be a second layer of overfitting on top of the")
    logger.info("first. This table simply presents the data for visual pattern recognition.")
    logger.info("")
    
    return result_df
