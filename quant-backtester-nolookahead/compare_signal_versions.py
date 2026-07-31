"""Compare vectorized vs loop-based earnings momentum signal for equivalence."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from src.data.loader import load_ohlcv
from src.signals.earnings_momentum import load_earnings_surprises, earnings_momentum_signal


def earnings_momentum_signal_vectorized(
    prices_dict: dict[str, pd.Series],
    earnings_df: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
    hold_days: int = 60,
) -> pd.DataFrame:
    """
    Vectorized earnings momentum signal.

    For each date, a ticker's position is based on its most recent earnings
    surprise within the lookback window (hold_days * 2 calendar days).
    Portfolio: long top_n by surprise, short bottom_n, equal weighted, dollar neutral.
    """
    tickers = list(prices_dict.keys())

    # Common dates across all price series
    all_dates = None
    for prices in prices_dict.values():
        idx = set(prices.index)
        all_dates = idx if all_dates is None else all_dates.intersection(idx)
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    signal_tz = all_dates.tz
    # Work in tz-naive space internally; index aligns 1:1 with all_dates
    dates_naive = all_dates.tz_localize(None)

    # Filter earnings to available tickers; use naive dates
    ef = earnings_df[earnings_df['ticker'].isin(tickers)].copy()
    ef['date'] = pd.to_datetime(ef['date'])
    if ef['date'].dt.tz is not None:
        ef['date'] = ef['date'].dt.tz_convert(signal_tz).dt.tz_localize(None)

    # --- Build surprise matrix (dates x tickers): surprise on announcement date ---
    pivot = ef.pivot_table(index='date', columns='ticker', values='surprise', aggfunc='mean')
    pivot = pivot.reindex(columns=tickers)
    surprise_mat = pivot.reindex(dates_naive)

    # --- Build announcement-date matrix: value = date of last announcement per ticker ---
    ann_mat = pd.DataFrame(index=dates_naive, columns=tickers, dtype='datetime64[ns]')
    for t in tickers:
        sub = ef[ef['ticker'] == t]
        if len(sub) > 0:
            dts = pd.DatetimeIndex(sub['date']).drop_duplicates()
            tmp = pd.Series(dts, index=dts)
            ann_mat[t] = tmp.reindex(dates_naive)

    # --- Forward-fill last known surprise and last announcement date ---
    last_surprise = surprise_mat.ffill()
    last_ann = ann_mat.ffill()

    # --- Staleness mask: last announcement within lookback window ---
    lookback = pd.Timedelta(days=hold_days * 2)
    age = last_ann.sub(dates_naive, axis=0)
    fresh = age <= lookback

    active = last_surprise.where(fresh)

    # --- Cross-sectional ranking per date (columns axis) ---
    ranked = active.rank(axis=1, method='first', ascending=True)
    valid_count = active.notna().sum(axis=1)
    enough = valid_count >= (top_n + bottom_n)

    long_mask = ranked.ge(valid_count - top_n + 1, axis=0)
    short_mask = ranked.le(bottom_n, axis=0)

    # Enforce the "enough valid tickers" constraint per date
    long_mask = long_mask & enough.to_numpy()[:, None]
    short_mask = short_mask & enough.to_numpy()[:, None]

    signal = pd.DataFrame(0.0, index=all_dates, columns=tickers)
    signal[long_mask.to_numpy()] = 1.0 / top_n
    signal[short_mask.to_numpy()] = -1.0 / bottom_n

    return signal


def main():
    print("Loading data...")
    earnings_df = load_earnings_surprises()
    tickers = earnings_df['ticker'].unique()
    prices_dict = {t: load_ohlcv(t, start='2020-01-01')['Close'] for t in tickers}

    print("Running original (loop) version...")
    t0 = time.time()
    sig_old = earnings_momentum_signal(prices_dict, earnings_df, top_n=5, bottom_n=5, hold_days=60).fillna(0)
    t_old = time.time() - t0
    print(f"  Loop version: {t_old:.2f}s")

    print("Running vectorized version...")
    t0 = time.time()
    sig_new = earnings_momentum_signal_vectorized(prices_dict, earnings_df, top_n=5, bottom_n=5, hold_days=60)
    t_new = time.time() - t0
    print(f"  Vectorized: {t_new:.2f}s")

    # Compare
    sig_old = sig_old.reindex(index=sig_new.index, columns=sig_new.columns)
    diff = (sig_old - sig_new).abs().max().max()
    n_diff = (sig_old != sig_new).sum().sum()
    print(f"\nMax abs diff: {diff:.6f}")
    print(f"Cells differing: {n_diff} / {sig_old.shape[0] * sig_old.shape[1]}")
    print(f"Non-zero old: {(sig_old != 0).sum().sum()}, new: {(sig_new != 0).sum().sum()}")

    # Per-date position count comparison
    old_per_date = (sig_old != 0).sum(axis=1)
    new_per_date = (sig_new != 0).sum(axis=1)
    pos_diff = (old_per_date - new_per_date).abs().max()
    print(f"Max per-date position-count diff: {pos_diff}")

    if n_diff == 0:
        print("\nRESULT: VECTORIZED VERSION IS IDENTICAL TO LOOP VERSION")
    else:
        print("\nRESULT: VECTORIZED VERSION DIFFERS - NEEDS INVESTIGATION")


if __name__ == "__main__":
    main()
