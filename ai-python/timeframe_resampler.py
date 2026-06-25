"""
timeframe_resampler.py — Multi-Timeframe OHLCV Aggregation
Converts base 5-minute candles into higher timeframes using Pandas resample.
Supports: 15m, 1H, 3H, 4H, 1D, 1W
"""

import pandas as pd
from config import TIMEFRAMES


# Aggregation rules for OHLCV resampling
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_single(df, timeframe_key):
    """
    Resample a DataFrame to a single target timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and OHLCV columns.
    timeframe_key : str
        Timeframe key from TIMEFRAMES config (e.g., '15m', '1H', '1D').

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame.
    """
    if timeframe_key not in TIMEFRAMES:
        raise ValueError(
            f"Unknown timeframe '{timeframe_key}'. "
            f"Available: {list(TIMEFRAMES.keys())}"
        )

    freq = TIMEFRAMES[timeframe_key]

    # If base resolution requested, return as-is
    if timeframe_key == "5m":
        return df.copy()

    resampled = df.resample(freq).agg(OHLCV_AGG)

    # Drop rows where close is NaN (empty bars from weekends/holidays)
    resampled = resampled.dropna(subset=["close"])

    return resampled


def resample_all(df, timeframes=None):
    """
    Resample to all configured timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        Base OHLCV DataFrame (5-minute bars).
    timeframes : list of str, optional
        List of timeframe keys to resample. Defaults to all in TIMEFRAMES.

    Returns
    -------
    dict of str → pd.DataFrame
        Dictionary keyed by timeframe label, values are resampled DataFrames.
    """
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())

    result = {}

    for tf_key in timeframes:
        print(f"[Resampler] Resampling to {tf_key}...", end=" ")
        resampled = resample_single(df, tf_key)
        result[tf_key] = resampled
        print(f"{len(resampled):,} bars")

    return result


if __name__ == "__main__":
    from data_loader import load_csv

    df = load_csv()

    print("\n=== Resampling all timeframes ===\n")
    all_tf = resample_all(df)

    for label, tf_df in all_tf.items():
        print(f"\n--- {label} ---")
        print(f"  Records: {len(tf_df):,}")
        print(f"  Date range: {tf_df.index.min()} -> {tf_df.index.max()}")
        print(f"  Sample:\n{tf_df.tail(3)}\n")
