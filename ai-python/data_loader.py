"""
data_loader.py — Load & Preprocess Historical Stock Market Data
Processes 117,000+ records from nifty.csv using Pandas & NumPy.
"""

import pandas as pd
import numpy as np
from config import CSV_PATH, DATE_COLUMN, OHLCV_COLUMNS


def load_csv(filepath=None):
    """
    Load OHLCV data from CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Defaults to CSV_PATH from config.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with DatetimeIndex and OHLCV columns.
    """
    if filepath is None:
        filepath = CSV_PATH

    print(f"[DataLoader] Loading data from: {filepath}")

    # Read CSV with date parsing
    df = pd.read_csv(
        filepath,
        parse_dates=[DATE_COLUMN],
        index_col=DATE_COLUMN,
    )

    print(f"[DataLoader] Raw records loaded: {len(df):,}")

    # Normalize column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Ensure required columns exist
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"[DataLoader] Missing required column: '{col}'")

    # Keep only OHLCV columns
    df = df[OHLCV_COLUMNS].copy()

    # Clean the data
    df = _clean_data(df)

    print(f"[DataLoader] Clean records after preprocessing: {len(df):,}")
    print(f"[DataLoader] Date range: {df.index.min()} -> {df.index.max()}")

    return df


def _clean_data(df):
    """
    Clean and preprocess the raw DataFrame.

    Steps:
    - Remove timezone info for consistent handling
    - Convert columns to float64 / int64
    - Remove rows with zero or negative prices
    - Remove duplicate indices
    - Forward-fill missing values
    - Sort by datetime index
    """

    # Strip timezone info if present (for consistent resampling)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    else:
        # Handle string-based timezone in index
        try:
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        except Exception:
            df.index = pd.to_datetime(df.index)

    # Convert to numeric, coercing errors
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(np.int64)

    # Remove rows where any price is zero, negative, or NaN
    price_cols = ["open", "high", "low", "close"]
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]

    # Remove zero-volume rows (likely invalid ticks)
    df = df[df["volume"] > 0]

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Sort chronologically
    df = df.sort_index()

    # Forward-fill any remaining gaps (unlikely after above cleaning)
    df = df.ffill()

    return df


def load_and_split(filepath=None, train_ratio=0.80):
    """
    Load data and split into train/test sets.

    Parameters
    ----------
    filepath : str, optional
    train_ratio : float
        Fraction of data for training.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_df, test_df)
    """
    df = load_csv(filepath)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"[DataLoader] Train samples: {len(train_df):,}")
    print(f"[DataLoader] Test samples:  {len(test_df):,}")

    return train_df, test_df


if __name__ == "__main__":
    df = load_csv()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nHead:\n{df.head()}")
    print(f"\nTail:\n{df.tail()}")
    print(f"\nDescribe:\n{df.describe()}")
