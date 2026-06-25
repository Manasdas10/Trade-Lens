"""
realtime_feed.py — Real-Time Market Data Feed
Fetches live data from yfinance with caching to avoid rate limit throttling.
"""

import time
import pandas as pd
import yfinance as yf
from config import CACHE_TTL_SECONDS, YFINANCE_INTERVALS, YFINANCE_PERIODS

# Simple in-memory cache: { (symbol, interval): (timestamp, df) }
_cache = {}


def fetch_live_data(symbol, interval="1D", force_refresh=False):
    """
    Fetch live OHLCV data for a given symbol and interval.
    Uses in-memory cache to prevent frequent external API requests.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. '^NSEI', 'RELIANCE.NS').
    interval : str
        Timeframe label (e.g. '5m', '15m', '1H', '4H', '1D', '1W').
    force_refresh : bool, optional
        Force fetch fresh data even if cached data is within TTL.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data.
    """
    now = time.time()
    cache_key = (symbol, interval)

    # Check cache
    if not force_refresh and cache_key in _cache:
        cached_time, df = _cache[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            print(f"[RealtimeFeed] Returning cached data for {symbol} ({interval})")
            return df.copy()

    # Get yfinance specific interval & period
    # If 4H is requested, we download 1H and aggregate
    actual_interval = "1H" if interval == "4H" else interval
    yf_interval = YFINANCE_INTERVALS.get(actual_interval, "1d")
    yf_period = YFINANCE_PERIODS.get(actual_interval, "1y")

    print(
        f"[RealtimeFeed] Fetching live data for {symbol} from yfinance "
        f"(interval={yf_interval}, period={yf_period})..."
    )

    try:
        df = yf.download(
            symbol,
            period=yf_period,
            interval=yf_interval,
            progress=False,
        )

        if df.empty:
            print(f"[RealtimeFeed] Warning: yfinance returned empty data for {symbol}")
            return pd.DataFrame()

        # Fix multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = df.columns.str.strip().str.lower()

        # Resample 1H to 4H if needed (yfinance doesn't support 4h directly)
        if interval == "4H":
            from timeframe_resampler import OHLCV_AGG

            df = df.resample("4h").agg(OHLCV_AGG).dropna()

        # Ensure index is DatetimeIndex and clean timezone
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Forward fill and drop NaNs in prices
        df = df.ffill()
        df = df.dropna(subset=["close"])

        # Update cache
        _cache[cache_key] = (now, df)
        return df.copy()

    except Exception as e:
        print(f"[RealtimeFeed] Error fetching live data for {symbol}: {e}")
        # Return stale cache if available, else empty DataFrame
        if cache_key in _cache:
            print(f"[RealtimeFeed] Returning stale cached data for {symbol} due to fetch error")
            return _cache[cache_key][1].copy()
        return pd.DataFrame()


if __name__ == "__main__":
    # Quick test
    df = fetch_live_data("^NSEI", "1D")
    print(f"Data retrieved: {len(df)} rows")
    print(df.tail(3))
