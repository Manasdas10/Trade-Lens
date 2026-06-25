"""
features.py — Technical Indicator Feature Engineering
Computes 14+ indicators using Pandas & NumPy for trend identification
across multiple timeframes (15m, 1H, 3H, 4H, 1D, 1W).
"""

import pandas as pd
import numpy as np
from config import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD,
    ATR_PERIOD, STOCH_K_PERIOD, STOCH_D_PERIOD,
    ADX_PERIOD, ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU,
    CCI_PERIOD, WILLIAMS_R_PERIOD, ROC_PERIOD, LAG_PERIODS,
)


def compute_all_features(df):
    """
    Compute the full suite of technical indicators on an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with all indicator columns appended.
        NaN rows at the start (from lookback periods) are dropped.
    """
    df = df.copy()

    # --- Simple Moving Averages ---
    for period in SMA_PERIODS:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

    # --- Exponential Moving Averages ---
    for period in EMA_PERIODS:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # --- RSI (Wilder's Smoothing) ---
    df["rsi"] = _compute_rsi(df["close"], RSI_PERIOD)

    # --- MACD ---
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Bollinger Bands ---
    sma_bb = df["close"].rolling(window=BOLLINGER_PERIOD).mean()
    std_bb = df["close"].rolling(window=BOLLINGER_PERIOD).std()
    df["bb_upper"] = sma_bb + (BOLLINGER_STD * std_bb)
    df["bb_middle"] = sma_bb
    df["bb_lower"] = sma_bb - (BOLLINGER_STD * std_bb)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # --- ATR (Average True Range) ---
    df["atr"] = _compute_atr(df, ATR_PERIOD)

    # --- Stochastic Oscillator (%K, %D) ---
    low_min = df["low"].rolling(window=STOCH_K_PERIOD).min()
    high_max = df["high"].rolling(window=STOCH_K_PERIOD).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["stoch_d"] = df["stoch_k"].rolling(window=STOCH_D_PERIOD).mean()

    # --- OBV (On-Balance Volume) ---
    df["obv"] = _compute_obv(df)
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()

    # --- ADX (Average Directional Index) ---
    df["adx"], df["di_plus"], df["di_minus"] = _compute_adx(df, ADX_PERIOD)

    # --- Ichimoku Cloud ---
    df["ichi_tenkan"] = (
        df["high"].rolling(window=ICHIMOKU_TENKAN).max()
        + df["low"].rolling(window=ICHIMOKU_TENKAN).min()
    ) / 2

    df["ichi_kijun"] = (
        df["high"].rolling(window=ICHIMOKU_KIJUN).max()
        + df["low"].rolling(window=ICHIMOKU_KIJUN).min()
    ) / 2

    df["ichi_senkou_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2).shift(ICHIMOKU_KIJUN)

    df["ichi_senkou_b"] = (
        (
            df["high"].rolling(window=ICHIMOKU_SENKOU).max()
            + df["low"].rolling(window=ICHIMOKU_SENKOU).min()
        ) / 2
    ).shift(ICHIMOKU_KIJUN)

    df["ichi_chikou"] = df["close"].shift(-ICHIMOKU_KIJUN)

    # --- VWAP ---
    df["vwap"] = _compute_vwap(df)

    # --- Rate of Change ---
    df["roc"] = df["close"].pct_change(periods=ROC_PERIOD) * 100

    # --- Williams %R ---
    high_max_wr = df["high"].rolling(window=WILLIAMS_R_PERIOD).max()
    low_min_wr = df["low"].rolling(window=WILLIAMS_R_PERIOD).min()
    df["williams_r"] = -100 * (high_max_wr - df["close"]) / (high_max_wr - low_min_wr)

    # --- CCI (Commodity Channel Index) ---
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(window=CCI_PERIOD).mean()
    mean_dev = typical_price.rolling(window=CCI_PERIOD).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["cci"] = (typical_price - sma_tp) / (0.015 * mean_dev)

    # --- Price-based features ---
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)

    # --- Volume features ---
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
    df["obv_slope"] = df["obv"].diff(5)

    # --- Lag features ---
    for lag in LAG_PERIODS:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    # --- Trend labels ---
    df["trend_sma"] = np.where(
        df["sma_10"] > df["sma_50"], 1, np.where(df["sma_10"] < df["sma_50"], -1, 0)
    )
    df["trend_adx_strong"] = np.where(df["adx"] > 25, 1, 0)
    df["trend_ichimoku"] = np.where(
        df["close"] > df["ichi_senkou_a"], 1,
        np.where(df["close"] < df["ichi_senkou_b"], -1, 0)
    )

    # --- Target: next-period direction (1 = up, 0 = down) ---
    df["target_direction"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    df["target_return"] = df["close"].shift(-1) / df["close"] - 1

    return df


def get_feature_columns(df):
    """Return list of feature column names (excludes OHLCV, target, and raw columns)."""
    exclude = {
        "open", "high", "low", "close", "volume",
        "target_direction", "target_return",
        "ichi_chikou",  # future-shifted, can't use as input
    }
    return [c for c in df.columns if c not in exclude and not df[c].isna().all()]


def prepare_for_model(df, dropna=True):
    """
    Prepare feature matrix for ML models.

    Returns
    -------
    tuple of (pd.DataFrame, list)
        (df_clean, feature_columns)
    """
    feature_cols = get_feature_columns(df)
    df_clean = df[feature_cols + ["close", "target_direction", "target_return"]].copy()

    if dropna:
        df_clean = df_clean.dropna()

    return df_clean, feature_cols


# ============================================================
# PRIVATE HELPER FUNCTIONS
# ============================================================

def _compute_rsi(series, period):
    """Compute RSI using Wilder's Exponential Moving Average."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _compute_atr(df, period):
    """Compute Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return atr


def _compute_obv(df):
    """Compute On-Balance Volume."""
    obv = np.where(
        df["close"] > df["close"].shift(1),
        df["volume"],
        np.where(df["close"] < df["close"].shift(1), -df["volume"], 0),
    )
    return pd.Series(np.cumsum(obv), index=df.index, name="obv")


def _compute_adx(df, period):
    """Compute Average Directional Index with DI+ and DI-."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    )

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


def _compute_vwap(df):
    """Compute Volume Weighted Average Price (cumulative intraday)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    return vwap


if __name__ == "__main__":
    from data_loader import load_csv
    from timeframe_resampler import resample_single

    df = load_csv()

    # Test on daily timeframe
    daily = resample_single(df, "1D")
    print(f"\nDaily bars: {len(daily):,}")

    featured = compute_all_features(daily)
    feature_cols = get_feature_columns(featured)

    print(f"Total features computed: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"\nSample (last 3 rows):\n{featured[feature_cols].tail(3)}")
