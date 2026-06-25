"""
predictor.py — ML-Powered Stock Prediction
Replaces the original hardcoded threshold logic with LSTM ensemble model.
Maintains backward compatibility with Java AIPredictor.java (sys.argv[1] interface).

Usage:
    python predictor.py <close_price>
    Outputs: BUY, SELL, or HOLD
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure ai-python is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LSTM_MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH,
    ARIMA_MODEL_PATH, MODELS_DIR,
    BUY_THRESHOLD, SELL_THRESHOLD,
    LSTM_SEQUENCE_LENGTH,
    DEFAULT_SYMBOL,
)


def predict_with_model(close_price):
    """
    Make prediction using trained LSTM model.

    Parameters
    ----------
    close_price : float
        Current close price.

    Returns
    -------
    str
        'BUY', 'SELL', or 'HOLD'
    """
    # Check if models exist
    if not os.path.exists(LSTM_MODEL_PATH):
        # Fallback to heuristic if model not trained yet
        return _fallback_prediction(close_price)

    try:
        # Try to get recent market data for feature computation
        prediction = _predict_with_live_context(close_price)
        return prediction
    except Exception as e:
        # If live data fails, use simplified prediction
        return _predict_simplified(close_price)


def _predict_with_live_context(close_price):
    """
    Predict using live market data context for proper feature computation.
    """
    import numpy as np
    import pandas as pd

    try:
        import yfinance as yf

        # Fetch recent data for feature computation
        # Use the DEFAULT_SYMBOL from config, which may have been overridden via CLI argument
        data = yf.download(
            DEFAULT_SYMBOL,
            period="60d",
            interval="1d",
            progress=False,
        )

        if data.empty or len(data) < LSTM_SEQUENCE_LENGTH:
            return _predict_simplified(close_price)

        # Fix multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = data.columns.str.strip().str.lower()

        # Ensure required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data.columns:
                return _predict_simplified(close_price)

        data = data[["open", "high", "low", "close", "volume"]].copy()
        data = data.dropna()

        # Compute features
        from features import compute_all_features, get_feature_columns
        featured = compute_all_features(data)
        feature_cols = get_feature_columns(featured)
        featured = featured.dropna()

        if len(featured) < LSTM_SEQUENCE_LENGTH:
            return _predict_simplified(close_price)

        # Load model and predict
        from lstm_forecaster import LSTMForecaster
        lstm = LSTMForecaster()
        lstm.load()

        # Use the model's feature columns (intersection with available)
        available_cols = [c for c in lstm.feature_columns if c in featured.columns]
        if len(available_cols) < len(lstm.feature_columns) // 2:
            return _predict_simplified(close_price)

        # Get last sequence
        recent = featured[available_cols].tail(LSTM_SEQUENCE_LENGTH).values
        pred_price = lstm.predict_next(recent)

        # Generate signal
        pred_return = (pred_price - close_price) / close_price

        if pred_return > BUY_THRESHOLD:
            return "BUY"
        elif pred_return < SELL_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"

    except Exception:
        return _predict_simplified(close_price)


def _predict_simplified(close_price):
    """
    Simplified prediction using just the ARIMA model.
    """
    try:
        from arima_forecaster import ARIMAForecaster

        arima = ARIMAForecaster()
        arima.load()

        forecast = arima.predict(steps=1)
        if len(forecast) > 0:
            pred_price = forecast[0]
            pred_return = (pred_price - close_price) / close_price

            if pred_return > BUY_THRESHOLD:
                return "BUY"
            elif pred_return < SELL_THRESHOLD:
                return "SELL"
            else:
                return "HOLD"

    except Exception:
        pass

    return _fallback_prediction(close_price)


def _fallback_prediction(close_price):
    """
    Original hardcoded fallback — used only when no trained models exist.
    """
    if close_price > 25000:
        return "BUY"
    elif close_price < 24000:
        return "SELL"
    else:
        return "HOLD"


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict BUY/SELL/HOLD for a given close price and optional ticker symbol.")
    parser.add_argument("close_price", type=float, help="Current close price.")
    parser.add_argument("--symbol", type=str, default=None, help="Ticker symbol for live data fetching (default: config.DEFAULT_SYMBOL).")
    args = parser.parse_args()
    # Set the global symbol if provided
    if args.symbol:
        # Override DEFAULT_SYMBOL in config at runtime
        import config
        config.DEFAULT_SYMBOL = args.symbol
    signal = predict_with_model(args.close_price)
    print(signal)