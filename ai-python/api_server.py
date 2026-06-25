"""
api_server.py — FastAPI REST API for Real-Time Forecasting & Signals
Provides model inference, technical indicators, and signal generation.
"""

import os
import sys
import json
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure ai-python is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_SYMBOL, LSTM_SEQUENCE_LENGTH,
    EVALUATION_REPORT_PATH,
)
from realtime_feed import fetch_live_data
from features import compute_all_features, get_feature_columns
from lstm_forecaster import LSTMForecaster
from arima_forecaster import ARIMAForecaster
from ensemble_forecaster import EnsembleForecaster

app = FastAPI(
    title="Market Engine AI API Server",
    description="Real-time stock prediction and financial analytics REST endpoints.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper function to compute features on live data
def get_live_featured_data(symbol, interval):
    df = fetch_live_data(symbol, interval)
    if df.empty or len(df) < LSTM_SEQUENCE_LENGTH + 20:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient historical data found for {symbol} on {interval} timeframe."
        )
    featured = compute_all_features(df)
    featured = featured.dropna()
    return featured


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "market-engine-ai-server"}


@app.get("/api/predict/{symbol}")
def get_prediction(symbol: str = DEFAULT_SYMBOL, interval: str = "1D"):
    """Get real-time prediction and signal using ARIMA + LSTM Ensemble."""
    try:
        featured = get_live_featured_data(symbol, interval)

        # Initialize models
        lstm = LSTMForecaster()
        lstm.load()

        arima = ARIMAForecaster()
        arima.load()

        ensemble = EnsembleForecaster()

        # LSTM prediction input
        available_cols = [c for c in lstm.feature_columns if c in featured.columns]
        recent = featured[available_cols].tail(LSTM_SEQUENCE_LENGTH).values
        lstm_pred = lstm.predict_next(recent)

        # ARIMA prediction input
        arima_pred = arima.predict(steps=1)[0]

        # Current close price
        current_price = float(featured["close"].iloc[-1])

        # Blend ensemble
        prediction_result = ensemble.predict_single(lstm_pred, arima_pred, current_price)

        return {
            "symbol": symbol,
            "interval": interval,
            "timestamp": str(featured.index[-1]),
            **prediction_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/{symbol}")
def get_forecast(symbol: str = DEFAULT_SYMBOL):
    """Get multi-timeframe forecasts for the next step across major intervals."""
    forecasts = {}
    intervals = ["15m", "1H", "4H", "1D"]

    # Initialize models once
    lstm = LSTMForecaster()
    try:
        lstm.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LSTM model: {e}")

    for interval in intervals:
        try:
            featured = fetch_live_data(symbol, interval)
            if featured.empty or len(featured) < LSTM_SEQUENCE_LENGTH + 20:
                continue

            featured = compute_all_features(featured).dropna()
            available_cols = [c for c in lstm.feature_columns if c in featured.columns]
            recent = featured[available_cols].tail(LSTM_SEQUENCE_LENGTH).values
            lstm_pred = lstm.predict_next(recent)

            current_price = float(featured["close"].iloc[-1])
            pred_return = (lstm_pred - current_price) / current_price

            forecasts[interval] = {
                "current_price": current_price,
                "predicted_price": lstm_pred,
                "predicted_return": pred_return,
                "signal": "BUY" if pred_return > 0.002 else "SELL" if pred_return < -0.002 else "HOLD"
            }
        except Exception:
            # Skip if timeframe fails
            continue

    return {
        "symbol": symbol,
        "forecasts": forecasts
    }


@app.get("/api/indicators/{symbol}")
def get_indicators(symbol: str = DEFAULT_SYMBOL, interval: str = "1D"):
    """Get live technical indicator values for a symbol."""
    try:
        featured = get_live_featured_data(symbol, interval)
        last_row = featured.iloc[-1]

        # Extract main technical indicators
        indicators = {
            "close": float(last_row["close"]),
            "sma_10": float(last_row.get("sma_10", 0)),
            "sma_50": float(last_row.get("sma_50", 0)),
            "sma_200": float(last_row.get("sma_200", 0)),
            "ema_9": float(last_row.get("ema_9", 0)),
            "ema_21": float(last_row.get("ema_21", 0)),
            "rsi": float(last_row.get("rsi", 50)),
            "macd": float(last_row.get("macd", 0)),
            "macd_signal": float(last_row.get("macd_signal", 0)),
            "macd_hist": float(last_row.get("macd_hist", 0)),
            "bb_upper": float(last_row.get("bb_upper", 0)),
            "bb_middle": float(last_row.get("bb_middle", 0)),
            "bb_lower": float(last_row.get("bb_lower", 0)),
            "atr": float(last_row.get("atr", 0)),
            "stoch_k": float(last_row.get("stoch_k", 50)),
            "stoch_d": float(last_row.get("stoch_d", 50)),
            "adx": float(last_row.get("adx", 0)),
            "tenkan_sen": float(last_row.get("ichi_tenkan", 0)),
            "kijun_sen": float(last_row.get("ichi_kijun", 0)),
            "senkou_span_a": float(last_row.get("ichi_senkou_a", 0)),
            "senkou_span_b": float(last_row.get("ichi_senkou_b", 0)),
            "vwap": float(last_row.get("vwap", 0)),
            "roc": float(last_row.get("roc", 0)),
            "williams_r": float(last_row.get("williams_r", -50)),
            "cci": float(last_row.get("cci", 0)),
        }

        # Trend labels
        trends = {
            "trend_sma": int(last_row.get("trend_sma", 0)),
            "trend_adx_strong": int(last_row.get("trend_adx_strong", 0)),
            "trend_ichimoku": int(last_row.get("trend_ichimoku", 0)),
        }

        return {
            "symbol": symbol,
            "interval": interval,
            "timestamp": str(featured.index[-1]),
            "indicators": indicators,
            "trends": trends
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluate")
def get_evaluation():
    """Retrieve saved evaluation metrics for the trained models."""
    if not os.path.exists(EVALUATION_REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found. Please run the training pipeline first."
        )

    try:
        with open(EVALUATION_REPORT_PATH, "r") as f:
            report = json.load(f)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/{symbol}")
def get_signals(symbol: str = DEFAULT_SYMBOL):
    """Retrieve detailed trade signals across multiple timeframes."""
    signals = {}
    intervals = ["15m", "1H", "4H", "1D"]

    lstm = LSTMForecaster()
    try:
        lstm.load()
    except Exception:
        raise HTTPException(status_code=500, detail="LSTM model is not loaded.")

    for interval in intervals:
        try:
            featured = get_live_featured_data(symbol, interval)
            available_cols = [c for c in lstm.feature_columns if c in featured.columns]
            recent = featured[available_cols].tail(LSTM_SEQUENCE_LENGTH).values
            lstm_pred = lstm.predict_next(recent)

            current_price = float(featured["close"].iloc[-1])
            pred_return = (lstm_pred - current_price) / current_price

            strength = min(abs(pred_return) / 0.01, 1.0)  # Strength normalized to 0-1

            if pred_return > 0.002:
                signal = "BUY"
            elif pred_return < -0.002:
                signal = "SELL"
            else:
                signal = "HOLD"

            signals[interval] = {
                "signal": signal,
                "strength": strength,
                "current_price": current_price,
                "predicted_price": lstm_pred,
                "predicted_return": pred_return,
            }
        except Exception:
            continue

    return {
        "symbol": symbol,
        "signals": signals
    }


@app.get("/api/trade-levels/{symbol}")
def get_trade_levels(symbol: str = DEFAULT_SYMBOL, interval: str = "1D"):
    """Retrieve detailed trade entry zones, stop losses, and target levels across timeframes."""
    try:
        from levels_analyzer import (
            calculate_pivot_points,
            get_trade_setup,
            synthesize_multi_timeframe_levels
        )
        
        # 1. Fetch live data for active timeframe
        df = fetch_live_data(symbol, interval)
        if df.empty or len(df) < 10:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data found for {symbol} on {interval} timeframe."
            )
            
        current_price = float(df["close"].iloc[-1])
        
        # Determine trend for active timeframe (e.g. from LSTM/ARIMA if available, else EMA)
        trend_signal = "HOLD"
        try:
            lstm = LSTMForecaster()
            lstm.load()
            featured = compute_all_features(df).dropna()
            available_cols = [c for c in lstm.feature_columns if c in featured.columns]
            recent = featured[available_cols].tail(LSTM_SEQUENCE_LENGTH).values
            if len(recent) == LSTM_SEQUENCE_LENGTH:
                lstm_pred = lstm.predict_next(recent)
                if (lstm_pred - current_price) / current_price > 0.002:
                    trend_signal = "BUY"
                elif (lstm_pred - current_price) / current_price < -0.002:
                    trend_signal = "SELL"
        except Exception:
            # Fallback
            ema9 = df["close"].ewm(span=9).mean().iloc[-1]
            ema21 = df["close"].ewm(span=21).mean().iloc[-1]
            trend_signal = "BUY" if ema9 > ema21 else "SELL"
            
        # Calculate ATR for volatility scaling
        high_low_diff = df["high"] - df["low"]
        atr_val = float(high_low_diff.rolling(14).mean().iloc[-1])
        if math.isnan(atr_val) or atr_val <= 0:
            atr_val = current_price * 0.01
            
        # Calculate levels and setup
        levels = calculate_pivot_points(df)
        setup = get_trade_setup(current_price, trend_signal, atr_val, levels)
        
        # 2. Get multi-timeframe execution grid
        multi_timeframe_grid = synthesize_multi_timeframe_levels(
            symbol=symbol,
            fetch_live_data_fn=fetch_live_data
        )
        
        return {
            "symbol": symbol,
            "active_interval": interval,
            "spot_price": current_price,
            "trend_signal": trend_signal,
            "levels": levels,
            "setup": {
                "pivot_type": setup["pivot_type"],
                "buy_zone_min": setup["buy_zone"][0],
                "buy_zone_max": setup["buy_zone"][1],
                "sell_zone_min": setup["sell_zone"][0],
                "sell_zone_max": setup["sell_zone"][1],
                "stop_loss": setup["stop_loss"],
                "target_1": setup["target_1"],
                "target_2": setup["target_2"],
                "risk_reward_ratio": setup["risk_reward_ratio"],
                "rationale": setup["rationale"]
            },
            "multi_timeframe_grid": multi_timeframe_grid
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

