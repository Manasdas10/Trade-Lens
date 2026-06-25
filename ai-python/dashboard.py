import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys
import json

# Ensure ai-python is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_SYMBOL, TIMEFRAMES,
    LSTM_MODEL_PATH, ARIMA_MODEL_PATH,
    EVALUATION_REPORT_PATH,
    LSTM_SEQUENCE_LENGTH,
)
from realtime_feed import fetch_live_data
from features import compute_all_features, get_feature_columns
from lstm_forecaster import LSTMForecaster
from arima_forecaster import ARIMAForecaster
from ensemble_forecaster import EnsembleForecaster
from levels_analyzer import (
    calculate_pivot_points,
    get_trade_setup,
    synthesize_multi_timeframe_levels
)

# PAGE CONFIG
st.set_page_config(
    page_title="TradeLens — Stock Forecasting & Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR CONTROLS
st.sidebar.header("⚙️ Configuration")
theme_choice = st.sidebar.selectbox("Dashboard Theme", ["Classic Dark", "Monochrome Dark", "Minimalist Light"])
symbol = st.sidebar.text_input("Active Ticker Symbol", value=DEFAULT_SYMBOL, help="Enter any stock ticker (e.g. ^NSEI, BTC-USD, RELIANCE.NS)")
timeframe_label = st.sidebar.selectbox("Select Timeframe", options=list(TIMEFRAMES.keys()), index=4, help="Choose the candle interval for the main chart.")

# DYNAMIC THEME SELECTOR & VARIABLE DEFINITION
if theme_choice == "Classic Dark":
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    card_bg = "rgba(255, 255, 255, 0.03)"
    card_border = "rgba(255, 255, 255, 0.05)"
    sidebar_bg = "#161b22"
    accent_color = "#3b82f6"
    title_gradient = "linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899)"
    grid_active_bg = "rgba(59, 130, 246, 0.08)"
    grid_active_border = "2px solid #3b82f6"
    grid_inactive_bg = "rgba(255, 255, 255, 0.02)"
    grid_inactive_border = "1px solid rgba(255,255,255,0.05)"
    text_muted = "#888888"
    setup_card_buy_bg = "rgba(16, 185, 129, 0.08)"
    setup_card_buy_border = "#10b981"
    setup_card_sell_bg = "rgba(239, 68, 68, 0.08)"
    setup_card_sell_border = "#ef4444"
    card_text_color = "#ffffff"
    plotly_template = "plotly_dark"
    plotly_grid_color = "rgba(255, 255, 255, 0.06)"
    trend_buy_bg = "#10b981"
    trend_buy_text = "#ffffff"
    trend_sell_bg = "#ef4444"
    trend_sell_text = "#ffffff"
    trend_hold_bg = "#6b7280"
    trend_hold_text = "#ffffff"
    gauge_steps = [
        {'range': [-1, -0.2], 'color': "rgba(239, 68, 68, 0.15)"},
        {'range': [-0.2, 0.2], 'color': "rgba(255, 255, 255, 0.05)"},
        {'range': [0.2, 1.0], 'color': "rgba(16, 185, 129, 0.15)"}
    ]
elif theme_choice == "Monochrome Dark":
    bg_color = "#000000"
    text_color = "#ffffff"
    card_bg = "#111111"
    card_border = "#333333"
    sidebar_bg = "#111111"
    accent_color = "#ffffff"
    title_gradient = "linear-gradient(90deg, #ffffff, #666666)"
    grid_active_bg = "#1a1a1a"
    grid_active_border = "2px solid #ffffff"
    grid_inactive_bg = "#0c0c0c"
    grid_inactive_border = "1px solid #222222"
    text_muted = "#888888"
    setup_card_buy_bg = "#111111"
    setup_card_buy_border = "#ffffff"
    setup_card_sell_bg = "#0c0c0c"
    setup_card_sell_border = "#444444"
    card_text_color = "#ffffff"
    plotly_template = "plotly_dark"
    plotly_grid_color = "rgba(255, 255, 255, 0.04)"
    trend_buy_bg = "#ffffff"
    trend_buy_text = "#000000"
    trend_sell_bg = "#444444"
    trend_sell_text = "#ffffff"
    trend_hold_bg = "#222222"
    trend_hold_text = "#ffffff"
    gauge_steps = [
        {'range': [-1, -0.2], 'color': "rgba(255, 255, 255, 0.05)"},
        {'range': [-0.2, 0.2], 'color': "rgba(255, 255, 255, 0.12)"},
        {'range': [0.2, 1.0], 'color': "rgba(255, 255, 255, 0.22)"}
    ]
else: # Minimalist Light
    bg_color = "#ffffff"
    text_color = "#111111"
    card_bg = "#f9f9f9"
    card_border = "#e0e0e0"
    sidebar_bg = "#f0f2f6"
    accent_color = "#111111"
    title_gradient = "linear-gradient(90deg, #111111, #888888)"
    grid_active_bg = "#eaeaea"
    grid_active_border = "2px solid #111111"
    grid_inactive_bg = "#f5f5f5"
    grid_inactive_border = "1px solid #e0e0e0"
    text_muted = "#666666"
    setup_card_buy_bg = "rgba(16, 185, 129, 0.05)"
    setup_card_buy_border = "#10b981"
    setup_card_sell_bg = "rgba(239, 68, 68, 0.05)"
    setup_card_sell_border = "#ef4444"
    card_text_color = "#111111"
    plotly_template = "plotly_white"
    plotly_grid_color = "rgba(0, 0, 0, 0.06)"
    trend_buy_bg = "#111111"
    trend_buy_text = "#ffffff"
    trend_sell_bg = "#cccccc"
    trend_sell_text = "#111111"
    trend_hold_bg = "#e5e7eb"
    trend_hold_text = "#111111"
    gauge_steps = [
        {'range': [-1, -0.2], 'color': "rgba(239, 68, 68, 0.1)"},
        {'range': [-0.2, 0.2], 'color': "rgba(0, 0, 0, 0.04)"},
        {'range': [0.2, 1.0], 'color': "rgba(16, 185, 129, 0.1)"}
    ]

# Inject dynamic CSS styles
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Inter', sans-serif;
    }}
    
    h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p {{
        color: {text_color} !important;
    }}
    
    /* Metrics panel card design */
    div[data-testid="stMetricValue"] {{
        font-size: 28px !important;
        font-weight: bold;
        color: {text_color} !important;
    }}
    
    div[data-testid="metric-container"] {{
        background-color: {card_bg};
        border: 1px solid {card_border};
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {card_border};
    }}
    
    .stAlert {{
        border-radius: 10px;
        background-color: {card_bg};
        border: 1px solid {card_border};
        color: {text_color};
    }}
    
    /* Custom widgets adaptation */
    div[data-baseweb="select"] > div {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {card_border} !important;
    }}
    div[data-baseweb="input"] > div {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {card_border} !important;
    }}
    
    /* Tab styling */
    button[data-baseweb="tab"] {{
        color: {text_muted} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text_color} !important;
        border-bottom-color: {accent_color} !important;
    }}
    
    /* Radio button styling */
    div[data-testid="stRadio"] label {{
        color: {text_color} !important;
    }}
    
    /* Button styling */
    button[data-testid="stBaseButton-secondary"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        color: {text_color} !important;
    }}
    button[data-testid="stBaseButton-secondary"]:hover {{
        border-color: {accent_color} !important;
        color: {accent_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# LIVE MODEL TRAINING SECTION (NON-TECH FRIENDLY)
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Live Model Training")
st.sidebar.write("Train the Artificial Intelligence models directly on any asset's live market data.")

train_symbol = st.sidebar.text_input("Train Symbol Ticker", value=symbol, help="Enter the ticker symbol you want to train models on.")
if st.sidebar.button("Train AI on Live Data"):
    if not train_symbol or not train_symbol.strip():
        st.sidebar.error("⚠️ Please specify a symbol ticker in 'Train Symbol Ticker' before training.")
    else:
        with st.spinner(f"Training AI on {train_symbol}... This downloads live data, builds technical indicators, and fits ARIMA + LSTM models. Takes ~1 minute."):
            try:
                import subprocess
                py_path = sys.executable
                # Run the training script in a subprocess
                result = subprocess.run(
                    [py_path, "train_model.py", "--symbol", train_symbol.strip()],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.sidebar.success(f"AI trained successfully on {train_symbol}!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Model Training Status")
models_trained = os.path.exists(LSTM_MODEL_PATH) and os.path.exists(ARIMA_MODEL_PATH)

if models_trained:
    st.sidebar.success("ARIMA + LSTM Models Trained")
else:
    st.sidebar.warning("Models Not Trained yet.")

# MAIN TITLE WITH UNIQUE LOGO
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 25px; border-bottom: 1px solid {card_border}; padding-bottom: 15px;">
<svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 15px;">
<!-- Candlestick 1 (Green/Bullish) -->
<line x1="4.5" y1="4" x2="4.5" y2="20" stroke="#10b981" stroke-width="2" stroke-linecap="round"/>
<rect x="2.5" y="8" width="4" height="8" rx="1" fill="#10b981"/>
<!-- Candlestick 2 (Red/Bearish) -->
<line x1="10.5" y1="2" x2="10.5" y2="22" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/>
<rect x="8.5" y="6" width="4" height="12" rx="1" fill="#ef4444"/>
<!-- Telescope / Lens overlay (Accent Color) -->
<circle cx="17" cy="10" r="4.5" stroke="{accent_color}" stroke-width="2.5" fill="none"/>
<line x1="20" y1="13" x2="23" y2="16" stroke="{accent_color}" stroke-width="2.5" stroke-linecap="round"/>
</svg>
<div style="display: flex; align-items: baseline; flex-wrap: wrap;">
<h1 style="margin: 0; padding: 0; font-family: 'Outfit', sans-serif; font-size: 32px; font-weight: 800; background: {title_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;">
TradeLens
</h1>
<span style="font-size: 18px; font-weight: 400; color: {text_muted}; font-family: 'Inter', sans-serif; letter-spacing: -0.5px; margin-left: 10px; display: inline-block;">
Stock Forecasting & Analytics
</span>
</div>
</div>
""", unsafe_allow_html=True)

# FETCH DATA FOR SELECT TIME FRAME
if not symbol or not symbol.strip():
    st.warning("⚠️ Please enter a valid Ticker Symbol in the configuration sidebar (e.g., ^NSEI, BTC-USD, RELIANCE.NS).")
    st.stop()

with st.spinner("Fetching market data..."):
    df = fetch_live_data(symbol.strip(), timeframe_label)

if df.empty:
    st.error(f"No market data found for {symbol} on {timeframe_label} timeframe.")
    st.stop()

# FEATURE ENGINEERING
with st.spinner("Engineering technical indicators..."):
    featured = compute_all_features(df)
    featured_clean = featured.dropna()

# --- TOP STATS PANEL ---
col1, col2, col3, col4 = st.columns(4)
latest_row = df.iloc[-1]
latest_close = float(latest_row["close"])
prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else latest_close
change = latest_close - prev_close
pct_change = (change / prev_close) * 100

with col1:
    st.metric(
        label=f"Latest Close ({symbol})",
        value=f"{latest_close:,.2f}",
        delta=f"{change:+.2f} ({pct_change:+.2f}%)"
    )

# Compute current signals using ML models if available, else fallback
prediction_result = None
if models_trained:
    try:
        # Predict using live context
        lstm = LSTMForecaster()
        lstm.load()

        arima = ARIMAForecaster()
        arima.load()

        ensemble = EnsembleForecaster()

        # Get inputs
        available_cols = [c for c in lstm.feature_columns if c in featured.columns]
        recent = featured_clean[available_cols].tail(LSTM_SEQUENCE_LENGTH).values

        if len(recent) == LSTM_SEQUENCE_LENGTH:
            lstm_pred = lstm.predict_next(recent)
            arima_pred = arima.predict(steps=1)[0]
            prediction_result = ensemble.predict_single(lstm_pred, arima_pred, latest_close)
    except Exception as e:
        st.sidebar.error(f"Prediction failed: {e}")

# Display signal in Col 2
with col2:
    if prediction_result:
        sig = prediction_result["signal"]
        conf = prediction_result["confidence"]
        if sig == "BUY":
            st.metric("Ensemble Signal", "BUY", f"Confidence: {conf*100:.1f}%", delta_color="normal")
        elif sig == "SELL":
            st.metric("Ensemble Signal", "SELL", f"Confidence: {conf*100:.1f}%", delta_color="inverse")
        else:
            st.metric("Ensemble Signal", "HOLD", f"Confidence: {conf*100:.1f}%", delta_color="off")
    else:
        # Fallback ema signal
        ema9 = df["close"].ewm(span=9).mean()
        ema21 = df["close"].ewm(span=21).mean()
        sig = "BUY" if ema9.iloc[-1] > ema21.iloc[-1] else "SELL"
        st.metric("EMA Crossover Signal", sig, "Models not trained fallback")

with col3:
    # Volume spike calculation
    vol_sma = df["volume"].rolling(20).mean()
    vol_ratio = float(df["volume"].iloc[-1] / vol_sma.iloc[-1]) if not vol_sma.empty else 1.0
    st.metric(
        label="Volume Ratio (vs 20 MA)",
        value=f"{vol_ratio:.2f}x",
        delta=f"Vol: {int(df['volume'].iloc[-1]):,}"
    )

with col4:
    # RSI indicator value
    rsi_val = featured["rsi"].iloc[-1] if "rsi" in featured.columns else 50.0
    st.metric(
        label="14-Period RSI",
        value=f"{rsi_val:.1f}",
        delta="Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral",
        delta_color="off"
    )

# Calculate Pivot levels and execution setup for the active symbol & timeframe
levels = calculate_pivot_points(df)
# Estimate ATR for volatility sizing
high_low_diff = df["high"] - df["low"]
atr_val = float(high_low_diff.rolling(14).mean().iloc[-1]) if len(df) >= 14 else (latest_close * 0.01)
if np.isnan(atr_val) or atr_val <= 0:
    atr_val = latest_close * 0.01

active_setup = get_trade_setup(latest_close, sig, atr_val, levels)

# --- EXECUTION LEVELS & MULTI-TIMEFRAME GRID ---
st.markdown("---")
st.markdown("### 🎯 Live Execution Levels & Multi-Timeframe Matrix")

# Calculate multi-timeframe grid data
with st.spinner("Synthesizing multi-timeframe levels..."):
    mtf_grid = synthesize_multi_timeframe_levels(
        symbol=symbol,
        fetch_live_data_fn=fetch_live_data
    )

# Render columns for each timeframe
if mtf_grid:
    mtf_cols = st.columns(len(mtf_grid))
    for idx, (tf, data) in enumerate(mtf_grid.items()):
        with mtf_cols[idx]:
            # Highlight active timeframe
            is_active = (tf == timeframe_label)
            active_border = f"border: {grid_active_border};" if is_active else f"border: {grid_inactive_border};"
            mtf_card_bg = grid_active_bg if is_active else grid_inactive_bg
            
            # Trend badge styling
            if data["trend"] == "BUY":
                trend_color = trend_buy_bg
                trend_text_color = trend_buy_text
            elif data["trend"] == "SELL":
                trend_color = trend_sell_bg
                trend_text_color = trend_sell_text
            else:
                trend_color = trend_hold_bg
                trend_text_color = trend_hold_text
            
            st.markdown(f"""
            <div style="{active_border} background-color: {mtf_card_bg}; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 800; font-size: 15px; color: {text_color if is_active else text_muted}">{tf} {'★' if is_active else ''}</span>
                    <span style="background-color: {trend_color}; color: {trend_text_color}; padding: 2px 8px; border-radius: 6px; font-weight: bold; font-size: 11px;">{data['trend']}</span>
                </div>
                <div style="font-size: 20px; font-weight: bold; color: {text_color}; margin-bottom: 12px;">{data['price']:,.2f}</div>
                <div style="font-size: 11px; margin-bottom: 2px; color: {text_muted};">[+] <b>Buy Entry:</b></div>
                <div style="font-size: 13px; font-family: monospace; color: {setup_card_buy_border}; margin-bottom: 6px; font-weight: bold;">{data['buy_zone_min']:,.2f} - {data['buy_zone_max']:,.2f}</div>
                <div style="font-size: 11px; margin-bottom: 2px; color: {text_muted};">[-] <b>Short Entry:</b></div>
                <div style="font-size: 13px; font-family: monospace; color: {setup_card_sell_border}; margin-bottom: 6px; font-weight: bold;">{data['sell_zone_min']:,.2f} - {data['sell_zone_max']:,.2f}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px; margin-top: 8px; border-top: 1px solid {card_border}; padding-top: 8px;">
                    <div>🛡️ <span style="color: {text_color}; font-weight: bold;">SL:</span><br/><span style="font-family: monospace; font-size: 11px; color: {setup_card_sell_border};">{data['stop_loss']:,.2f}</span></div>
                    <div>🎯 <span style="color: {text_color}; font-weight: bold;">T1:</span><br/><span style="font-family: monospace; font-size: 11px; color: {setup_card_buy_border};">{data['target_1']:,.2f}</span></div>
                </div>
                <div style="font-size: 11px; margin-top: 4px;">🎯 <span style="color: {text_color}; font-weight: bold;">T2:</span> <span style="font-family: monospace; font-size: 11px; font-weight: bold; color: {setup_card_buy_border};">{data['target_2']:,.2f}</span></div>
                <div style="font-size: 11px; color: {text_muted}; margin-top: 6px; text-align: right;">R/R: {data['rr_ratio']:.1f}x</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Insufficient data to build multi-timeframe level grid.")

# --- CHART AND FORECAST PANEL ---
chart_col, forecast_col = st.columns([3, 1])

with chart_col:
    st.subheader("📊 Price Chart & Technical Overlays")
    chart_type = st.radio("Chart Style", ["Candlestick", "Line"], horizontal=True)
    overlay_options = st.multiselect(
        "Technical Indicators Overlay",
        ["EMA 9", "EMA 21", "SMA 50", "SMA 200", "Bollinger Bands", "Ichimoku Cloud", "Camarilla Pivots", "Fibonacci Pivots"],
        default=["EMA 9", "EMA 21"]
    )

    fig = go.Figure()

    # Price chart
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="OHLC Price",
            increasing_line_color="#10b981", # Standard trading green
            decreasing_line_color="#ef4444", # Standard trading red
            increasing_fillcolor="#10b981",
            decreasing_fillcolor="#ef4444"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"], mode="lines", name="Price", line=dict(color=accent_color, width=2)
        ))

    # Overlays
    if "EMA 9" in overlay_options and "ema_9" in featured.columns:
        fig.add_trace(go.Scatter(x=df.index, y=featured["ema_9"], mode="lines", name="EMA 9", line=dict(color="#FF9800", width=1.2)))
    if "EMA 21" in overlay_options and "ema_21" in featured.columns:
        fig.add_trace(go.Scatter(x=df.index, y=featured["ema_21"], mode="lines", name="EMA 21", line=dict(color="#E91E63", width=1.2)))
    if "SMA 50" in overlay_options and "sma_50" in featured.columns:
        fig.add_trace(go.Scatter(x=df.index, y=featured["sma_50"], mode="lines", name="SMA 50", line=dict(color="#4CAF50", width=1.2)))
    if "SMA 200" in overlay_options and "sma_200" in featured.columns:
        fig.add_trace(go.Scatter(x=df.index, y=featured["sma_200"], mode="lines", name="SMA 200", line=dict(color="#9C27B0", width=1.5)))

    if "Bollinger Bands" in overlay_options and "bb_upper" in featured.columns:
        fig.add_trace(go.Scatter(x=df.index, y=featured["bb_upper"], line=dict(color="rgba(173,216,230,0.4)", width=0.8), name="BB Upper"))
        fig.add_trace(go.Scatter(x=df.index, y=featured["bb_lower"], line=dict(color="rgba(173,216,230,0.4)", width=0.8), fill="tonexty", fillcolor="rgba(173,216,230,0.05)", name="BB Lower"))

    if "Ichimoku Cloud" in overlay_options and "ichi_senkou_a" in featured.columns:
        fig.add_trace(go.Scatter(x=df.index, y=featured["ichi_senkou_a"], line=dict(color="rgba(76,175,80,0.3)", width=0.8), name="Senkou A"))
        fig.add_trace(go.Scatter(x=df.index, y=featured["ichi_senkou_b"], line=dict(color="rgba(244,67,54,0.3)", width=0.8), fill="tonexty", fillcolor="rgba(128,128,128,0.05)", name="Senkou B"))

    if "Camarilla Pivots" in overlay_options:
        cam = levels.get("camarilla", {})
        if cam:
            fig.add_hline(y=cam["R4"], line_dash="dash", line_color="#ef4444", annotation_text="Cam R4 (Breakout)", annotation_position="top left")
            fig.add_hline(y=cam["R3"], line_dash="dot", line_color="#f87171", annotation_text="Cam R3 (Short Zone)", annotation_position="top left")
            fig.add_hline(y=cam["S3"], line_dash="dot", line_color="#34d399", annotation_text="Cam S3 (Buy Zone)", annotation_position="bottom left")
            fig.add_hline(y=cam["S4"], line_dash="dash", line_color="#10b981", annotation_text="Cam S4 (Breakdown)", annotation_position="bottom left")

    if "Fibonacci Pivots" in overlay_options:
        fib = levels.get("fibonacci", {})
        if fib:
            fig.add_hline(y=fib["R3"], line_dash="dash", line_color="#b91c1c", annotation_text="Fib R3 (Target 2)", annotation_position="top right")
            fig.add_hline(y=fib["R2"], line_dash="dot", line_color="#ef4444", annotation_text="Fib R2 (Target 1)", annotation_position="top right")
            fig.add_hline(y=fib["P"], line_color="#9ca3af", annotation_text="Fib Pivot P", annotation_position="top right")
            fig.add_hline(y=fib["S2"], line_dash="dot", line_color="#10b981", annotation_text="Fib S2 (Support)", annotation_position="bottom right")
            fig.add_hline(y=fib["S3"], line_dash="dash", line_color="#047857", annotation_text="Fib S3 (Deep Support)", annotation_position="bottom right")

    fig.update_layout(
        height=550,
        template=plotly_template,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color
    )
    
    # Custom axes formatting to ensure crisp grid lines and readable ticks
    fig.update_xaxes(
        gridcolor=plotly_grid_color,
        zerolinecolor=plotly_grid_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color)
    )
    fig.update_yaxes(
        gridcolor=plotly_grid_color,
        zerolinecolor=plotly_grid_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color)
    )
    st.plotly_chart(fig, use_container_width=True)

with forecast_col:
    st.subheader("🎯 Active Trade Setup")
    
    # Active setup card
    setup_action = active_setup["trend_signal"]
    
    if setup_action == "BUY":
        st.markdown(f"""
        <div style="background-color: {setup_card_buy_bg}; border: 2px solid {setup_card_buy_border}; padding: 15px; border-radius: 12px; margin-bottom: 15px;">
            <h4 style="color: {setup_card_buy_border}; margin: 0 0 10px 0;">[+] BUY / LONG SETUP</h4>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Buy Zone:</b> <span style="font-family: monospace; font-weight: bold; color: {card_text_color};">{active_setup['buy_zone'][0]:,.2f} - {active_setup['buy_zone'][1]:,.2f}</span></div>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Stop Loss:</b> <span style="font-family: monospace; color: {setup_card_sell_border}; font-weight: bold;">{active_setup['stop_loss']:,.2f}</span></div>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Target 1:</b> <span style="font-family: monospace; color: {setup_card_buy_border}; font-weight: bold;">{active_setup['target_1']:,.2f}</span></div>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Target 2:</b> <span style="font-family: monospace; color: {setup_card_buy_border}; font-weight: bold;">{active_setup['target_2']:,.2f}</span></div>
            <div style="font-size: 12px; margin-top: 10px; color: {text_muted}; border-top: 1px solid {card_border}; padding-top: 6px;">
                R/R Ratio: <b>{active_setup['risk_reward_ratio']:.2f}x</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif setup_action == "SELL":
        st.markdown(f"""
        <div style="background-color: {setup_card_sell_bg}; border: 2px solid {setup_card_sell_border}; padding: 15px; border-radius: 12px; margin-bottom: 15px;">
            <h4 style="color: {setup_card_sell_border}; margin: 0 0 10px 0;">[-] SHORT / SELL SETUP</h4>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Short Zone:</b> <span style="font-family: monospace; font-weight: bold; color: {card_text_color};">{active_setup['sell_zone'][0]:,.2f} - {active_setup['sell_zone'][1]:,.2f}</span></div>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Stop Loss:</b> <span style="font-family: monospace; color: {setup_card_buy_border}; font-weight: bold;">{active_setup['stop_loss']:,.2f}</span></div>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Target 1:</b> <span style="font-family: monospace; color: {setup_card_sell_border}; font-weight: bold;">{active_setup['target_1']:,.2f}</span></div>
            <div style="font-size: 13px; margin-bottom: 6px; color: {text_muted};"><b>Target 2:</b> <span style="font-family: monospace; color: {setup_card_sell_border}; font-weight: bold;">{active_setup['target_2']:,.2f}</span></div>
            <div style="font-size: 12px; margin-top: 10px; color: {text_muted}; border-top: 1px solid {card_border}; padding-top: 6px;">
                R/R Ratio: <b>{active_setup['risk_reward_ratio']:.2f}x</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 15px; border-radius: 12px; margin-bottom: 15px;">
            <h4 style="color: {text_muted}; margin: 0 0 10px 0;">[=] RANGEBOUND SETUP</h4>
            <div style="font-size: 12px; line-height: 1.4; color: {card_text_color};">{active_setup['rationale']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.info(active_setup["rationale"])
    st.markdown("---")

    st.subheader("🤖 ML Forecasts")
    if prediction_result:
        st.write(f"**LSTM Predict:** {prediction_result['lstm_prediction']:,.2f}")
        st.write(f"**ARIMA Predict:** {prediction_result['arima_prediction']:,.2f}")
        st.write(f"**Blend Price:** {prediction_result['ensemble_prediction']:,.2f}")
        st.write(f"**Predicted Return:** {prediction_result['predicted_return']*100:+.2f}%")
        
        # Signal Gauge
        sig_val = prediction_result["predicted_return"]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sig_val * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Return Forecast %", 'font': {'size': 13, 'color': text_color}},
            number={'font': {'color': text_color}},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': text_color, 'tickfont': {'color': text_color}},
                'bar': {'color': accent_color},
                'steps': gauge_steps,
                'threshold': {
                    'line': {'color': text_color, 'width': 3},
                    'thickness': 0.75,
                    'value': sig_val * 100
                }
            }
        ))
        fig_g.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), template=plotly_template, paper_bgcolor=bg_color)
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("Models not trained yet. Showing fallback EMA indicators.")
        st.write("Train ARIMA & LSTM model in training pipeline to get ensemble model forecasts with confidence levels.")

# --- LOWER DETAILS PANEL (INDICATORS, METRICS, SIGNALS) ---
st.markdown("---")
details_tab1, details_tab2, details_tab3 = st.tabs(["🚀 Technical Indicators Info", "📈 Model Evaluation Report", "📝 Recent Signals History"])

with details_tab1:
    st.subheader("Indicators Summary")
    st.write("Below are the values of standard technical indicators calculated on the active timeframe.")
    if not featured_clean.empty:
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        with ind_col1:
            st.markdown("**Oscillators (Momentum)**")
            st.write(f"**RSI (Relative Strength Index):** {featured_clean['rsi'].iloc[-1]:.2f}")
            st.caption("Measures speed and change of price moves. >70 is Overbought (Sell pressure), <30 is Oversold (Buy pressure).")
            st.write(f"**MACD Histogram:** {featured_clean['macd_hist'].iloc[-1]:.4f}")
            st.caption("Shows momentum shifts. Positive histogram indicates bullish momentum; negative is bearish.")
            st.write(f"**Stochastic %K:** {featured_clean['stoch_k'].iloc[-1]:.2f}")
            st.caption("Compares close price to range over time. >80 indicates overbought; <20 indicates oversold.")
            st.write(f"**CCI (Commodity Channel Index):** {featured_clean['cci'].iloc[-1]:.2f}")
            st.caption("Identifies new trends or extreme conditions. >100 is strong uptrend; <-100 is strong downtrend.")
        with ind_col2:
            st.markdown("**Trend & Momentum**")
            st.write(f"**ADX (Trend Strength):** {featured_clean['adx'].iloc[-1]:.2f}")
            st.caption("Measures overall trend strength (not direction). >25 is a strong trend; <20 is a weak or flat trend.")
            st.write(f"**ROC (Rate of Change):** {featured_clean['roc'].iloc[-1]:.2f}%")
            st.caption("Percentage change in price over last 10 periods. Positive shows rising speed; negative shows falling.")
            st.write(f"**Williams %R:** {featured_clean['williams_r'].iloc[-1]:.2f}")
            st.caption("Similar to Stochastic. Reflected scale (-100 to 0). >-20 is overbought; <-80 is oversold.")
        with ind_col3:
            st.markdown("**Volatility & Volume**")
            st.write(f"**ATR (Average True Range):** {featured_clean['atr'].iloc[-1]:.2f}")
            st.caption("Measures market volatility. Higher values indicate higher price swings.")
            st.write(f"**VWAP (Volume Weighted Avg):** {featured_clean['vwap'].iloc[-1]:,.2f}")
            st.caption("Average price weighted by volume. Trading above is bullish; below is bearish.")
            st.write(f"**OBV Slope (On-Balance Volume):** {featured_clean['obv_slope'].iloc[-1]:,.0f}")
            st.caption("Slope of cumulative volume. Confirming price rises with rising volume slope is bullish.")
    else:
        st.write("Insufficient data for full indicators computation.")

with details_tab2:
    st.subheader("Performance Evaluation Metrics")
    st.write("These metrics evaluate how accurately the AI models predicted price moves on test data.")
    if os.path.exists(EVALUATION_REPORT_PATH):
        with open(EVALUATION_REPORT_PATH, "r") as f:
            metrics_report = json.load(f)

        m_cols = st.columns(len(metrics_report))
        for idx, (model_name, metrics) in enumerate(metrics_report.items()):
            with m_cols[idx]:
                st.markdown(f"##### {model_name}")
                st.write(f"**RMSE (Avg Price Error):** {metrics['rmse']:.4f}")
                st.caption("Root Mean Square Error. Measures average dollar prediction error. Lower is better.")
                st.write(f"**MAE (Median Price Error):** {metrics['mae']:.4f}")
                st.caption("Mean Absolute Error. Measures absolute average difference. Lower is better.")
                st.write(f"**MAPE (Percent Error):** {metrics['mape']:.2f}%")
                st.caption("Mean Absolute Percentage Error. Tells you the percentage prediction error (e.g. 3% MAPE means 97% accurate). Lower is better.")
                st.write(f"**Directional Accuracy:** {metrics['directional_accuracy']:.2f}%")
                st.caption("How often the model correctly guessed whether the price would go Up or Down.")
                st.write(f"**R² Score:** {metrics['r_squared']:.4f}")
                st.caption("R-squared. Tells you how much variance of price the model captures. 1.0 is a perfect fit; 0.0 is random.")
    else:
        st.info("No evaluation report found. Run model training to generate the performance metrics.")

with details_tab3:
    st.subheader("Recent Generated Signals")
    st.write("The recent signal history captures direction indicators based on SMA crossovers and Ichimoku cloud positions.")
    # Generate signals table
    if not featured_clean.empty:
        signals_df = featured_clean[["close", "trend_sma", "trend_ichimoku"]].tail(10).copy()
        signals_df.columns = ["Price", "SMA Trend Direction", "Ichimoku Trend Direction"]
        signals_df["SMA Trend Direction"] = signals_df["SMA Trend Direction"].map({1: "Bullish (Up)", -1: "Bearish (Down)", 0: "Neutral"})
        signals_df["Ichimoku Trend Direction"] = signals_df["Ichimoku Trend Direction"].map({1: "Above Cloud (Bullish)", -1: "Below Cloud (Bearish)", 0: "Neutral"})
        st.dataframe(signals_df.style.format({"Price": "{:,.2f}"}), use_container_width=True)
    else:
        st.write("No signal history available.")