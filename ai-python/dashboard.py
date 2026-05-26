import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("TradeLens Live Signals")

# FETCH DATA
data = yf.download("^NSEI", period="5d", interval="5m")

# CHECK EMPTY
if data.empty:
    st.error("Market data unavailable right now.")
    st.stop()

# CLOSE PRICES
close_prices = data["Close"]

# FIX MULTI-DIMENSION ISSUE
if hasattr(close_prices, "columns"):
    close_prices = close_prices.iloc[:, 0]

close_prices = close_prices.dropna()

# EMPTY CHECK
if len(close_prices) == 0:
    st.error("No live market data available.")
    st.stop()

# LATEST PRICE
latest_close = close_prices.iloc[-1]

# AVERAGE PRICE
avg_price = close_prices.mean()

# SIGNAL
if latest_close > avg_price:
    st.success("BUY SIGNAL")
else:
    st.error("SELL SIGNAL")

# METRIC
st.metric(
    label="Latest Price",
    value=round(float(latest_close), 2)
)