import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# PAGE
st.set_page_config(
    page_title="TradeLens Live Signals",
    layout="wide"
)

st.title("TradeLens Live Signals")

# FETCH LIVE DATA
data = yf.download(
    "^NSEI",
    period="1d",
    interval="5m"
)

# CHECK DATA
if data.empty:
    st.error("No market data found")
    st.stop()

# FIX MULTI COLUMN ISSUE
close_prices = data["Close"]

if hasattr(close_prices, "columns"):
    close_prices = close_prices.iloc[:, 0]

close_prices = close_prices.dropna()

# EMA
ema9 = close_prices.ewm(span=9).mean()
ema21 = close_prices.ewm(span=21).mean()

# CREATE CHART
fig = go.Figure()

# PRICE
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=close_prices,
        mode="lines",
        name="NIFTY",
    )
)

# EMA 9
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=ema9,
        mode="lines",
        name="EMA 9",
    )
)

# EMA 21
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=ema21,
        mode="lines",
        name="EMA 21",
    )
)

# CHART DESIGN
fig.update_layout(
    height=500,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=40, b=20)
)

# SHOW CHART
st.plotly_chart(
    fig,
    width="stretch"
)

# SIGNAL
latest_close = float(close_prices.iloc[-1])

if ema9.iloc[-1] > ema21.iloc[-1]:
    st.success("BUY SIGNAL")
else:
    st.error("SELL SIGNAL")

# METRIC
st.metric(
    label="Latest Price",
    value=round(latest_close, 2)
)