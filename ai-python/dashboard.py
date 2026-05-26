import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("TradeLens Live Signals")

# FETCH DATA
data = yf.download("^NSEI", period="5d", interval="5m")

# FIX MULTI-INDEX COLUMNS
if isinstance(data.columns, tuple) or hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

# CLOSE PRICES
close_prices = data['Close']

# CHART
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=close_prices,
        mode='lines',
        name='NIFTY'
    )
)

st.plotly_chart(fig, use_container_width=True)

# LATEST PRICE
latest_close = float(close_prices.iloc[-1])

# SIMPLE AI SIGNAL
if latest_close > float(close_prices.mean()):
    st.success("BUY SIGNAL")
else:
    st.error("SELL SIGNAL")

st.metric(
    label="Latest Price",
    value=round(latest_close, 2)
)