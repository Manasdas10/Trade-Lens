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
close_prices = data["Close"].dropna()

if len(close_prices) == 0:
    st.error("No live market data available.")
    st.stop()
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

st.plotly_chart(fig, width='stretch')

# LATEST PRICE
latest_close = float(close_prices.values[-1])
# SIGNAL
if latest_close > float(close_prices.mean()):
    st.success("BUY SIGNAL")
else:
    st.error("SELL SIGNAL")

# METRIC
st.metric(
    label="Latest Price",
    value=round(latest_close, 2)
)