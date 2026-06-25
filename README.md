# TradeLens — Java Quant Trading Engine (Stock Price Forcsting)

TradeLens is a Java-based algorithmic trading and market analysis engine designed for realtime signal generation, quantitative research, and AI-powered market prediction.

The platform integrates:

- Java trading engine
- Technical indicators
- Backtesting framework
- Performance analytics
- Monte Carlo simulations
- Python AI integration
- Live market dashboards
- Cloud deployment

TradeLens combines Java backend performance with Python AI capabilities to create a lightweight quantitative trading platform.

---

# Live Deployment

## Realtime Dashboard

https://trade-lens-yhjrfh2lecepvhcvdgcqmg.streamlit.app/

---

# Core Features

## Java Trading Engine

✅ Modular strategy engine  
✅ Realtime signal generation  
✅ Historical market analysis  
✅ Candle-based processing  
✅ Trade execution simulation  
✅ Portfolio management  
✅ Risk modules  
✅ Indicator framework  

---

## Quantitative Analysis

✅ EMA crossover strategy  
✅ Performance analytics  
✅ Monte Carlo simulations  
✅ Backtesting framework  
✅ Statistical trade evaluation  

---

## AI + Python Integration

✅ Python prediction modules  
✅ AI signal generation  
✅ Streamlit realtime dashboard  
✅ Plotly chart visualization  
✅ Live market deployment  

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Java 17 | Core trading engine |
| Maven | Dependency management |
| Python | AI integration |
| Streamlit | Live dashboard |
| Plotly | Interactive charts |
| yFinance | Market data |
| GitHub | Source control |
| Streamlit Cloud | Deployment |

---

# Architecture

```text
                +-------------------+
                |   Market Data     |
                |  Yahoo Finance    |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Java Trading Core |
                | Strategy Engine   |
                +---------+---------+
                          |
        +-----------------+-----------------+
        |                                   |
        v                                   v
+---------------+                 +------------------+
| Indicators    |                 | Risk Management  |
| EMA / SMA     |                 | Position Sizing  |
+---------------+                 +------------------+
        |
        v
+-------------------+
| Signal Generation |
+-------------------+
        |
        v
+-------------------+
| Python AI Layer   |
| Predictor Models  |
+-------------------+
        |
        v
+-------------------+
| Streamlit UI      |
| Live Dashboard    |
+-------------------+


**Java Engine Features**

1. Strategy Engine

-The engine supports pluggable trading strategies.

Current implementation:

- EMA crossover strategy
- Signal-based execution
- Historical candle analysis

2. Market Data Integration

- Market data is fetched using:
YahooFinance.get("^NSEI");

Supported:

Historical candles
Daily intervals
Intraday intervals
OHLCV processing

3. Backtesting Framework

- TradeLens includes a Java-based backtesting engine for:

Historical strategy testing
Trade simulation
Profit/loss calculation
Equity curve generation

4. Performance Analytics

- Performance metrics include:

Total return
Win rate
Profit factor
Trade statistics
Drawdown analysis

5. Monte Carlo Simulation

- Monte Carlo simulations are used for:

Risk analysis
Probability estimation
Strategy robustness testing
Trading Strategy
EMA Crossover Strategy
BUY Signal

- Generated when:

EMA 9 > EMA 21
SELL Signal

- Generated when:

EMA 9 < EMA 21

Installation

Clone Repository
git clone https://github.com/Manasdas10/Trade-Lens.git

Open Project
cd Trade-Lens
Java Setup
Build Maven Project
mvn clean install
Run Java Engine
mvn exec:java -Dexec.mainClass="com.manas.engine.Main"
Python Dashboard Setup
Install Dependencies
pip install -r ai-python/requirements.txt
Run Dashboard
python -m streamlit run ai-python/dashboard.py
Live Dashboard Features

- The deployed dashboard includes:

✅ Live NIFTY chart
✅ Realtime BUY/SELL signals
✅ EMA indicators
✅ Latest price feed
✅ Interactive market visualization

Future Roadmap

Planned upgrades:

LSTM forecasting models
Reinforcement learning strategies
TradingView webhook integration
Zerodha Kite API integration
Live broker execution
Portfolio optimization
Multi-asset trading
WebSocket market feeds
Order management system
AI-based risk engine
Deployment Pipeline
VS Code
   ↓
GitHub
   ↓
Streamlit Cloud
   ↓
Live Trading Dashboard


Author
Manas Das

GitHub:
https://github.com/Manasdas10
