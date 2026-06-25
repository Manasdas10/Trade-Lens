"""
config.py — Central Configuration for Market Engine AI
All paths, hyperparameters, timeframe definitions, and feature settings.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_ROOT, "src", "main", "resources", "nifty.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
ARIMA_MODEL_PATH = os.path.join(MODELS_DIR, "arima_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
EVALUATION_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# DATA SETTINGS
# ============================================================
DATE_COLUMN = "date"
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "close"

TRAIN_TEST_SPLIT = 0.80  # 80% train, 20% test

# ============================================================
# TIMEFRAME DEFINITIONS
# ============================================================
# Maps user-friendly labels to Pandas resample frequency strings
TIMEFRAMES = {
    "5m": "5min",      # base resolution (raw data)
    "15m": "15min",
    "1H": "1h",
    "3H": "3h",
    "4H": "4h",
    "1D": "1D",
    "1W": "1W",
}

DEFAULT_TIMEFRAME = "1D"  # primary timeframe for model training

# ============================================================
# TECHNICAL INDICATOR PARAMETERS
# ============================================================
SMA_PERIODS = [10, 20, 50, 200]
EMA_PERIODS = [9, 21, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ADX_PERIOD = 14
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU = 52
CCI_PERIOD = 20
WILLIAMS_R_PERIOD = 14
ROC_PERIOD = 10
LAG_PERIODS = [1, 5, 10]

# ============================================================
# LSTM HYPERPARAMETERS
# ============================================================
LSTM_SEQUENCE_LENGTH = 60          # lookback window (timesteps)
LSTM_UNITS_LAYER_1 = 128
LSTM_UNITS_LAYER_2 = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001
LSTM_EARLY_STOPPING_PATIENCE = 10
LSTM_LR_REDUCE_PATIENCE = 5
LSTM_LR_REDUCE_FACTOR = 0.5

# ============================================================
# ARIMA HYPERPARAMETERS
# ============================================================
ARIMA_DEFAULT_ORDER = (5, 1, 2)  # fallback (p,d,q)
ARIMA_MAX_P = 7
ARIMA_MAX_D = 2
ARIMA_MAX_Q = 7
ARIMA_SEASONAL = False
ARIMA_FORECAST_STEPS = 10  # number of steps to forecast

# ============================================================
# ENSEMBLE WEIGHTS
# ============================================================
ENSEMBLE_LSTM_WEIGHT = 0.60
ENSEMBLE_ARIMA_WEIGHT = 0.40

# ============================================================
# REAL-TIME / API SETTINGS
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
CACHE_TTL_SECONDS = 60     # data cache time-to-live
DEFAULT_SYMBOL = "^NSEI"   # NIFTY 50

# yfinance interval mappings for live data
YFINANCE_INTERVALS = {
    "5m": "5m",
    "15m": "15m",
    "1H": "60m",
    "4H": "60m",   # fetch 60m and resample to 4H
    "1D": "1d",
    "1W": "1wk",
}

YFINANCE_PERIODS = {
    "5m": "60d",
    "15m": "60d",
    "1H": "730d",
    "4H": "730d",
    "1D": "5y",
    "1W": "10y",
}

# ============================================================
# PREDICTION THRESHOLDS
# ============================================================
BUY_THRESHOLD = 0.002    # predicted return > 0.2% → BUY
SELL_THRESHOLD = -0.002  # predicted return < -0.2% → SELL
