import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "ai-python")

from config import DEFAULT_TIMEFRAME, TRAIN_TEST_SPLIT, LSTM_SEQUENCE_LENGTH
from data_loader import load_csv
from timeframe_resampler import resample_all
from features import compute_all_features, prepare_for_model
from lstm_forecaster import LSTMForecaster
from arima_forecaster import ARIMAForecaster
from ensemble_forecaster import EnsembleForecaster

def test():
    # Load data
    df_raw = load_csv()
    all_timeframes = resample_all(df_raw)
    df_tf = all_timeframes[DEFAULT_TIMEFRAME]
    df_featured = compute_all_features(df_tf)
    df_clean, feature_cols = prepare_for_model(df_featured)
    
    split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
    test_df = df_clean.iloc[split_idx:]
    
    # Load models
    lstm = LSTMForecaster()
    lstm.load()
    
    arima = ARIMAForecaster()
    arima.load()
    
    ensemble = EnsembleForecaster()
    
    # LSTM predictions
    lstm_preds = lstm.predict(test_df)
    
    # Align prices
    test_close = test_df["close"].values
    lstm_offset = LSTM_SEQUENCE_LENGTH
    
    actual = test_close[lstm_offset: lstm_offset + len(lstm_preds)]
    prev_actual = test_close[lstm_offset - 1: lstm_offset - 1 + len(lstm_preds)]
    
    # Calculate directional accuracy: predicted_next vs prev_actual
    actual_dir = np.sign(actual - prev_actual)
    pred_dir = np.sign(lstm_preds - prev_actual)
    
    mask = actual_dir != 0
    accuracy = np.sum(actual_dir[mask] == pred_dir[mask]) / mask.sum() * 100
    print(f"Correctly computed LSTM Directional Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test()
