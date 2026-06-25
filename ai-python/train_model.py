"""
train_model.py — Full Training Pipeline Orchestration
Loads 117K+ records, engineers features, trains ARIMA & LSTM,
evaluates with RMSE/MAE/MAPE/R²/Directional Accuracy, and saves models.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Ensure ai-python is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_TEST_SPLIT, DEFAULT_TIMEFRAME,
    LSTM_SEQUENCE_LENGTH,
)
from data_loader import load_csv
from timeframe_resampler import resample_all
from features import compute_all_features, get_feature_columns, prepare_for_model
from arima_forecaster import ARIMAForecaster
from lstm_forecaster import LSTMForecaster
from ensemble_forecaster import EnsembleForecaster
from evaluator import (
    evaluate, evaluate_all, save_report,
    plot_actual_vs_predicted, plot_comparison,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None, help="Live yfinance ticker symbol to train on (e.g. BTC-USD)")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("  MARKET ENGINE — MODEL TRAINING PIPELINE")
    print("=" * 60)
    print()

    # =========================================================
    # STEP 1: Load Data
    # =========================================================
    print("[Step 1/7] Loading historical data...")
    if args.symbol:
        print(f"  Fetching historical data for live training on {args.symbol}...")
        import yfinance as yf
        df_raw = yf.download(args.symbol, period="5y", interval="1d", progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        df_raw = df_raw.dropna(subset=["close"])
        print(f"  Raw records loaded for {args.symbol}: {len(df_raw):,}")
    else:
        df_raw = load_csv()
    print()

    # =========================================================
    # STEP 2: Resample to Multiple Timeframes / Prepare
    # =========================================================
    if args.symbol:
        print("[Step 2/7] Live training mode: Using daily close prices directly.")
        df_tf = df_raw.copy()
    else:
        print("[Step 2/7] Resampling to multiple timeframes...")
        all_timeframes = resample_all(df_raw)
        df_tf = all_timeframes[DEFAULT_TIMEFRAME]
    print()

    # =========================================================
    # STEP 3: Feature Engineering
    # =========================================================
    print(f"[Step 3/7] Engineering features on '{DEFAULT_TIMEFRAME}' timeframe...")
    df_featured = compute_all_features(df_tf)
    df_clean, feature_cols = prepare_for_model(df_featured)

    print(f"  Features computed: {len(feature_cols)}")
    print(f"  Clean samples: {len(df_clean):,}")
    print()

    # =========================================================
    # STEP 4: Train/Test Split
    # =========================================================
    print("[Step 4/7] Splitting data into train/test...")
    split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    print(f"  Train: {len(train_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    print()

    # =========================================================
    # STEP 5: Train ARIMA
    # =========================================================
    print("[Step 5/7] Training ARIMA model...")
    arima = ARIMAForecaster()

    # Use daily close prices for ARIMA
    train_close = train_df["close"].values
    test_close = test_df["close"].values

    arima.fit(train_close)

    # ARIMA in-sample evaluation: forecast each test point
    print("[ARIMA] Generating test predictions (walk-forward)...")
    arima_test_preds = []
    history = list(train_close)

    # For efficiency, use a sliding ARIMA re-fit every N steps
    refit_interval = max(1, len(test_close) // 50)  # ~50 re-fits

    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

    current_model = None
    for i in range(len(test_close)):
        if i % refit_interval == 0 or current_model is None:
            try:
                model = StatsARIMA(history, order=arima.order)
                current_model = model.fit()
            except Exception:
                pass

        try:
            yhat = current_model.forecast(steps=1)[0]
        except Exception:
            yhat = history[-1]

        arima_test_preds.append(yhat)
        history.append(test_close[i])

        if i % 50 == 0:
            print(f"  ARIMA test step {i}/{len(test_close)}")

    arima_test_preds = np.array(arima_test_preds)
    arima.save()
    print()

    # =========================================================
    # STEP 6: Train LSTM
    # =========================================================
    print("[Step 6/7] Training LSTM model...")
    lstm = LSTMForecaster()

    lstm.fit(train_df, feature_cols, target_column="close")

    # LSTM test predictions
    print("[LSTM] Generating test predictions...")
    lstm_test_preds = lstm.predict(test_df)

    lstm.save()
    print()

    # =========================================================
    # STEP 7: Evaluate All Models
    # =========================================================
    print("[Step 7/7] Evaluating models...")
    print()

    # Align predictions to same length
    # LSTM predictions start from LSTM_SEQUENCE_LENGTH into the test set
    lstm_offset = LSTM_SEQUENCE_LENGTH
    if len(lstm_test_preds) < len(test_close):
        # LSTM predictions start after sequence_length warmup
        test_actual_lstm = test_close[lstm_offset: lstm_offset + len(lstm_test_preds)]
    else:
        test_actual_lstm = test_close[:len(lstm_test_preds)]

    # ARIMA predictions are 1-to-1 with test set
    test_actual_arima = test_close[:len(arima_test_preds)]

    # Evaluate individual models
    arima_metrics = evaluate(test_actual_arima, arima_test_preds, "ARIMA")
    lstm_metrics = evaluate(test_actual_lstm, lstm_test_preds, "LSTM")

    # Ensemble: align both predictions
    min_len = min(len(arima_test_preds), len(lstm_test_preds))
    ensemble = EnsembleForecaster()

    # Align ARIMA predictions to match LSTM's offset window
    if len(arima_test_preds) > len(lstm_test_preds):
        arima_aligned = arima_test_preds[lstm_offset: lstm_offset + min_len]
    else:
        arima_aligned = arima_test_preds[:min_len]

    lstm_aligned = lstm_test_preds[:min_len]
    ensemble_preds = ensemble.blend(lstm_aligned, arima_aligned)

    # Use the LSTM-aligned actual values for ensemble evaluation
    actual_aligned = test_actual_lstm[:min_len]
    ensemble_metrics = evaluate(actual_aligned, ensemble_preds, "Ensemble (ARIMA+LSTM)")

    # Save evaluation report
    all_metrics = {
        "ARIMA": arima_metrics,
        "LSTM": lstm_metrics,
        "Ensemble": ensemble_metrics,
    }
    save_report(all_metrics)

    # Generate plots
    print("\n[Plots] Generating evaluation charts...")
    plot_actual_vs_predicted(test_actual_arima, arima_test_preds, "ARIMA")
    plot_actual_vs_predicted(test_actual_lstm, lstm_test_preds, "LSTM")
    plot_actual_vs_predicted(actual_aligned, ensemble_preds, "Ensemble")

    # Comparison chart
    plot_comparison(
        actual_aligned,
        {
            "ARIMA": arima_aligned[:len(actual_aligned)],
            "LSTM": lstm_aligned[:len(actual_aligned)],
            "Ensemble": ensemble_preds[:len(actual_aligned)],
        },
    )

    # =========================================================
    # SUMMARY
    # =========================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Records processed: {len(df_raw):,} (raw)")
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print()
    print("  MODEL PERFORMANCE SUMMARY:")
    print(f"  {'Model':<25} {'RMSE':<12} {'MAE':<12} {'MAPE':<10} {'Dir.Acc':<10} {'R²':<10}")
    print(f"  {'-'*79}")
    for name, m in all_metrics.items():
        print(
            f"  {name:<25} "
            f"{m['rmse']:<12.4f} "
            f"{m['mae']:<12.4f} "
            f"{m['mape']:<10.2f}% "
            f"{m['directional_accuracy']:<10.2f}% "
            f"{m['r_squared']:<10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
