"""
ensemble_forecaster.py — Weighted ARIMA + LSTM Ensemble
Combines both models for improved prediction accuracy.
"""

import numpy as np
from config import (
    ENSEMBLE_LSTM_WEIGHT, ENSEMBLE_ARIMA_WEIGHT,
    BUY_THRESHOLD, SELL_THRESHOLD,
)


class EnsembleForecaster:
    """
    Ensemble model blending ARIMA and LSTM predictions.
    Default weights: 60% LSTM + 40% ARIMA.
    """

    def __init__(self, lstm_weight=None, arima_weight=None):
        self.lstm_weight = lstm_weight or ENSEMBLE_LSTM_WEIGHT
        self.arima_weight = arima_weight or ENSEMBLE_ARIMA_WEIGHT

        # Normalize weights
        total = self.lstm_weight + self.arima_weight
        self.lstm_weight /= total
        self.arima_weight /= total

    def blend(self, lstm_predictions, arima_predictions):
        """
        Blend LSTM and ARIMA predictions using weighted average.

        Parameters
        ----------
        lstm_predictions : array-like
            LSTM predicted values.
        arima_predictions : array-like
            ARIMA predicted values.

        Returns
        -------
        np.ndarray
            Blended predictions.
        """
        lstm_preds = np.array(lstm_predictions, dtype=np.float64)
        arima_preds = np.array(arima_predictions, dtype=np.float64)

        # Align lengths (use shorter)
        min_len = min(len(lstm_preds), len(arima_preds))
        lstm_preds = lstm_preds[:min_len]
        arima_preds = arima_preds[:min_len]

        ensemble = (
            self.lstm_weight * lstm_preds
            + self.arima_weight * arima_preds
        )

        return ensemble

    def generate_signals(self, current_prices, predicted_prices):
        """
        Generate BUY/SELL/HOLD signals based on predicted vs current price.

        Parameters
        ----------
        current_prices : array-like
            Current close prices.
        predicted_prices : array-like
            Predicted next close prices.

        Returns
        -------
        list of dict
            Signal details with prediction, return, confidence, and signal.
        """
        current = np.array(current_prices, dtype=np.float64)
        predicted = np.array(predicted_prices, dtype=np.float64)

        min_len = min(len(current), len(predicted))
        current = current[:min_len]
        predicted = predicted[:min_len]

        signals = []
        for i in range(min_len):
            pred_return = (predicted[i] - current[i]) / current[i]

            if pred_return > BUY_THRESHOLD:
                signal = "BUY"
            elif pred_return < SELL_THRESHOLD:
                signal = "SELL"
            else:
                signal = "HOLD"

            confidence = min(abs(pred_return) / 0.01, 1.0)  # normalize to [0, 1]

            signals.append({
                "current_price": float(current[i]),
                "predicted_price": float(predicted[i]),
                "predicted_return": float(pred_return),
                "signal": signal,
                "confidence": float(confidence),
            })

        return signals

    def compute_confidence(self, lstm_pred, arima_pred):
        """
        Compute confidence score based on model agreement.

        Parameters
        ----------
        lstm_pred : float
            LSTM predicted price.
        arima_pred : float
            ARIMA predicted price.

        Returns
        -------
        float
            Confidence between 0 and 1. Higher = more agreement.
        """
        if lstm_pred == 0 and arima_pred == 0:
            return 1.0

        avg = (lstm_pred + arima_pred) / 2
        if avg == 0:
            return 0.0

        divergence = abs(lstm_pred - arima_pred) / abs(avg)

        # Confidence decreases as divergence increases
        # At 0% divergence → 1.0, at 5% divergence → 0.0
        confidence = max(0.0, 1.0 - (divergence / 0.05))

        return float(confidence)

    def predict_single(self, lstm_pred, arima_pred, current_price):
        """
        Generate a single ensemble prediction with signal.

        Returns
        -------
        dict
            Complete prediction with signal, confidence, and individual model outputs.
        """
        ensemble_pred = (
            self.lstm_weight * lstm_pred
            + self.arima_weight * arima_pred
        )

        pred_return = (ensemble_pred - current_price) / current_price
        confidence = self.compute_confidence(lstm_pred, arima_pred)

        if pred_return > BUY_THRESHOLD:
            signal = "BUY"
        elif pred_return < SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "ensemble_prediction": float(ensemble_pred),
            "lstm_prediction": float(lstm_pred),
            "arima_prediction": float(arima_pred),
            "current_price": float(current_price),
            "predicted_return": float(pred_return),
            "signal": signal,
            "confidence": float(confidence),
            "lstm_weight": float(self.lstm_weight),
            "arima_weight": float(self.arima_weight),
        }
