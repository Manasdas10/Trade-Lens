"""
arima_forecaster.py — ARIMA Model for Time Series Forecasting
Uses pmdarima auto_arima for optimal (p,d,q) selection with fallback.
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from config import (
    ARIMA_MODEL_PATH, ARIMA_DEFAULT_ORDER,
    ARIMA_MAX_P, ARIMA_MAX_D, ARIMA_MAX_Q,
    ARIMA_SEASONAL, ARIMA_FORECAST_STEPS,
)

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """ARIMA model wrapper with auto-tuning and fallback."""

    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
        self.is_fitted = False

    def fit(self, series):
        """
        Fit ARIMA model to time series data.

        Parameters
        ----------
        series : pd.Series or np.ndarray
            Close price series for fitting.
        """
        series = np.array(series, dtype=np.float64)

        # Remove NaN/Inf
        series = series[np.isfinite(series)]

        print(f"[ARIMA] Fitting on {len(series):,} data points...")

        try:
            import pmdarima as pm

            self.model = pm.auto_arima(
                series,
                start_p=1, start_q=1,
                max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q,
                max_d=ARIMA_MAX_D,
                seasonal=ARIMA_SEASONAL,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                n_fits=30,
            )
            self.order = self.model.order
            self.is_fitted = True

            print(f"[ARIMA] Auto-fitted with order: {self.order}")
            print(f"[ARIMA] AIC: {self.model.aic():.2f}")

        except Exception as e:
            print(f"[ARIMA] Auto-ARIMA failed: {e}")
            print(f"[ARIMA] Falling back to manual order: {ARIMA_DEFAULT_ORDER}")
            self._fit_manual(series, ARIMA_DEFAULT_ORDER)

    def _fit_manual(self, series, order):
        """Fit ARIMA with manual order specification."""
        from statsmodels.tsa.arima.model import ARIMA

        try:
            model = ARIMA(series, order=order)
            self.fitted_model = model.fit()
            self.order = order
            self.is_fitted = True
            print(f"[ARIMA] Manual fit successful with order: {order}")

        except Exception as e:
            print(f"[ARIMA] Manual fit also failed: {e}")
            self.is_fitted = False

    def predict(self, steps=None):
        """
        Forecast future values.

        Parameters
        ----------
        steps : int, optional
            Number of steps to forecast. Defaults to ARIMA_FORECAST_STEPS.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if not self.is_fitted:
            raise RuntimeError("[ARIMA] Model not fitted. Call fit() first.")

        if steps is None:
            steps = ARIMA_FORECAST_STEPS

        try:
            if self.model is not None:
                # pmdarima model
                forecast = self.model.predict(n_periods=steps)
            else:
                # statsmodels fallback
                forecast = self.fitted_model.forecast(steps=steps)

            return np.array(forecast, dtype=np.float64)

        except Exception as e:
            print(f"[ARIMA] Prediction error: {e}")
            return np.array([])

    def predict_in_sample(self, series):
        """
        Generate in-sample predictions for evaluation.

        Parameters
        ----------
        series : array-like
            Original series used for fitting.

        Returns
        -------
        np.ndarray
            In-sample predicted values.
        """
        if not self.is_fitted:
            raise RuntimeError("[ARIMA] Model not fitted.")

        series = np.array(series, dtype=np.float64)
        series = series[np.isfinite(series)]
        n = len(series)

        # Walk-forward prediction: use rolling window
        predictions = []
        train_size = max(int(n * 0.8), 100)

        print(f"[ARIMA] Generating walk-forward predictions ({n - train_size} steps)...")

        from statsmodels.tsa.arima.model import ARIMA

        history = list(series[:train_size])

        for i in range(train_size, n):
            try:
                model = ARIMA(history, order=self.order)
                fitted = model.fit()
                yhat = fitted.forecast(steps=1)[0]
                predictions.append(yhat)
            except Exception:
                # Use last known value as fallback
                predictions.append(history[-1])

            history.append(series[i])

            if (i - train_size) % 100 == 0:
                print(f"  ... step {i - train_size}/{n - train_size}")

        return np.array(predictions, dtype=np.float64)

    def save(self, filepath=None):
        """Save model to disk."""
        if filepath is None:
            filepath = ARIMA_MODEL_PATH

        data = {
            "model": self.model,
            "fitted_model": self.fitted_model,
            "order": self.order,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"[ARIMA] Model saved to: {filepath}")

    def load(self, filepath=None):
        """Load model from disk."""
        if filepath is None:
            filepath = ARIMA_MODEL_PATH

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.fitted_model = data["fitted_model"]
        self.order = data["order"]
        self.is_fitted = data["is_fitted"]

        print(f"[ARIMA] Model loaded from: {filepath} (order={self.order})")
