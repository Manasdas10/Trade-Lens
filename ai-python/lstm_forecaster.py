"""
lstm_forecaster.py — PyTorch Deep Learning Model for Stock Price Forecasting
Architecture: 2 LSTM layers (128 → 64 units) + Dropout + Dense
Uses PyTorch with custom EarlyStopping and Learning Rate scheduling on validation loss.
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    LSTM_MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH,
    LSTM_SEQUENCE_LENGTH, LSTM_UNITS_LAYER_1, LSTM_UNITS_LAYER_2,
    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE, LSTM_EARLY_STOPPING_PATIENCE,
    LSTM_LR_REDUCE_PATIENCE, LSTM_LR_REDUCE_FACTOR,
)

warnings.filterwarnings("ignore")


class PyTorchLSTMModel(nn.Module):
    """PyTorch implementation of the 2-layer LSTM model."""

    def __init__(self, input_dim, lstm_units_1=128, lstm_units_2=64, dropout=0.2):
        super(PyTorchLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, lstm_units_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm_units_1, lstm_units_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units_2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        # Select last timestep's output (equivalent to return_sequences=False)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LSTMForecaster:
    """LSTM model wrapper for multi-feature stock price prediction using PyTorch."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.input_dim = None
        self.is_fitted = False
        self.history = None
        # Detect GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LSTM] Using device: {self.device}")

    def build_model(self, input_shape):
        """
        Build the LSTM architecture.

        Parameters
        ----------
        input_shape : tuple
            (sequence_length, num_features)
        """
        _, num_features = input_shape
        self.input_dim = num_features
        self.model = PyTorchLSTMModel(
            input_dim=num_features,
            lstm_units_1=LSTM_UNITS_LAYER_1,
            lstm_units_2=LSTM_UNITS_LAYER_2,
            dropout=LSTM_DROPOUT,
        )
        self.model.to(self.device)
        return self.model

    def prepare_sequences(self, data, target, seq_length=None):
        """
        Create sliding window sequences for LSTM input.

        Parameters
        ----------
        data : np.ndarray
            Feature matrix (n_samples, n_features). Already scaled.
        target : np.ndarray
            Target values (n_samples,). Already scaled.
        seq_length : int, optional
            Lookback window size. Defaults to LSTM_SEQUENCE_LENGTH.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            X: (n_sequences, seq_length, n_features)
            y: (n_sequences,)
        """
        if seq_length is None:
            seq_length = LSTM_SEQUENCE_LENGTH

        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length: i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def fit(self, feature_df, feature_columns, target_column="close"):
        """
        Train the LSTM model.

        Parameters
        ----------
        feature_df : pd.DataFrame
            DataFrame with feature columns and target.
        feature_columns : list of str
            Column names to use as features.
        target_column : str
            Column to predict.
        """
        from sklearn.preprocessing import MinMaxScaler

        self.feature_columns = feature_columns

        print(f"[LSTM] Preparing data with {len(feature_columns)} features...")
        print(f"[LSTM] Total samples: {len(feature_df):,}")

        # Extract feature matrix and target
        features = feature_df[feature_columns].values.astype(np.float64)
        target = feature_df[target_column].values.astype(np.float64).reshape(-1, 1)

        # Scale features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = self.scaler.fit_transform(features)

        # Scale target separately
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled = self.target_scaler.fit_transform(target).flatten()

        # Create sequences
        X, y = self.prepare_sequences(features_scaled, target_scaled)

        print(f"[LSTM] Sequences created: X={X.shape}, y={y.shape}")

        # Train/val split (last 20% of sequences for validation)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        print(f"[LSTM] Train: {len(X_train):,} | Val: {len(X_val):,}")

        # Convert to PyTorch Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.build_model(input_shape)

        # Optimizer and Loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)

        # Training loop parameters
        best_val_loss = float("inf")
        best_model_weights = None
        early_stop_counter = 0
        lr_reduce_counter = 0
        current_lr = LSTM_LEARNING_RATE

        history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        print(f"[LSTM] Training for up to {LSTM_EPOCHS} epochs...")
        for epoch in range(1, LSTM_EPOCHS + 1):
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            total_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                total_batches += 1

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_x, val_y = X_val_tensor.to(self.device), y_val_tensor.to(self.device)
                val_outputs = self.model(val_x)
                val_loss = criterion(val_outputs, val_y).item()
                val_mae = torch.mean(torch.abs(val_outputs - val_y)).item()

            epoch_train_loss = train_loss / total_batches
            epoch_train_mae = train_mae / total_batches

            history["loss"].append(epoch_train_loss)
            history["val_loss"].append(val_loss)
            history["mae"].append(epoch_train_mae)
            history["val_mae"].append(val_mae)

            if epoch % 5 == 0 or epoch == 1 or epoch == LSTM_EPOCHS:
                print(
                    f"Epoch {epoch:02d}/{LSTM_EPOCHS:02d} | "
                    f"Loss: {epoch_train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val MAE: {val_mae:.6f}"
                )

            # Early stopping & Learning rate reduction check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                early_stop_counter = 0
                lr_reduce_counter = 0
            else:
                early_stop_counter += 1
                lr_reduce_counter += 1

                # Reduce LR
                if lr_reduce_counter >= LSTM_LR_REDUCE_PATIENCE:
                    current_lr *= LSTM_LR_REDUCE_FACTOR
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    print(f"  [LSTM] Reducing learning rate to {current_lr:.6f} at epoch {epoch}")
                    lr_reduce_counter = 0

                # Early Stopping
                if early_stop_counter >= LSTM_EARLY_STOPPING_PATIENCE:
                    print(f"  [LSTM] Early stopping triggered at epoch {epoch}")
                    break

        # Restore best weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            self.model.to(self.device)

        self.is_fitted = True
        self.history = history

        print(f"[LSTM] Training complete. Best Val Loss: {best_val_loss:.6f}")
        return self.history

    def predict(self, feature_df):
        """
        Generate predictions on new data.

        Parameters
        ----------
        feature_df : pd.DataFrame
            DataFrame with same feature columns as training data.

        Returns
        -------
        np.ndarray
            Predicted close prices (unscaled).
        """
        if not self.is_fitted:
            raise RuntimeError("[LSTM] Model not fitted. Call fit() first.")

        features = feature_df[self.feature_columns].values.astype(np.float64)
        features_scaled = self.scaler.transform(features)

        # Create sequences
        X, _ = self.prepare_sequences(
            features_scaled,
            np.zeros(len(features_scaled)),  # dummy target
        )

        if len(X) == 0:
            return np.array([])

        # Predict
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy().flatten()

        # Inverse transform
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        return predictions

    def predict_next(self, recent_features):
        """
        Predict the next single value given recent feature data.

        Parameters
        ----------
        recent_features : np.ndarray
            Shape: (seq_length, n_features) — already in the correct format.
            Must be raw (unscaled) values.

        Returns
        -------
        float
            Predicted next close price.
        """
        if not self.is_fitted:
            raise RuntimeError("[LSTM] Model not fitted.")

        # Scale
        scaled = self.scaler.transform(recent_features)
        X = scaled.reshape(1, scaled.shape[0], scaled.shape[1])

        # Predict
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy().flatten()[0]

        # Inverse scale
        pred = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]

        return float(pred)

    def save(self, model_path=None, scaler_path=None, feature_cols_path=None):
        """Save model state dict, scalers, and feature columns."""
        if model_path is None:
            model_path = LSTM_MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_PATH
        if feature_cols_path is None:
            feature_cols_path = FEATURE_COLUMNS_PATH

        # Save PyTorch state dictionary and model info
        state = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
        }
        torch.save(state, model_path)
        print(f"[LSTM] PyTorch Model weights saved to: {model_path}")

        with open(scaler_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "target_scaler": self.target_scaler,
            }, f)
        print(f"[LSTM] Scalers saved to: {scaler_path}")

        with open(feature_cols_path, "wb") as f:
            pickle.dump(self.feature_columns, f)
        print(f"[LSTM] Feature columns saved to: {feature_cols_path}")

    def load(self, model_path=None, scaler_path=None, feature_cols_path=None):
        """Load model state dict, scalers, and feature columns."""
        if model_path is None:
            model_path = LSTM_MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_PATH
        if feature_cols_path is None:
            feature_cols_path = FEATURE_COLUMNS_PATH

        # Load scalers first
        with open(scaler_path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.target_scaler = data["target_scaler"]
        print(f"[LSTM] Scalers loaded from: {scaler_path}")

        # Load feature columns
        with open(feature_cols_path, "rb") as f:
            self.feature_columns = pickle.load(f)
        print(f"[LSTM] Feature columns loaded ({len(self.feature_columns)} features)")

        # Load state dictionary
        state = torch.load(model_path, map_location=self.device)
        self.input_dim = state["input_dim"]

        # Build architecture
        self.build_model((LSTM_SEQUENCE_LENGTH, self.input_dim))

        # Load weights
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        print(f"[LSTM] PyTorch Model weights loaded from: {model_path}")

        self.is_fitted = True
