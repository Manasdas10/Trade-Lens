"""
evaluator.py — Model Performance Evaluation
Computes RMSE, MAE, MAPE, R², and Directional Accuracy.
Generates evaluation reports and Matplotlib comparison charts.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from config import EVALUATION_REPORT_PATH, MODELS_DIR
import os


def rmse(actual, predicted):
    """Root Mean Square Error."""
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    """Mean Absolute Error."""
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    # Avoid division by zero
    mask = actual != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def r_squared(actual, predicted):
    """R² — Coefficient of Determination."""
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def directional_accuracy(actual, predicted):
    """
    Directional Accuracy — percentage of correctly predicted up/down moves.
    Compares direction of change (not absolute values).
    """
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)

    if len(actual) < 2:
        return 0.0

    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))

    # Ignore flat periods (direction = 0)
    mask = actual_direction != 0
    if mask.sum() == 0:
        return 0.0

    correct = np.sum(actual_direction[mask] == predicted_direction[mask])
    total = mask.sum()

    return float(correct / total * 100)


def max_error(actual, predicted):
    """Maximum absolute error."""
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    return float(np.max(np.abs(actual - predicted)))


def evaluate(actual, predicted, model_name="Model"):
    """
    Run full evaluation suite and print results.

    Parameters
    ----------
    actual : array-like
        Actual values.
    predicted : array-like
        Predicted values.
    model_name : str
        Label for the model being evaluated.

    Returns
    -------
    dict
        Dictionary of all metrics.
    """
    metrics = {
        "model": model_name,
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
        "r_squared": r_squared(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "max_error": max_error(actual, predicted),
        "sample_count": len(actual),
    }

    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT — {model_name}")
    print(f"{'='*55}")
    print(f"  Samples Evaluated:      {metrics['sample_count']:,}")
    print(f"  RMSE:                   {metrics['rmse']:.4f}")
    print(f"  MAE:                    {metrics['mae']:.4f}")
    print(f"  MAPE:                   {metrics['mape']:.2f}%")
    print(f"  R²:                     {metrics['r_squared']:.4f}")
    print(f"  Directional Accuracy:   {metrics['directional_accuracy']:.2f}%")
    print(f"  Max Error:              {metrics['max_error']:.4f}")
    print(f"{'='*55}\n")

    return metrics


def evaluate_all(actual, predictions_dict):
    """
    Evaluate multiple models and return all results.

    Parameters
    ----------
    actual : array-like
        Ground truth values.
    predictions_dict : dict
        {model_name: predicted_values}

    Returns
    -------
    dict
        {model_name: metrics_dict}
    """
    all_metrics = {}
    for name, preds in predictions_dict.items():
        # Align lengths
        min_len = min(len(actual), len(preds))
        all_metrics[name] = evaluate(actual[:min_len], preds[:min_len], name)

    return all_metrics


def save_report(all_metrics, filepath=None):
    """Save evaluation report as JSON."""
    if filepath is None:
        filepath = EVALUATION_REPORT_PATH

    # Convert numpy types to native Python types
    clean = {}
    for model_name, metrics in all_metrics.items():
        clean[model_name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
            for k, v in metrics.items()
        }

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"[Evaluator] Report saved to: {filepath}")


def plot_actual_vs_predicted(actual, predicted, model_name="Model", save_path=None):
    """Generate Actual vs Predicted comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Model Evaluation — {model_name}", fontsize=14, fontweight="bold")

    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)

    # 1. Actual vs Predicted time series
    ax1 = axes[0, 0]
    ax1.plot(actual, label="Actual", color="#2196F3", alpha=0.8, linewidth=0.8)
    ax1.plot(predicted, label="Predicted", color="#FF5722", alpha=0.8, linewidth=0.8)
    ax1.set_title("Actual vs Predicted")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(actual, predicted, alpha=0.3, s=5, color="#4CAF50")
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Perfect Fit")
    ax2.set_title("Scatter: Actual vs Predicted")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residuals
    residuals = actual - predicted
    ax3 = axes[1, 0]
    ax3.plot(residuals, color="#9C27B0", alpha=0.6, linewidth=0.5)
    ax3.axhline(y=0, color="red", linewidth=1, linestyle="--")
    ax3.set_title("Residuals (Actual - Predicted)")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Residual")
    ax3.grid(True, alpha=0.3)

    # 4. Error Distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=50, color="#FF9800", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax4.axvline(x=0, color="red", linewidth=1, linestyle="--")
    ax4.set_title("Error Distribution")
    ax4.set_xlabel("Error")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"eval_{model_name.lower().replace(' ', '_')}.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Chart saved to: {save_path}")

    return save_path


def plot_comparison(actual, predictions_dict, save_path=None):
    """Generate comparison chart for multiple models."""
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(actual, label="Actual", color="#2196F3", linewidth=1.2)

    colors = ["#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(preds, label=name, color=color, alpha=0.8, linewidth=0.8)

    ax.set_title("Model Comparison — Actual vs Predictions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "eval_comparison.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Comparison chart saved to: {save_path}")

    return save_path
