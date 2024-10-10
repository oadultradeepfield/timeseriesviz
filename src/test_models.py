import numpy as np
from typing import Dict
from sklearn.metrics import (mean_absolute_error, 
                             root_mean_squared_error,
                             r2_score)

def calculate_residuals(y_true: np.ndarray, 
                      y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate percentage residuals between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth (actual) values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Percentage residuals, handling division by zero.
    """
    return np.where(y_true != 0, np.abs(y_true - y_pred) / np.abs(y_true) * 100, 0)
    
def calculate_metrics(y_true: np.ndarray, 
                      y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate residuals and regression performance metrics.

    Args:
        y_true (np.ndarray): Ground truth (actual) values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: A dictionary containing:
            - 'mae': Mean Absolute Error (MAE)
            - 'rmse': Root Mean Squared Error (RMSE)
            - 'r2': R-squared (R2) score
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
