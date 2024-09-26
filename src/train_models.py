import numpy as np
import pandas as pd
from typing import List, Tuple
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def get_arima_prediction(data: pd.Series,
                         target_column: str,
                         p: int,
                         d: int,
                         q: int) -> np.ndarray:
    """
    Generate ARIMA model predictions.

    Args:
        data (pd.Series): Input data as a pandas Series.
        target_column (str): The target variable column name.
        p (int): The number of lag observations included in the model (AR term).
        d (int): The number of times that the raw observations are differenced (I term).
        q (int): The size of the moving average window (MA term).

    Returns:
        np.ndarray: Predictions for the validation set.
    """
    model = ARIMA(data[f"{target_column}_imputed"], order=(p, d, q)).fit()
    return model.predict(start=0, end=len(data) - 1)

def get_ml_prediction(model_name: str,
                      data_train: pd.DataFrame,
                      data_validation: pd.DataFrame,
                      features: List[str],
                      target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions on both the training and validation data using a specified machine learning model.

    Args:
        model_name (str): The name of the model to use ('Random Forests' or 'LightGBM').
        data_train (pd.DataFrame): Training data as a pandas DataFrame.
        data_validation (pd.DataFrame): Validation data as a pandas DataFrame.
        features (List[str]): List of feature column names for training.
        target_column (str): The target variable column name.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing predictions for the training set,
                                       predictions for the validation set, 
                                       and feature importances.
    """
    model_name_to_model = {
        "Random Forests": RandomForestRegressor(random_state=0),
        "LightGBM": LGBMRegressor(random_state=0)
    }

    if model_name not in model_name_to_model:
        raise ValueError(f"Model '{model_name}' is not supported. Choose 'Random Forests' or 'LightGBM'.")

    model = model_name_to_model[model_name]

    model.fit(data_train[features], data_train[target_column])

    train_predictions = model.predict(data_train[features])
    validation_predictions = model.predict(data_validation[features])

    feature_importances = model.feature_importances_

    return np.concatenate((train_predictions, validation_predictions), axis=0), feature_importances