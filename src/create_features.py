import pandas as pd
from typing import Callable, Dict

# This file is only used for random forests and xgboost

def create_lag_features(data: pd.DataFrame,
                        column: str,
                        number_of_steps: int) -> pd.DataFrame:
    """
    Create lag features for the specified column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to create lag features for.
        number_of_steps (int): The number of lag steps to create.

    Returns:
        pd.DataFrame: The DataFrame with additional lag features.
    """
    for step in range(1, number_of_steps + 1):
        data[f"lag_{step}"] = data[f"{column}_imputed"].shift(step)
    
    return data

def create_rolling_statistics(data: pd.DataFrame, 
                              column: str,
                              statistics: str,
                              number_of_steps: int) -> pd.DataFrame:
    """
    Create rolling statistics for the specified column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to compute rolling statistics for.
        statistics (str): The type of rolling statistic to compute ('Mean' or 'Standard Deviation').
        number_of_steps (int): The window size for rolling statistics.

    Returns:
        pd.DataFrame: The DataFrame with additional rolling statistics.
    """
    statistics_to_method: Dict[str, Callable[[pd.Series], pd.Series]] = {
        "Mean": lambda s: s.rolling(window=number_of_steps).mean(),
        "SD": lambda s: s.rolling(window=number_of_steps).std(),
    }
    data[f"rolling_{statistics.lower()}"] = statistics_to_method[statistics](data[f"{column}_imputed"])
    return data
