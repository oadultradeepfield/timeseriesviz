import numpy as np
import pandas as pd
from typing import Callable, Dict

def reverse_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse the dataframe index if the index is in non-ascending order.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame which is reversed.
    """
    if data.index.is_monotonic_decreasing:
        data = data.iloc[::-1]
    
    return data
    

def load_toy_dataset(option: str, missing_fraction: float = 0.1) -> pd.DataFrame:
    """
    Load a toy dataset based on the specified option, with randomly inserted missing data.

    Args:
        option (str): The name of the dataset to load.
        missing_fraction (float): The fraction of values to randomly replace with NaN. Default is 0.1 (10%).

    Returns:
        pd.DataFrame: The loaded dataset with random missing data.
    """
    option_to_path = {
        "Catfish": "src/data/catfish.csv", 
        "Ice Cream vs. Heater": "src/data/ice_cream_vs_heater.csv", 
        "Sales": "src/data/sales.csv",
    }
    
    selected_data = pd.read_csv(option_to_path[option])
    
    if missing_fraction > 0:
        mask = np.random.rand(*selected_data.shape) < missing_fraction
        selected_data.iloc[:, 1:] = selected_data.iloc[:, 1:].mask(mask)

    return selected_data

def format_datetime_column(data: pd.DataFrame, 
                           datetime_column: str) -> pd.DataFrame:
    """
    Convert a column to datetime format and add it as a new column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        datetime_column (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The DataFrame with the new formatted datetime column.
    """
    data["formatted_datetime"] = pd.to_datetime(data[datetime_column])
    data = data.set_index('formatted_datetime')
    return reverse_data(data)

def impute_missing_data(data: pd.DataFrame, 
                        column: str,
                        method: str) -> pd.DataFrame:
    """
    Impute missing values in a column using the specified method.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to impute.
        method (str): The imputation method to use. Options are 'No Imputation', 'Forward Fill', 'Backward Fill', or 'Linear Interpolation'.

    Returns:
        pd.DataFrame: The DataFrame with the imputed column.
    """
    method_to_function: Dict[str, Callable[[pd.Series], pd.Series]] = {
        "No Imputation": lambda s: s,
        "Forward Fill": lambda s: s.ffill(),
        "Backward Fill": lambda s: s.bfill(),
        "Linear Interpolation": lambda s: s.interpolate(),
    }
    data[f"{column}_imputed"] = method_to_function[method](data[column])
    return data

def resampling_data(data: pd.DataFrame,
                    target_frequency: str) -> pd.DataFrame:
    """
    Resample a time series column to the target frequency.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        target_frequency (str): The target frequency for resampling (e.g., 'D' for daily).

    Returns:
        pd.DataFrame: The DataFrame with the resampled column.
    """
    if target_frequency != "Default":
        data = data.select_dtypes(include='number').resample(target_frequency).mean()
    return data
