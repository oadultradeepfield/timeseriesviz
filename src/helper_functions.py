from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def is_column_contain_missing_values(data: pd.DataFrame, 
                                     column: str) -> bool:
    """
    Check if a specified column in the DataFrame contains missing values.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to check for missing values.

    Returns:
        bool: True if the column contains missing values, False otherwise.
    """
    return data[column].isnull().any()

def resample_checker(data: pd.DataFrame):
    """
    Initialize a function that checks if the DataFrame can be resampled to the target frequency.

    Args:
        data (pd.DataFrame): The DataFrame containing the data with a datetime column.

    Returns:
        function: A function that accepts a target_frequency and checks resampling eligibility.
    """
    frequency_to_alias = {
        "Every Nanosecond": "N",
        "Every Microsecond": "U",
        "Every Millisecond": "L",
        "Every Second": "s",
        "Every Minute": "min",
        "Hourly": "h",
        "Daily": "D",
        "Business Day": "BD",
        "Weekly": "W",
        "Monthly Start": "MS",
        "Quarterly Start": "QS",
        "Semi-Annually": "6M",
        "Yearly Start": "YS",
        "Business Year Start": "BYS",
    }

    sampling_frequencies = [
        "N",  # Nanosecond
        "U",  # Microsecond
        "L",  # Millisecond
        "s",  # Second
        "min",  # Minute
        "h",  # Hour
        "D",  # Day
        "BD",  # Business Day
        "W",  # Week
        "MS",  # Month Start
        "QS",  # Quarter Start
        "6M",  # Semi-Annually
        "YS",  # Year Start
        "BYS",  # Business Year Start
    ]

    def is_able_to_resample(target_frequency: str) -> bool:
        """
        Determine if the DataFrame can be resampled to the target frequency.

        Args:
            target_frequency (str): The desired frequency to resample to.

        Returns:
            bool: True if resampling is possible, False otherwise.
        """
        current_frequency_alias = pd.infer_freq(data.index)
        
        if not current_frequency_alias:
            return False
        
        available_frequencies = sampling_frequencies[sampling_frequencies.index(current_frequency_alias):]
        target_alias = frequency_to_alias.get(target_frequency)
        return target_alias in available_frequencies

    return is_able_to_resample

def calculate_maximum_lookback(data: pd.DataFrame) -> int:
    """
    Calculate the maximum lookback period as 10% of the dataset length.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        int: The maximum lookback period as an integer.
    """
    return int(0.1 * len(data))
    
def plot_univariate_time_series(data: pd.DataFrame, 
                                column: str) -> go.Figure:
    """
    Create a line plot for a specified univariate time series.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot.

    Returns:
        go.Figure: The plotly figure object containing the time series plot.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name=column.split("_")[0]
        )
    )
    fig.update_layout(
        title=column.capitalize().split("_")[0],
        xaxis_title="Time",
        yaxis_title=column.capitalize().split("_")[0]
    )
    return fig

def split_dataset(data: pd.DataFrame, 
                  split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into two parts based on the specified ratio.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to split.
        split_ratio (float): The ratio to split the data, between 0 and 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The two split DataFrames.
    """
    split_index = int(split_ratio * len(data))
    return data.iloc[:split_index], data.iloc[split_index:]

def add_shaded_regions_for_splits(fig: go.Figure, 
                                  data: pd.DataFrame, 
                                  split_ratio: float) -> go.Figure:
    """
    Add shaded regions to indicate dataset splits in the figure.

    Args:
        fig (go.Figure): The plotly figure to add shaded regions to.
        data (pd.DataFrame): The DataFrame containing the data.
        split_ratio (float): The ratio used to determine the split point.

    Returns:
        go.Figure: The updated figure with shaded regions.
    """
    split_index = int(split_ratio * len(data))
    boundary = data.index[split_index]

    fig.add_shape(
        type="rect",
        x0=data.index[0],
        x1=boundary,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="lightblue",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    fig.add_shape(
        type="rect",
        x0=boundary,
        x1=data.index[-1],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="lightcoral",
        opacity=0.3,
        layer="below",
        line_width=0,
    )
    
    return fig

def visualize_model_predictions(fig: go.Figure, 
                                data: pd.DataFrame, 
                                y_pred: np.ndarray) -> go.Figure:
    """
    Visualize model predictions on the time series plot.

    Args:
        fig (go.Figure): The plotly figure to add predictions to.
        data (pd.DataFrame): The DataFrame containing the data.
        y_pred (np.ndarray): The array of predicted values.

    Returns:
        go.Figure: The updated figure with model predictions.
    """
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=y_pred,
            mode="lines",
            name="Prediction",
            line=dict(color='orange')
        )    
    )
    return fig

def visualize_residuals(data: pd.DataFrame, 
                        residuals: np.ndarray) -> go.Figure:
    """
    Create a plot of residuals from the model predictions.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        residuals (np.ndarray): The array of residuals to plot.

    Returns:
        go.Figure: The plotly figure showing the residuals.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=residuals,
            mode='lines',
            name='Residuals'
        )
    )
    fig.update_layout(
        title="Errors Made by the Model",
        xaxis_title="Time",
        yaxis_title="Residuals (%)"
    )
    return fig
