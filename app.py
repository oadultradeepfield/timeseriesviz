import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.load_and_preprocess_data import (load_toy_dataset, 
                                          format_datetime_column,
                                          impute_missing_data,
                                          resampling_data)
from src.helper_functions import (plot_univariate_time_series, 
                                  resample_checker, 
                                  calculate_maximum_lookback, 
                                  add_shaded_regions_for_splits, 
                                  split_dataset, 
                                  visualize_model_predictions, 
                                  visualize_residuals,
                                  is_column_contain_missing_values)
from src.create_features import create_lag_features, create_rolling_statistics
from src.train_models import get_arima_prediction, get_ml_prediction
from src.test_models import calculate_metrics, calculate_residuals

# Load your styles.css file
with open('styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
frequency_to_alias = {
    "Every Nanosecond": "N",
    "Every Microsecond": "U",
    "Every Millisecond": "L",
    "Every Second": "s",
    "Every Minute": "min",
    "Hourly": "h",
    "Daily": "D",
    "Business Day": "B",
    "Weekly": "W",
    "Monthly Start": "MS",
    "Quarterly Start": "QS",
    "Semi-Annually": "6M",
    "Yearly Start": "YS",
    "Business Year Start": "BYS",
}

def main_components(dataframe: pd.DataFrame, choice_column: str, fig: go.Figure):
    imputation_method = "No Imputation"
                
    if is_column_contain_missing_values(dataframe, choice_column):
        st.markdown(
            f"""
            <div style="margin-top: -12px; margin-bottom: 12px">
                Your dataset contains missing row(s), consider imputing the methods below:
            </div>
            """,
            unsafe_allow_html=True
        )
        
        choice_impute = st.selectbox("Imputation Options", 
                                        options=["Please select",
                                                "No Imputation",
                                                "Forward Fill",
                                                "Backward Fill",
                                                "Linear Interpolation"], 
                                        key="imputation")

        if choice_impute != "Please select":
            imputation_method = choice_impute

    dataframe = impute_missing_data(dataframe, choice_column, imputation_method)
    resampling_options = list(filter(resample_checker(dataframe), list(frequency_to_alias.keys())))
    targeted_frequency = "Default"
    fig = plot_univariate_time_series(dataframe, choice_column)
    
    if len(resampling_options) > 0:
        st.markdown(
            """
            <div style="margin-top: -6px; margin-bottom: 12px">
                <span>Your data may not be indexed with the desired frequency, consider downsampling it as needed. Select the available downsampling options below to proceed.</span> 
                <span style="color: #3494fa; font-weight:bold">The default option is the current frequency.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        choice_resampling = st.selectbox("Resampling Options", options=resampling_options, key="resampling")
        targeted_frequency = frequency_to_alias[choice_resampling]
        
    dataframe = resampling_data(dataframe, targeted_frequency)
    fig = plot_univariate_time_series(dataframe, f"{choice_column}_imputed")
        
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="margin-top: -24px; margin-bottom: 12px">
            <h2>
                Train and Evaluate Model
            </h2>
            <span>
                Select the model you want to use to forecast the time series of the value:
            </span>
            <span style="font-weight: bold; color: #3494fa">
                {choice_column}.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    choice_model = st.selectbox("Model Options", options=["Please select", "ARIMA", "Random Forests", "LightGBM"], key="model")

    if choice_model == "ARIMA":
        st.markdown(
            f"""
            <div style="margin-top: -6px; margin-bottom: 12px">
                Please specify the parameters of ARIMA to use below:
            </div>
            """,
            unsafe_allow_html=True
        )
        
        p, d, q = st.columns(3)
        p_val = p.number_input(label="p (AR terms)", min_value=0, max_value=calculate_maximum_lookback(dataframe))
        d_val = d.number_input(label="d (I terms)", min_value=0, max_value=calculate_maximum_lookback(dataframe))
        q_val = q.number_input(label="q (MA terms)", min_value=0, max_value=calculate_maximum_lookback(dataframe))
        
        arima_predictions = get_arima_prediction(dataframe, 
                                                    choice_column,
                                                    p_val, d_val, q_val)
        
        if p_val != 0 or d_val != 0 or q_val != 0:
            metrics = calculate_metrics(dataframe[f"{choice_column}_imputed"], arima_predictions)
            residuals = calculate_residuals(dataframe[f"{choice_column}_imputed"], arima_predictions)
            
            result_fig = visualize_model_predictions(fig, dataframe, arima_predictions)
            st.plotly_chart(result_fig, use_container_width=True)
            
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Mean Absolute Error", f"{metrics['mae']:.0f}")
            metric2.metric("Root Mean Squared Error", f"{metrics['rmse']:.0f}")
            metric3.metric("R-Squared Score", f"{metrics['r2']:.4f}")
            
            residual_fig = visualize_residuals(dataframe, residuals)
            st.plotly_chart(residual_fig, use_container_width=True)

    elif choice_model != "Please select":
        st.markdown(
            f"""
            <div style="margin-top: 6px; margin-bottom: 12px">
                <h2>
                    Create Train and Validation Set
                </h2>
                <span>
                    Select the ratio of training and validation set below:
                </span>
                <span style="font-weight: bold; color: #3494fa">
                    {choice_column}.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        split_ratio = st.slider(label="Training and validation split (%)", min_value=0, max_value=99, value=50, step=5)

        if split_ratio != 0:
            fig = add_shaded_regions_for_splits(fig, dataframe, split_ratio / 100)
        
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                f"""
                <div style="margin-top: -24px; margin-bottom: 12px">
                    <h2>
                        Create Features
                    </h2>
                    Please specify the number of previous steps to consider when generating the features. A value of zero means the feature will not be created. For lag features, the features are generated for each step from 1 up to N, where N is the specified input number.
                </div>
                """,
                unsafe_allow_html=True
            )
            
            lag, mean, sd = st.columns(3)
            lag_val = lag.number_input(label="Lag Features", min_value=1, max_value=calculate_maximum_lookback(dataframe))
            mean_val = mean.number_input(label="Rolling Mean", min_value=0, max_value=calculate_maximum_lookback(dataframe))
            sd_val = sd.number_input(label="Rolling Standard Deviation", min_value=0, max_value=calculate_maximum_lookback(dataframe))
            
            dataframe = create_lag_features(dataframe, choice_column, lag_val)
            dataframe = create_rolling_statistics(dataframe, choice_column, "Mean", lag_val)
            dataframe = create_rolling_statistics(dataframe, choice_column, "SD", lag_val)
            
            train_dataframe, valid_dataframe = split_dataset(dataframe, split_ratio / 100)
            
            features = [f"lag_{i}" for i in range(1, lag_val + 1)]
            
            if mean_val != 0:
                features += ["rolling_mean"] 
            
            if sd_val != 0:
                features += ["rolling_sd"]
            
            ml_results = get_ml_prediction(choice_model, 
                                        train_dataframe, 
                                        valid_dataframe,
                                        features,
                                        choice_column)
            
            metrics = calculate_metrics(dataframe[f"{choice_column}_imputed"], ml_results)
            residuals = calculate_residuals(dataframe[f"{choice_column}_imputed"], ml_results)
            
            result_fig = visualize_model_predictions(fig, dataframe, ml_results)
            st.plotly_chart(result_fig, use_container_width=True)
            
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
            metric2.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
            metric3.metric("R-Squared Score", f"{metrics['r2']:.3f}")
            
            residual_fig = visualize_residuals(dataframe, residuals)
            st.plotly_chart(residual_fig, use_container_width=True)

if __name__ == "__main__":
    st.markdown(
        """
        <h1 style="margin-top: -54px">
            <img src="https://iili.io/ds4RP6b.png" alt="Logo" style="vertical-align: middle; width: 48px; filter: drop-shadow(3px 3px 5px rgba(0, 0, 0, 0.2));">
            <span style="color: #333333; font-style: italic; vertical-align: middle; margin-left: 6px;">TimeSeries</span><span style="color: #3494fa; vertical-align: middle;">Viz</span>
        </h1>
        <div class="text-container" style="margin-top: 8px">
            <p style="font-size: 16px;">Made with ❤️📚 by <a href="https://www.instagram.com/oadultradeepfield/">@oadultradeepfield</a></p>
            <p style"margin-top: 64px">
                Learn time series processing concepts like resampling, lag features, rolling statistics, and modeling techniques such as ARIMA, random forests, and LightGBM, all enhanced by clear visualizations!
            </p>
        </div>
        <hr>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2>
            Load and Preprocess Data
        </h2>
        <div style="margin-bottom: 12px">
            Choose the following toy datasets or upload your own to get started.
        </div>
        """,
        unsafe_allow_html=True
    )

    choice = st.selectbox("Select Dataset", options=["Please select",
                                                    "Catfish", 
                                                    "Ice Cream vs. Heater", 
                                                    "Sales", 
                                                    "Upload a File"],
                        key="dataset")

    if choice != "Please select":
        if choice != "Upload a File":
            st.markdown(
                f"""
                <div style="margin-top: -6px; margin-bottom: 12px">
                    <span>You have selected:</span>
                    <span style="color: #3494fa; font-weight: bold">{choice}</span>
                    <span>as the dataset, please select the column to visualize.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            dataframe = load_toy_dataset(choice)
            choice_column = st.selectbox("Select Column", options=["Please select"] + list(dataframe.columns.values)[1:], key="column")
            
            if choice_column != "Please select":
                dataframe = format_datetime_column(dataframe, list(dataframe.columns.values)[0])
                fig = plot_univariate_time_series(dataframe, choice_column)
                main_components(dataframe, choice_column, fig)
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type="csv")
            if uploaded_file is not None:
                dataframe = pd.read_csv(uploaded_file)
                st.markdown(
                    f"""
                    <div style="margin-top: 12px; margin-bottom: 12px">
                        <span>You have uploaded:</span>
                        <span style="color: #3494fa; font-weight: bold">{uploaded_file.name}</span>
                        <span>as the dataset, please select the column to visualize.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                time, target = st.columns(2)
                time_column = time.selectbox("Select Date and Time Column", options=["Please select"] + list(dataframe.columns.values), key="date_upload")
                choice_column = target.selectbox("Select Targeted Column", options=["Please select"] + list(dataframe.columns.values), key="column_upload")
                
                if time_column != "Please select" and choice_column != "Please select":
                    st.markdown(
                        """
                        <div style="margin-top: -6px; margin-bottom: 12px">
                            <span>Your data may not be indexed with the desired frequency, consider downsampling it as needed. Select the available downsampling options below to proceed.</span> 
                            <span style="color: #3494fa; font-weight:bold">The default option is the current frequency.</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    dataframe = format_datetime_column(dataframe, time_column)
                    fig = plot_univariate_time_series(dataframe, choice_column)
                    main_components(dataframe, choice_column, fig)
