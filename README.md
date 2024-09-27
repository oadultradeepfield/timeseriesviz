<div align="center">
  <img src="https://github.com/oadultradeepfield/timeseriesviz/blob/main/logo.png" width="64px"><br> 
  <h1>
    <i>TimeSeries</i>Viz
  </h1>
  <p>
    Time Series Visualization Tool for Processing and Modeling
  </p>
</div>

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-ff4b4b?logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-2.1.1-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.2-f7931e?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-00ff00?logo=LightGBM&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24.1-3f4f75?logo=plotly&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14.3-009999?logo=statsmodels&logoColor=white)

## Overview

**TimeSeriesViz** is a Streamlit app designed for interactive visualization and analysis of time series data. It allows users to easily explore the effects of various preprocessing techniques and model hyperparameters through dynamic plots and key numerical metrics.

Originally developed as a hands-on tool for the **NUS Fintech Society Machine Learning Training AY24/25**, it provides an intuitive platform for learning and applying time series concepts such as:

- **Resampling**, **Handling Missing Data**, **Lag Features**, **Rolling Statistics**
- **ARIMA**, **Random Forest**, and **LightGBM** modeling techniques

🔗 **[Try the live app here!](https://timeseriesviz.streamlit.app/)**

## Preview

![App Preview](https://github.com/oadultradeepfield/timeseriesviz/blob/main/preview.gif)

## Key Features

- **Interactive Visualization**: Visualize the impact of preprocessing methods on time series data in real time.
- **Modeling Insights**: Evaluate model performance through clear plots and metrics.
- **Beginner-Friendly**: Simplified interface for beginners to grasp key time series concepts quickly.

## Installation

To run the app locally:

```bash
git clone https://github.com/oadultradeepfield/timeseriesviz.git
cd timeseriesviz
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Upload your time series dataset.
2. Experiment with various preprocessing methods like data imputation, resampling, and feature creation.
3. Select models such as ARIMA or LightGBM and tune hyperparameters.
4. Visualize the impact of each change in real-time.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/oadultradeepfield/timeseriesviz/blob/main/LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to enhance the tool.

### Acknowledgements
This app was built for the NUS Fintech Society Machine Learning Training AY24/25 as a part of an interactive hands-on curriculum. NUS Fintech Society was founded in 2018 in collaboration with NUS Fintech Lab under the NUS School of Computing. It has a mission to educate students with Fintech knowledge through industry projects, and connect and establish relationships with industry partners.

The toy dataset used in this project is sourced and adapted from this Kaggle repository: https://www.kaggle.com/datasets/yekahaaagayeham/time-series-toy-data-set. I greatly appreciate the author for sharing it with us.
