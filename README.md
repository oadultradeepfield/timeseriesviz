# TimeSeriesViz

![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

## Overview

**TimeSeriesViz** is a Streamlit app designed for interactive visualization and analysis of time series data. It allows users to easily explore the effects of various preprocessing techniques and model hyperparameters through dynamic plots and key numerical metrics.

Originally developed as a hands-on tool for the **NUS Fintech Society Machine Learning Training AY24/25**, it provides an intuitive platform for learning and applying time series concepts such as:

- **Resampling**, **Handling Missing Data**, **Lag Features**, **Rolling Statistics**
- **ARIMA**, **Random Forest**, and **LightGBM** modeling techniques

As of December 2024, I’ve decided to revamp the deployment process by containerizing the app with Docker and deploying it serverlessly on Google Cloud Run. The live version is now available [here](tsviz.phanuphats.com).

## Preview

![App Preview](/src/assets/preview.gif)

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

Alternatively, you can run the app using Docker (recommended):

```bash
git clone https://github.com/oadultradeepfield/timeseriesviz.git
cd timeseriesviz
docker build -t timeseriesviz .
docker run -p 8080:8080 timeseriesviz
```

## Usage

1. Upload your time series dataset.
2. Experiment with various preprocessing methods like data imputation, resampling, and feature creation.
3. Select models such as ARIMA or LightGBM and tune hyperparameters.
4. Visualize the impact of each change in real-time.

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to enhance the tool.

## Acknowledgements

This app was built for the NUS Fintech Society Machine Learning Training AY24/25 as a part of an interactive hands-on curriculum. NUS Fintech Society was founded in 2018 in collaboration with NUS Fintech Lab under the NUS School of Computing. It has a mission to educate students with Fintech knowledge through industry projects, and connect and establish relationships with industry partners.

The toy dataset used in this project is sourced and adapted from this Kaggle repository: https://www.kaggle.com/datasets/yekahaaagayeham/time-series-toy-data-set. I greatly appreciate the author for sharing it.
