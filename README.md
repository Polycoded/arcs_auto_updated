<h1 align="center">Time Series Algorithm Recommender</h1>

<h4 align="center">Interactive Streamlit app that analyzes your time series, explains patterns in plain English, and recommends, trains, and benchmarks forecasting algorithms.</h4>

---

## Overview

This project is a **time series analysis and forecasting assistant** that takes a CSV file and walks you end-to-end from raw data to model selection and code generation. [file:21]

It automatically:
- Profiles your time series (trend, seasonality, stationarity, autocorrelation, volatility). [file:21]
- Explains the patterns in plain English for non-technical users. [file:21]
- Recommends suitable algorithms using a hybrid of AI meta-learning and rule-based heuristics. [file:21]
- Trains multiple benchmark models, compares metrics, and highlights the best one. [file:21]
- Generates starter Python code customized for your data and chosen algorithm. [file:21]
- Lets you save and reload trained models for later forecasting. [file:21]

The app is built with Streamlit, Plotly, and a custom meta-learning module trained on around 300 time series with known best-performing algorithms. [file:21]

---

## Features

- **Upload & Preprocessing**
  - Upload your own CSV or use demo datasets from the sidebar. [file:21]
  - Automatically infer time and target columns, handle missing values, detect outliers, and create a processed series. [file:21]
  - Keep track of all preprocessing steps in a human-readable log. [file:21]

- **Visualization & Evidence**
  - Side-by-side plots of original vs processed series. [file:21]
  - Simple trend view and seasonality view for non-technical users. [file:21]
  - Advanced decomposition (trend/seasonal/residual) for power users. [file:21]
  - ACF/PACF plots to inspect autocorrelation structure. [file:21]

- **Plain-English Evidence Summary**
  - Diagnostic cards summarizing key statistics. [file:21]
  - Narrative evidence sections for trend, seasonality, stationarity, and autocorrelation. [file:21]
  - Overall pattern summary in concise sentences. [file:21]

- **Algorithm Recommendation**
  - Task selector: Forecasting, Classification, Anomaly Detection, Clustering (UI is present; core deep logic is strongest for Forecasting). [file:21]
  - AI meta-model prediction (if enabled) with confidence score and training accuracy. [file:21]
  - Rule-based recommendations using statistical tests and time series characteristics. [file:21]
  - Final hybrid recommendation combining 80% AI and 20% rules, with detailed decision logic explanation. [file:21]
  - Alternative algorithm suggestions with pros/cons and “Why this algorithm?” reasoning. [file:21]

- **Model Training & Benchmarking (Forecasting)**
  - Quick single-algorithm training for a preferred method (SARIMA, ARIMA, Prophet, ETS, XGBoost, STL+ETS Hybrid, ARIMA+ETS+GBDT Hybrid). [file:21]
  - Full benchmark mode: train multiple models and compare MAE, RMSE, MAPE, and R. [file:21]
  - Visual comparison of metrics in subplot bar charts. [file:21]
  - Forecast vs actual plots for single and multiple models, highlighting the best model with thicker lines. [file:21]
  - Ability to save the best model to disk and reload it for forecasting on a new test CSV. [file:21]

- **Code Generation**
  - Generate starter Python code for a recommended algorithm tailored to your dataset’s structure. [file:21]
  - One-click download of the generated code file. [file:21]

- **Hackathon-Friendly UX**
  - Step-by-step “How this app works” section and “Key Features” summary. [file:21]
  - “Quick Actions” buttons for: Train this algorithm, Generate code, See all algorithms. [file:21]
  - Built with hackathon workflows in mind (upload → analyze → recommend/train → deploy). [file:21]

---

## Tech Stack

- **Frontend & Orchestration**
  - Streamlit (multi-tab interface, layout, interactivity). [file:21]
  - Plotly (interactive time series, bar charts, model comparison plots). [file:21]

- **Core Analytics & Modeling**
  - Pandas, NumPy (data handling and feature computation). [file:21]
  - Statsmodels (ARIMA, SARIMA, ETS and decomposition). [file:21]
  - XGBoost or similar gradient-boosting library for tree-based forecasting. [file:21]

- **Meta-Learning & Rules**
  - Custom `MetaPredictor` class from `utils.metapredictor` for AI-based model recommendation. [file:21]
  - Rule-based recommendation engine using extracted statistical features. [file:21]

- **Model Persistence**
  - Custom `savemodel` / `loadmodel` helpers for saving and reusing trained models. [file:21]

---

## Project Structure (High-Level)

Key logical components visible in the code: [file:21]

- `app` (main Streamlit script)
  - Handles layout, tabs, and user interactions. [file:21]
  - Tabs: Visualization, Evidence Summary, Recommendations, Benchmark Models, Code. [file:21]

- **Visualization Tab**
  - `plottimeseries` for original and processed series. [file:21]
  - Trend, seasonality, decomposition, and ACF/PACF plotting utilities. [file:21]

- **Evidence Tab**
  - `createstatscards` for diagnostic snapshot cards. [file:21]
  - `generatetrendevidence`, `generateseasonalityevidence`, `generatestationarityevidence`, `generateautocorrelationevidence`. [file:21]
  - `summarizepatterns` for plain-English pattern summaries. [file:21]

- **Recommendations Tab**
  - `MetaPredictor` (AI meta-model) and feature extraction consistent with its training. [file:21]
  - Rule-based `getrecommendations(features, task)`. [file:21]
  - Hybrid logic combining AI and rules with explicit reasoning. [file:21]

- **Benchmark Models Tab**
  - `ModelTrainer` class with methods like `trainsarima`, `trainarima`, `trainprophet`, `trainets`, `trainxgboost`, `trainstlets`, `trainarimaetsgbdt`. [file:21]
  - Full benchmark pipeline producing metrics table and forecast comparison plots. [file:21]
  - Save/load best model and test on new data. [file:21]

- **Code Tab**
  - `generatecodesnippet(algorithm, features, task, timecol, targetcol)` for starter code. [file:21]
  - Download button for generated script. [file:21]

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
