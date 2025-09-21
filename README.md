# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Features

### Stock Analysis & Monte Carlo Simulation
The main Streamlit app provides:
- Interactive stock analysis with multiple tickers
- Real-time data from Yahoo Finance (with demo data fallback)
- Monte Carlo simulation for price forecasting
- Risk metrics calculation (VaR, CVaR, Sharpe Ratio)

### XGBoost Machine Learning Training
The repository includes an XGBoost training script (`train_xgb.py`) that:
- Trains both classification and regression models
- Uses time series cross-validation
- Supports multiple forecast horizons (30, 60, 90, 120, 180 days)
- Automatically selects features based on naming patterns
- Saves trained models using joblib

#### Usage
```bash
# Ensure you have a features_base.csv file with appropriate columns
python train_xgb.py
```

The script expects a CSV file with:
- Features starting with "nv_" (numerical variables)
- Features starting with "corr_" (correlation features)
- Specific technical indicators: "rsi_14", "macd_hist", "excess_ret_20"
- Target columns: "y_cls" (classification), "y_reg" (regression)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
