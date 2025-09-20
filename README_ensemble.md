# Single Ensemble Financial Analysis Tool

This repository contains a comprehensive financial ensemble analysis tool (`single_ensemble.py`) that combines multiple machine learning models to provide investment recommendations.

## Features

The tool includes the following models:
- **XGBoost** - Gradient boosting classifier for price direction prediction
- **LSTM** - Long Short-Term Memory neural network for time series analysis  
- **GRU** - Gated Recurrent Unit neural network for sequence modeling
- **Prophet** - Facebook's time series forecasting tool (optional)
- **ARIMA + GARCH** - Statistical time series and volatility models (optional) 
- **Monte Carlo** - Simulation-based price movement analysis
- **Technical Analysis** - RSI, MACD, SMA indicators
- **Fundamental Analysis** - Placeholder for fundamental metrics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Optional dependencies (may have compatibility issues):
```bash
pip install pmdarima arch prophet
```

## Usage

### Basic Usage
```bash
python single_ensemble.py --ticker XLC
```

### Fast Mode (Reduced Training)
```bash
python single_ensemble.py --ticker NVDA --fast
```

### Custom Weights
```bash
python single_ensemble.py --ticker AAPL --weights '{"XGBoost":0.2,"LSTM":0.15,"GRU":0.1,"Prophet":0.1,"ARIMA_GARCH":0.1,"MonteCarlo":0.1,"Technical":0.15,"Fundamental":0.1}'
```

### Other Options
```bash
# Different time periods
python single_ensemble.py --ticker MSFT --period 1y

# Override fundamental score
python single_ensemble.py --ticker GOOGL --fundamental_override 0.75

# Adjust neural network parameters
python single_ensemble.py --ticker TSLA --lookback 60 --epochs_lstm 10
```

## Command Line Arguments

- `--ticker`: Stock symbol (default: XLC)
- `--period`: Data period (1y, 5y, max, etc. - default: 5y)
- `--interval`: Data interval (1d, 1h, 1wk - default: 1d)
- `--lookback`: Sequence length for LSTM/GRU (default: 80)
- `--epochs_lstm`: Training epochs for LSTM (default: 6)
- `--epochs_gru`: Training epochs for GRU (default: 6)
- `--fast`: Enable fast mode with reduced training
- `--fundamental_override`: Override fundamental score (0-1)
- `--weights`: JSON string with custom model weights

## Output

The tool provides:
- Individual model scores (0-1 range)
- Final ensemble score (weighted average)
- Investment recommendation:
  - Strong Buy (≥0.70)
  - Buy (≥0.60)
  - Hold (≥0.50)  
  - Sell (≥0.40)
  - Strong Sell (<0.40)

## Testing

Run the test suite to verify functionality:
```bash
python test_ensemble.py
```

## Technical Details

### Feature Engineering
- Simple Moving Averages (SMA 20, 50)
- Relative Strength Index (RSI)
- MACD and Signal Line
- Price Volatility (20-day rolling)

### Model Architecture
- **XGBoost**: Classification with 120 estimators, depth 5
- **LSTM/GRU**: 48→24 units with dropout, sequence-to-one prediction
- **Technical**: Rule-based scoring using RSI and MACD signals

### Default Weights
```json
{
    "XGBoost": 0.15,
    "LSTM": 0.15,
    "GRU": 0.10,
    "Prophet": 0.10,
    "ARIMA_GARCH": 0.10,
    "MonteCarlo": 0.10,
    "Technical": 0.15,
    "Fundamental": 0.15
}
```

## Dependencies

Core requirements:
- pandas, numpy - Data manipulation
- yfinance - Financial data
- xgboost - Gradient boosting  
- tensorflow - Deep learning
- scikit-learn - Machine learning utilities

Optional (may have compatibility issues):
- pmdarima - ARIMA modeling
- arch - GARCH volatility models
- prophet - Time series forecasting

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always do your own research and consult with financial professionals before making investment decisions.

**Hebrew text in output**: אין באמור ייעוץ השקעות - "This does not constitute investment advice"