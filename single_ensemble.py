#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-file Hybrid Ensemble (Minimal) – גרסה בקובץ אחד
-----------------------------------------------------
מה כולל:
- הורדת נתונים (yfinance)
- הפקת פיצ'רים טכניים בסיסיים
- מודלים: XGBoost, LSTM, GRU, Prophet (אם מותקן), ARIMA + GARCH, Monte Carlo, Technical Block, Fundamental Stub
- שקלול משקולות + המלצה
- פרמטרים דרך שורת הפקודה

שימוש בסיסי:
    python single_ensemble.py --ticker XLC
שימוש מהיר (פחות אימונים):
    python single_ensemble.py --ticker NVDA --fast
החלפת משקולות (JSON inline):
    python single_ensemble.py --ticker AAPL --weights '{"XGBoost":0.2,"LSTM":0.15,"GRU":0.1,"Prophet":0.1,"ARIMA_GARCH":0.1,"MonteCarlo":0.1,"Technical":0.15,"Fundamental":0.1}'
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from datetime import datetime, timezone

# --- Imports (חלקם אופציונליים) ---
import pandas as pd
import numpy as np
import yfinance as yf

# מודלים סטטיסטיים
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# XGBoost
import xgboost as xgb

# Deep Learning (LSTM / GRU)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Prophet (לא חובה)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False


# ========= Feature Engineering =========
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Volatility
    df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    return df

def build_feature_matrix(df: pd.DataFrame):
    features = [
        'Open','High','Low','Close','Volume',
        'SMA_20','SMA_50','RSI','MACD','MACD_Signal','Volatility_20'
    ]
    avail = [f for f in features if f in df.columns]
    return df[avail].copy(), avail


# ========= Models =========
def model_xgboost(df, feature_cols):
    temp = df.copy()
    temp['Target'] = (temp['Close'].shift(-1) > temp['Close']).astype(int)
    temp.dropna(inplace=True)
    if temp['Target'].nunique() < 2:
        return 0.5
    X = temp[feature_cols]
    y = temp['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    return float(model.predict_proba(X.iloc[[-1]])[0][1])

def _seq_builder(values, lookback):
    X, y = [], []
    for i in range(len(values) - lookback - 1):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback])
    return np.array(X), np.array(y)

def _train_seq(close_series, lookback, epochs, cell='LSTM'):
    if len(close_series) < lookback + 20:
        return 0.5
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(close_series.values.reshape(-1,1))
    X, y = _seq_builder(arr.flatten(), lookback)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    if cell == 'LSTM':
        model.add(LSTM(48, return_sequences=True, input_shape=(lookback,1)))
        model.add(Dropout(0.25))
        model.add(LSTM(24))
    else:
        model.add(GRU(48, return_sequences=True, input_shape=(lookback,1)))
        model.add(Dropout(0.25))
        model.add(GRU(24))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    last = arr[-lookback:].reshape(1,lookback,1)
    pred_scaled = model.predict(last, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    current = close_series.values[-1]
    # heuristic scoring
    if pred > current:
        score = min(1.0, (pred-current)/current * 8)
    else:
        score = max(0.0, 1 - (current-pred)/current * 8)
    return float(score)

def model_lstm(df, lookback, epochs):
    return _train_seq(df['Close'], lookback, epochs, 'LSTM')

def model_gru(df, lookback, epochs):
    return _train_seq(df['Close'], lookback, epochs, 'GRU')

def model_prophet(df):
    if not HAS_PROPHET:
        return 0.5
    s = df[['Close']].reset_index()
    s.columns = ['ds','y']
    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m.fit(s)
        future = m.make_future_dataframe(periods=1)
        fc = m.predict(future).iloc[-1]['yhat']
    except Exception:
        return 0.5
    last = df['Close'].iloc[-1]
    if fc > last:
        return min(1.0,(fc-last)/last*8)
    else:
        return max(0,(1-(last-fc)/last*8))

def model_arima(df):
    if not HAS_PMDARIMA:
        return 0.5
    try:
        model = pm.auto_arima(df['Close'], seasonal=False, suppress_warnings=True)
        fc = model.predict(1)[0]
    except Exception:
        return 0.5
    last = df['Close'].iloc[-1]
    if fc > last:
        return min(1.0,(fc-last)/last*8)
    else:
        return max(0,(1-(last-fc)/last*8))

def model_garch(df):
    if not HAS_ARCH:
        return 0.5
    rets = df['Close'].pct_change().dropna()
    if len(rets) < 80:
        return 0.5
    try:
        am = arch_model(rets, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        f = res.forecast(horizon=1)
        vol = float((f.variance.values[-1,0])**0.5)
        score = 1 - min(1, vol * 25)
        return float(max(0, min(1, score)))
    except Exception:
        return 0.5

def model_monte_carlo(df, sims=400):
    log_r = np.log(1 + df['Close'].pct_change().dropna())
    if len(log_r) < 50:
        return 0.5
    mu = log_r.mean()
    sigma = log_r.std()
    last = df['Close'].iloc[-1]
    ups = 0
    for _ in range(sims):
        step = np.random.normal(mu, sigma)
        price = last * np.exp(step)
        if price > last:
            ups += 1
    return float(ups/sims)

def model_technical(df):
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    sig = df['MACD_Signal'].iloc[-1]
    score = 0.5
    if rsi < 30: score += 0.12
    if rsi > 70: score -= 0.12
    if macd > sig: score += 0.05
    else: score -= 0.05
    return float(max(0,min(1,score)))

def model_fundamental_stub(override=None):
    if override is not None:
        return float(override)
    return 0.65  # קבוע זמני


# ========= Weighting & Classification =========
def classify(score: float) -> str:
    if score >= 0.70: return "Strong Buy"
    if score >= 0.60: return "Buy"
    if score >= 0.50: return "Hold"
    if score >= 0.40: return "Sell"
    return "Strong Sell"

DEFAULT_WEIGHTS = {
    "XGBoost": 0.15,
    "LSTM": 0.15,
    "GRU": 0.10,
    "Prophet": 0.10,
    "ARIMA_GARCH": 0.10,
    "MonteCarlo": 0.10,
    "Technical": 0.15,
    "Fundamental": 0.15
}

def normalize_weights(wdict):
    s = sum(wdict.values())
    if s == 0:
        return wdict
    return {k:v/s for k,v in wdict.items()}


# ========= Main Runner =========
def run_single(ticker: str,
               period: str = "5y",
               interval: str = "1d",
               lookback: int = 80,
               epochs_lstm: int = 6,
               epochs_gru: int = 6,
               fast: bool = False,
               fundamental_override=None,
               custom_weights=None):

    weights = custom_weights if custom_weights else DEFAULT_WEIGHTS
    weights = normalize_weights(weights)

    print(f"\n--- הורדת נתונים: {ticker} period={period} interval={interval} ---")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("לא נמצאו נתונים לטיקר")

    df = add_technical_indicators(df)
    X, feat_cols = build_feature_matrix(df)

    if fast:
        epochs_lstm = max(1, min(epochs_lstm, 2))
        epochs_gru = max(1, min(epochs_gru, 2))

    scores = {}

    # XGBoost
    scores['XGBoost'] = model_xgboost(df, feat_cols)

    # LSTM
    scores['LSTM'] = model_lstm(df, lookback, epochs_lstm)

    # GRU
    scores['GRU'] = model_gru(df, lookback, epochs_gru)

    # Prophet
    scores['Prophet'] = model_prophet(df)

    # ARIMA + GARCH
    ar = model_arima(df)
    ga = model_garch(df)
    scores['ARIMA_GARCH'] = (ar + ga)/2

    # Monte Carlo
    scores['MonteCarlo'] = model_monte_carlo(df, sims=200 if fast else 400)

    # Technical
    scores['Technical'] = model_technical(df)

    # Fundamental Stub
    scores['Fundamental'] = model_fundamental_stub(fundamental_override)

    # Weighted aggregation
    total = 0
    wsum = 0
    for k,v in scores.items():
        if k in weights:
            total += v * weights[k]
            wsum += weights[k]
    final_score = total / wsum if wsum > 0 else 0
    decision = classify(final_score)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "ticker": ticker,
        "timestamp": ts,
        "scores": scores,
        "final_score": final_score,
        "decision": decision,
        "weights_used": weights
    }


def print_report(result: dict):
    print("\n===== תוצאות מודלים =====")
    for k,v in sorted(result['scores'].items(), key=lambda x: x[0]):
        print(f"{k:12s}: {v:.4f}")
    print("-------------------------")
    print(f"Final Ensemble Score: {result['final_score']:.4f}")
    print(f"Recommendation: {result['decision']}")
    print(f"Timestamp: {result['timestamp']}")
    print("\n(אין באמור ייעוץ השקעות)")


# ========= CLI =========
def main():
    parser = argparse.ArgumentParser(description="Single-File Hybrid Ensemble")
    parser.add_argument("--ticker", type=str, default="XLC", help="סימול (טיקר)")
    parser.add_argument("--period", type=str, default="5y", help="טווח (למשל 1y, 5y, max)")
    parser.add_argument("--interval", type=str, default="1d", help="1d / 1h / 1wk ...")
    parser.add_argument("--lookback", type=int, default=80, help="אורך רצף ל-LSTM/GRU")
    parser.add_argument("--epochs_lstm", type=int, default=6)
    parser.add_argument("--epochs_gru", type=int, default=6)
    parser.add_argument("--fast", action="store_true", help="מצב מהיר (פחות אימונים / סימולציות)")
    parser.add_argument("--fundamental_override", type=float, default=None, help="דריסת ציון פונדמנטלי (0-1)")
    parser.add_argument("--weights", type=str, default=None, help='JSON של משקולות (מחרוזת). למשל: {"XGBoost":0.2,...}')
    args = parser.parse_args()

    custom_weights = None
    if args.weights:
        try:
            custom_weights = json.loads(args.weights)
        except Exception:
            print("אזהרה: משקולות לא בפורמט JSON תקין. מתעלם.")

    result = run_single(
        ticker=args.ticker.upper(),
        period=args.period,
        interval=args.interval,
        lookback=args.lookback,
        epochs_lstm=args.epochs_lstm,
        epochs_gru=args.epochs_gru,
        fast=args.fast,
        fundamental_override=args.fundamental_override,
        custom_weights=custom_weights
    )
    print_report(result)


if __name__ == "__main__":
    main()