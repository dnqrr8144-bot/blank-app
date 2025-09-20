import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import ensemble model functions
import warnings
warnings.filterwarnings("ignore")

# Machine Learning models
import xgboost as xgb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Optional models
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except:
    HAS_PMDARIMA = False

try:
    from arch import arch_model
    HAS_ARCH = True
except:
    HAS_ARCH = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    HAS_PROPHET = False

# Configuration class equivalent to the React config
class Config:
    TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'SPY']
    API_KEY = os.environ.get('REACT_APP_STOCK_API_KEY', 'DEMO_PLACEHOLDER')
    MONTE_CARLO_SIMS = 5000
    RISK_FREE_RATE_DAILY = 0.03 / 252  # 3% annual -> daily

# Default weights for ensemble models
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
    """Normalize weights to sum to 1"""
    s = sum(wdict.values())
    if s == 0:
        return wdict
    return {k:v/s for k,v in wdict.items()}

# Initialize page config
st.set_page_config(
    page_title="📈 Stock Analysis & Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hebrew support styling
st.markdown("""
<style>
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("📈 Stock Analysis & Monte Carlo Simulation")
st.markdown("<div class='rtl-text'>ניתוח מניות וסימולציות מונטה קרלו</div>", unsafe_allow_html=True)

# Sidebar for stock selection
st.sidebar.title("Stock Selection")
selected_stocks = st.sidebar.multiselect(
    "Select stocks to analyze:",
    Config.TICKERS,
    default=['NVDA', 'AAPL', 'SPY']
)

# Sidebar for analysis options
st.sidebar.title("Analysis Options")
enable_ensemble = st.sidebar.checkbox("Enable Hybrid Ensemble Analysis", value=True)
fast_mode = st.sidebar.checkbox("Fast Mode (reduced training)", value=True)

# Model weights configuration (collapsible)
with st.sidebar.expander("🔧 Model Weights Configuration"):
    st.markdown("**Adjust model weights for ensemble:**")
    
    weight_xgb = st.slider("XGBoost", 0.0, 1.0, DEFAULT_WEIGHTS["XGBoost"], 0.05)
    weight_lstm = st.slider("LSTM", 0.0, 1.0, DEFAULT_WEIGHTS["LSTM"], 0.05)
    weight_gru = st.slider("GRU", 0.0, 1.0, DEFAULT_WEIGHTS["GRU"], 0.05)
    weight_prophet = st.slider("Prophet", 0.0, 1.0, DEFAULT_WEIGHTS["Prophet"], 0.05)
    weight_arima = st.slider("ARIMA+GARCH", 0.0, 1.0, DEFAULT_WEIGHTS["ARIMA_GARCH"], 0.05)
    weight_mc = st.slider("Monte Carlo", 0.0, 1.0, DEFAULT_WEIGHTS["MonteCarlo"], 0.05)
    weight_tech = st.slider("Technical", 0.0, 1.0, DEFAULT_WEIGHTS["Technical"], 0.05)
    weight_fund = st.slider("Fundamental", 0.0, 1.0, DEFAULT_WEIGHTS["Fundamental"], 0.05)
    
    custom_weights = {
        "XGBoost": weight_xgb,
        "LSTM": weight_lstm,
        "GRU": weight_gru,
        "Prophet": weight_prophet,
        "ARIMA_GARCH": weight_arima,
        "MonteCarlo": weight_mc,
        "Technical": weight_tech,
        "Fundamental": weight_fund
    }
    custom_weights = normalize_weights(custom_weights)

# Date range selection
st.sidebar.title("Date Range")
end_date = st.sidebar.date_input("End Date", datetime.now())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=90))

# Monte Carlo parameters
st.sidebar.title("Monte Carlo Parameters")
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, Config.MONTE_CARLO_SIMS, step=1000)
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 252, step=30)

@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data from Yahoo Finance or generate demo data"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data is not None and not data.empty:
            return data
        else:
            raise Exception("No data returned from Yahoo Finance")
    except Exception as e:
        st.warning(f"Unable to fetch real data from Yahoo Finance: {str(e)}")
        st.info("🎲 Generating demo data for demonstration purposes...")
        return generate_demo_data(tickers, start_date, end_date)

def generate_demo_data(tickers, start_date, end_date):
    """Generate realistic demo stock data"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # Filter out weekends for market days only
    dates = dates[dates.weekday < 5]
    
    # Base prices for different stocks (realistic starting prices)
    base_prices = {
        'NVDA': 120.0,
        'AAPL': 180.0,
        'GOOGL': 165.0,
        'MSFT': 420.0,
        'TSLA': 250.0,
        'AMZN': 140.0,
        'SPY': 540.0
    }
    
    # Volatility parameters for different stocks
    volatilities = {
        'NVDA': 0.035,   # High volatility for tech stock
        'AAPL': 0.022,   # Moderate volatility
        'GOOGL': 0.025,  # Moderate volatility
        'MSFT': 0.020,   # Lower volatility
        'TSLA': 0.040,   # Very high volatility
        'AMZN': 0.028,   # Moderate-high volatility
        'SPY': 0.015     # Low volatility (ETF)
    }
    
    # Trend parameters (daily drift)
    trends = {
        'NVDA': 0.0008,  # Slight upward trend
        'AAPL': 0.0005,  # Slight upward trend
        'GOOGL': 0.0003, # Small upward trend
        'MSFT': 0.0006,  # Moderate upward trend
        'TSLA': 0.0002,  # Small upward trend (volatile)
        'AMZN': 0.0004,  # Small upward trend
        'SPY': 0.0004    # Steady upward trend
    }
    
    data_dict = {}
    
    for ticker in tickers:
        if ticker not in base_prices:
            continue
            
        np.random.seed(42 + hash(ticker) % 1000)  # Consistent but different seed per ticker
        
        n_days = len(dates)
        returns = np.random.normal(trends[ticker], volatilities[ticker], n_days)
        
        # Generate price series
        prices = [base_prices[ticker]]
        for i in range(1, n_days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Create OHLC data (simplified)
        high_factor = np.random.uniform(1.005, 1.02, n_days)
        low_factor = np.random.uniform(0.98, 0.995, n_days)
        open_factor = np.random.uniform(0.995, 1.005, n_days)
        
        ticker_data = pd.DataFrame({
            'Open': np.array(prices) * open_factor,
            'High': np.array(prices) * high_factor,
            'Low': np.array(prices) * low_factor,
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 50000000, n_days)
        }, index=dates)
        
        data_dict[ticker] = ticker_data
    
    if len(tickers) == 1:
        return data_dict[tickers[0]]
    else:
        # Multi-level column structure like yfinance
        result = pd.concat(data_dict, axis=1)
        result.columns = pd.MultiIndex.from_tuples([
            (col, ticker) for ticker in tickers for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        ])
        return result

def calculate_returns(data):
    """Calculate daily returns"""
    if len(data.columns.levels) > 1:  # Multi-ticker data
        returns = data['Adj Close'].pct_change().dropna()
    else:  # Single ticker data
        returns = data['Adj Close'].pct_change().dropna()
    return returns

def monte_carlo_simulation(returns, last_price, num_sims, forecast_days):
    """Run Monte Carlo simulation"""
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Generate random returns
    random_returns = np.random.normal(mean_return, std_return, (forecast_days, num_sims))
    
    # Calculate cumulative returns
    price_paths = np.zeros((forecast_days + 1, num_sims))
    price_paths[0] = last_price
    
    for t in range(1, forecast_days + 1):
        price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
    
    return price_paths

def calculate_risk_metrics(price_paths, initial_price):
    """Calculate various risk metrics"""
    final_prices = price_paths[-1]
    returns = (final_prices - initial_price) / initial_price
    
    metrics = {
        'Expected Return': np.mean(returns),
        'Volatility': np.std(returns),
        'VaR 95%': np.percentile(returns, 5),
        'VaR 99%': np.percentile(returns, 1),
        'CVaR 95%': np.mean(returns[returns <= np.percentile(returns, 5)]),
        'CVaR 99%': np.mean(returns[returns <= np.percentile(returns, 1)]),
        'Sharpe Ratio': (np.mean(returns) - Config.RISK_FREE_RATE_DAILY * forecast_days) / np.std(returns) if np.std(returns) != 0 else 0
    }
    
    return metrics

# ========= Ensemble Model Functions =========

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to stock data"""
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
    """Build feature matrix for ML models"""
    features = [
        'Open','High','Low','Close','Volume',
        'SMA_20','SMA_50','RSI','MACD','MACD_Signal','Volatility_20'
    ]
    avail = [f for f in features if f in df.columns]
    return df[avail].copy(), avail

def model_xgboost(df, feature_cols):
    """XGBoost model for price direction prediction"""
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
    """Build sequences for RNN models"""
    X, y = [], []
    for i in range(len(values) - lookback - 1):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback])
    return np.array(X), np.array(y)

def _train_seq(close_series, lookback, epochs, cell='LSTM'):
    """Train sequence model (LSTM/GRU)"""
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

def model_lstm(df, lookback=60, epochs=3):
    """LSTM model for time series prediction"""
    return _train_seq(df['Close'], lookback, epochs, 'LSTM')

def model_gru(df, lookback=60, epochs=3):
    """GRU model for time series prediction"""
    return _train_seq(df['Close'], lookback, epochs, 'GRU')

def model_prophet(df):
    """Prophet model for time series forecasting"""
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
    """ARIMA model for time series forecasting"""
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
    """GARCH model for volatility prediction"""
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

def model_monte_carlo(df, sims=200):
    """Monte Carlo simulation for price movement"""
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
    """Technical analysis model"""
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
    """Fundamental analysis stub"""
    if override is not None:
        return float(override)
    return 0.65  # Placeholder for NVDA fundamentals

def classify_recommendation(score: float) -> str:
    """Classify ensemble score into recommendation"""
    if score >= 0.70: return "Strong Buy"
    if score >= 0.60: return "Buy"
    if score >= 0.50: return "Hold"
    if score >= 0.40: return "Sell"
    return "Strong Sell"

def get_recommendation_color(recommendation: str) -> str:
    """Get color for recommendation display"""
    colors = {
        "Strong Buy": "#00C851",
        "Buy": "#00FF00", 
        "Hold": "#FFC107",
        "Sell": "#FF5722",
        "Strong Sell": "#CC0000"
    }
    return colors.get(recommendation, "#000000")

@st.cache_data
def run_ensemble_analysis(stock_data, ticker, fast_mode=True):
    """Run the hybrid ensemble analysis on stock data"""
    
    # Add technical indicators
    df_with_indicators = add_technical_indicators(stock_data.copy())
    X, feat_cols = build_feature_matrix(df_with_indicators)
    
    weights = normalize_weights(DEFAULT_WEIGHTS)
    scores = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # XGBoost
        status_text.text("Running XGBoost model...")
        progress_bar.progress(1/8)
        scores['XGBoost'] = model_xgboost(df_with_indicators, feat_cols)
    
        # LSTM
        status_text.text("Training LSTM neural network...")
        progress_bar.progress(2/8)
        epochs = 2 if fast_mode else 6
        scores['LSTM'] = model_lstm(df_with_indicators, epochs=epochs)
    
        # GRU
        status_text.text("Training GRU neural network...")
        progress_bar.progress(3/8)
        scores['GRU'] = model_gru(df_with_indicators, epochs=epochs)
    
        # Prophet
        status_text.text("Running Prophet forecasting...")
        progress_bar.progress(4/8)
        scores['Prophet'] = model_prophet(df_with_indicators)
    
        # ARIMA + GARCH
        status_text.text("Running ARIMA and GARCH models...")
        progress_bar.progress(5/8)
        ar = model_arima(df_with_indicators)
        ga = model_garch(df_with_indicators)
        scores['ARIMA_GARCH'] = (ar + ga)/2
    
        # Monte Carlo
        status_text.text("Running Monte Carlo simulation...")
        progress_bar.progress(6/8)
        sims = 100 if fast_mode else 400
        scores['MonteCarlo'] = model_monte_carlo(df_with_indicators, sims=sims)
    
        # Technical
        status_text.text("Analyzing technical indicators...")
        progress_bar.progress(7/8)
        scores['Technical'] = model_technical(df_with_indicators)
    
        # Fundamental
        status_text.text("Evaluating fundamental factors...")
        progress_bar.progress(8/8)
        scores['Fundamental'] = model_fundamental_stub()
        
        # Calculate weighted ensemble
        total = 0
        wsum = 0
        for k,v in scores.items():
            if k in weights:
                total += v * weights[k]
                wsum += weights[k]
        final_score = total / wsum if wsum > 0 else 0
        
        recommendation = classify_recommendation(final_score)
        
        progress_bar.empty()
        status_text.empty()
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "scores": scores,
            "final_score": final_score,
            "recommendation": recommendation,
            "weights_used": weights
        }
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error running ensemble analysis: {str(e)}")
        return None

# Main application logic
if selected_stocks:
    with st.spinner('Fetching stock data...'):
        stock_data = fetch_stock_data(selected_stocks, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        st.success(f"Successfully fetched data for {len(selected_stocks)} stocks")
        
        # Display stock prices chart
        st.subheader("📊 Stock Price Evolution")
        
        fig = go.Figure()
        
        if len(selected_stocks) == 1:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Adj Close'],
                mode='lines',
                name=selected_stocks[0],
                line=dict(width=2)
            ))
        else:
            for ticker in selected_stocks:
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Adj Close'][ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Stock Price History",
            xaxis_title="Date",
            yaxis_title="Adjusted Close Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display returns
        returns = calculate_returns(stock_data)
        
        # Display returns chart
        st.subheader("📈 Daily Returns")
        
        fig_returns = go.Figure()
        
        if len(selected_stocks) == 1:
            fig_returns.add_trace(go.Scatter(
                x=returns.index,
                y=returns,
                mode='lines',
                name=f"{selected_stocks[0]} Returns",
                line=dict(width=1)
            ))
        else:
            for ticker in selected_stocks:
                fig_returns.add_trace(go.Scatter(
                    x=returns.index,
                    y=returns[ticker],
                    mode='lines',
                    name=f"{ticker} Returns",
                    line=dict(width=1)
                ))
        
        fig_returns.update_layout(
            title="Daily Returns",
            xaxis_title="Date",
            yaxis_title="Daily Return",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Hybrid Ensemble Analysis section
        if enable_ensemble:
            st.subheader("🤖 Hybrid Ensemble Analysis")
            st.markdown("<div class='rtl-text'>ניתוח אינסמבל היברידי - שילוב מודלים סטטיסטיים</div>", unsafe_allow_html=True)
            
            # Run ensemble analysis for each stock
            ensemble_results = {}
            
            for ticker in selected_stocks:
                st.markdown(f"### 🎯 {ticker} - Ensemble Analysis")
                
                # Get single stock data
                if len(selected_stocks) == 1:
                    ticker_data = stock_data
                else:
                    # Extract single ticker data from multi-index DataFrame
                    ticker_data = pd.DataFrame({
                        'Open': stock_data['Open'][ticker],
                        'High': stock_data['High'][ticker], 
                        'Low': stock_data['Low'][ticker],
                        'Close': stock_data['Close'][ticker],
                        'Adj Close': stock_data['Adj Close'][ticker],
                        'Volume': stock_data['Volume'][ticker]
                    })
                
                if len(ticker_data) < 50:
                    st.warning(f"Not enough data for {ticker} ensemble analysis")
                    continue
                
                # Run ensemble analysis
                with st.spinner(f'Running hybrid ensemble analysis for {ticker}...'):
                    result = run_ensemble_analysis(ticker_data, ticker, fast_mode)
                
                if result:
                    ensemble_results[ticker] = result
                    
                    # Display recommendation prominently
                    recommendation = result['recommendation']
                    final_score = result['final_score']
                    color = get_recommendation_color(recommendation)
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style='background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;'>
                            {recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Ensemble Score", f"{final_score:.3f}")
                        
                    with col3:
                        confidence = "High" if abs(final_score - 0.5) > 0.2 else "Medium" if abs(final_score - 0.5) > 0.1 else "Low"
                        st.metric("Confidence", confidence)
                    
                    # Individual model scores
                    st.markdown("#### 📊 Individual Model Scores")
                    
                    # Create two columns for model scores
                    score_col1, score_col2 = st.columns(2)
                    
                    model_items = list(result['scores'].items())
                    mid_point = len(model_items) // 2
                    
                    with score_col1:
                        for model, score in model_items[:mid_point]:
                            st.metric(model, f"{score:.3f}")
                    
                    with score_col2:
                        for model, score in model_items[mid_point:]:
                            st.metric(model, f"{score:.3f}")
                    
                    # Model scores visualization
                    fig_scores = go.Figure(data=[
                        go.Bar(
                            x=list(result['scores'].keys()),
                            y=list(result['scores'].values()),
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                        )
                    ])
                    
                    fig_scores.update_layout(
                        title=f"{ticker} - Individual Model Scores",
                        xaxis_title="Models",
                        yaxis_title="Score (0-1)",
                        showlegend=False,
                        height=400
                    )
                    
                    # Add horizontal line for ensemble score
                    fig_scores.add_hline(y=final_score, line_dash="dash", line_color="red", 
                                        annotation_text=f"Ensemble: {final_score:.3f}")
                    
                    st.plotly_chart(fig_scores, use_container_width=True)
                    
                    # Weights visualization 
                    fig_weights = go.Figure(data=[
                        go.Pie(
                            labels=list(result['weights_used'].keys()),
                            values=list(result['weights_used'].values()),
                            hole=0.3
                        )
                    ])
                    
                    fig_weights.update_layout(
                        title=f"{ticker} - Model Weights Distribution",
                        height=400
                    )
                    
                    st.plotly_chart(fig_weights, use_container_width=True)
                    
                    # Analysis summary
                    with st.expander("📈 Analysis Details"):
                        st.markdown(f"""
                        **Analysis Timestamp:** {result['timestamp']}
                        
                        **Key Insights:**
                        - **Ensemble Score:** {final_score:.3f} (0=Strong Sell, 1=Strong Buy)
                        - **Recommendation:** {recommendation}
                        - **Top Performing Model:** {max(result['scores'], key=result['scores'].get)} ({max(result['scores'].values()):.3f})
                        - **Lowest Performing Model:** {min(result['scores'], key=result['scores'].get)} ({min(result['scores'].values()):.3f})
                        
                        **Model Breakdown:**
                        - **Technical Analysis:** RSI, MACD, and moving average signals
                        - **Deep Learning:** LSTM and GRU neural networks for pattern recognition
                        - **Machine Learning:** XGBoost for classification-based prediction
                        - **Statistical Models:** ARIMA, GARCH, and Prophet for time series forecasting
                        - **Simulation:** Monte Carlo methods for probability assessment
                        - **Fundamental:** Basic fundamental analysis (placeholder)
                        """)
                        
                        st.markdown("<div class='rtl-text'><small>הערה: הניתוח מבוסס על נתונים היסטוריים ואינו מהווה ייעוץ השקעות</small></div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # Summary comparison if multiple stocks
            if len(ensemble_results) > 1:
                st.subheader("📊 Portfolio Comparison")
                
                comparison_data = []
                for ticker, result in ensemble_results.items():
                    comparison_data.append({
                        'Stock': ticker,
                        'Recommendation': result['recommendation'],
                        'Ensemble Score': result['final_score'],
                        'Color': get_recommendation_color(result['recommendation'])
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Create comparison chart
                fig_comp = go.Figure(data=[
                    go.Bar(
                        x=comp_df['Stock'],
                        y=comp_df['Ensemble Score'],
                        text=comp_df['Recommendation'],
                        textposition='outside',
                        marker_color=comp_df['Color']
                    )
                ])
                
                fig_comp.update_layout(
                    title="Portfolio Recommendations Comparison",
                    xaxis_title="Stocks",
                    yaxis_title="Ensemble Score",
                    showlegend=False,
                    height=400
                )
                
                fig_comp.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                  annotation_text="Neutral (0.5)")
                
                st.plotly_chart(fig_comp, use_container_width=True)
        
        # Monte Carlo Simulation section
        st.subheader("🎲 Monte Carlo Simulation")
        
        # Run simulation for each selected stock
        for ticker in selected_stocks:
            st.markdown(f"### {ticker} Analysis")
            
            if len(selected_stocks) == 1:
                ticker_returns = returns
                last_price = stock_data['Adj Close'].iloc[-1]
            else:
                ticker_returns = returns[ticker].dropna()
                last_price = stock_data['Adj Close'][ticker].iloc[-1]
            
            if len(ticker_returns) < 10:
                st.warning(f"Not enough data for {ticker}")
                continue
            
            # Run Monte Carlo simulation
            with st.spinner(f'Running Monte Carlo simulation for {ticker}...'):
                price_paths = monte_carlo_simulation(
                    ticker_returns, last_price, num_simulations, forecast_days
                )
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(price_paths, last_price)
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Return", f"{risk_metrics['Expected Return']:.2%}")
                st.metric("Volatility", f"{risk_metrics['Volatility']:.2%}")
            
            with col2:
                st.metric("VaR 95%", f"{risk_metrics['VaR 95%']:.2%}")
                st.metric("VaR 99%", f"{risk_metrics['VaR 99%']:.2%}")
            
            with col3:
                st.metric("CVaR 95%", f"{risk_metrics['CVaR 95%']:.2%}")
                st.metric("CVaR 99%", f"{risk_metrics['CVaR 99%']:.2%}")
            
            with col4:
                st.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.3f}")
                st.metric("Current Price", f"${last_price:.2f}")
            
            # Plot Monte Carlo paths
            fig_mc = go.Figure()
            
            # Sample a subset of paths for visualization
            sample_paths = price_paths[:, ::max(1, num_simulations//100)]
            
            for i in range(sample_paths.shape[1]):
                fig_mc.add_trace(go.Scatter(
                    x=list(range(forecast_days + 1)),
                    y=sample_paths[:, i],
                    mode='lines',
                    line=dict(width=0.5, color='lightblue'),
                    showlegend=False,
                    hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
            
            # Add mean path
            mean_path = np.mean(price_paths, axis=1)
            fig_mc.add_trace(go.Scatter(
                x=list(range(forecast_days + 1)),
                y=mean_path,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Path'
            ))
            
            # Add confidence intervals
            upper_95 = np.percentile(price_paths, 97.5, axis=1)
            lower_95 = np.percentile(price_paths, 2.5, axis=1)
            
            fig_mc.add_trace(go.Scatter(
                x=list(range(forecast_days + 1)),
                y=upper_95,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper 95%'
            ))
            
            fig_mc.add_trace(go.Scatter(
                x=list(range(forecast_days + 1)),
                y=lower_95,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                name='95% Confidence Interval'
            ))
            
            fig_mc.update_layout(
                title=f"{ticker} - Monte Carlo Price Simulation ({num_simulations:,} simulations)",
                xaxis_title="Days",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Distribution of final prices
            final_prices = price_paths[-1]
            
            fig_hist = px.histogram(
                x=final_prices,
                nbins=50,
                title=f"{ticker} - Distribution of Final Prices (Day {forecast_days})"
            )
            fig_hist.update_layout(
                xaxis_title="Final Price ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("---")
    
    else:
        st.error("Failed to fetch stock data. Please check your selection and try again.")

else:
    st.info("Please select at least one stock from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div class='rtl-text'>
<small>
הערה: האפליקציה משתמשת בנתוני Yahoo Finance וסימולציות מונטה קרלו לצורכי הדגמה בלבד.
לא מהווה ייעוץ השקעות.
</small>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<small>
Note: This application uses Yahoo Finance data and Monte Carlo simulations for demonstration purposes only.
Not investment advice.
</small>
""", unsafe_allow_html=True)
