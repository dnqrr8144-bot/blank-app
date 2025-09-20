import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration class equivalent to the React config
class Config:
    TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'SPY']
    API_KEY = os.environ.get('REACT_APP_STOCK_API_KEY', 'DEMO_PLACEHOLDER')
    MONTE_CARLO_SIMS = 5000
    RISK_FREE_RATE_DAILY = 0.03 / 252  # 3% annual -> daily

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
