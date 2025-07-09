import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Stock Price Predictor & Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Custom CSS for 3D/shadow and dark mode tweaks ---
st.markdown(
    """
    <style>
    .main {
        background: #181818;
        color: #f0f0f0;
    }
    .stApp {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
    }
    .stPlotlyChart > div {
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border-radius: 16px;
        background: #222831;
        padding: 1.5rem;
    }
    .stTextInput, .stDateInput {
        box-shadow: 0 4px 16px 0 rgba(0,0,0,0.25);
        border-radius: 8px;
    }
    .stButton > button {
        box-shadow: 0 2px 8px 0 rgba(0,0,0,0.25);
        border-radius: 8px;
        background: #393e46;
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Inputs ---
st.sidebar.title("Stock Analyzer Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., MSFT, TSLA, GOOGL)", value="MSFT")
years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
start_date = date.today() - timedelta(days=years*365)
end_date = date.today()

# --- Download Data ---
@st.cache_data(show_spinner=True)
def download_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, timeout=30)
        if data is None or data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None

# --- Technical Indicators ---
def calculate_technical_indicators(data, ma_short=50, ma_long=200, rsi_period=14):
    data[f'MA_{ma_short}'] = data['Close'].rolling(window=ma_short).mean()
    data[f'MA_{ma_long}'] = data['Close'].rolling(window=ma_long).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# --- Prepare Data for ML ---
def prepare_data_for_ml(data, ma_short=50, ma_long=200):
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', f'MA_{ma_short}', f'MA_{ma_long}', 'RSI']
    X = data[features]
    y = data['Target']
    return X, y, features

# --- Train Model ---
def train_and_predict(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, y_test, y_pred, rmse

# --- Main App ---
st.title(f"📈 Stock Price Predictor & Analyzer: {ticker.upper()}")

# Download and process data
data_load_state = st.info(f"Loading data for {ticker.upper()}...")
try:
    data = download_data(ticker, start_date, end_date)
    data_load_state.empty()
    
    if data is None or data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        st.stop()
    
    # Show data info for debugging
    st.success(f"✅ Successfully loaded {len(data)} trading days of data for {ticker.upper()}")
    st.write(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    
    # Show a small preview of the data
    with st.expander("📊 Data Preview"):
        st.dataframe(data.head(10))
        st.write(f"Columns: {list(data.columns)}")
    
except Exception as e:
    data_load_state.empty()
    st.error(f"Error downloading data for {ticker}: {str(e)}")
    st.info("Try a different stock ticker or check your internet connection.")
    st.stop()

# Calculate indicators
data = calculate_technical_indicators(data)
X, y, features = prepare_data_for_ml(data)
model, y_test, y_pred, rmse = train_and_predict(X, y)

# --- Date Picker for Accurate Price ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🔎 Price at a Specific Date")
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    selected_date = st.date_input("Select Date", value=max_date, min_value=min_date, max_value=max_date)
    if pd.Timestamp(selected_date) in data.index:
        price_at_date = float(data.loc[pd.Timestamp(selected_date), 'Close'])
        st.success(f"Close Price on {selected_date}: ${price_at_date:.2f}")
    else:
        st.warning("No trading data for this date.")

# --- Next Day Prediction ---
last_day_features = X.iloc[-1:].values.reshape(1, -1)
next_day_prediction = model.predict(last_day_features)
last_known_date = X.index[-1]

st.markdown(f"### Prediction for Next Trading Day after {last_known_date.date()}:")
st.info(f"Predicted closing price: **${next_day_prediction[0]:.2f}**")

# --- Plots ---
# 1. Price & Moving Averages
trace_close = go.Scatter(
    x=data.index, 
    y=data['Close'], 
    mode='lines', 
    name='Close Price', 
    line=dict(color='#00d4ff', width=2),
    opacity=0.9
)
trace_ma_short = go.Scatter(
    x=data.index, 
    y=data['MA_50'], 
    mode='lines', 
    name='50-Day MA', 
    line=dict(color='#ffa500', width=1.5, dash='dash'),
    opacity=0.8
)
trace_ma_long = go.Scatter(
    x=data.index, 
    y=data['MA_200'], 
    mode='lines', 
    name='200-Day MA', 
    line=dict(color='#ff4444', width=1.5, dash='dash'),
    opacity=0.8
)

layout1 = go.Layout(
    title=dict(
        text=f"{ticker.upper()} Stock Price & Moving Averages",
        font=dict(size=20, color='#f0f0f0')
    ),
    xaxis=dict(
        title='Date',
        gridcolor='#404040',
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title='Price (USD)',
        gridcolor='#404040',
        showgrid=True,
        zeroline=False
    ),
    plot_bgcolor='#1a1a1a',
    paper_bgcolor='#1a1a1a',
    font=dict(color='#f0f0f0'),
    showlegend=True,
    legend=dict(
        bgcolor='#2a2a2a',
        bordercolor='#404040',
        borderwidth=1
    ),
    margin=dict(l=50, r=50, t=80, b=50),
    hovermode='x unified'
)
fig1 = go.Figure(data=[trace_close, trace_ma_short, trace_ma_long], layout=layout1)

# 2. RSI
trace_rsi = go.Scatter(
    x=data.index, 
    y=data['RSI'], 
    mode='lines', 
    name='RSI', 
    line=dict(color='#9c27b0', width=2),
    fill='tonexty',
    fillcolor='rgba(156, 39, 176, 0.1)'
)
layout2 = go.Layout(
    title=dict(
        text=f"{ticker.upper()} Relative Strength Index (RSI)",
        font=dict(size=20, color='#f0f0f0')
    ),
    xaxis=dict(
        title='Date',
        gridcolor='#404040',
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title='RSI Value', 
        range=[0, 100],
        gridcolor='#404040',
        showgrid=True,
        zeroline=False
    ),
    plot_bgcolor='#1a1a1a',
    paper_bgcolor='#1a1a1a',
    font=dict(color='#f0f0f0'),
    shapes=[
        dict(type='line', xref='paper', x0=0, x1=1, y0=70, y1=70, line=dict(color='#ff4444', dash='dash', width=1)),
        dict(type='line', xref='paper', x0=0, x1=1, y0=30, y1=30, line=dict(color='#44ff44', dash='dash', width=1)),
    ],
    annotations=[
        dict(x=0.02, y=0.95, xref='paper', yref='paper', text='Overbought', showarrow=False, font=dict(color='#ff4444')),
        dict(x=0.02, y=0.05, xref='paper', yref='paper', text='Oversold', showarrow=False, font=dict(color='#44ff44')),
    ],
    legend=dict(
        bgcolor='#2a2a2a',
        bordercolor='#404040',
        borderwidth=1
    ),
    margin=dict(l=50, r=50, t=80, b=50),
    hovermode='x unified'
)
fig2 = go.Figure(data=[trace_rsi], layout=layout2)

# 3. Actual vs Predicted (Test Set)
test_dates = y_test.index
trace_actual = go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual Price', line=dict(color='royalblue'))
trace_pred = go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Predicted Price', line=dict(color='red', dash='dash'))
layout3 = go.Layout(
    title=f"{ticker.upper()} Model Performance: Actual vs. Predicted (Test Set)",
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price (USD)'),
    plot_bgcolor='#232526',
    paper_bgcolor='#232526',
    font=dict(color='#f0f0f0'),
    showlegend=True,
    margin=dict(l=40, r=40, t=60, b=40),
)
fig3 = go.Figure(data=[trace_actual, trace_pred], layout=layout3)

# --- Show Plots ---
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)
