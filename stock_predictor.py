# =============================================================================
# Stock Price Predictor & Technical Analyzer
#
# Author: [Your Name]
# Date: [Current Date]
#
# Description:
# This script downloads historical stock data using the yfinance library,
# calculates common technical indicators (Moving Averages, RSI), trains a
# Linear Regression model to predict the next day's closing price, and
# visualizes the data and model performance in a multi-panel plot.
#
# Disclaimer:
# This script is for educational purposes ONLY. It is not financial advice.
# Do not use this tool for making actual investment decisions.
# =============================================================================

# 1. Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- SCRIPT PARAMETERS ---
# You can change these settings to analyze different stocks or adjust indicators.
TICKER = 'AAPL'      # Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
START_DATE = date.today() - timedelta(days=5*365) # 5 years of data
END_DATE = date.today()
MA_SHORT = 50        # Short-term moving average window
MA_LONG = 200        # Long-term moving average window
RSI_PERIOD = 14      # RSI calculation period
TEST_SET_SIZE = 0.2  # Proportion of data to be used for testing (e.g., 20%)


def download_data(ticker, start, end):
    """Downloads historical stock data from Yahoo Finance."""
    print(f"Downloading historical data for {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=True)
    if data.empty:
        print(f"Error: No data found for ticker '{ticker}'. Please check the symbol.")
        return None
    print(f"Successfully downloaded {len(data)} trading days of data.")
    return data

def calculate_technical_indicators(data):
    """Calculates Moving Averages and RSI for the given data."""
    print("Calculating technical indicators...")
    # Simple Moving Averages
    data[f'MA_{MA_SHORT}'] = data['Close'].rolling(window=MA_SHORT).mean()
    data[f'MA_{MA_LONG}'] = data['Close'].rolling(window=MA_LONG).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def prepare_data_for_ml(data):
    """Prepares data for machine learning by creating features and target."""
    print("Preparing data for machine learning...")
    # Create the target variable: next day's closing price
    data['Target'] = data['Close'].shift(-1)
    
    # Remove rows with NaN values (from indicators and target shift)
    data.dropna(inplace=True)
    
    # Define features (X) and target (y)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', f'MA_{MA_SHORT}', f'MA_{MA_LONG}', 'RSI']
    X = data[features]
    y = data['Target']
    
    return X, y, features

def train_and_evaluate(X, y):
    """Splits data, trains a linear regression model, and evaluates it."""
    # Split data chronologically for time-series forecasting
    split_index = int(len(X) * (1 - TEST_SET_SIZE))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Create and train the model
    print("\nTraining the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model Performance on Test Set (RMSE): ${rmse:.2f}")
    
    return model, y_test, y_pred

def plot_results(data, y_test, y_pred, ticker):
    """Generates a multi-panel plot of the stock data and model results."""
    print("\nGenerating plots...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
    fig.suptitle(f'{ticker} Stock Analysis and Price Prediction', fontsize=20, y=0.93)

    # Plot 1: Stock Price and Moving Averages
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.plot(data.index, data[f'MA_{MA_SHORT}'], label=f'{MA_SHORT}-Day MA', color='orange', linestyle='--')
    ax1.plot(data.index, data[f'MA_{MA_LONG}'], label=f'{MA_LONG}-Day MA', color='red', linestyle='--')
    ax1.set_ylabel('Price (USD)')
    ax1.set_title('Stock Price and Moving Averages')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot 2: Relative Strength Index (RSI)
    ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
    ax2.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
    ax2.set_ylim([0, 100])
    ax2.set_ylabel('RSI Value')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Plot 3: Actual vs. Predicted Prices (on Test Set)
    test_dates = y_test.index
    ax3.plot(test_dates, y_test, label='Actual Price', color='blue', alpha=0.7)
    ax3.plot(test_dates, y_pred, label='Predicted Price', color='red', linestyle='--')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price (USD)')
    ax3.set_title('Model Performance: Actual vs. Predicted Prices (Test Set)')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make room for suptitle
    plt.show()

def main():
    """Main function to run the entire analysis pipeline."""
    # Step 1: Download data
    stock_data = download_data(TICKER, START_DATE, END_DATE)
    if stock_data is None:
        return

    # Step 2: Calculate indicators
    data_with_indicators = calculate_technical_indicators(stock_data)
    
    # Step 3: Prepare data for ML
    X, y, features = prepare_data_for_ml(data_with_indicators)
    
    # Step 4: Train and evaluate the model
    model, y_test, y_pred = train_and_evaluate(X, y)
    
    # Step 5: Predict the next trading day's price
    last_day_features = X.iloc[-1:].values.reshape(1, -1)
    next_day_prediction = model.predict(last_day_features)
    
    print("\n--- Prediction for the Next Trading Day ---")
    last_known_date = X.index[-1].date()
    print(f"Based on data from: {last_known_date}")
    print(f"Predicted closing price for the next trading day: ${next_day_prediction[0]:.2f}")
    
    # Step 6: Visualize everything
    plot_results(data_with_indicators, y_test, y_pred, TICKER)
    
    print("\n--- IMPORTANT DISCLAIMER ---")
    print("This analysis is for educational purposes only and should not be considered financial advice.")


if __name__ == "__main__":
    main()