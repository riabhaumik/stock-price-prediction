# 📈 Stock Price Predictor & Technical Analyzer

A modern, interactive web application for stock analysis and price prediction using machine learning and technical indicators.

## 🚀 Features

##  Features


- **Interactive Stock Analysis**: Enter any stock ticker and get instant analysis
- **Technical Indicators**: Moving Averages (50-day & 200-day) and RSI
- **Machine Learning Prediction**: Linear Regression model for next-day price prediction
- **Beautiful Dark Mode UI**: Modern 3D-styled charts with shadows
- **Date-Specific Price Lookup**: Find exact stock prices at any historical date
- **Real-time Data**: Live stock data from Yahoo Finance
- **Responsive Design**: Works on desktop and mobile devices


## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📊 How to Use

1. **Enter Stock Ticker**: Type any valid stock symbol (e.g., MSFT, TSLA, GOOGL, NFLX)
2. **Adjust Time Period**: Use the slider to select years of historical data (1-10 years)
3. **View Analysis**: See price charts, moving averages, and RSI indicators
4. **Check Predictions**: View the predicted price for the next trading day
5. **Historical Lookup**: Use the date picker to find prices at specific dates


## 🎯 Technical Indicators


### Moving Averages
- **50-Day Moving Average**: Short-term trend indicator
- **200-Day Moving Average**: Long-term trend indicator
- **Golden Cross**: When 50-day MA crosses above 200-day MA (bullish signal)
- **Death Cross**: When 50-day MA crosses below 200-day MA (bearish signal)

### Relative Strength Index (RSI)
- **Overbought Level**: RSI > 70 (potential sell signal)
- **Oversold Level**: RSI < 30 (potential buy signal)
- **Neutral Zone**: RSI between 30-70


##  Machine Learning Model


The application uses a **Linear Regression** model trained on:
- Historical price data (Open, High, Low, Close)
- Trading volume
- Technical indicators (Moving Averages, RSI)

**Model Performance**: Shows RMSE (Root Mean Square Error) on test data

## 📁 Project Structure

```
stock-price-predictor/
├── app.py                 # Main Streamlit application
├── stock_predictor.py     # Original script version
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── screenshot.png        # Application screenshot
└── venv/                 # Virtual environment (not in repo)
```

## 📈 Supported Stocks

The application works with any stock available on Yahoo Finance, including:
- **US Stocks**: AAPL, MSFT, GOOGL, TSLA, AMZN, NFLX, etc.
- **International Stocks**: Many global markets supported
- **ETFs**: SPY, QQQ, VTI, etc.

## ⚠️ Disclaimer

**This application is for educational purposes only.** 

- The predictions are based on historical data and technical analysis
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- This tool should not be considered as financial advice

##  Troubleshooting
### Common Issues

1. **"No data found for ticker"**
   - Check if the stock symbol is correct
   - Try a different stock ticker
   - Verify internet connection

2. **Installation errors**
   - Make sure Python 3.7+ is installed
   - Try upgrading pip: `pip install --upgrade pip`
   - Use virtual environment to avoid conflicts

3. **Charts not loading**
   - Refresh the browser page
   - Check browser console for errors
   - Try a different browser

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
=======
##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments


- **Yahoo Finance**: For providing stock data
- **Streamlit**: For the amazing web framework
- **Plotly**: For interactive charts
- **yfinance**: For easy stock data access
- **scikit-learn**: For machine learning capabilities

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Ensure all dependencies are properly installed

---

**Happy Trading! 📈💰**

