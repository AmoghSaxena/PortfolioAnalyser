#  Portfolio Analyzer - Advanced Stock Analysis Dashboard

A comprehensive Streamlit-based stock analysis application that provides TradingView-style charts with advanced technical indicators and multi-user session management.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

##  Features

###  Technical Indicators
- **RSI (Relative Strength Index)** - Momentum oscillator with customizable overbought/oversold levels
- **MACD** - Moving Average Convergence Divergence with signal line and histogram
- **Moving Averages** - 50-day and 200-day SMA with Golden/Death Cross signals
- **Bollinger Bands** - Volatility indicator with customizable periods and standard deviation
- **Volume Analysis** - Trading volume correlation with price movements

###  Multi-User Session Management
- **Individual User Sessions** - Each user gets isolated data and analysis
- **Session Tracking** - Monitor analysis history and activity duration
- **Data Privacy** - Complete isolation between different users
- **Session Persistence** - Data remains during browser session
- **Analysis History** - Track recently analyzed stocks with timestamps

###  Advanced Features
- **TradingView-style Charts** - Professional candlestick charts with dark theme
- **Real-time Data** - Live stock data from Yahoo Finance
- **Customizable Settings** - Adjust all indicator parameters via sidebar
- **Trading Signals** - Color-coded buy/sell/neutral signals with interpretations
- **Company Information** - Display key metrics and company details
- **Multiple Time Periods** - Analysis from 1 month to 5 years
- **Raw Data Display** - Detailed data tables with statistics

##  Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Clone the Repository
`ash
git clone https://github.com/AmoghSaxena/PortfolioAnalyser.git
cd PortfolioAnalyser
`

### Install Dependencies
`ash
pip install -r requirements.txt
`

### Run the Application
`ash
streamlit run app.py
`

The application will open in your default web browser at http://localhost:8501

##  Dependencies

- **streamlit** - Web application framework
- **yfinance** - Yahoo Finance API for stock data
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **plotly** - Interactive charting library

##  Usage

### Basic Usage
1. **Enter Stock Symbol** - Type any valid stock symbol (e.g., AAPL, MSFT, GOOGL)
2. **Select Time Period** - Choose from 1 month to 5 years
3. **Customize Indicators** - Adjust parameters in the sidebar
4. **Analyze Stock** - Click the "Analyze Stock" button
5. **Interpret Results** - Review charts, signals, and metrics

### Popular Stock Symbols
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- **TSLA** - Tesla Inc.
- **AMZN** - Amazon.com Inc.
- **NVDA** - NVIDIA Corporation
- **META** - Meta Platforms Inc.
- **NFLX** - Netflix Inc.

### Technical Indicator Interpretation

#### RSI (Relative Strength Index)
- **RSI > 70**: Potentially overbought (consider selling)
- **RSI < 30**: Potentially oversold (consider buying)
- **RSI  50**: Neutral momentum

#### MACD
- **MACD above Signal Line**: Bullish momentum
- **MACD below Signal Line**: Bearish momentum
- **Histogram**: Shows strength of momentum

#### Moving Averages
- **Golden Cross** (MA50 > MA200): Bullish long-term signal
- **Death Cross** (MA50 < MA200): Bearish long-term signal

#### Bollinger Bands
- **Price above Upper Band**: Potentially overbought
- **Price below Lower Band**: Potentially oversold
- **Price within Bands**: Normal trading range

##  Configuration

### Customizable Parameters
- **RSI Period**: 5-30 days (default: 14)
- **RSI Levels**: Overbought (60-90), Oversold (10-40)
- **MACD**: Fast (5-20), Slow (20-35), Signal (5-15)
- **Bollinger Bands**: Period (10-30), Standard Deviation (1.0-3.0)

### Session Management
- Each browser session gets a unique ID
- Analysis history is maintained per session
- Session data automatically clears when browser is closed
- Use "Clear Session" button to reset data manually

##  Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Disclaimer

This application is for educational and informational purposes only. The data and analysis provided should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.

##  Contact

**Amogh Saxena** - [GitHub Profile](https://github.com/AmoghSaxena)

Project Link: [https://github.com/AmoghSaxena/PortfolioAnalyser](https://github.com/AmoghSaxena/PortfolioAnalyser)

##  Acknowledgments

- **Yahoo Finance** for providing free stock data API
- **Streamlit** for the amazing web application framework
- **Plotly** for interactive charting capabilities
- **Pandas & NumPy** for data processing and analysis

---

*Built with  using Python and Streamlit*
