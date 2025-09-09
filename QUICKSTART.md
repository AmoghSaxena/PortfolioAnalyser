#  Quick Start Guide

## Running the Application Locally

1. **Clone the repository:**
   `ash
   git clone https://github.com/AmoghSaxena/PortfolioAnalyser.git
   cd PortfolioAnalyser
   `

2. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `

3. **Run the application:**
   `ash
   streamlit run app.py
   `

4. **Open your browser:**
   - The app will automatically open at http://localhost:8501
   - If not, manually navigate to this URL

## First Time Usage

1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Choose your analysis period:
   - **Quick Periods**: Select from preset timeframes (1 Day to 5 Years)
   - **Custom Date Range**: Pick any specific date range with quick presets (Last 7/30/90 days, YTD) or select exact dates
3. Adjust technical indicator settings in the sidebar
4. Click "Analyze Stock" to generate your first analysis
5. Explore the interactive charts and trading signals

## Features to Try

- ⚡ Analyze different stocks (AAPL, TSLA, NVDA, etc.)
- 📅 **NEW: Custom Date Ranges!**
  - **Quick Periods**: 1 Day, 1 Week, 1 Month, 3 Months, 6 Months, 1 Year, 2 Years, 5 Years
  - **Custom Ranges**: Pick any date range within the last 5 years
  - **Quick Presets**: Last 7/30/90 days, Year-to-Date analysis
- 🔧 Adjust RSI, MACD, and Bollinger Band parameters
- 📊 Toggle between different indicators
- 📈 Enable/disable various technical indicators
- 📋 View raw data and statistics
- 👤 Check your session info and analysis history

**Smart Feature:** All technical indicators are calculated using 5 years of historical data for maximum accuracy, regardless of the display period you choose!

Enjoy your stock analysis! 
