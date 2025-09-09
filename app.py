import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import uuid
import hashlib

def safe_date_filter(data, start_date, end_date):
    """Safely filter pandas DataFrame by date range, handling timezone issues"""
    try:
        # Convert dates to pandas datetime
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        
        # Handle timezone compatibility
        if data.index.tz is not None:
            # Data is timezone-aware
            if start_datetime.tz is None:
                start_datetime = start_datetime.tz_localize('UTC').tz_convert(data.index.tz)
            if end_datetime.tz is None:
                end_datetime = end_datetime.tz_localize('UTC').tz_convert(data.index.tz)
        else:
            # Data is timezone-naive
            if start_datetime.tz is not None:
                start_datetime = start_datetime.tz_localize(None)
            if end_datetime.tz is not None:
                end_datetime = end_datetime.tz_localize(None)
        
        # Filter the data
        filtered_data = data[start_datetime:end_datetime]
        return filtered_data
        
    except Exception as e:
        # Fallback to string-based filtering
        try:
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            return data.loc[start_str:end_str]
        except:
            # Final fallback - return empty DataFrame
            return data.iloc[0:0]

# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Stock Analysis Dashboard")
st.markdown("*TradingView-style stock analysis with technical indicators*")

# Sidebar for inputs
st.sidebar.header("📊 Stock Selection")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL")

# Time period selection
st.sidebar.subheader("📅 Time Period Selection")

# Choice between preset periods and custom date range
period_choice = st.sidebar.radio(
    "Choose analysis method:",
    ["Quick Periods", "Custom Date Range"],
    help="Quick Periods for common timeframes, Custom Date Range for specific analysis"
)

if period_choice == "Quick Periods":
    period_options = {
        "1 Day": "1d",
        "1 Week": "1wk", 
        "1 Month": "1mo",
        "2 Months": "2mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    selected_period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Set custom dates to None for preset periods
    custom_start_date = None
    custom_end_date = None
    
else:  # Custom Date Range
    st.sidebar.markdown("*Select your custom analysis period:*")
    
    # Calculate available date range (5 years back from today)
    today = date.today()
    five_years_ago = today - timedelta(days=5*365)
    
    # Quick preset buttons for common custom ranges
    st.sidebar.markdown("**Quick Presets:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📅 Last 7 Days", key="last_7_days"):
            st.session_state.quick_start = today - timedelta(days=7)
            st.session_state.quick_end = today
        if st.button("📅 Last 30 Days", key="last_30_days"):
            st.session_state.quick_start = today - timedelta(days=30)
            st.session_state.quick_end = today
            
    with col2:
        if st.button("📅 Last 90 Days", key="last_90_days"):
            st.session_state.quick_start = today - timedelta(days=90)
            st.session_state.quick_end = today
        if st.button("📅 YTD", key="ytd"):
            st.session_state.quick_start = date(today.year, 1, 1)
            st.session_state.quick_end = today
    
    # Custom date inputs
    custom_start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.get('quick_start', today - timedelta(days=30)),
        min_value=five_years_ago,
        max_value=today,
        help="Select the start date for your analysis"
    )
    
    custom_end_date = st.sidebar.date_input(
        "End Date", 
        value=st.session_state.get('quick_end', today),
        min_value=custom_start_date,
        max_value=today,
        help="Select the end date for your analysis"
    )
    
    # Validate date range
    if custom_start_date >= custom_end_date:
        st.sidebar.error("Start date must be before end date!")
    
    # Set period variables for custom range
    selected_period = f"Custom ({custom_start_date} to {custom_end_date})"
    period = "custom"

# Indicator settings
st.sidebar.header("🔧 Technical Indicators Settings")

# Indicator calculation method
st.sidebar.subheader("📊 Calculation Method")
indicator_calc_method = st.sidebar.radio(
    "Choose indicator calculation approach:",
    ["Smart Hybrid", "Period-Only", "Full-Context"],
    index=0,
    help="""
    • Smart Hybrid: Period-based for short-term indicators, full-context for long-term
    • Period-Only: All indicators calculated on selected period only
    • Full-Context: All indicators calculated on 5-year historical data
    """
)

# Explanation of selected method
if indicator_calc_method == "Smart Hybrid":
    st.sidebar.info("🧠 RSI & MACD: Period-based | MA & BB: Full-context")
elif indicator_calc_method == "Period-Only":
    st.sidebar.warning("⚠️ May have insufficient data for MA200 on short periods")
else:  # Full-Context
    st.sidebar.info("📈 All indicators use 5-year historical context")

# RSI settings
rsi_enabled = st.sidebar.checkbox("RSI (Relative Strength Index)", value=True)
if rsi_enabled:
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_overbought = st.sidebar.slider("RSI Overbought Level", 60, 90, 70)
    rsi_oversold = st.sidebar.slider("RSI Oversold Level", 10, 40, 30)

# MACD settings
macd_enabled = st.sidebar.checkbox("MACD", value=True)
if macd_enabled:
    macd_fast = st.sidebar.slider("MACD Fast Period", 5, 20, 12)
    macd_slow = st.sidebar.slider("MACD Slow Period", 20, 35, 26)
    macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, 9)

# Moving Averages
ma_enabled = st.sidebar.checkbox("Moving Averages (50 & 200)", value=True)

# Bollinger Bands
bb_enabled = st.sidebar.checkbox("Bollinger Bands", value=True)
if bb_enabled:
    bb_period = st.sidebar.slider("Bollinger Bands Period", 10, 30, 20)
    bb_std = st.sidebar.slider("Bollinger Bands Std Dev", 1.0, 3.0, 2.0, 0.1)

# Volume analysis
volume_enabled = st.sidebar.checkbox("Volume Analysis", value=True)

@st.cache_data
def fetch_stock_data(symbol, display_period, custom_start_date=None, custom_end_date=None):
    """Fetch stock data from Yahoo Finance - Always fetch 5 years for accurate indicators"""
    try:
        stock = yf.Ticker(symbol)
        
        # Always fetch 5 years of data for accurate technical indicators
        full_data = stock.history(period="5y")
        
        if full_data.empty:
            st.warning(f"No data available for {symbol}. Try a different symbol.")
            return None, None, None
        
        # Calculate the date range for display based on selected period
        if custom_start_date and custom_end_date:
            # Custom date range using safe filtering
            display_data = safe_date_filter(full_data, custom_start_date, custom_end_date)
            
            if display_data.empty:
                st.warning(f"No data available for {symbol} in the selected date range ({custom_start_date} to {custom_end_date}). Try a different range.")
                return None, None, None
        else:
            # Preset periods
            end_date = full_data.index[-1]  # Most recent date
            
            if display_period == "1d":
                # For 1 day, get intraday data separately for better granularity
                try:
                    intraday_data = stock.history(period="1d", interval="5m")
                    if not intraday_data.empty:
                        display_data = intraday_data
                    else:
                        display_data = full_data.tail(1)
                except:
                    display_data = full_data.tail(1)
            elif display_period == "1wk":
                display_data = full_data.tail(5)  # Approximately 1 week (5 trading days)
            elif display_period == "1mo":
                display_data = full_data.tail(22)  # Approximately 1 month (22 trading days)
            elif display_period == "2mo":
                display_data = full_data.tail(44)  # Approximately 2 months (44 trading days)
            elif display_period == "3mo":
                display_data = full_data.tail(66)  # Approximately 3 months
            elif display_period == "6mo":
                display_data = full_data.tail(132)  # Approximately 6 months
            elif display_period == "1y":
                display_data = full_data.tail(252)  # Approximately 1 year (252 trading days)
            elif display_period == "2y":
                display_data = full_data.tail(504)  # Approximately 2 years
            else:  # 5y
                display_data = full_data
        
        # Ensure we have at least one data point for display
        if len(display_data) == 0:
            st.warning(f"Insufficient data for {symbol}. Try a different symbol.")
            return None, None, None
        
        # Get company info
        try:
            info = stock.info
        except:
            info = {"longName": symbol.upper()}
            
        return full_data, display_data, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

def calculate_rsi(prices, period=14):
    """Calculate RSI using pandas"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD using pandas"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands using pandas"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    bollinger_upper = sma + (std * std_dev)
    bollinger_lower = sma - (std * std_dev)
    return bollinger_upper, sma, bollinger_lower

def calculate_indicators(full_data, display_data, calc_method="Smart Hybrid"):
    """Calculate all technical indicators using selected calculation method"""
    indicators = {}
    
    # Check if we have enough data
    if len(full_data) == 0 or len(display_data) == 0:
        return indicators
    
    # Determine which dataset to use for each indicator based on method
    if calc_method == "Period-Only":
        # Use display_data for all indicators
        rsi_data = display_data
        macd_data = display_data
        ma_data = display_data
        bb_data = display_data
    elif calc_method == "Full-Context":
        # Use full_data for all indicators
        rsi_data = full_data
        macd_data = full_data
        ma_data = full_data
        bb_data = full_data
    else:  # Smart Hybrid
        # Short-term indicators: use display period
        rsi_data = display_data
        macd_data = display_data
        # Long-term indicators: use full context
        ma_data = full_data
        bb_data = full_data
    
    # Calculate indicators based on selected datasets
    full_indicators = {}
    
    # RSI - calculate on selected dataset
    if rsi_enabled and len(rsi_data) >= rsi_period:
        full_indicators['rsi'] = calculate_rsi(rsi_data['Close'], rsi_period)
    
    # MACD - calculate on selected dataset
    if macd_enabled and len(macd_data) >= macd_slow:
        macd, macd_signal_line, macd_histogram = calculate_macd(
            macd_data['Close'], macd_fast, macd_slow, macd_signal
        )
        full_indicators['macd'] = macd
        full_indicators['macd_signal'] = macd_signal_line
        full_indicators['macd_histogram'] = macd_histogram
    
    # Moving Averages - calculate on selected dataset
    if ma_enabled:
        if len(ma_data) >= 50:
            full_indicators['ma50'] = ma_data['Close'].rolling(window=50).mean()
        if len(ma_data) >= 200:
            full_indicators['ma200'] = ma_data['Close'].rolling(window=200).mean()
    
    # Bollinger Bands - calculate on selected dataset
    if bb_enabled and len(bb_data) >= bb_period:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            bb_data['Close'], bb_period, bb_std
        )
        full_indicators['bb_upper'] = bb_upper
        full_indicators['bb_middle'] = bb_middle
        full_indicators['bb_lower'] = bb_lower
    
    # Extract indicators for the display period
    if calc_method == "Period-Only":
        # Indicators are already for the display period
        indicators = full_indicators
    else:
        # Filter indicators to match display period
        display_start_date = display_data.index[0]
        display_end_date = display_data.index[-1]
        
        for indicator_name, indicator_values in full_indicators.items():
            try:
                # Use safe date filtering for indicators too
                indicators[indicator_name] = indicator_values[display_start_date:display_end_date]
            except:
                # Fallback: take the last N values matching display data length
                indicators[indicator_name] = indicator_values.tail(len(display_data))
    
    return indicators

def create_main_chart(data, indicators, company_info):
    """Create the main stock chart with all indicators"""
    
    # Determine number of subplots
    subplot_count = 1  # Main price chart
    subplot_titles = [f"{symbol.upper()} Stock Price"]
    
    if rsi_enabled and 'rsi' in indicators:
        subplot_count += 1
        subplot_titles.append("RSI")
    
    if macd_enabled and 'macd' in indicators and 'macd_signal' in indicators and 'macd_histogram' in indicators:
        subplot_count += 1
        subplot_titles.append("MACD")
    
    if volume_enabled:
        subplot_count += 1
        subplot_titles.append("Volume")
    
    # Create subplots
    row_heights = [0.6] + [0.4/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1]
    
    fig = make_subplots(
        rows=subplot_count, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if ma_enabled and 'ma50' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['ma50'],
                name="MA 50",
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    if ma_enabled and 'ma200' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['ma200'],
                name="MA 200",
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if bb_enabled and 'bb_upper' in indicators and 'bb_lower' in indicators and 'bb_middle' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['bb_upper'],
                name="BB Upper",
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['bb_lower'],
                name="BB Lower",
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['bb_middle'],
                name="BB Middle",
                line=dict(color='blue', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    current_row = 2
    
    # RSI subplot
    if rsi_enabled and 'rsi' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['rsi'],
                name="RSI",
                line=dict(color='purple', width=2)
            ),
            row=current_row, col=1
        )
        
        # RSI levels
        fig.add_hline(
            y=rsi_overbought, 
            line=dict(color='red', width=1, dash='dash'),
            row=current_row, col=1
        )
        fig.add_hline(
            y=rsi_oversold, 
            line=dict(color='green', width=1, dash='dash'),
            row=current_row, col=1
        )
        fig.add_hline(
            y=50, 
            line=dict(color='gray', width=1, dash='dot'),
            row=current_row, col=1
        )
        
        current_row += 1
    
    # MACD subplot
    if macd_enabled and 'macd' in indicators and 'macd_signal' in indicators and 'macd_histogram' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['macd'],
                name="MACD",
                line=dict(color='blue', width=2)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['macd_signal'],
                name="MACD Signal",
                line=dict(color='red', width=2)
            ),
            row=current_row, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in indicators['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=indicators['macd_histogram'],
                name="MACD Histogram",
                marker_color=colors,
                opacity=0.6
            ),
            row=current_row, col=1
        )
        
        current_row += 1
    
    # Volume subplot
    if volume_enabled:
        colors = ['green' if close >= open_price else 'red' 
                 for close, open_price in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=current_row, col=1
        )
    
    # Update layout
    company_name = company_info.get('longName', symbol.upper()) if company_info else symbol.upper()
    fig.update_layout(
        title=f"{company_name} - Stock Analysis",
        xaxis_title="Date",
        template="plotly_dark",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    
    current_row = 2
    if rsi_enabled and 'rsi' in indicators:
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])
        current_row += 1
    
    if macd_enabled and 'macd' in indicators and 'macd_signal' in indicators and 'macd_histogram' in indicators:
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1
    
    if volume_enabled:
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
    
    return fig

def display_stock_info(data, company_info):
    """Display stock information and key metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    
    # Safe handling for previous close price
    if len(data) >= 2:
        prev_close = data['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
    else:
        # For single data point (like intraday data), compare with open price
        prev_close = data['Open'].iloc[-1] if len(data) > 0 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{change:.2f} ({change_pct:.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Day High",
            value=f"${data['High'].iloc[-1]:.2f}"
        )
    
    with col3:
        st.metric(
            label="Day Low",
            value=f"${data['Low'].iloc[-1]:.2f}"
        )
    
    with col4:
        st.metric(
            label="Volume",
            value=f"{data['Volume'].iloc[-1]:,.0f}"
        )
    
    # Company information
    if company_info:
        st.subheader("📋 Company Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Company:** {company_info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
            
        with col2:
            market_cap = company_info.get('marketCap', 0)
            if market_cap:
                st.write(f"**Market Cap:** ${market_cap:,.0f}")
            else:
                st.write("**Market Cap:** N/A")
            st.write(f"**P/E Ratio:** {company_info.get('trailingPE', 'N/A')}")
            st.write(f"**52W High:** ${company_info.get('fiftyTwoWeekHigh', 'N/A')}")

def get_trading_signals(data, indicators):
    """Generate trading signals based on indicators"""
    signals = {}
    
    # Check if we have enough data
    if len(data) == 0:
        return signals
    
    if rsi_enabled and 'rsi' in indicators and len(indicators['rsi'].dropna()) > 0:
        current_rsi = indicators['rsi'].iloc[-1]
        if not pd.isna(current_rsi):
            if current_rsi > rsi_overbought:
                signals['RSI'] = ("🔴 Overbought", "Consider selling")
            elif current_rsi < rsi_oversold:
                signals['RSI'] = ("🟢 Oversold", "Consider buying")
            else:
                signals['RSI'] = ("🟡 Neutral", "No clear signal")
    
    if macd_enabled and 'macd' in indicators and 'macd_signal' in indicators:
        if len(indicators['macd'].dropna()) > 0 and len(indicators['macd_signal'].dropna()) > 0:
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            if not pd.isna(current_macd) and not pd.isna(current_signal):
                if current_macd > current_signal:
                    signals['MACD'] = ("🟢 Bullish", "Upward momentum")
                else:
                    signals['MACD'] = ("🔴 Bearish", "Downward momentum")
    
    if ma_enabled and 'ma50' in indicators and 'ma200' in indicators:
        if len(indicators['ma50'].dropna()) > 0 and len(indicators['ma200'].dropna()) > 0:
            ma50_current = indicators['ma50'].iloc[-1]
            ma200_current = indicators['ma200'].iloc[-1]
            if not pd.isna(ma50_current) and not pd.isna(ma200_current):
                if ma50_current > ma200_current:
                    signals['Moving Average'] = ("🟢 Golden Cross", "Bullish trend")
                else:
                    signals['Moving Average'] = ("🔴 Death Cross", "Bearish trend")
    
    if bb_enabled and 'bb_upper' in indicators and 'bb_lower' in indicators:
        if len(indicators['bb_upper'].dropna()) > 0 and len(indicators['bb_lower'].dropna()) > 0:
            current_price = data['Close'].iloc[-1]
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                if current_price > bb_upper:
                    signals['Bollinger Bands'] = ("🔴 Above Upper Band", "Potentially overbought")
            elif current_price < bb_lower:
                signals['Bollinger Bands'] = ("🟢 Below Lower Band", "Potentially oversold")
            else:
                signals['Bollinger Bands'] = ("🟡 Within Bands", "Normal range")
    
    return signals

# Main app logic
# Enhanced Session Management for Multiple Users
def initialize_session():
    """Initialize session state with unique user identification"""
    # Generate unique session ID if not exists
    if 'session_id' not in st.session_state:
        # Create a unique session ID for this user
        timestamp = datetime.now().isoformat()
        random_uuid = str(uuid.uuid4())
        session_string = f"{timestamp}_{random_uuid}"
        st.session_state.session_id = hashlib.md5(session_string.encode()).hexdigest()[:12]
        st.session_state.session_start_time = datetime.now()
        st.session_state.analysis_count = 0
    
    # Initialize analysis data if not exists
    if 'analyzed_data' not in st.session_state:
        st.session_state.analyzed_data = None
        st.session_state.analyzed_indicators = None
        st.session_state.analyzed_info = None
        st.session_state.analyzed_symbol = None
        st.session_state.last_analysis_time = None
        st.session_state.analysis_history = []  # Track analyzed symbols

def display_session_info():
    """Display session information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Session Info")
    
    # Session duration
    if 'session_start_time' in st.session_state:
        duration = datetime.now() - st.session_state.session_start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
            
        st.sidebar.text(f"Session: {duration_str}")
        st.sidebar.text(f"Analyses: {st.session_state.get('analysis_count', 0)}")
        
        if st.session_state.analyzed_symbol:
            st.sidebar.text(f"Current: {st.session_state.analyzed_symbol}")
            
        # Show analysis history
        if st.session_state.get('analysis_history'):
            st.sidebar.markdown("**Recent Analyses:**")
            for hist_item in st.session_state.analysis_history[-3:]:  # Show last 3
                st.sidebar.text(f"• {hist_item['symbol']} ({hist_item['time'].strftime('%H:%M')})")
    
    # Clear session button
    if st.sidebar.button("🗑️ Clear Session"):
        clear_session_data()
        st.rerun()

def clear_session_data():
    """Clear analysis data from session"""
    st.session_state.analyzed_data = None
    st.session_state.analyzed_indicators = None
    st.session_state.analyzed_info = None
    st.session_state.analyzed_symbol = None
    st.session_state.last_analysis_time = None
    st.session_state.analysis_history = []

def update_analysis_stats(symbol, period_selected):
    """Update analysis statistics"""
    st.session_state.analysis_count = st.session_state.get('analysis_count', 0) + 1
    st.session_state.last_analysis_time = datetime.now()
    
    # Add to analysis history
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Add current analysis to history
    hist_item = {
        'symbol': symbol,
        'time': datetime.now(),
        'period': period_selected
    }
    
    # Remove if already exists (to avoid duplicates)
    st.session_state.analysis_history = [h for h in st.session_state.analysis_history if h['symbol'] != symbol]
    
    # Add to beginning of list
    st.session_state.analysis_history.insert(0, hist_item)
    
    # Keep only last 10 analyses
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[:10]

# Initialize session for this user
initialize_session()

# Display session info in sidebar
display_session_info()

if st.sidebar.button("📊 Analyze Stock", type="primary"):
    if not symbol.strip():
        st.error("Please enter a stock symbol")
    elif period_choice == "Custom Date Range" and custom_start_date >= custom_end_date:
        st.error("Please select a valid date range (start date must be before end date)")
    else:
        with st.spinner(f"Fetching data for {symbol.upper()}..."):
            if period_choice == "Custom Date Range":
                full_data, display_data, company_info = fetch_stock_data(
                    symbol.upper(), period, custom_start_date, custom_end_date
                )
            else:
                full_data, display_data, company_info = fetch_stock_data(symbol.upper(), period)
        
        if display_data is not None and not display_data.empty:
            # Store in session state
            st.session_state.analyzed_data = display_data
            st.session_state.analyzed_full_data = full_data
            st.session_state.analyzed_info = company_info
            st.session_state.analyzed_symbol = symbol.upper()
            st.session_state.analyzed_period = selected_period
            st.session_state.analyzed_period_choice = period_choice
            if period_choice == "Custom Date Range":
                st.session_state.custom_start_date = custom_start_date
                st.session_state.custom_end_date = custom_end_date
            
            # Calculate indicators using selected calculation method
            indicators = calculate_indicators(full_data, display_data, indicator_calc_method)
            st.session_state.analyzed_indicators = indicators
            st.session_state.analyzed_calc_method = indicator_calc_method
            
            # Update analysis statistics
            update_analysis_stats(symbol.upper(), selected_period)
            
            st.success(f"✅ Analysis completed for {symbol.upper()}")
        else:
            st.error(f"❌ Could not fetch data for symbol: {symbol.upper()}")
            st.info("💡 Please check the symbol and try again. Examples: AAPL, MSFT, GOOGL, TSLA")

# Display analysis if data exists in session state
if st.session_state.analyzed_data is not None:
    data = st.session_state.analyzed_data
    company_info = st.session_state.analyzed_info
    indicators = st.session_state.analyzed_indicators
    
    # Analysis info in expandable section (minimized by default)
    with st.expander("📊 Analysis Details", expanded=False):
        if st.session_state.last_analysis_time:
            analysis_time = st.session_state.last_analysis_time.strftime("%Y-%m-%d %H:%M:%S")
            period_name = st.session_state.get('analyzed_period', selected_period)
            period_choice_used = st.session_state.get('analyzed_period_choice', 'Quick Periods')
            st.info(f"📊 **Current Analysis:** {st.session_state.analyzed_symbol} | **Period:** {period_name} | **Generated:** {analysis_time}")
            
            # Debug information
            full_data_len = len(st.session_state.get('analyzed_full_data', []))
            calc_method_used = st.session_state.get('analyzed_calc_method', 'Full-Context')
            st.info(f"📋 **Data Info:** Showing {len(data)} data points from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')} (Full dataset: {full_data_len} points)")
            
            # Indicator calculation method info
            if indicators:
                sample_indicator = list(indicators.keys())[0]
                indicator_len = len(indicators[sample_indicator])
                if calc_method_used == "Smart Hybrid":
                    st.info(f"🧠 **Smart Hybrid:** Short-term indicators (RSI, MACD) on {len(data)} points | Long-term indicators (MA, BB) on {full_data_len} points")
                elif calc_method_used == "Period-Only":
                    st.info(f"📊 **Period-Only:** All indicators calculated on {len(data)} points of selected period")
                else:
                    st.info(f"📈 **Full-Context:** All indicators calculated on {full_data_len} points, filtered to {indicator_len} points")
            
            # Add enhanced notice about the approach
            if period_choice_used == "Custom Date Range":
                days_analyzed = len(data)
                start_date = data.index[0].strftime('%Y-%m-%d')
                end_date = data.index[-1].strftime('%Y-%m-%d')
                st.success(f"📅 **Custom Analysis**: {days_analyzed} trading days from {start_date} to {end_date} with indicators calculated using 5 years of historical data!")
            elif period_name in ["1 Day", "1 Week"]:
                st.success(f"✨ **Smart Analysis**: Displaying {period_name} data with indicators calculated using 5 years of historical data for maximum accuracy!")
            else:
                st.success(f"📈 **Professional Analysis**: {period_name} view with comprehensive technical indicators!")
    
    # Display stock information
    display_stock_info(data, company_info)
    
    # Create and display chart
    st.subheader("📈 Stock Chart with Technical Indicators")
    fig = create_main_chart(data, indicators, company_info)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display trading signals
    st.subheader("🎯 Trading Signals")
    signals = get_trading_signals(data, indicators)
    
    if signals:
        cols = st.columns(len(signals))
        for i, (indicator, (signal, description)) in enumerate(signals.items()):
            with cols[i]:
                st.markdown(f"**{indicator}**")
                st.markdown(f"{signal}")
                st.caption(description)
    else:
        st.info("No signals available - ensure indicators are enabled and data is sufficient")
    
    # Display current indicator values
    st.subheader("📊 Current Indicator Values")
    
    # Create a table data structure
    indicator_table_data = []
    
    if rsi_enabled and 'rsi' in indicators and len(indicators['rsi'].dropna()) > 0:
        rsi_val = indicators['rsi'].iloc[-1]
        if not pd.isna(rsi_val):
            indicator_table_data.append({
                "Indicator": "RSI",
                "Value": f"{rsi_val:.2f}",
                "Signal": "Overbought" if rsi_val > rsi_overbought else "Oversold" if rsi_val < rsi_oversold else "Neutral"
            })
    
    if macd_enabled and 'macd' in indicators and len(indicators['macd'].dropna()) > 0:
        macd_val = indicators['macd'].iloc[-1]
        if not pd.isna(macd_val):
            macd_signal_val = indicators.get('macd_signal', {}).iloc[-1] if 'macd_signal' in indicators else None
            signal = "Bullish" if macd_signal_val and macd_val > macd_signal_val else "Bearish" if macd_signal_val else "N/A"
            indicator_table_data.append({
                "Indicator": "MACD",
                "Value": f"{macd_val:.4f}",
                "Signal": signal
            })
    
    if ma_enabled:
        if 'ma50' in indicators and len(indicators['ma50'].dropna()) > 0:
            ma50_val = indicators['ma50'].iloc[-1]
            if not pd.isna(ma50_val):
                indicator_table_data.append({
                    "Indicator": "MA50",
                    "Value": f"${ma50_val:.2f}",
                    "Signal": "Support/Resistance"
                })
        
        if 'ma200' in indicators and len(indicators['ma200'].dropna()) > 0:
            ma200_val = indicators['ma200'].iloc[-1]
            if not pd.isna(ma200_val):
                indicator_table_data.append({
                    "Indicator": "MA200",
                    "Value": f"${ma200_val:.2f}",
                    "Signal": "Long-term Trend"
                })
    
    if bb_enabled and 'bb_upper' in indicators and 'bb_lower' in indicators:
        if len(indicators['bb_upper'].dropna()) > 0 and len(indicators['bb_lower'].dropna()) > 0:
            bb_upper_val = indicators['bb_upper'].iloc[-1]
            bb_lower_val = indicators['bb_lower'].iloc[-1]
            bb_middle_val = indicators.get('bb_middle', {}).iloc[-1] if 'bb_middle' in indicators else None
            
            if not pd.isna(bb_upper_val) and not pd.isna(bb_lower_val):
                current_price = data['Close'].iloc[-1]
                if current_price > bb_upper_val:
                    bb_signal = "Above Upper (Overbought)"
                elif current_price < bb_lower_val:
                    bb_signal = "Below Lower (Oversold)"
                else:
                    bb_signal = "Within Bands (Normal)"
                
                indicator_table_data.append({
                    "Indicator": "BB Upper",
                    "Value": f"${bb_upper_val:.2f}",
                    "Signal": bb_signal
                })
                indicator_table_data.append({
                    "Indicator": "BB Lower", 
                    "Value": f"${bb_lower_val:.2f}",
                    "Signal": bb_signal
                })
                
                if bb_middle_val and not pd.isna(bb_middle_val):
                    indicator_table_data.append({
                        "Indicator": "BB Middle",
                        "Value": f"${bb_middle_val:.2f}",
                        "Signal": "Moving Average"
                    })
    
    # Display the table
    if indicator_table_data:
        df_indicators = pd.DataFrame(indicator_table_data)
        st.dataframe(
            df_indicators,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Indicator": st.column_config.TextColumn("Indicator", width="medium"),
                "Value": st.column_config.TextColumn("Current Value", width="medium"),
                "Signal": st.column_config.TextColumn("Signal/Status", width="large")
            }
        )
    else:
        st.info("No indicator values available - ensure indicators are enabled and sufficient data exists")
    
    # Show data table in expandable section (minimized by default)
    with st.expander("📋 Raw Stock Data & Summary", expanded=False):
        # Show appropriate amount of data based on period
        if len(data) <= 50:
            # For short periods, show all data
            st.write(f"**Showing all {len(data)} data points for selected period:**")
            st.dataframe(data)
        else:
            # For longer periods, show first 10, last 10, and offer full download
            st.write(f"**Showing recent data (last 20 of {len(data)} total points):**")
            st.dataframe(data.tail(20))
            
            with st.expander("🔍 View Complete Dataset"):
                st.dataframe(data)
        
        # Additional data insights
        st.subheader("📊 Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Statistics (Selected Period):**")
            st.write(f"- Highest Price: ${data['High'].max():.2f}")
            st.write(f"- Lowest Price: ${data['Low'].min():.2f}")
            st.write(f"- Average Volume: {data['Volume'].mean():,.0f}")
            
        with col2:
            st.write("**Period Performance:**")
            period_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            st.write(f"- Period Return: {period_return:.2f}%")
            st.write(f"- Total Trading Days: {len(data)}")
            st.write(f"- Data Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

else:
    # Default display with personalized welcome
    session_id = st.session_state.get('session_id', 'Unknown')
    analysis_count = st.session_state.get('analysis_count', 0)
    
    if analysis_count == 0:
        st.info(f"👋 **Welcome to your Stock Analysis Dashboard!** (Session: {session_id[:8]})")
        st.info("👆 Select a stock symbol and click 'Analyze Stock' to start your first analysis")
    else:
        st.info(f"📊 **Welcome back!** You've performed {analysis_count} analysis(es) in this session")
        st.info("👆 Select a new stock symbol or adjust settings and click 'Analyze Stock'")
    
    st.markdown("""
    ### 🚀 Features of this Advanced Multi-User Stock Analysis Dashboard:
    
    #### 🧠 **Smart Data Processing:**
    - **Hybrid Indicator Calculation** - Choose between Smart Hybrid, Period-Only, or Full-Context
    - **Smart Hybrid (Recommended)** - Short-term indicators on selected period, long-term on 5-year data
    - **Flexible Time Periods** - Quick periods (1 day to 5 years) or custom date ranges
    - **Custom Date Analysis** - Select any specific date range within the last 5 years
    - **Quick Date Presets** - Last 7/30/90 days, Year-to-Date options
    - **Real-time Analysis** - Latest data with comprehensive historical context
    - **Optimized Performance** - Single API call provides data for all time periods
    
    #### 🔐 **Session Management:**
    - **Individual User Sessions** - Each user gets their own isolated data
    - **Session Tracking** - Monitor your analysis history and activity
    - **Data Privacy** - Your analysis data is only visible to you
    - **Session Persistence** - Data remains during your browser session
    
    #### 📈 **Advanced Technical Indicators:**
    - **🧠 Smart Hybrid Mode (Default):**
        - **RSI & MACD**: Calculated on selected period for immediate trend signals
        - **MA50/MA200 & Bollinger Bands**: Calculated on 5-year data for reliable context
    - **📊 Period-Only Mode:**
        - All indicators calculated only on selected time period
        - May have insufficient data warnings for short periods
    - **📈 Full-Context Mode:**
        - All indicators calculated on 5-year historical data
        - Maximum accuracy but may not reflect short-term changes
        - Golden Cross (MA50 > MA200): Bullish signal
        - Death Cross (MA50 < MA200): Bearish signal
    - **Bollinger Bands** - Volatility indicator
        - Price above upper band: Potentially overbought
        - Price below lower band: Potentially oversold
    - **Volume Analysis** - Trading volume with price correlation
        - Green bars: Price closed higher than opened
        - Red bars: Price closed lower than opened
    
    #### 🎯 **Key Features:**
    - **TradingView-style charts** with interactive plotly graphs
    - **Customizable indicator settings** via sidebar
    - **Real-time data** from Yahoo Finance
    - **Quick Periods & Custom Date Ranges** - Full flexibility in time selection
    - **Professional Analysis Tools** - Same quality as premium trading platforms
    - **Company information** and key metrics
    - **Trading signals** with color-coded alerts
    - **Current indicator values** display
    
    #### 📊 **How to Use:**
    1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL, TSLA)
    2. Choose **Quick Periods** (1 day to 5 years) or **Custom Date Range**
    3. For custom ranges: Use quick presets or select specific dates
    4. Adjust indicator settings in the sidebar
    5. Click "Analyze Stock" to generate the analysis
    6. Use expandable sections below to view analysis details and raw data
    
    #### 💡 **Popular Stock Symbols:**
    - **AAPL** - Apple Inc.
    - **MSFT** - Microsoft Corporation
    - **GOOGL** - Alphabet Inc.
    - **TSLA** - Tesla Inc.
    - **AMZN** - Amazon.com Inc.
    - **NVDA** - NVIDIA Corporation
    - **META** - Meta Platforms Inc.
    - **NFLX** - Netflix Inc.
    """)

# Footer
st.markdown("---")
session_info = f"Session: {st.session_state.get('session_id', 'Unknown')[:8]} | " if 'session_id' in st.session_state else ""
# st.markdown(f"*{session_info}Built with Streamlit, Yahoo Finance, and custom technical indicators | Multi-user sessions supported | Data is for educational purposes only*")
