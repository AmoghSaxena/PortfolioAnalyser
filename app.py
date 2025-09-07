import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import uuid
import hashlib

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
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
selected_period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))
period = period_options[selected_period]

# Indicator settings
st.sidebar.header("🔧 Technical Indicators Settings")

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

# Show raw data option
show_raw_data = st.sidebar.checkbox("📋 Show Raw Data", value=False)

@st.cache_data
def fetch_stock_data(symbol, period):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return None, None
        
        # Get company info
        try:
            info = stock.info
        except:
            info = {"longName": symbol.upper()}
        return data, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

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

def calculate_indicators(data):
    """Calculate all technical indicators"""
    indicators = {}
    
    # RSI
    if rsi_enabled:
        indicators['rsi'] = calculate_rsi(data['Close'], rsi_period)
    
    # MACD
    if macd_enabled:
        macd, macd_signal_line, macd_histogram = calculate_macd(
            data['Close'], macd_fast, macd_slow, macd_signal
        )
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal_line
        indicators['macd_histogram'] = macd_histogram
    
    # Moving Averages
    if ma_enabled:
        indicators['ma50'] = data['Close'].rolling(window=50).mean()
        indicators['ma200'] = data['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    if bb_enabled:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            data['Close'], bb_period, bb_std
        )
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
    
    return indicators

def create_main_chart(data, indicators, company_info):
    """Create the main stock chart with all indicators"""
    
    # Determine number of subplots
    subplot_count = 1  # Main price chart
    subplot_titles = [f"{symbol.upper()} Stock Price"]
    
    if rsi_enabled:
        subplot_count += 1
        subplot_titles.append("RSI")
    
    if macd_enabled:
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
    if bb_enabled and 'bb_upper' in indicators:
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
    if macd_enabled and 'macd' in indicators:
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
    if rsi_enabled:
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])
        current_row += 1
    
    if macd_enabled:
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1
    
    if volume_enabled:
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
    
    return fig

def display_stock_info(data, company_info):
    """Display stock information and key metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    
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
    
    if rsi_enabled and 'rsi' in indicators:
        current_rsi = indicators['rsi'].iloc[-1]
        if not pd.isna(current_rsi):
            if current_rsi > rsi_overbought:
                signals['RSI'] = ("🔴 Overbought", "Consider selling")
            elif current_rsi < rsi_oversold:
                signals['RSI'] = ("🟢 Oversold", "Consider buying")
            else:
                signals['RSI'] = ("🟡 Neutral", "No clear signal")
    
    if macd_enabled and 'macd' in indicators:
        current_macd = indicators['macd'].iloc[-1]
        current_signal = indicators['macd_signal'].iloc[-1]
        if not pd.isna(current_macd) and not pd.isna(current_signal):
            if current_macd > current_signal:
                signals['MACD'] = ("🟢 Bullish", "Upward momentum")
            else:
                signals['MACD'] = ("🔴 Bearish", "Downward momentum")
    
    if ma_enabled and 'ma50' in indicators and 'ma200' in indicators:
        ma50_current = indicators['ma50'].iloc[-1]
        ma200_current = indicators['ma200'].iloc[-1]
        if not pd.isna(ma50_current) and not pd.isna(ma200_current):
            if ma50_current > ma200_current:
                signals['Moving Average'] = ("🟢 Golden Cross", "Bullish trend")
            else:
                signals['Moving Average'] = ("🔴 Death Cross", "Bearish trend")
    
    if bb_enabled and 'bb_upper' in indicators and 'bb_lower' in indicators:
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
    else:
        with st.spinner(f"Fetching data for {symbol.upper()}..."):
            data, company_info = fetch_stock_data(symbol.upper(), period)
        
        if data is not None and not data.empty:
            # Store in session state
            st.session_state.analyzed_data = data
            st.session_state.analyzed_info = company_info
            st.session_state.analyzed_symbol = symbol.upper()
            
            # Calculate indicators
            indicators = calculate_indicators(data)
            st.session_state.analyzed_indicators = indicators
            
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
    
    # Show analysis timestamp
    if st.session_state.last_analysis_time:
        analysis_time = st.session_state.last_analysis_time.strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"📊 **Current Analysis:** {st.session_state.analyzed_symbol} | **Generated:** {analysis_time}")
    
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
    
    indicator_cols = []
    if rsi_enabled and 'rsi' in indicators:
        indicator_cols.append(("RSI", f"{indicators['rsi'].iloc[-1]:.2f}" if not pd.isna(indicators['rsi'].iloc[-1]) else "N/A"))
    
    if macd_enabled and 'macd' in indicators:
        macd_val = indicators['macd'].iloc[-1]
        indicator_cols.append(("MACD", f"{macd_val:.4f}" if not pd.isna(macd_val) else "N/A"))
    
    if ma_enabled and 'ma50' in indicators and 'ma200' in indicators:
        ma50_val = indicators['ma50'].iloc[-1]
        ma200_val = indicators['ma200'].iloc[-1]
        if not pd.isna(ma50_val) and not pd.isna(ma200_val):
            indicator_cols.append(("MA50/MA200", f"${ma50_val:.2f} / ${ma200_val:.2f}"))
        else:
            indicator_cols.append(("MA50/MA200", "N/A"))
    
    if bb_enabled and 'bb_upper' in indicators:
        bb_upper_val = indicators['bb_upper'].iloc[-1]
        bb_lower_val = indicators['bb_lower'].iloc[-1]
        if not pd.isna(bb_upper_val) and not pd.isna(bb_lower_val):
            indicator_cols.append(("BB Range", f"${bb_lower_val:.2f} - ${bb_upper_val:.2f}"))
        else:
            indicator_cols.append(("BB Range", "N/A"))
    
    if indicator_cols:
        cols = st.columns(len(indicator_cols))
        for i, (label, value) in enumerate(indicator_cols):
            with cols[i]:
                st.metric(label=label, value=value)
    
    # Show data table if enabled
    if show_raw_data:
        st.subheader("📋 Raw Stock Data")
        st.dataframe(data.tail(20))
        
        # Additional data insights
        st.subheader("📊 Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Statistics:**")
            st.write(f"- Highest Price: ${data['High'].max():.2f}")
            st.write(f"- Lowest Price: ${data['Low'].min():.2f}")
            st.write(f"- Average Volume: {data['Volume'].mean():,.0f}")
            
        with col2:
            st.write("**Recent Performance:**")
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
    ### 🚀 Features of this Multi-User Stock Analysis Dashboard:
    
    #### 🔐 **Session Management:**
    - **Individual User Sessions** - Each user gets their own isolated data
    - **Session Tracking** - Monitor your analysis history and activity
    - **Data Privacy** - Your analysis data is only visible to you
    - **Session Persistence** - Data remains during your browser session
    
    #### 📈 **Technical Indicators:**
    - **RSI (Relative Strength Index)** - Momentum oscillator (0-100)
        - Values above 70: Potentially overbought
        - Values below 30: Potentially oversold
    - **MACD** - Moving Average Convergence Divergence
        - MACD above signal line: Bullish momentum
        - MACD below signal line: Bearish momentum
    - **Moving Averages** - 50-day and 200-day simple moving averages
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
    - **Multiple time periods** (1 month to 5 years)
    - **Company information** and key metrics
    - **Trading signals** with color-coded alerts
    - **Current indicator values** display
    
    #### 📊 **How to Use:**
    1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL, TSLA)
    2. Select your preferred time period
    3. Adjust indicator settings in the sidebar
    4. Click "Analyze Stock" to generate the analysis
    5. Use "Show Raw Data" checkbox in sidebar to view detailed data
    
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
