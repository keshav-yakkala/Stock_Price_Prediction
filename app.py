import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import threading
import queue
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Indian Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Style */
    .stMetric, .stContainer {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2d3748;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e2e8f0;
        color: #2b6cb0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION & STATE ---
DEFAULT_TICKERS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "BSE SENSEX": "^BSESN",
    "NIFTY 50": "^NSEI"
}

if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []

# --- MODELS (Cached) ---
@st.cache_resource
def load_sentiment_models():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    vader = SentimentIntensityAnalyzer()
    return finbert, vader

finbert_pipeline, vader_analyzer = load_sentiment_models()

# --- HELPER FUNCTIONS ---
def get_stock_data(ticker, period="1d", interval="1m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Calculate Technical Indicators
        if not df.empty and len(df) > 20:
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.ema(length=20, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.bbands(length=20, append=True)
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except:
        return {}

def get_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        # Fallback if yfinance returns empty list
        if not news:
            # Try fetching from a general market news source if specific stock news fails
            # For now, we return a placeholder to avoid UI breaking, or we could try a different ticker like ^BSESN
            return []
        return news
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def create_lstm_model(input_shape):
    model = Sequential()
    # Adjust units based on input size to prevent overfitting on small data
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(df, days_ahead=30):
    if df.empty:
        return None, None, None
        
    data_len = len(df)
    
    # Dynamic Lookback Window
    if data_len < 60:
        prediction_days = max(5, data_len // 4) # Use smaller window for short data
    else:
        prediction_days = 60
        
    if data_len <= prediction_days + 5: # Need at least a few points for training
         return None, None, None

    # Preprocessing
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    
    # Use all available data for training
    for i in range(prediction_days, data_len):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    if len(x_train) == 0:
        return None, None, None
        
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build Model
    model = create_lstm_model((x_train.shape[1], 1))
    
    # Train
    # Adjust epochs based on data size
    epochs = 5 if data_len > 200 else 10 
    model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    
    # Predict Future
    # Start with the last window
    test_data = scaled_data[-prediction_days:]
    x_input = test_data.reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    
    lst_output = []
    n_steps = prediction_days
    i = 0
    
    while(i < days_ahead):
        if(len(temp_input) > n_steps):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1
            
    predicted_prices = scaler.inverse_transform(lst_output)
    
    # Future Dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    return future_dates, predicted_prices.flatten(), model

def analyze_sentiment(text):
    if not text:
        return 0.0
    try:
        # FinBERT
        # Truncate text to 512 tokens if needed, but for titles it's fine
        fb_res = finbert_pipeline(text[:512])[0]
        fb_score = fb_res['score'] if fb_res['label'] == 'positive' else -fb_res['score'] if fb_res['label'] == 'negative' else 0
        
        # VADER
        vader_score = vader_analyzer.polarity_scores(text)['compound']
        
        return (fb_score + vader_score) / 2
    except Exception as e:
        # print(f"Sentiment Error: {e}") # Debug
        return 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🇮🇳 Market Settings")
    selected_stock_name = st.selectbox("Select Stock", list(DEFAULT_TICKERS.keys()))
    ticker = DEFAULT_TICKERS[selected_stock_name]
    
    st.subheader("Analysis Parameters")
    period = st.select_slider("History Period", options=['1d', '5d', '1mo', '3mo', '1y', '5y'], value='1y')
    interval = st.select_slider("Interval", options=['1m', '5m', '15m', '1d', '1wk'], value='1d')
    
    st.subheader("Technical Indicators")
    show_sma = st.checkbox("SMA (20 & 50)", value=True)
    show_ema = st.checkbox("EMA (20)")
    show_bb = st.checkbox("Bollinger Bands")
    
    st.info(f"Selected: **{selected_stock_name}** ({ticker})")

# --- MAIN CONTENT ---
st.title(f"📊 {selected_stock_name} Dashboard")

# Fetch Data
data = get_stock_data(ticker, period=period, interval=interval)

if not data.empty:
    # Calculate Metrics
    latest_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else latest_close
    change = latest_close - prev_close
    pct_change = (change / prev_close) * 100
    
    # Top Metrics Row (Fundamentals)
    info = get_stock_info(ticker)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{latest_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    col2.metric("Market Cap", f"₹{info.get('marketCap', 'N/A'):,}" if isinstance(info.get('marketCap'), (int, float)) else "N/A")
    col3.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
    col4.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volume", f"{int(data['Volume'].iloc[-1]):,}")
    col2.metric("Day High", f"₹{float(data['High'].iloc[-1]):.2f}")
    col3.metric("Day Low", f"₹{float(data['Low'].iloc[-1]):.2f}")
    col4.metric("Sector", f"{info.get('sector', 'N/A')}")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Real-Time Chart", "🔮 Prediction Studio", "📰 News & Sentiment"])

    with tab1:
        st.subheader("Price Movement")
        # Interactive Plotly Chart
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price'))
        
        # Indicators
        if show_sma and 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)))
            
        if show_ema and 'EMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='purple', width=1)))
            
        if show_bb and 'BBL_20_2.0' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['BBU_20_2.0'], name='Upper BB', line=dict(color='gray', width=0.5), showlegend=False))
            fig.add_trace(go.Scatter(x=data.index, y=data['BBL_20_2.0'], name='Lower BB', line=dict(color='gray', width=0.5), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False))

        fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False, 
                          title=f"{selected_stock_name} Technical Chart")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Price Forecast (LSTM Neural Network)")
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
        
        if st.button("Generate Prediction"):
            with st.spinner("Training LSTM model (this may take a moment)..."):
                future_dates, predictions, model = train_lstm_model(data, days_ahead=forecast_days)
                
                if future_dates:
                    pred_fig = go.Figure()
                    # Historical
                    pred_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical', line=dict(color='blue')))
                    # Prediction
                    pred_fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast', line=dict(color='red', dash='dash')))
                    
                    pred_fig.update_layout(title=f"{selected_stock_name} Price Prediction", height=500, template="plotly_white")
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Recommendation
                    trend = "UP" if predictions[-1] > data['Close'].iloc[-1] else "DOWN"
                    st.success(f"LSTM Model predicts an **{trend}** trend over the next {forecast_days} days.")
                else:
                    st.error("Not enough data to train LSTM model. Please select a longer history period (e.g., 1y or 5y).")

    with tab3:
        st.subheader("Latest Market News")
        st.subheader("Latest Market News (Live)")
        news_data = get_stock_news(ticker)
        
        if news_data:
            for item in news_data[:5]: # Show top 5 news
                # Handle nested structure if present
                if 'content' in item:
                    news_item = item['content']
                else:
                    news_item = item
                
                title = news_item.get('title', 'No Title')
                
                # Link might be in clickThroughUrl or canonicalUrl
                link = '#'
                if 'clickThroughUrl' in news_item and news_item['clickThroughUrl']:
                     link = news_item['clickThroughUrl'].get('url', '#')
                elif 'canonicalUrl' in news_item and news_item['canonicalUrl']:
                     link = news_item['canonicalUrl'].get('url', '#')
                else:
                     link = news_item.get('link', '#')

                # Publisher
                publisher = 'Unknown'
                if 'provider' in news_item and news_item['provider']:
                    publisher = news_item['provider'].get('displayName', 'Unknown')
                else:
                    publisher = news_item.get('publisher', 'Unknown')
                
                sent_score = analyze_sentiment(title)
                sent_label = "Positive" if sent_score > 0.2 else "Negative" if sent_score < -0.2 else "Neutral"
                color = "green" if sent_label == "Positive" else "red" if sent_label == "Negative" else "gray"
                
                with st.container():
                    col_news, col_sent = st.columns([3, 1])
                    with col_news:
                        st.markdown(f"**[{title}]({link})**")
                        st.caption(f"Source: {publisher}")
                    with col_sent:
                        st.markdown(f":{color}[{sent_label}]")
                        st.progress((sent_score + 1) / 2) # Normalize -1 to 1 -> 0 to 1
        else:
            st.info("No recent news found for this stock.")

else:
    st.warning("No data available. Please check the ticker or try a different period.")
