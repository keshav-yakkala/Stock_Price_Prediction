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
