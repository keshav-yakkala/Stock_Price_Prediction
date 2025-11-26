# Indian Stock Market Dashboard

A comprehensive Streamlit application for analyzing Indian stock market data.

## Features

- **Real-time Data**: Fetches live stock data using `yfinance`.
- **Technical Analysis**: Interactive charts with SMA, EMA, and Bollinger Bands.
- **Price Prediction**: LSTM Neural Network model to forecast future stock prices.
- **News & Sentiment**: Aggregates latest news and performs sentiment analysis using FinBERT and VADER.
- **Fundamental Metrics**: Displays key financial metrics like PE Ratio, Market Cap, and 52-Week High/Low.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Dependencies

- streamlit
- yfinance
- pandas
- numpy
- plotly
- pandas_ta
- scikit-learn
- tensorflow
- transformers
- torch
- vaderSentiment
