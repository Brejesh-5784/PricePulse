# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from model import lstm_prediction

st.set_page_config(page_title="📈 Stock Forecasting Dashboard", layout="centered")
st.title("📈 Stock Forecasting Dashboard")

# Input section
stock_code = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper()
forecast_days = st.slider("Select Forecast Period (Days)", min_value=5, max_value=30, value=10)

# Fetch stock data
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker):
    df = yf.download(ticker, start="2015-01-01", progress=False, threads=False, timeout=10)
    if df.empty or "Close" not in df.columns:
        raise ValueError("No data found for this ticker.")
    df = df[["Close"]].dropna()
    return df

# Predict and plot
if st.button("🔍 Forecast"):
    try:
        df = fetch_stock_data(stock_code)
        predictions, future_dates = lstm_prediction(df, forecast_days)

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index[-100:], df["Close"].values[-100:], label="Historical")
        ax.plot(future_dates, predictions, label="Forecast", linestyle='--', color="green")
        ax.set_title(f"{stock_code} Stock Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Show table
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions.flatten()})
        forecast_df.set_index("Date", inplace=True)
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")
