import streamlit as st
import pandas as pd
import yfinance as yf
from plotly import graph_objs as go
from datetime import datetime as dt

# Import the LSTM prediction function from the model.py file
from model import lstm_prediction

# Streamlit App Title and Config
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B7BE5;'>Stock Forecasting Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for User Inputs
st.sidebar.title("Stock Data Inputs")
stock_code = st.sidebar.text_input('Enter Stock Code:', 'AAPL')
start_date = st.sidebar.date_input('Start Date', dt(2023, 1, 1))
end_date = st.sidebar.date_input('End Date', dt.now())
forecast_days = st.sidebar.slider('Forecast Days', 1, 30, 7)

# Initialize session state for storing the fetched data
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Fetch stock data
if st.sidebar.button('Fetch Data'):
    df = yf.download(stock_code, start=start_date, end=end_date)
    
    if df.empty:
        st.error(f"No data found for {stock_code}. Please try a different stock code.")
    else:
        st.success(f"Data fetched for {stock_code} from {start_date} to {end_date}.")
        st.session_state['df'] = df  # Save the data to session state
        
        # Display stock data
        st.write(f"Stock data for {stock_code}:")
        st.dataframe(df.tail())

        # Plot the closing prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='#4B7BE5')))
        fig.update_layout(title=f"Closing Prices for {stock_code}", xaxis_title='Date', yaxis_title='Price (USD)', 
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# Forecast the stock prices
if st.sidebar.button('Forecast Stock Price'):
    if st.session_state['df'] is not None and not st.session_state['df'].empty:
        df = st.session_state['df']  # Get the stored data
        
        # Perform LSTM Prediction using the function from model.py
        predictions, future_dates = lstm_prediction(df, forecast_days)
        
        # Plot the forecasted prices
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Prices'))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig_forecast.update_layout(title=f"Predicted Closing Prices for {stock_code}", xaxis_title='Date', yaxis_title='Price (USD)', 
                                   template='plotly_white')
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.error("Please fetch data before forecasting.")
