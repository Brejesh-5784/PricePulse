import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

def fetch_stock_data(ticker):
    """
    Fetches historical stock data for the given ticker using yfinance.
    """
    df = yf.download(ticker, start='2015-01-01', progress=False, threads=False, timeout=10)
    if df.empty or 'Close' not in df.columns:
        raise ValueError("No data found for this ticker.")
    df = df[['Close']]
    df.dropna(inplace=True)
    return df


def lstm_prediction(df, forecast_days):
    """
    Trains an LSTM model and predicts stock prices for the given number of days.
    """
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

    look_back = 100
    if len(scaled_data) <= look_back:
        raise ValueError("Not enough data to train the LSTM model. Try a different stock or increase date range.")

    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    predictions = []
    last_data = scaled_data[-look_back:]
    last_data = np.reshape(last_data, (1, look_back, 1))

    for _ in range(forecast_days):
        pred = model.predict(last_data)[0, 0]
        predictions.append(pred)
        last_data = np.append(last_data[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    return predictions, future_dates

