import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import date, timedelta
import plotly.graph_objs as go

def lstm_prediction(stock, n_days):
    # Load and preprocess the data
    df = yf.download(stock, period='60d')
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Prepare training data
    train_data = scaled_data[:-n_days]
    X_train, y_train = [], []
    for i in range(len(train_data) - 1):
        X_train.append(train_data[i:i + 1])
        y_train.append(train_data[i + 1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    # Predict future values
    last_data = scaled_data[-1:]
    predictions = []
    for _ in range(n_days):
        pred = model.predict(np.expand_dims(last_data, axis=0))[0, -1]
        predictions.append(pred)
        last_data = np.concatenate((last_data[1:], np.expand_dims(pred, axis=0)))
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Generate future dates for the forecast
    dates = [date.today() + timedelta(days=i) for i in range(n_days)]
    
    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=dates, y=predictions.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(title=f"Predicted Close Price for Next {n_days} Days", xaxis_title="Date", yaxis_title="Close Price")
    
    return fig
