import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

def lstm_prediction(df, forecast_days):
    """
    This function trains an LSTM model on the stock data and returns future predictions.

    Parameters:
    df (pd.DataFrame): Stock data with a 'Close' column.
    forecast_days (int): The number of future days to predict.

    Returns:
    predictions (np.array): Predicted stock prices for the forecast period.
    future_dates (list): Dates for the forecasted prices.
    """
    # Preprocess the data
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Prepare training data using 60-day window
    look_back = 60
    X_train, y_train = [], []
    
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshaping for LSTM
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Predict future prices
    predictions = []
    last_data = scaled_data[-look_back:]
    last_data = np.reshape(last_data, (1, look_back, 1))
    
    for _ in range(forecast_days):
        pred = model.predict(last_data)[0, 0]
        predictions.append(pred)
        last_data = np.append(last_data[:, 1:, :], [[[pred]]], axis=1)  # Update sliding window
    
    # Inverse scale the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create future dates for the forecast
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    
    return predictions, future_dates
