


# Stock Price Prediction using LSTM

This project leverages **Long Short-Term Memory (LSTM)**, a type of Recurrent Neural Network (RNN), to predict stock prices based on historical data. The model is trained using historical stock price data and is capable of predicting future stock prices over a user-defined number of days.

## Features
- **Data Fetching**: Fetches historical stock data using the `yfinance` library.
- **LSTM Model**: Uses an LSTM model to train and predict stock prices.
- **Future Predictions**: Predicts stock prices for a given number of future days.
- **Data Preprocessing**: Scales the stock data using MinMaxScaler to prepare for LSTM training.

## How It Works

### 1. Fetch Stock Data:
The `fetch_stock_data(ticker)` function downloads historical stock data for a given stock ticker using `yfinance`. The function retrieves data from **2015-01-01** and extracts the 'Close' price for training. If no data is found, an error is raised.

### 2. LSTM Model Setup:
The `lstm_prediction(df, forecast_days)` function takes in the stock data (`df`) and a desired number of forecast days (`forecast_days`). The data is scaled using **MinMaxScaler** to normalize the values between 0 and 1, which is required by the LSTM model.

- **Look-back Window**: The LSTM model is trained with a look-back window of 100 days. This means that the model considers the past 100 days of stock data to predict the next day's stock price.
- **Model Architecture**: The model is created using two **LSTM layers** followed by a **Dense layer** to output the predicted price.

### 3. Training the Model:
- The data is split into training sequences where each sequence consists of 100 past days of stock prices.
- The model is compiled with the **Adam optimizer** and trained using the **Mean Squared Error (MSE)** loss function.

### 4. Prediction:
After training, the model predicts stock prices for the given number of forecast days. The predictions are transformed back to the original price range using the inverse scaling. The forecasted dates are computed based on the last date of the stock data.

### 5. Output:
The function returns:
- **Predictions**: The predicted stock prices for the given forecast days.
- **Future Dates**: A list of future dates corresponding to the predicted prices.

## Technologies Used
- **Python**: Programming language for implementing the model.
- **TensorFlow/Keras**: Framework for building the LSTM model.
- **yfinance**: Library for fetching historical stock data.
- **scikit-learn**: Used for scaling the data with **MinMaxScaler**.
- **NumPy & Pandas**: For data manipulation and processing.




## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

