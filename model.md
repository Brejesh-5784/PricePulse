### Documentation for LSTM Prediction Function

#### Overview
The `lstm_prediction` function is designed to train a Long Short-Term Memory (LSTM) model on historical stock price data and generate predictions for future prices. It employs data preprocessing techniques to prepare the data, constructs an LSTM model, and utilizes it to forecast stock prices over a specified period.

#### Function Definition

```python
def lstm_prediction(df, forecast_days):
```

#### Parameters
- **df (pd.DataFrame)**: A Pandas DataFrame containing historical stock data, specifically with a 'Close' column representing the closing prices of the stock.
- **forecast_days (int)**: An integer indicating the number of future days for which stock prices should be predicted.

#### Returns
- **predictions (np.array)**: An array containing the predicted stock prices for the specified forecast period.
- **future_dates (list)**: A list of dates corresponding to the forecasted stock prices.

#### Function Workflow

1. **Data Preprocessing**:
   - The function first extracts the 'Close' prices from the input DataFrame.
   - It then initializes a `MinMaxScaler` to scale the price data to a range of 0 to 1, enhancing the model's training efficiency.

2. **Training Data Preparation**:
   - A look-back window of 60 days is used to create training samples:
     - `X_train`: A collection of input sequences (60 days of closing prices).
     - `y_train`: The corresponding output values (the next day's closing price).
   - The data is reshaped to meet the input requirements of the LSTM model.

3. **Model Construction**:
   - A Sequential LSTM model is built:
     - Two LSTM layers with 50 units each, where the first layer returns sequences for the second layer.
     - A Dense layer is added as the output layer to predict a single value (the closing price).
   - The model is compiled using the Adam optimizer and mean squared error loss function.

4. **Model Training**:
   - The model is trained on the prepared training data for 20 epochs with a batch size of 32.

5. **Price Prediction**:
   - The function prepares to predict future prices by using the last 60 days of scaled data.
   - For each day in the forecast period, the model predicts the next closing price and updates the sliding window with the new prediction.
   - This process continues for the specified number of forecast days.

6. **Inverse Scaling**:
   - The predicted stock prices are inverse scaled to their original values using the same `MinMaxScaler`.

7. **Future Date Generation**:
   - The function computes future dates for the forecasted prices by adding days to the last date in the input DataFrame.

#### Usage
This function can be utilized within a broader stock forecasting application to generate and visualize future stock price predictions, aiding investors in their decision-making processes based on machine learning insights. The integration of LSTM networks allows the model to capture complex temporal dependencies in the stock price data effectively.