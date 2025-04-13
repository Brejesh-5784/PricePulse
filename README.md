
```markdown
# 📈 Stock Forecasting Dashboard

This is a stock price forecasting application built using **Streamlit**, **LSTM (Long Short-Term Memory)**, and **yfinance**. The app predicts the future stock prices of a company based on historical stock data and visualizes the results on an interactive dashboard.

## Features
- **Stock Price Prediction**: Predict future stock prices for any ticker using an LSTM model.
- **Historical Data Visualization**: Displays the last 100 days of historical stock prices.
- **Interactive UI**: Easy-to-use interface built with Streamlit that allows users to select a stock ticker and forecast days.
- **Forecasting Period**: Select the number of days (5 to 30) for which the stock price is predicted.
- **Graphical Representation**: The dashboard shows the historical stock data along with the predicted prices.

## 🛠️ Technologies Used

- **Python**: Core language for backend logic.
- **Streamlit**: For building the interactive dashboard.
- **Keras (TensorFlow)**: For implementing the LSTM model.
- **yfinance**: To fetch historical stock data from Yahoo Finance.
- **Matplotlib**: For plotting the stock price data and predictions.
- **pandas**: For data manipulation.

## Requirements

Make sure you have the following dependencies installed:
```bash
pip install numpy pandas yfinance scikit-learn tensorflow streamlit matplotlib
```

## How to Run the Project

1. **Clone this repository**:
    ```bash
    git clone https://github.com/yourusername/stock-forecasting-dashboard.git
    cd stock-forecasting-dashboard
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

4. **Access the Dashboard**: Open your browser and go to `http://localhost:8501` to access the stock forecasting dashboard.

## How It Works

1. **Data Fetching**: The user enters a stock ticker symbol (e.g., AAPL, MSFT, GOOGL) and the app fetches historical stock data (from 2015 onwards) using the `yfinance` library.
2. **Preprocessing**: The stock data is preprocessed using **MinMaxScaler** to normalize the prices before feeding them into the LSTM model.
3. **LSTM Model**: The LSTM model is trained on the historical data and used to predict the future stock prices.
4. **Prediction**: After training, the model predicts stock prices for a user-defined number of days.
5. **Visualization**: The historical stock prices and the predicted stock prices are plotted using **Matplotlib** and displayed on the Streamlit dashboard.

## Example Usage

1. Enter the stock ticker of the company (e.g., `AAPL` for Apple, `MSFT` for Microsoft).
2. Choose the number of forecast days (e.g., 5 to 30 days).
3. Click on the "🔍 Forecast" button to generate predictions.
4. The app will display:
   - A graph showing the historical stock prices and the predicted future prices.
   - A table with the predicted prices and corresponding dates.

## Contributing

Feel free to fork this repository, make improvements, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Streamlit**: For building the interactive web application.
- **yfinance**: For providing easy access to Yahoo Finance data.
- **TensorFlow/Keras**: For enabling LSTM-based stock price prediction.
- **Matplotlib**: For plotting and visualizing the data.



