
# Stock Prediction Web Application using Dash and Machine Learning

This project provides a web application designed to help stock market investors visualize company stock data and make predictions based on machine learning models. Built with the Dash framework, this tool offers an interactive user interface for analyzing and forecasting stock prices.

## Project Structure

The project structure consists of the following files:

1. `app.py` - The main application file containing the Dash app instance, layout, and callbacks. This file sets up the web interface and handles user interactions.
2. `model.py` - Contains functions to fetch stock data using the yfinance library and to train and predict stock prices using machine learning algorithms.
3. `assets/styles.css` -  Provides custom styling for the web application, enhancing its visual appeal.
4. `requirements.txt` - Lists all the dependencies required to run the application.

## Features 

Stock Data Visualization: Allows users to input a company stock code and a date range to visualize historical stock prices.
Stock Price Prediction: Utilizes machine learning models to forecast future stock prices based on historical data.
Interactive User Interface: Built with Dash, offering a seamless user experience with options to view historical data, indicators, and forecasts.

## Usage

1. Install Dependencies: Ensure you have the required packages installed. 
You can install them using the following command: pip install -r requirements.txt

2. Run the Application: Start the Dash application by running app.py:python app.py

3. Interact with the Application:
Stock Code Input: Enter the stock code of the company you want to analyze.
Date Range Picker: Select the start and end dates for the historical data you want to visualize.
Stock Price Button: View the stock price data for the selected date range.
Indicators Button: View technical indicators such as Exponential Moving Averages.
Forecast Button: Get predictions for future stock prices based on historical data.


## Dependencies

The following Python libraries are required to run the application:

Dash: For building the web application interface.
yfinance: For fetching stock data.
pandas: For data manipulation and analysis.
scikit-learn: For machine learning algorithms.
plotly: For interactive data visualizations.
tensorflow: For training LSTM models.
Install these dependencies using the command : pip install -r requirements.txt

## Future Enhancements

1.Incorporate additional machine learning algorithms for better predictions.
2.Add more technical indicators and analytics tools.
3.Implement user authentication and personalized dashboards.

## Conclusion

This project provides a simple yet powerful tool for investors to visualize and analyze stock data of a specific company using machine learning models. It is a great project for beginners to learn about web application development, data visualization, and machine learning in Python.
