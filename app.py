import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import date, timedelta

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

# Define the layout components

# Navigation component
item1 = html.Div(
    [
        html.P("Welcome to the Stock Forecasting App using Dash Framework!", className="start"),

        html.Div([
            # stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Submit', id='submit-button')
        ], className="stock-input"),

        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range', start_date=dt(2023, 1, 1).date(), end_date=dt.now().date(), className='date-input')
        ]),
        html.Div([
            # Stock price button
            html.Button('Stock Price', id='stock-price-button'),

            # Indicators button
            html.Button('Indicators', id='indicators-button'),

            # Number of days of forecast input
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),

            # Forecast button
            html.Button('Get Forecast', id='forecast-button')
        ], className="selectors")
    ],
    className="nav"
)

# Content component
item2 = html.Div(
    [
        html.Div(
            [
                html.Img(id='logo', className='logo'),
                html.H1(id='company-name', className='company-name')
            ],
            className="header"),
        html.Div(id="description"),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ],
    className="content"
)

# Set the layout
app.layout = html.Div(className='container', children=[item1, item2])

# Callbacks

# Callback to update the data based on the submitted stock code
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("company-name", "children"),
        Output("stock-price-button", "n_clicks"),
        Output("indicators-button", "n_clicks"),
        Output("forecast-button", "n_clicks")
    ],
    [Input("submit-button", "n_clicks")],
    [State("stock-code", "value")]
)
def update_data(n, val):
    if n is None:
        return None, None, None, None, None, None
    else:
        if val is None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            if 'logo_url' not in inf:
                return None, None, None, None, None, None
            else:
                name = inf['longName']
                logo_url = inf['logo_url']
                description = inf['longBusinessSummary']
                return description, logo_url, name, None, None, None


# Callback for displaying stock price graphs
@app.callback(
    [Output("graphs-content", "children")],
    [
        Input("stock-price-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        raise PreventUpdate
    else:
        if start_date is not None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace=True)
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    return [dcc.Graph(figure=fig)]


# Callback for displaying indicators
@app.callback(
    [Output("main-content", "children")],
    [
        Input("indicators-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        return [""]
    if start_date is None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


# Callback for displaying forecast
@app.callback(
    [Output("forecast-content", "children")],
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    if n is None:
        return [""]
    if val is None:
        raise PreventUpdate
    fig = lstm_prediction(val, int(n_days))
    return [dcc.Graph(figure=fig)]


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
    
    # Plot results
    dates = [date.today() + timedelta(days=i) for i in range(n_days)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=dates, y=predictions.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(title=f"Predicted Close Price for Next {n_days} Days", xaxis_title="Date", yaxis_title="Close Price")
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
