import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def preprocess_data(data):
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Daily_Return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data = data.dropna()
    return data

def calculate_rsi(prices, window=14):
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def select_features(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA10', 'MA50', 'RSI']]
    return features

def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def select_model():
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2]
    }

    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)
    return grid_search

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

def make_prediction(model, features):
    prediction = model.predict(features)
    return prediction

def visualize_results(actual, predicted):
    plt.plot(actual.index, actual, label='Actual Price', color='blue')
    plt.plot(predicted.index, predicted, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def run_prediction():
    symbol = symbol_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    stock_data = get_stock_data(symbol, start_date, end_date)
    processed_data = preprocess_data(stock_data)
    features = select_features(processed_data)
    target = processed_data['Close']

    X_train, X_test, y_train, y_test = split_data(features, target)
    model = select_model()
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    predictions = make_prediction(model, features)
    visualize_results(target, pd.Series(predictions, index=target.index))

root = tk.Tk()
root.title("Stock Price Prediction")
symbol_label = ttk.Label(root, text="Stock Symbol:")
symbol_label.grid(row=0, column=0, padx=10, pady=10)
symbol_entry = ttk.Entry(root)
symbol_entry.grid(row=0, column=1, padx=10, pady=10)
start_date_label = ttk.Label(root, text="Start Date (YYYY-MM-DD):")
start_date_label.grid(row=1, column=0, padx=10, pady=10)
start_date_entry = ttk.Entry(root)
start_date_entry.grid(row=1, column=1, padx=10, pady=10)
end_date_label = ttk.Label(root, text="End Date (YYYY-MM-DD):")
end_date_label.grid(row=2, column=0, padx=10, pady=10)
end_date_entry = ttk.Entry(root)
end_date_entry.grid(row=2, column=1, padx=10, pady=10)
run_button = ttk.Button(root, text="Run Prediction", command=run_prediction)
run_button.grid(row=3, column=0, columnspan=2, pady=20)
def set_default_dates():
    today = datetime.today().strftime('%Y-%m-%d')
    one_year_ago = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
    start_date_entry.delete(0, tk.END)
    start_date_entry.insert(0, one_year_ago)
    end_date_entry.delete(0, tk.END)
    end_date_entry.insert(0, today)
default_dates_button = ttk.Button(root, text="Set Default Dates", command=set_default_dates)
default_dates_button.grid(row=4, column=0, columnspan=2, pady=10)
root.mainloop()
