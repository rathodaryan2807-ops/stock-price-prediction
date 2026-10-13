""""
Trains the LSTM used by the Streamlit app and saves:
  - lstm_stock_model.keras
  - scaler.pkl
 
Run:  python stock_price_prediction.py
Settings below must match the app (WINDOW, single 'Close' feature).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
 
TICKER = "RELIANCE.NS"
PERIOD = "10y"
WINDOW = 60  # same as app.py
MODEL_PATH = "lstm_stock_model.keras"
SCALER_PATH = "scaler.pkl"
 
 
def load_close(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        raise SystemExit(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()
 
 
def make_windows(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i, 0])
        y.append(arr[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    return X, np.array(y)
 
 
def main():
    df = load_close(TICKER, PERIOD)
    values = df.values  # shape (n, 1), same as the app
 
    split = int(len(values) * 0.8)
    train_raw, test_raw = values[:split], values[split - WINDOW:]
 
    # fit scaler on train only (no leakage)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)
 
    X_train, y_train = make_windows(train_scaled, WINDOW)
    X_test, y_test = make_windows(test_scaled, WINDOW)
 
    model = Sequential([
        Input(shape=(WINDOW, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
 
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )
 
    pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mape = float(np.mean(np.abs((pred - actual) / actual)) * 100)
    print(f"Test RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
 
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved {MODEL_PATH} and {SCALER_PATH}")
 
 
if __name__ == "__main__":
    main()
 
