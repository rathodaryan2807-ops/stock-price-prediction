"""
Stock Price Prediction using LSTM
Standalone training/evaluation script.

Run:
    python stock_price_prediction.py

For the interactive web app:
    streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

TICKER = "RELIANCE.NS"
PERIOD = "10y"
WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32
MODEL_PATH = Path("lstm_stock_model.keras")
SCALER_PATH = Path("scaler.pkl")


def download_data(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"No data received for {ticker}. Check the ticker symbol.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Close" not in data.columns:
        raise RuntimeError("The downloaded data does not contain a Close column.")
    return data[["Close"]].dropna()


def make_sequences(values: np.ndarray, window: int):
    x, y = [], []
    for i in range(window, len(values)):
        x.append(values[i-window:i, 0])
        y.append(values[i, 0])
    return np.asarray(x).reshape(-1, window, 1), np.asarray(y)


def main():
    df = download_data(TICKER, PERIOD)
    if len(df) < WINDOW + 50:
        raise RuntimeError("Not enough historical data to train the model.")

    values = df[["Close"]].values.astype("float32")
    split = int(len(values) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(values[:split])
    scaled = scaler.transform(values)

    x_train, y_train = make_sequences(scaled[:split], WINDOW)
    x_test, y_test = make_sequences(scaled[split-WINDOW:], WINDOW)

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(
        x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.1, callbacks=[early_stop], verbose=1,
    )

    pred = scaler.inverse_transform(model.predict(x_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    print(f"Ticker: {TICKER}")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

    test_dates = df.index[split:]
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates[:len(actual)], actual, label="Actual")
    plt.plot(test_dates[:len(pred)], pred, label="LSTM predicted")
    plt.title(f"{TICKER}: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
