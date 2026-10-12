"""
Stock Price Prediction using LSTM (Senior Project)
---------------------------------------------------
Install:  pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
Run:      python stock_price_prediction.py
Web app:  streamlit run app.py   (see bottom of this file for app code)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- 1. CONFIG ----------------
TICKER = "RELIANCE.NS"      # Indian stock (NSE). US example: "AAPL"
START, END = "2015-01-01", "2026-10-01"
WINDOW = 60                 # last 60 days -> predict next day
EPOCHS, BATCH = 50, 32

# ---------------- 2. DATA ----------------
df = yf.download(TICKER, start=START, end=END)
df = df[["Close"]].dropna()
print(df.tail())

# Quick EDA plot
df["Close"].plot(figsize=(12, 5), title=f"{TICKER} Closing Price")
plt.show()

# Moving averages (EDA)
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()
df[["Close", "MA50", "MA200"]].plot(figsize=(12, 5), title="Moving Averages")
plt.show()

# ---------------- 3. PREPROCESSING ----------------
data = df[["Close"]].values
split = int(len(data) * 0.8)          # 80% train, 20% test (time order maintain)

scaler = MinMaxScaler()
scaler.fit(data[:split])              # sirf train pe fit -> data leakage nahi
scaled = scaler.transform(data)

def make_sequences(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i, 0])
        y.append(arr[i, 0])
    return np.array(X).reshape(-1, window, 1), np.array(y)

train_scaled = scaled[:split]
test_scaled = scaled[split - WINDOW:]   # window ke liye overlap

X_train, y_train = make_sequences(train_scaled, WINDOW)
X_test, y_test = make_sequences(test_scaled, WINDOW)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------------- 4. LSTM MODEL ----------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH,
    validation_split=0.1, callbacks=[early_stop], verbose=1,
)

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend(); plt.title("Training vs Validation Loss"); plt.show()

# ---------------- 5. EVALUATION ----------------
pred = scaler.inverse_transform(model.predict(X_test))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)
mape = np.mean(np.abs((actual - pred) / actual)) * 100
r2 = r2_score(actual, pred)
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R2: {r2:.3f}")

# Baseline: "kal ka price = aaj ka price" (naive model). LSTM isse better hona chahiye.
naive_rmse = np.sqrt(mean_squared_error(actual[1:], actual[:-1]))
print(f"Naive baseline RMSE: {naive_rmse:.2f}")

test_dates = df.index[split:]
plt.figure(figsize=(12, 5))
plt.plot(test_dates, actual, label="Actual")
plt.plot(test_dates, pred, label="LSTM Predicted")
plt.title(f"{TICKER}: Actual vs Predicted"); plt.legend(); plt.show()

# ---------------- 6. FUTURE FORECAST (next 30 days) ----------------
last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
future = []
for _ in range(30):
    nxt = model.predict(last_window, verbose=0)[0, 0]
    future.append(nxt)
    last_window = np.append(last_window[:, 1:, :], [[[nxt]]], axis=1)
future = scaler.inverse_transform(np.array(future).reshape(-1, 1))
print("Next 30 days forecast:", future.flatten().round(2))

# ---------------- 7. SAVE MODEL ----------------
model.save("lstm_stock_model.keras")
import joblib
joblib.dump(scaler, "scaler.pkl")

# =====================================================================
# STREAMLIT APP  (isko alag file "app.py" mein save karo)
# =====================================================================
APP_CODE = '''
import streamlit as st, numpy as np, yfinance as yf, joblib
from tensorflow.keras.models import load_model

st.title("Stock Price Prediction (LSTM)")
ticker = st.text_input("Ticker", "RELIANCE.NS")
days = st.slider("Forecast days", 1, 30, 7)

if st.button("Predict"):
    df = yf.download(ticker, period="2y")[["Close"]].dropna()
    st.line_chart(df)
    model = load_model("lstm_stock_model.keras")
    scaler = joblib.load("scaler.pkl")
    window = scaler.transform(df.values)[-60:].reshape(1, 60, 1)
    preds = []
    for _ in range(days):
        p = model.predict(window, verbose=0)[0, 0]
        preds.append(p)
        window = np.append(window[:, 1:, :], [[[p]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    st.subheader("Forecast")
    st.line_chart(preds.flatten())
    st.caption("Educational project only. Not financial advice.")
'''
