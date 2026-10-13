import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")
st.caption("Educational project only — predictions are estimates, not financial advice.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock ticker", "RELIANCE.NS").strip().upper()
    period = st.selectbox("Historical data", ["2y", "5y", "10y"], index=1)
    forecast_days = st.slider("Forecast days", min_value=1, max_value=30, value=7)
    epochs = st.slider("Training epochs", min_value=5, max_value=50, value=15, step=5)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, history_period):
    data = yf.download(symbol, period=history_period, auto_adjust=False, progress=False)
    if data.empty:
        return pd.DataFrame()
    # yfinance may return MultiIndex columns for some ticker/version combinations.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Close" not in data.columns:
        return pd.DataFrame()
    result = data[["Close"]].copy()
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    return result.dropna()

def build_sequences(values, window):
    x, y = [], []
    for i in range(window, len(values)):
        x.append(values[i-window:i, 0])
        y.append(values[i, 0])
    return np.asarray(x).reshape(-1, window, 1), np.asarray(y)

if st.button("Train model and predict", type="primary", use_container_width=True):
    if not ticker:
        st.error("Please enter a stock ticker.")
        st.stop()

    try:
        with st.spinner(f"Downloading historical data for {ticker}..."):
            df = load_data(ticker, period)

        if df.empty or len(df) < 100:
            st.error("Not enough historical data was returned. Check the ticker (for example, RELIANCE.NS or AAPL) and try again.")
            st.stop()

        window = 60
        values = df[["Close"]].values.astype("float32")
        split = int(len(values) * 0.8)
        if split <= window or len(values) - split < 10:
            st.error("Not enough data to train and test the model. Choose a longer historical period.")
            st.stop()

        scaler = MinMaxScaler()
        scaler.fit(values[:split])
        scaled = scaler.transform(values)

        x_train, y_train = build_sequences(scaled[:split], window)
        test_values = scaled[split-window:]
        x_test, y_test = build_sequences(test_values, window)

        with st.spinner("Training LSTM model... this may take a few minutes on first run."):
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(window, 1)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0,
            )
            pred_scaled = model.predict(x_test, verbose=0)
            pred = scaler.inverse_transform(pred_scaled).flatten()
            actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest closing price", f"{values[-1, 0]:,.2f}")
        c2.metric("Test RMSE", f"{rmse:,.2f}")
        c3.metric("Test MAE", f"{mae:,.2f}")

        st.subheader(f"{ticker} — historical closing price")
        st.line_chart(df["Close"])

        st.subheader("Test period: actual vs predicted")
        test_dates = df.index[split:]
        comparison = pd.DataFrame({"Actual": actual, "LSTM predicted": pred}, index=test_dates[:len(actual)])
        st.line_chart(comparison)

        # Iteratively forecast future values from the most recent 60 observations.
        current_window = scaled[-window:].reshape(1, window, 1)
        future_scaled = []
        for _ in range(forecast_days):
            next_value = float(model.predict(current_window, verbose=0)[0, 0])
            future_scaled.append(next_value)
            current_window = np.concatenate(
                [current_window[:, 1:, :], np.array(next_value, dtype="float32").reshape(1, 1, 1)],
                axis=1,
            )
        future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast = pd.DataFrame({"Forecast closing price": future_prices}, index=future_dates)

        st.subheader(f"Next {forecast_days} business-day forecast")
        st.line_chart(forecast)
        st.dataframe(forecast.style.format({"Forecast closing price": "{:,.2f}"}), use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            data=forecast.to_csv(index_label="Date").encode("utf-8"),
            file_name=f"{ticker.lower().replace('.', '_')}_forecast.csv",
            mime="text/csv",
        )
        st.info("Stock markets are uncertain. This model uses historical closing prices only and should not be used alone for investment decisions.")

    except Exception as exc:
        st.error("The app could not complete the prediction.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.caption("If this is a deployment error, check the app logs and the Python/package versions in requirements.txt.")
else:
    st.write("Choose a ticker and settings in the sidebar, then click **Train model and predict**.")
    st.markdown("**Examples:** `RELIANCE.NS` (Reliance Industries, NSE), `TCS.NS`, `INFY.NS`, `AAPL` (Apple).")
