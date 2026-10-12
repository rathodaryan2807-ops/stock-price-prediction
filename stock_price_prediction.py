import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model

WINDOW = 60
MODEL_PATH = "lstm_stock_model.keras"
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="wide")
st.title("Stock Price Prediction (LSTM)")


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.download(ticker, period="2y", progress=False)
    if df is None or df.empty:
        return None
    # newer yfinance returns MultiIndex columns like ("Close", "RELIANCE.NS")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error(
        "lstm_stock_model.keras ya scaler.pkl nahi mila. "
        "Pehle stock_price_prediction.py chalao aur dono files repo mein upload karo."
    )
    st.stop()

ticker = st.text_input("Ticker", "RELIANCE.NS").strip().upper()
days = st.slider("Forecast days", 1, 30, 7)

if st.button("Predict"):
    df = get_data(ticker)
    if df is None or len(df) < WINDOW:
        st.error(f"'{ticker}' ka kaafi data nahi mila (kam se kam {WINDOW} din chahiye).")
        st.stop()

    model, scaler = load_artifacts()

    window = scaler.transform(df.values)[-WINDOW:].reshape(1, WINDOW, 1)
    preds = []
    for _ in range(days):
        p = model.predict(window, verbose=0)[0, 0]
        preds.append(p)
        window = np.append(window[:, 1:, :], [[[p]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({"Forecast": preds}, index=future_dates)

    col1, col2 = st.columns(2)
    col1.metric("Last close", f"{float(df['Close'].iloc[-1]):.2f}")
    col2.metric(f"Predicted (day {days})", f"{preds[-1]:.2f}")

    st.subheader("Historical price (2 years)")
    st.line_chart(df)

    st.subheader("Forecast")
    st.line_chart(forecast)
    st.dataframe(forecast.round(2))

    st.caption("Educational project only. Not financial advice.")
