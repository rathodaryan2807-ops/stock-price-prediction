import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction & Future Forecast")
st.write("Search a company by typing the beginning of its name, select it, and forecast future prices.")
st.caption("Educational project only — model forecasts are estimates, not financial advice.")

# Searchable company list. Type the first letters in the dropdown to filter it.
STOCKS = {
    "Aarti Industries — AARTIIND.NS": "AARTIIND.NS",
    "Adani Enterprises — ADANIENT.NS": "ADANIENT.NS",
    "Adani Ports — ADANIPORTS.NS": "ADANIPORTS.NS",
    "Asian Paints — ASIANPAINT.NS": "ASIANPAINT.NS",
    "Axis Bank — AXISBANK.NS": "AXISBANK.NS",
    "Bajaj Finance — BAJFINANCE.NS": "BAJFINANCE.NS",
    "Bharti Airtel — BHARTIARTL.NS": "BHARTIARTL.NS",
    "Cipla — CIPLA.NS": "CIPLA.NS",
    "Coal India — COALINDIA.NS": "COALINDIA.NS",
    "HCL Technologies — HCLTECH.NS": "HCLTECH.NS",
    "HDFC Bank — HDFCBANK.NS": "HDFCBANK.NS",
    "Hindustan Unilever — HINDUNILVR.NS": "HINDUNILVR.NS",
    "ICICI Bank — ICICIBANK.NS": "ICICIBANK.NS",
    "Infosys — INFY.NS": "INFY.NS",
    "ITC — ITC.NS": "ITC.NS",
    "Kotak Mahindra Bank — KOTAKBANK.NS": "KOTAKBANK.NS",
    "Larsen & Toubro — LT.NS": "LT.NS",
    "Maruti Suzuki — MARUTI.NS": "MARUTI.NS",
    "NTPC — NTPC.NS": "NTPC.NS",
    "Power Grid — POWERGRID.NS": "POWERGRID.NS",
    "Reliance Industries — RELIANCE.NS": "RELIANCE.NS",
    "State Bank of India — SBIN.NS": "SBIN.NS",
    "Sun Pharmaceutical — SUNPHARMA.NS": "SUNPHARMA.NS",
    "Tata Consultancy Services — TCS.NS": "TCS.NS",
    "Tata Motors — TATAMOTORS.NS": "TATAMOTORS.NS",
    "Tata Steel — TATASTEEL.NS": "TATASTEEL.NS",
    "Tech Mahindra — TECHM.NS": "TECHM.NS",
    "Wipro — WIPRO.NS": "WIPRO.NS",
    "Apple — AAPL": "AAPL",
    "Amazon — AMZN": "AMZN",
    "Alphabet (Google) — GOOGL": "GOOGL",
    "Microsoft — MSFT": "MSFT",
    "NVIDIA — NVDA": "NVDA",
    "Tesla — TSLA": "TSLA",
}
labels = sorted(STOCKS.keys())
with st.sidebar:
    st.header("🔎 Find a stock")
    selected_label = st.selectbox(
        "Type the first letters of a company name",
        labels,
        index=labels.index("Reliance Industries — RELIANCE.NS"),
        help="Click the box and type e.g. Reliance, Tata, Infosys, Apple, or NVIDIA.",
        filter_mode="prefix",
    )
    custom_ticker = st.checkbox("Use another ticker symbol")
    if custom_ticker:
        ticker = st.text_input("Enter ticker symbol", "RELIANCE.NS").strip().upper()
    else:
        ticker = STOCKS[selected_label]
    st.caption(f"Selected ticker: **{ticker or '—'}**")
    period = st.selectbox("Historical data", ["2y", "5y", "10y"], index=1)
    forecast_days = st.slider("Future business days", min_value=1, max_value=30, value=7)
    epochs = st.slider("Training epochs", min_value=5, max_value=50, value=15, step=5)
    show_indicators = st.checkbox("Show moving averages", value=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, history_period):
    data = yf.download(symbol, period=history_period, auto_adjust=False, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Close" not in data.columns:
        return pd.DataFrame()
    result = data[["Close"]].copy()
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    return result.dropna()

def make_sequences(values, window):
    x, y = [], []
    for i in range(window, len(values)):
        x.append(values[i-window:i, 0])
        y.append(values[i, 0])
    return np.asarray(x).reshape(-1, window, 1), np.asarray(y)

if st.button("🚀 Train model and predict", type="primary", use_container_width=True):
    if not ticker:
        st.error("Enter a valid ticker symbol.")
        st.stop()
    try:
        with st.spinner(f"Downloading historical prices for {ticker}..."):
            df = load_data(ticker, period)
        if df.empty or len(df) < 100:
            st.error("No data or not enough data returned. Check the ticker symbol and try again.")
            st.stop()

        df = df.copy()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        latest = float(df["Close"].iloc[-1])

        st.subheader(f"{ticker} — historical price")
        st.metric("Latest available closing price", f"{latest:,.2f}", help="Latest close returned by the data provider; it may not be today's live price.")
        if show_indicators:
            st.line_chart(df[["Close", "MA20", "MA50"]])
        else:
            st.line_chart(df[["Close"]])

        values = df[["Close"]].values.astype("float32")
        window = 60
        split = int(len(values) * 0.8)
        if split <= window or len(values) - split < 10:
            st.error("Not enough historical data to train and test.")
            st.stop()

        scaler = MinMaxScaler()
        scaler.fit(values[:split])
        scaled = scaler.transform(values)
        x_train, y_train = make_sequences(scaled[:split], window)
        x_test, y_test = make_sequences(scaled[split-window:], window)

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

        with st.spinner("Training the LSTM model. First run can take a few minutes..."):
            model.fit(
                x_train, y_train, epochs=epochs, batch_size=32,
                validation_split=0.1, callbacks=[early_stop], verbose=0,
            )
            pred_scaled = model.predict(x_test, verbose=0)
            pred = scaler.inverse_transform(pred_scaled).flatten()
            actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))
        col1, col2, col3 = st.columns(3)
        col1.metric("Test RMSE", f"{rmse:,.2f}")
        col2.metric("Test MAE", f"{mae:,.2f}")
        col3.metric("Test observations", f"{len(actual):,}")

        st.subheader("Model check: actual vs predicted")
        test_dates = df.index[split:]
        comparison = pd.DataFrame(
            {"Actual close": actual, "Predicted close": pred},
            index=test_dates[:len(actual)]
        )
        st.line_chart(comparison)

        # Iterative multi-step forecast; later days compound model error.
        current_window = scaled[-window:].reshape(1, window, 1)
        future_scaled = []
        with st.spinner(f"Creating the next {forecast_days} business-day forecast..."):
            for _ in range(forecast_days):
                next_value = float(model.predict(current_window, verbose=0)[0, 0])
                future_scaled.append(next_value)
                current_window = np.concatenate(
                    [current_window[:, 1:, :], np.array(next_value, dtype="float32").reshape(1, 1, 1)],
                    axis=1,
                )
        future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast = pd.DataFrame({"Predicted close": future_prices}, index=future_dates)
        forecast.index.name = "Forecast date"
        change_pct = ((float(future_prices[-1]) - latest) / latest * 100) if latest else 0.0

        st.subheader(f"🔮 Future forecast — next {forecast_days} business days")
        f1, f2, f3 = st.columns(3)
        f1.metric("Last forecast estimate", f"{future_prices[-1]:,.2f}", f"{change_pct:+.2f}% vs latest close")
        f2.metric("Highest forecast estimate", f"{np.max(future_prices):,.2f}")
        f3.metric("Lowest forecast estimate", f"{np.min(future_prices):,.2f}")
        st.line_chart(forecast)
        st.dataframe(forecast.style.format({"Predicted close": "{:,.2f}"}), use_container_width=True)
        st.download_button(
            "⬇️ Download future forecast CSV",
            data=forecast.to_csv().encode("utf-8"),
            file_name=f"{ticker.lower().replace('.', '_')}_future_forecast.csv",
            mime="text/csv",
        )
        st.info("Important: these are model-generated estimates, not guaranteed future prices. Multi-day forecasts can become less reliable over time. Do not use this alone for investment decisions.")
    except Exception as exc:
        st.error("Prediction failed. Check the deployment logs for details.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.caption("If the error mentions TensorFlow or Python, fix the Python version in Streamlit Cloud before retrying.")
else:
    st.markdown("## Welcome to the Stock Price Prediction Project")
    st.write(
        "This seminar project demonstrates how historical stock-price data and an "
        "LSTM (Long Short-Term Memory) deep-learning model can be used to estimate "
        "possible future closing prices. It is designed for learning and experimentation."
    )
    st.markdown("### 🚀 How to use the website")
    st.markdown(
        "1. **Find a stock:** In the left sidebar, type the beginning of a company name, "
        "such as `Tata`, `Reliance`, `Infosys`, or `Apple`, then select a result.\n"
        "2. **Choose historical data:** Select 2 years, 5 years, or 10 years.\n"
        "3. **Set forecast options:** Choose 1–30 future business days and the training epochs.\n"
        "4. **Run prediction:** Click **Train model and predict** and wait for training to finish.\n"
        "5. **Review results:** Check the historical chart, actual-versus-predicted chart, RMSE, MAE, and future forecast.\n"
        "6. **Export results:** Download the forecast as a CSV file."
    )
    st.markdown("### ✨ Main features")
    feature1, feature2, feature3 = st.columns(3)
    with feature1:
        st.markdown("#### 🔎 Stock search")
        st.write("Search supported Indian and US tickers by company name.")
    with feature2:
        st.markdown("#### 🧠 LSTM model")
        st.write("Train a sequence model on historical closing prices and evaluate test predictions.")
    with feature3:
        st.markdown("#### 🔮 Future forecast")
        st.write("View estimates for up to 30 business days and download the results.")
    st.markdown("### 🔬 Project workflow")
    st.code(
        "Select stock → Download historical data → Scale prices → Train LSTM → "
        "Evaluate model → Generate future estimates",
        language="text",
    )
    st.markdown("### 🌱 Future scope")
    st.write(
        "Possible future improvements include technical indicators, model comparisons, "
        "prediction-confidence ranges, company-news or market-sentiment analysis, and optional price alerts. "
        "These are proposed enhancements, not all implemented features."
    )
    st.warning(
        "Educational use only: forecasts are estimates, not guaranteed prices or investment advice. "
        "The current model mainly learns from historical closing prices and may be inaccurate."
    )
