import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------
# Load Model
# -------------------------------
model = load_model("stock_GS_modell.h5")  # your trained LSTM model

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="PowerGrid Stock Forecasting", layout="wide")

st.title("⚡ PowerGrid Stock Price Forecasting with LSTM")

# Fixed ticker
ticker = "POWERGRID.NS"

# Fixed start and dynamic end
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime.today()

if st.button("Run Forecast"):
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("❌ No data found for PowerGrid.")
    else:
        st.success(f"✅ Data loaded for {ticker} from {start_date.date()} to {end_date.date()}")

        # Reset index
        df.reset_index(inplace=True)

        # Moving Averages
        df["MA100"] = df["Close"].rolling(100).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        # Show tail of DataFrame
        st.subheader("📄 Latest Stock Data")
        st.dataframe(df.tail())

        # -------------------------------
        # Plot Closing Price with MAs
        # -------------------------------
        st.subheader("📊 Closing Price with Moving Averages")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Date"], df["Close"], label="Closing Price", color="blue")
        ax.plot(df["Date"], df["MA100"], label="100-Day MA", color="red")
        ax.plot(df["Date"], df["MA200"], label="200-Day MA", color="green")
        ax.set_title("PowerGrid Stock Price with Moving Averages")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------
        # Prepare data for prediction
        # -------------------------------
        data = df[["Date", "Close"]]
        data = data.set_index("Date")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Train-test split
        train_size = int(len(scaled_data) * 0.8)
        test_data = scaled_data[train_size - 100:]

        x_test, y_test = [], []
        for i in range(100, len(test_data)):
            x_test.append(test_data[i - 100:i])
            y_test.append(test_data[i])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Prediction
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)

        # -------------------------------
        # Plot Prediction vs Actual
        # -------------------------------
        st.subheader("📈 Prediction vs Actual")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(y_test, label="Actual Price", color="blue")
        ax1.plot(predictions, label="Predicted Price", color="orange")
        ax1.set_title("PowerGrid Stock Price Prediction vs Actual")
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

        # -------------------------------
        # Forecast next 3 days
        # -------------------------------
        st.subheader("🔮 Next 3 Days Forecast")
        last_100_days = scaled_data[-100:]
        temp_input = list(last_100_days)
        future_prices = []

        for _ in range(3):  # Only 3 days forecast
            pred_price = model.predict(np.array(temp_input[-100:]).reshape(1, 100, 1))
            future_prices.append(pred_price[0, 0])
            temp_input.append([pred_price[0, 0]])

        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

        future_dates = pd.date_range(start=end_date, periods=4, freq="B")[1:]
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_prices.flatten()})

        st.dataframe(forecast_df)

        # -------------------------------
        # Plot Forecast
        # -------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(forecast_df["Date"], forecast_df["Forecast"], marker="o", linestyle="--", color="red", label="Forecast")
        ax2.set_title("PowerGrid 3-Day Forecast")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)
