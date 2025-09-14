import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("stock_GS_model.h5")

st.title("📈 Stock Price Predictor (LSTM)")

# File uploader for stock data
uploaded_file = st.file_uploader("Upload stock data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Stock Data (Head)")
    st.write(df.head())

    # --- Preprocessing ---
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # --- Prediction ---
    y_predicted = model.predict(x_test)

    # Rescale back to original prices
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # --- Plot ---
    st.subheader("Actual vs Predicted Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Original Price')
    ax.plot(y_predicted, label='Predicted Price')
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Prediction complete!")
