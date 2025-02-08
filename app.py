import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = pickle.load(open("rul_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚀 Spacecraft Predictive Maintenance AI")
st.subheader("Real-time RUL Prediction & Monitoring")

# Function to predict RUL
def predict_rul(new_data, model, scaler, threshold=20):
    if "RUL" in new_data.columns:
        new_data = new_data.drop(columns=["RUL"])

    # Select sensor columns
    sensor_cols = [col for col in new_data.columns if "sensor" in col]
    new_data[sensor_cols] = scaler.transform(new_data[sensor_cols])

    # Drop unnecessary columns
    X_new = new_data.drop(columns=["unit", "time"])

    # Predict RUL
    rul_prediction = model.predict(X_new)[0]
    alert = "⚠️ Maintenance Alert! Component failure expected soon!" if rul_prediction < threshold else "✅ System is healthy."

    return rul_prediction, alert

# Simulate real-time telemetry data
if st.button("📡 Get New Sensor Data"):
    df = pd.read_csv("nasa-cmaps/CMaps/train_FD001.txt", sep=" ", header=None)
    new_sample = df.sample(1)

    predicted_rul, alert_msg = predict_rul(new_sample, model, scaler)

    st.write(f"🔢 **Predicted RUL:** {predicted_rul:.2f} cycles")
    st.warning(alert_msg) if predicted_rul < 20 else st.success(alert_msg)
